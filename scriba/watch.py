"""Watched folder: drop an audio file in, a transcript comes out.

It also watches the Voice Memos library, which is where the recordings already
are and is the reason anybody would want this. That folder lives under
`~/Library/Group Containers/group.com.apple.VoiceMemos.shared/` and macOS protects
it: not even your own user can read it until Full Disk Access is granted to the
process doing the reading. `scriba memos` says so in as many words rather than
reporting an empty folder, which is what the operating system's error looks like
from the inside.

Nothing is ever written into a watched folder that will not take it. The ledger of
what has been processed normally sits in the folder itself, so it travels with it;
where the folder is read-only, or belongs to another application, it goes under
~/.scriba/watch instead. Apple's container is not ours to leave files in.

The polling is deliberately naive: a loop with `sleep`. FSEvents would notify sooner.
It also fires *while* a 200 MB file is still being copied, and at that point you would
transcribe half the audio. Here a file enters processing only once its size has stayed
identical for two rounds.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .audio import is_audio
from .config import DATA_DIR, Settings
from .pipeline import Job

DONE_MARK = ".scriba-done"

# Where macOS keeps what the Voice Memos app records. Same path on the machines
# that have the app at all; the app has not moved it in years.
VOICE_MEMOS = (Path.home() / "Library" / "Group Containers"
               / "group.com.apple.VoiceMemos.shared" / "Recordings")

# Where the mirror leaves what it copied out of that library.
#
# The mirror exists so that nothing else has to hold Full Disk Access. macOS
# charges an access to the process that asked for it, so a permission cannot be
# borrowed from Finder or from System Events by driving them: the request comes
# back charged to whoever was driving. It can only be held by a program with its
# own identity, started by launchd, which is what ScribaMemoMirror.app is. This
# folder is ordinary, and reading it needs nothing.
INBOX = DATA_DIR / "inbox"


def readable(folder: Path) -> tuple[bool, str]:
    """Whether this process can list the folder, and what to do when it cannot.

    macOS answers a protected folder with the same error it uses for a missing
    one, so a caller that only checks `exists()` reports an empty library to
    somebody who has two hundred recordings in it.
    """
    folder = Path(folder).expanduser()
    if not folder.exists():
        return False, f"there is no folder at {folder}"
    try:
        next(iter(folder.iterdir()), None)
    except PermissionError:
        return False, (
            f"macOS will not let this process read {folder}.\n"
            "  Give Full Disk Access to whatever runs scriba, in System Settings > "
            "Privacy & Security > Full Disk Access:\n"
            "  the Terminal if you run it from a terminal, or Scriba.app if you use "
            "the window. Then start it again.")
    except OSError as exc:
        return False, f"{folder} cannot be read: {exc}"
    return True, ""


def ledger_for(folder: Path) -> Path:
    """Where to record what has already been processed.

    Inside the folder when the folder will have it, because then the ledger
    travels with the recordings and moving them somewhere else does not cause two
    hundred transcriptions. Under ~/.scriba when it will not, which is the case
    for anything belonging to another application.
    """
    folder = Path(folder).expanduser().resolve()
    inside = folder / DONE_MARK
    try:
        inside.mkdir(exist_ok=True)
        probe = inside / ".writable"
        probe.touch()
        probe.unlink()
        return inside
    except OSError:
        pass
    import hashlib
    digest = hashlib.sha1(str(folder).encode()).hexdigest()[:10]
    outside = DATA_DIR / "watch" / f"{folder.name}-{digest}"
    outside.mkdir(parents=True, exist_ok=True)
    return outside


def watch(
    folder: Path,
    settings: Settings | None = None,
    *,
    interval: float = 5.0,
    report: Callable[[str], None] = print,
    create: bool = True,
) -> None:
    folder = Path(folder).expanduser().resolve()
    if create:
        # Only for a folder that is ours to make. Pointing this at somebody else's
        # library and having it created empty is how you end up watching a folder
        # the recordings are not in.
        folder.mkdir(parents=True, exist_ok=True)
    ok, why = readable(folder)
    if not ok:
        raise PermissionError(why)
    done_dir = ledger_for(folder)

    seen: dict[Path, int] = {}
    processed: set[str] = {p.name for p in done_dir.glob("*")}
    # Files that failed during this run, and the size they had when they did.
    #
    # In memory rather than on disk: not retrying every five seconds is right,
    # never retrying is not. Most of the ways this fails are the same for every
    # file and get fixed between runs, and one marker on disk cannot tell "this
    # file is unusable" from "ffmpeg was not on the PATH that afternoon". A folder
    # full of recordings and no transcripts, with nothing in the log, is the worst
    # version of this.
    #
    # The size is what makes the obvious fix work. Somebody whose export came out
    # truncated re-exports it and drops it in under the same name, which is the
    # first thing anyone tries. Keyed by name alone that does nothing until the
    # watcher is restarted; keyed by name and size, different bytes get a fresh
    # attempt and identical bytes still do not.
    failed: dict[str, int] = {}

    report(f"watching {folder}  (ctrl-c to stop)")
    if done_dir.parent != folder:
        report(f"keeping the list of what is done in {done_dir}, because that "
               "folder is not ours to write in")
    if processed:
        report(f"{len(processed)} files already processed earlier: skipping them")

    while True:
        try:
            for path in sorted(folder.iterdir()):
                try:
                    if not path.is_file() or not is_audio(path):
                        continue
                    if path.name in processed:
                        continue
                    size = path.stat().st_size
                except OSError:
                    # Moved, renamed or evicted by iCloud between the listing and
                    # here. It used to take the watcher down with it, and a watcher
                    # that has silently exited looks exactly like one with nothing
                    # to do. is_file() stats too, so it is inside the guard.
                    seen.pop(path, None)
                    continue

                if failed.get(path.name) == size:
                    continue
                if seen.get(path) != size:
                    # Still being copied (or still syncing from iCloud): retry next round.
                    seen[path] = size
                    continue

                report(f"── {path.name}")
                try:
                    job = Job(path, settings, report=lambda m: report(f"   {m}"))
                    res = job.run()
                    report(f"   {len(res.outputs)} files written")
                    if res.unresolved:
                        report(f"   unidentified voices: {', '.join(res.unresolved)} "
                               f"(read {res.dossier_path.name} and use `scriba name`)")
                    (done_dir / path.name).touch()
                    processed.add(path.name)
                except Exception as exc:
                    report(f"   error: {exc}")
                    report("   left in place: it will be tried again next time "
                           "you start the watcher")
                    failed[path.name] = size
                finally:
                    seen.pop(path, None)

            time.sleep(interval)
        except KeyboardInterrupt:
            report("\nstopping.")
            return
