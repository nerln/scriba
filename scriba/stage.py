"""Run one stage of a job in a process of its own, so that what it took goes back.

Transcription and alignment put gigabytes on the GPU: CTranslate2's Metal backend
for whisper, and MPSGraph programs that torch keeps for the life of the process,
one per input shape, for the aligner (whisperX feeds it a differently sized
segment every time, hundreds of shapes on a long file). None of that comes back
while the process lives. `del`, `gc.collect()` and `torch.mps.empty_cache()` do not
reach it. On a 16 GB Mac an 86 minute recording went through transcription and
failed at diarization, the next stage, with "MPS backend out of memory (MPS
allocated: 4.07 GiB, other allocations: 15.96 GiB)"; the rerun, which loaded the
cached transcript and never touched whisper, went through. A process that exits
gives everything back, so the stage that loads those models runs in one, and the
stage after it starts in a process that never held any of it.

`call` runs `module:function` in a child, hands it the keyword arguments as JSON,
and returns what it returned. Four things have to stay true across that boundary,
and each is here for a reason:

  Output. The child inherits stdout and stderr, so its lines (whisper's
  "Progress: N%", the phase lines) reach the same pipe unchanged. The macOS app
  reads them for its progress bar and cannot tell a child from the parent.
  What the stage reports through `report` travels on a pipe of its own and comes
  out of the parent's own `report`, so it is formatted by whoever owns the output.

  Failure. An exception in the child arrives in the parent with the same message,
  and the types the command line knows how to explain (ValueError, RuntimeError,
  FileNotFoundError) keep their type. Everything else becomes a RuntimeError
  carrying the same text, because the app shows the reason and a traceback is not
  one; the child prints its own traceback to stderr for those, as before.

  Stopping. The app's Stop sends SIGTERM to the parent only, and Python's default
  for SIGTERM is to die on the spot. Without more, the child would carry on as an
  orphan with the GPU, and the app, which waits for the pipes to close, would hang
  on it. So the parent catches SIGTERM for as long as the child lives, kills the
  child, waits until it is gone, and then dies the way it would have. For the
  deaths that cannot be caught (SIGKILL, a crash, a force quit) the child holds
  the read end of a pipe whose write end only the parent has, and exits when it
  reads end-of-file.

  Tests. They monkeypatch the stage function, and a patch does not cross a process
  boundary. SCRIBA_IN_PROCESS=1 runs the stage in the calling process instead.
  tests/conftest.py sets it; it is also what to use to debug a stage with pdb.
"""

from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

Reporter = Callable[[str], None]

IN_PROCESS = "SCRIBA_IN_PROCESS"

# The failures the command line explains in one line (cli._run_one and cli.main).
# Only these are rebuilt as themselves in the parent.
_EXPECTED: dict[str, type[Exception]] = {
    "ValueError": ValueError,
    "RuntimeError": RuntimeError,
    "FileNotFoundError": FileNotFoundError,
}

# What the child runs. `-c` and not `-m scriba.stage`: run as a script this module
# would be loaded twice, once as __main__ and once as scriba.stage. The folder
# goes on sys.path first so that the child imports the same copy of scriba the
# parent is running, whatever the working directory holds. The app sets
# PYTHONPATH and a working directory; a terminal may set neither.
_ENTRY = ("import sys; sys.path.insert(0, {root!r}); "
          "from scriba.stage import child_main; child_main()")


def _resolve(target: str) -> Callable[..., Any]:
    module, _, name = target.partition(":")
    return getattr(importlib.import_module(module), name)


def call(target: str, args: dict[str, Any], *, report: Reporter,
         what: str = "stage") -> Any:
    """Run `module:function(report=..., **args)` in a child process and return its value.

    `args` and the return value have to be JSON. `what` names the stage in the
    message for a child that vanished without saying why.
    """
    if os.environ.get(IN_PROCESS):
        return _resolve(target)(report=report, **args)

    report_r, report_w = os.pipe()
    life_r, life_w = os.pipe()
    spec = json.dumps({"target": target, "args": args,
                       "report_fd": report_w, "lifeline_fd": life_r})
    entry = _ENTRY.format(root=str(Path(__file__).resolve().parent.parent))
    try:
        proc = subprocess.Popen([sys.executable, "-c", entry, spec],
                                pass_fds=(report_w, life_r))
    except BaseException:
        os.close(report_r)
        os.close(life_w)
        raise
    finally:
        # The child has its own copies. Holding these would keep the pipes open
        # after it exits, and the loop below would never see the end of them.
        os.close(report_w)
        os.close(life_r)

    outcome: dict[str, Any] = {}
    restore = _stop_child_on_sigterm(proc)
    try:
        with os.fdopen(report_r, "r", encoding="utf-8") as lines:
            for line in lines:
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if "report" in message:
                    report(message["report"])
                else:
                    outcome = message
        proc.wait()
    finally:
        restore()
        if proc.returncode is None:
            # Out through an exception or Ctrl-C with the child still running.
            _stop(proc.pid)
        os.close(life_w)

    if "done" in outcome:
        # Whatever the exit code says. A child that finished its work and then
        # crashed while the GPU libraries were tearing down has still finished it.
        return outcome["done"]
    if "error" in outcome:
        raise _EXPECTED.get(outcome.get("kind", ""), RuntimeError)(outcome["error"])
    raise RuntimeError(f"The {what} process ended without a result: "
                       f"{_how_it_ended(proc.returncode)}.")


def _how_it_ended(code: int) -> str:
    if code >= 0:
        return f"it exited with code {code}"
    if -code == signal.SIGKILL:
        # Nobody sends this by hand. On a Mac it is the system taking a process
        # that used too much memory.
        return "it was killed, most likely by the system for lack of memory"
    try:
        return f"it was stopped by {signal.Signals(-code).name}"
    except ValueError:
        return f"it was stopped by signal {-code}"


def _stop_child_on_sigterm(proc: subprocess.Popen) -> Callable[[], None]:
    """Catch SIGTERM while the child lives; return the function that undoes it."""
    def on_term(signum: int, frame: Any) -> None:
        if proc.returncode is None:  # a child already reaped may have had its pid reused
            _stop(proc.pid)
        # Then die as the signal would have made this process die. The app tells
        # a Stop from a failure by how the process ended.
        signal.signal(signum, previous if previous is not None else signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    try:
        previous = signal.signal(signal.SIGTERM, on_term)
    except ValueError:
        # Not the main thread, where signals cannot be set. The lifeline still
        # takes the child down when this process dies.
        return lambda: None
    return lambda: signal.signal(signal.SIGTERM,
                                 previous if previous is not None else signal.SIG_DFL)


def _stop(pid: int) -> None:
    """Terminate the child and wait until it is gone, so its GPU memory is free.

    os.waitpid and not Popen.wait: this runs from a signal handler, possibly
    interrupting a Popen.wait on the same thread, and that one holds a lock the
    handler's own call would then wait on for ever.
    """
    for sig, patience in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 10.0)):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + patience
        while time.monotonic() < deadline:
            try:
                done, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                return
            if done:
                return
            time.sleep(0.02)


# --------------------------------------------------------------------- the child

def child_main() -> None:
    """The child's side of `call`: read the job from argv, run it, report, exit."""
    spec = json.loads(sys.argv[1])
    channel = os.fdopen(spec["report_fd"], "w", encoding="utf-8")

    def send(**message: Any) -> None:
        try:
            channel.write(json.dumps(message, ensure_ascii=False) + "\n")
            channel.flush()
        except OSError:
            pass  # the parent is gone, and the lifeline is about to end this

    _exit_with_parent(spec["lifeline_fd"])
    code = 0
    try:
        value = _resolve(spec["target"])(report=lambda m: send(report=m),
                                         **spec["args"])
        send(done=value)
    except KeyboardInterrupt:
        code = 130  # Ctrl-C reached the whole foreground group; the parent says so
    except Exception as exc:
        kind = type(exc).__name__
        send(error=str(exc).strip() or kind, kind=kind)
        if kind not in _EXPECTED:
            traceback.print_exc()
        code = 1
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except OSError:
            pass
    # _exit, not a normal return: tearing down torch and CTranslate2 at interpreter
    # exit has crashed before, after the work was done and written, and a crash on
    # the way out turns a finished stage into an exit code nobody asked for.
    os._exit(code)


def _exit_with_parent(fd: int) -> None:
    """End this process when the parent's end of the lifeline closes."""
    def watch() -> None:
        try:
            while os.read(fd, 1):
                pass
        except OSError:
            pass
        os._exit(1)

    threading.Thread(target=watch, name="lifeline", daemon=True).start()
