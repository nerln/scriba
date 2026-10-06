"""Keep the Mac from going to sleep in the middle of a transcription.

An hour of audio takes a good part of an hour on the CPU, and a folder of memos takes
the night. Run from a terminal that is nobody's foreground, a laptop that sees no
keyboard for a few minutes sleeps, and the run stops where it was: the model half
loaded, the diarizer halfway down the file, and nothing on screen saying why it is
late. The app keeps the Mac awake for the runs it starts. A run started from a shell
had nothing doing it.

macOS has a tool for this. `caffeinate -i` refuses idle sleep, and `-w <pid>` ties
that to a process: it lets go on its own when the process ends, by a clean exit, a
crash or a kill -9, so there is no release to forget and no assertion left behind.

Only idle sleep is held. The lid closing and the Sleep menu still work, because a
person asking for sleep is not the thing being guarded against.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Callable, Mapping

CAFFEINATE = "/usr/bin/caffeinate"

# `SCRIBA_KEEP_AWAKE=0` turns it off, for the person who would rather the laptop
# slept on battery and picked the run up again when the lid opens. Anything else, or
# nothing, leaves it on. The tests set it to 0 so that none of them starts a process.
SWITCH = "SCRIBA_KEEP_AWAKE"
_OFF = {"0", "off", "no", "false"}

# The caffeinate this process started, if it did. Module state because the point is
# one per process: a batch of recordings and a watcher that meets a hundred of them
# all go through Job.run, and each of those asking must find the first one still
# running instead of starting its own.
_held: Any = None


def wanted(
    *,
    platform: str | None = None,
    environ: Mapping[str, str] | None = None,
    runnable: Callable[[str], bool] | None = None,
) -> bool:
    """Should this process keep the Mac awake? Decides and starts nothing.

    Apart from `hold` so that the decision can be checked with every input made up,
    on any machine: the platform, the environment and whether the tool is there are
    all parameters, and the defaults read the real ones.
    """
    platform = sys.platform if platform is None else platform
    environ = os.environ if environ is None else environ
    runnable = (lambda path: os.access(path, os.X_OK)) if runnable is None else runnable
    # caffeinate is macOS's own. Linux has other tools and Windows has none worth
    # guessing at, and a missing binary on those is not an error to report.
    if platform != "darwin":
        return False
    if environ.get(SWITCH, "").strip().lower() in _OFF:
        return False
    return runnable(CAFFEINATE)


def command(pid: int | None = None) -> list[str]:
    """What is run: idle sleep refused for as long as `pid` (this process) lives."""
    return [CAFFEINATE, "-i", "-w", str(os.getpid() if pid is None else pid)]


def hold(*, spawn: Callable[..., Any] | None = None, **decision: Any) -> bool:
    """Keep the Mac awake until this process ends. True when it is being kept awake.

    Once per process: a second call finds the first one running and starts nothing.
    If somebody killed that caffeinate the next call starts another, since what
    matters is that an assertion is held, not that it is the same one.

    Never raises. Staying awake is a convenience, and a transcription that was about
    to run must not fail because a helper could not start. `spawn` and the keywords
    of `wanted` are there for the tests.
    """
    global _held
    if _held is not None and _held.poll() is None:
        return True
    _held = None
    if not wanted(**decision):
        return False
    # Looked up here and not in the signature, so that a test which replaces
    # subprocess.Popen to catch a real start is replacing the one that gets called.
    spawn = subprocess.Popen if spawn is None else spawn
    try:
        _held = spawn(command(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL)
    except OSError:
        return False
    return True
