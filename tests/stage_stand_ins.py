"""Functions for the stage tests to run in a child process.

`stage.call` takes a `module:function` and the child imports it, so a stand-in has
to live in a module the child can find: the tests put this folder on PYTHONPATH.
Nothing here loads a model. The names and the text are invented.
"""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path


def chatter(report, lines, value=None):
    """Print what the real stage prints, report something, and hand a value back."""
    for line in lines:
        print(line, flush=True)
    report("transcription: a stand-in in it (CPU, int8)")
    return {"pid": os.getpid(), "value": value}


def raises(report, kind, message):
    exc = {"ValueError": ValueError, "RuntimeError": RuntimeError,
           "FileNotFoundError": FileNotFoundError, "KeyError": KeyError,
           "OSError": OSError}[kind]
    raise exc(message)


def dies(report, how):
    """End without saying anything: the way an out-of-memory kill looks from outside."""
    if how == "kill":
        os.kill(os.getpid(), signal.SIGKILL)
    os._exit(3)


def sleeps(report, pidfile):
    """Say who it is and then hold on, as a stage holding the GPU would."""
    Path(pidfile).write_text(str(os.getpid()))
    report("started")
    time.sleep(120)


def fake_asr(report, wav, settings, out):
    """What scriba.asr.transcribe_to_file does, without the models."""
    print("Progress: 50.00%...", flush=True)
    report("transcription: a stand-in in " + settings["language"] + " (CPU, int8)")
    Path(out).with_name("child-pids.txt").open("a").write(f"{os.getpid()}\n")
    Path(out).write_text(json.dumps({
        "segments": [{"start": 0.0, "end": 2.0, "text": "Ada parla del lago."}],
        "language": settings["language"], "word_level": False,
    }))


def fake_asr_out_of_memory(report, wav, settings, out):
    raise RuntimeError("MPS backend out of memory (MPS allocated: 4.07 GiB)")
