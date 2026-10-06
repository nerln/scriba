"""Running transcription in a child process (scriba/stage.py).

The reason it exists is a number from a real run: on a 16 GB Mac, an 86 minute
recording failed at diarization with "MPS backend out of memory", because the
Metal memory that whisper and the aligner took stays with the process that took
it. These tests do not load a model and cannot reproduce that. They pin down what
the boundary has to keep true, with stand-in stages from stage_stand_ins.py that
run in a real child process:

  - what the child prints reaches the same stdout, and what it reports comes out
    of the parent's own report;
  - a failure in the child arrives with its message, with the type the command
    line explains;
  - a child that vanishes is said so, in words;
  - a child never outlives its parent: not on SIGTERM (the app's Stop), and not on
    SIGKILL either;
  - a cached transcript starts no child at all.

Every name and sentence is invented.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import dataclasses
import types

from scriba import asr, config, pipeline, stage
from scriba.config import Settings

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent


@pytest.fixture
def in_a_child(monkeypatch):
    """The real thing: no in-process switch, and the stand-ins findable by the child."""
    monkeypatch.delenv(stage.IN_PROCESS, raising=False)
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(TESTS), existing])))


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


# --------------------------------------------------------------------------- #
# what crosses the boundary
# --------------------------------------------------------------------------- #

def test_the_stage_runs_in_another_process_and_hands_its_value_back(in_a_child):
    out = stage.call("stage_stand_ins:chatter", {"lines": [], "value": ["ç", 3]},
                     report=lambda m: None)
    assert out["pid"] != os.getpid()
    assert out["value"] == ["ç", 3]


def test_the_switch_runs_it_in_this_process_instead(monkeypatch):
    monkeypatch.setenv(stage.IN_PROCESS, "1")
    monkeypatch.syspath_prepend(str(TESTS))
    out = stage.call("stage_stand_ins:chatter", {"lines": []}, report=lambda m: None)
    assert out["pid"] == os.getpid()


def test_the_childs_lines_reach_the_same_stdout_unchanged(in_a_child, capfd):
    lines = ["[scriba] transcription: large-v3 on Metal (float16)",
             "Progress: 25.00%...",
             "[scriba] alignment: word-level timestamps (it)",
             "Progress: 100.00%..."]
    stage.call("stage_stand_ins:chatter", {"lines": lines}, report=lambda m: None)
    assert capfd.readouterr().out.splitlines() == lines


def test_what_the_stage_reports_comes_out_of_the_parents_report(in_a_child):
    heard: list[str] = []
    stage.call("stage_stand_ins:chatter", {"lines": []}, report=heard.append)
    assert heard == ["transcription: a stand-in in it (CPU, int8)"]


def test_the_stage_can_run_where_signals_cannot_be_set(in_a_child):
    """A worker thread cannot install the SIGTERM handler. The call must still work."""
    box: dict = {}

    def work():
        box["out"] = stage.call("stage_stand_ins:chatter", {"lines": []},
                                report=lambda m: None)

    t = threading.Thread(target=work)
    t.start()
    t.join(60)
    assert box["out"]["pid"] != os.getpid()


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kind, type_", [
    ("ValueError", ValueError), ("RuntimeError", RuntimeError),
    ("FileNotFoundError", FileNotFoundError),
])
def test_an_error_the_command_line_explains_keeps_its_type_and_message(
        in_a_child, capfd, kind, type_):
    message = "MPS backend out of memory (MPS allocated: 4.07 GiB)\nsecond line, kept"
    with pytest.raises(type_) as caught:
        stage.call("stage_stand_ins:raises", {"kind": kind, "message": message},
                   report=lambda m: None)
    assert str(caught.value) == message
    # No stack trace for a failure that has its explanation already written.
    assert "Traceback" not in capfd.readouterr().err


@pytest.mark.parametrize("kind", ["KeyError", "OSError"])
def test_any_other_error_becomes_a_runtime_error_with_a_traceback_in_the_log(
        in_a_child, capfd, kind):
    with pytest.raises(RuntimeError) as caught:
        stage.call("stage_stand_ins:raises", {"kind": kind, "message": "disk went away"},
                   report=lambda m: None)
    assert "disk went away" in str(caught.value)
    err = capfd.readouterr().err
    assert "Traceback" in err and "stage_stand_ins.py" in err


def test_a_child_killed_for_memory_is_named_as_that(in_a_child):
    with pytest.raises(RuntimeError) as caught:
        stage.call("stage_stand_ins:dies", {"how": "kill"}, report=lambda m: None,
                   what="transcription")
    text = str(caught.value)
    assert text.startswith("The transcription process ended without a result")
    assert "lack of memory" in text


def test_a_child_that_exits_without_a_result_gives_its_exit_code(in_a_child):
    with pytest.raises(RuntimeError, match="exited with code 3"):
        stage.call("stage_stand_ins:dies", {"how": "exit"}, report=lambda m: None)


# --------------------------------------------------------------------------- #
# the child never outlives the parent
# --------------------------------------------------------------------------- #

# A parent of its own, so that the test can kill it. It asks for a stage that
# sleeps, which stands for whisper holding the GPU.
PARENT = """
import sys
sys.path.insert(0, {root!r})
from scriba import stage
stage.call("stage_stand_ins:sleeps", {{"pidfile": {pidfile!r}}}, report=print)
"""


def parent_with_a_sleeping_child(tmp_path, monkeypatch):
    """Start a parent that has started a child; return (parent, child pid)."""
    monkeypatch.delenv(stage.IN_PROCESS, raising=False)
    pidfile = tmp_path / "child.pid"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        filter(None, [str(TESTS), os.environ.get("PYTHONPATH")])))
    parent = subprocess.Popen(
        [sys.executable, "-c", PARENT.format(root=str(ROOT), pidfile=str(pidfile))],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 60
    while not (pidfile.exists() and pidfile.read_text().strip()):
        assert parent.poll() is None, parent.stderr.read().decode()
        assert time.monotonic() < deadline, "the child never started"
        time.sleep(0.05)
    return parent, int(pidfile.read_text())


def test_sigterm_to_the_parent_takes_the_child_with_it(tmp_path, monkeypatch):
    """The app's Stop. The child must be gone by the time the parent is."""
    parent, child = parent_with_a_sleeping_child(tmp_path, monkeypatch)
    try:
        parent.send_signal(signal.SIGTERM)
        parent.wait(timeout=60)
        # Not "soon": the parent waited for it, and an orphan holding the GPU
        # while the next run starts is exactly what this is for.
        assert not alive(child)
        # And it died the way an unhandled SIGTERM does, which is how the app
        # tells a Stop from a failure.
        assert parent.returncode == -signal.SIGTERM
    finally:
        if alive(child):
            os.kill(child, signal.SIGKILL)
        parent.stderr.close()


def test_sigkill_to_the_parent_still_takes_the_child_with_it(tmp_path, monkeypatch):
    """A force quit or a crash: nothing in the parent runs, the child notices."""
    parent, child = parent_with_a_sleeping_child(tmp_path, monkeypatch)
    try:
        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=60)
        deadline = time.monotonic() + 30
        while alive(child) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not alive(child), "the child outlived a parent that was killed"
    finally:
        if alive(child):
            os.kill(child, signal.SIGKILL)
        parent.stderr.close()


# --------------------------------------------------------------------------- #
# the job
# --------------------------------------------------------------------------- #

@pytest.fixture
def job(tmp_path, monkeypatch, in_a_child):
    root = tmp_path / "scriba-home"
    jobs = root / "jobs"
    jobs.mkdir(parents=True)
    monkeypatch.setenv("SCRIBA_HOME", str(root))
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(config, "VOICES_DIR", root / "voices")
    monkeypatch.setattr(config, "JOBS_DIR", jobs)
    monkeypatch.setattr(pipeline, "JOBS_DIR", jobs)
    monkeypatch.setattr(pipeline, "ASR_STAGE", "stage_stand_ins:fake_asr")

    source = tmp_path / "memos" / "Lake trip.m4a"
    source.parent.mkdir()
    source.write_bytes(b"not audio, nothing decodes it")

    log: list[str] = []
    j = pipeline.Job(source, Settings(language="it"), report=log.append)
    j.log = log  # type: ignore[attr-defined]
    return j


def children_of(job) -> list[int]:
    path = job.dir / "child-pids.txt"
    return [int(p) for p in path.read_text().split()] if path.exists() else []


def test_a_transcript_is_made_by_a_child_and_read_back(job, capfd):
    segments = job.transcribe()

    assert segments == [{"start": 0.0, "end": 2.0, "text": "Ada parla del lago."}]
    (child,) = children_of(job)
    assert child != os.getpid()
    # The transcript is where the cache looks for it, and the state describes it.
    on_disk = json.loads((job.dir / "transcript.json").read_text())
    assert on_disk["segments"] == segments
    assert job.state["asr_fingerprint"] == job._asr_fingerprint()
    assert job.state["word_level"] is False
    # Its output and its report both arrived.
    assert "Progress: 50.00%..." in capfd.readouterr().out
    assert "transcription: a stand-in in it (CPU, int8)" in job.log


def test_a_cached_transcript_starts_no_child(job):
    job.transcribe()
    assert len(children_of(job)) == 1

    again = pipeline.Job(job.source, Settings(language="it"), report=job.log.append)
    assert again.transcribe() == [{"start": 0.0, "end": 2.0, "text": "Ada parla del lago."}]
    assert "transcription: reusing the cache" in job.log
    assert len(children_of(job)) == 1, "a cache hit started a process"


def test_a_changed_setting_or_force_starts_a_new_child(job):
    job.transcribe()
    job.transcribe(force=True)
    assert len(children_of(job)) == 2
    assert len(set(children_of(job))) == 2

    other = pipeline.Job(job.source, Settings(language="it", beam_size=1),
                         report=job.log.append)
    other.transcribe()
    assert len(children_of(job)) == 3


def test_a_failing_child_surfaces_in_the_job_and_leaves_no_transcript(
        job, monkeypatch):
    monkeypatch.setattr(pipeline, "ASR_STAGE", "stage_stand_ins:fake_asr_out_of_memory")
    with pytest.raises(RuntimeError, match="MPS backend out of memory"):
        job.transcribe()
    assert not (job.dir / "transcript.json").exists()
    assert "asr_fingerprint" not in job.state


# --------------------------------------------------------------------------- #
# the stage itself, and the cheap complements in asr.py
# --------------------------------------------------------------------------- #

def test_the_transcription_stage_reports_where_it_runs_and_writes_the_file(
        tmp_path, monkeypatch):
    seen: list[Settings] = []

    def fake(wav, s, **kw):
        seen.append(s)
        return asr.Transcript(segments=[{"start": 0.0, "end": 1.0, "text": "Ciao."}],
                              language="it", word_level=True)

    monkeypatch.setattr(asr, "transcribe", fake)
    heard: list[str] = []
    out = tmp_path / "transcript.json"

    asr.transcribe_to_file(
        str(tmp_path / "a.wav"),
        dataclasses.asdict(Settings(asr_device="cpu", model="small", language="it")),
        str(out), report=heard.append)

    assert heard == ["transcription: small in it (CPU, int8)"]
    assert seen[0].model == "small"
    assert json.loads(out.read_text()) == {
        "segments": [{"start": 0.0, "end": 1.0, "text": "Ciao."}],
        "language": "it", "word_level": True}


def test_the_decoder_is_asked_to_unload_before_it_is_dropped():
    calls: list[str] = []
    ct2 = types.SimpleNamespace(unload_model=lambda: calls.append("unload"))
    pipeline_ = types.SimpleNamespace(model=types.SimpleNamespace(model=ct2))

    asr._unload_whisper(pipeline_)

    assert calls == ["unload"]


def test_a_build_without_unload_model_just_skips_it():
    asr._unload_whisper(types.SimpleNamespace())          # no .model at all
    asr._unload_whisper(types.SimpleNamespace(
        model=types.SimpleNamespace(model=object())))     # nothing to call


def fake_torch(monkeypatch, *, mps: bool, calls: list[str], fail: bool = False):
    def empty_cache():
        calls.append("empty_cache")
        if fail:
            raise RuntimeError("driver said no")

    torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
        mps=types.SimpleNamespace(empty_cache=empty_cache))
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_the_metal_cache_is_emptied_only_where_there_is_metal(monkeypatch):
    calls: list[str] = []
    fake_torch(monkeypatch, mps=True, calls=calls)
    asr._empty_mps_cache()
    assert calls == ["empty_cache"]

    calls.clear()
    fake_torch(monkeypatch, mps=False, calls=calls)
    asr._empty_mps_cache()
    assert calls == []


def test_a_failing_cache_flush_never_fails_the_transcription(monkeypatch):
    calls: list[str] = []
    fake_torch(monkeypatch, mps=True, calls=calls, fail=True)
    asr._empty_mps_cache()          # must not raise
    assert calls == ["empty_cache"]
