"""Apple's Audio Mix on recordings made with Spatial Audio, and the audio recipe.

The helper itself needs a real Spatial Audio recording, which has no place in a
public repository; these tests stand in for it and check what scriba does with
its answers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scriba import audio, spatial
from scriba import config as cfg
from scriba import pipeline
from scriba.config import Settings


@pytest.fixture
def job_home(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(pipeline, "JOBS_DIR", tmp_path / "jobs")
    return tmp_path


def _job(job_home, name="Lake trip.qta", **settings) -> pipeline.Job:
    src = job_home / name
    src.write_bytes(b"\0" * 2048)
    notes: list[str] = []
    job = pipeline.Job(src, Settings(**settings), report=notes.append)
    job.notes = notes
    return job


def _spatial(monkeypatch, found: dict, tracks=(1,)):
    monkeypatch.setattr(spatial, "info", lambda path: found)
    monkeypatch.setattr(audio, "layers", lambda path: list(tracks))


SPATIAL = {"spatial": True, "supported": True, "default_style": "standard",
           "default_intensity": 0.5}


def test_off_reads_the_stereo_track(job_home, monkeypatch):
    _spatial(monkeypatch, SPATIAL)
    assert _job(job_home)._audio_recipe() == "plain"


def test_studio_takes_the_intensity_the_phone_stored(job_home, monkeypatch):
    _spatial(monkeypatch, SPATIAL)
    assert _job(job_home, audio_mix="studio")._audio_recipe() == "studio@0.50"
    assert _job(job_home, name="Lake trip 2.qta", audio_mix="studio",
                audio_mix_intensity=1.0)._audio_recipe() == "studio@1.00"


def test_a_recording_without_spatial_audio_says_so(job_home, monkeypatch):
    _spatial(monkeypatch, {"spatial": False}, tracks=(0,))
    job = _job(job_home, audio_mix="studio")
    assert job._audio_recipe() == "plain"
    assert any("no Spatial Audio" in n for n in job.notes)


def test_a_mac_that_cannot_render_says_why(job_home, monkeypatch):
    _spatial(monkeypatch, {"spatial": False, "why": "needs macOS 26 for Audio Mix"})
    job = _job(job_home, audio_mix="studio")
    assert job._audio_recipe() == "plain"
    assert any("needs macOS 26" in n for n in job.notes)


def test_a_layered_recording_is_its_own_recipe(job_home, monkeypatch):
    _spatial(monkeypatch, {"spatial": False}, tracks=(0, 1))
    assert _job(job_home, layer_mix=0.7)._audio_recipe() == "layers@0.70"


def test_other_files_never_ask_the_helper(job_home, monkeypatch):
    def refuse(path):
        raise AssertionError("asked the Audio Mix helper about an .m4a")
    monkeypatch.setattr(spatial, "info", refuse)
    assert _job(job_home, name="memo.m4a", audio_mix="studio")._audio_recipe() == "plain"


def test_the_mix_is_rendered_then_converted_then_thrown_away(job_home, monkeypatch):
    _spatial(monkeypatch, SPATIAL)
    calls = []

    def render(src, dest, style, intensity):
        calls.append(("render", style, intensity))
        dest.write_bytes(b"RIFF")
        return dest

    def prepare(src, dest, **kw):
        calls.append(("prepare", src.name))
        dest.write_bytes(b"RIFF")
        return dest

    monkeypatch.setattr(spatial, "render", render)
    monkeypatch.setattr(audio, "prepare", prepare)
    monkeypatch.setattr(audio, "probe", lambda p: audio.AudioInfo(p, 60.0, 48000, 2, "aac"))
    monkeypatch.setattr(audio, "recorded_at", lambda info: (None, ""))

    job = _job(job_home, audio_mix="studio", audio_mix_intensity=1.0)
    job.prepare_audio()

    assert calls == [("render", "studio", 1.0), ("prepare", "audio-mix.wav")]
    assert not (job.dir / "audio-mix.wav").exists()
    assert job.state["audio_recipe"] == "studio@1.00"


def test_a_changed_mix_redoes_the_audio_and_what_was_heard(job_home, monkeypatch):
    _spatial(monkeypatch, SPATIAL)
    job = _job(job_home, audio_mix="studio")
    job.dir.mkdir(parents=True, exist_ok=True)
    for name in ("audio16k.wav", "transcript.json", "diarization.json", "words.json"):
        (job.dir / name).write_text("{}")
    job.state.update({"audio_recipe": "plain", "language": "it",
                      "names": {"SPEAKER_00": "Ada"}})

    job._drop_changed_recipe()

    assert not (job.dir / "audio16k.wav").exists()
    assert not (job.dir / "transcript.json").exists()
    assert not (job.dir / "words.json").exists()
    assert "language" not in job.state and "names" not in job.state
    assert any("plain last time and as studio@0.50 now" in n for n in job.notes)


def test_the_same_mix_keeps_the_cache(job_home, monkeypatch):
    _spatial(monkeypatch, SPATIAL)
    job = _job(job_home, audio_mix="studio")
    job.dir.mkdir(parents=True, exist_ok=True)
    (job.dir / "audio16k.wav").write_text("{}")
    job.state["audio_recipe"] = "studio@0.50"

    job._drop_changed_recipe()
    assert (job.dir / "audio16k.wav").exists()


def test_an_older_job_without_a_recipe_counts_as_plain(job_home, monkeypatch):
    _spatial(monkeypatch, SPATIAL)
    job = _job(job_home)
    job.dir.mkdir(parents=True, exist_ok=True)
    (job.dir / "audio16k.wav").write_text("{}")

    job._drop_changed_recipe()
    assert (job.dir / "audio16k.wav").exists()


@pytest.mark.parametrize("field, value", [("audio_mix", "loud"),
                                          ("audio_mix_intensity", 1.5),
                                          ("layer_mix", -0.1)])
def test_settings_refuse_what_the_sliders_cannot_do(field, value):
    with pytest.raises(ValueError, match=field):
        Settings(**{field: value}).validate()


def test_render_refuses_an_unknown_style(tmp_path):
    with pytest.raises(ValueError, match="unknown Audio Mix style"):
        spatial.render(tmp_path / "a.qta", tmp_path / "out.wav", "loud", 0.5)


def test_info_on_a_mac_that_cannot_run_it(monkeypatch, tmp_path):
    monkeypatch.setattr(spatial, "available", lambda: (False, "needs macOS 26"))
    assert spatial.info(tmp_path / "a.qta") == {"spatial": False, "why": "needs macOS 26"}


def test_the_helper_answer_is_read(monkeypatch, tmp_path):
    monkeypatch.setattr(spatial, "available", lambda: (True, ""))
    monkeypatch.setattr(spatial, "compile_helper", lambda *a: Path("/bin/helper"))

    class Done:
        returncode = 0
        stdout = json.dumps(SPATIAL)

    monkeypatch.setattr(spatial.subprocess, "run", lambda cmd, **kw: Done())
    assert spatial.info(tmp_path / "a.qta") == SPATIAL
