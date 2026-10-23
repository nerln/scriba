"""Tests for words.json, the track the app's player reads.

Invented text throughout.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from scriba import diarize


def test_every_word_keeps_its_time_and_speaker():
    segs = [{"start": 5.0, "end": 7.0, "speaker": "SPEAKER_01", "text": "see you",
             "words": [{"word": "see", "start": 5.0, "end": 5.3, "speaker": "SPEAKER_01"},
                       {"word": "you", "start": 5.4, "end": 5.8, "speaker": "SPEAKER_00"}]},
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "hi Ada",
             "words": [{"word": "hi", "start": 0.0, "end": 0.4},
                       {"word": "Ada", "start": 0.5, "end": 0.9}]}]
    assert diarize.word_track(segs) == [
        [0.0, 0.4, "hi", "SPEAKER_00"], [0.5, 0.9, "Ada", "SPEAKER_00"],
        [5.0, 5.3, "see", "SPEAKER_01"], [5.4, 5.8, "you", "SPEAKER_00"]]


def test_a_word_the_aligner_could_not_place_stays_in_order():
    segs = [{"start": 1.0, "end": 3.0, "speaker": "A", "text": "it was 42",
             "words": [{"word": "it", "start": 1.0, "end": 1.2},
                       {"word": "was", "start": 1.3, "end": 1.5}, {"word": "42"}]}]
    assert diarize.word_track(segs)[-1] == [1.5, 1.5, "42", "A"]


def test_a_segment_without_words_is_one_entry():
    segs = [{"start": 2.0, "end": 4.0, "speaker": "B", "text": " Lake trip. "}]
    assert diarize.word_track(segs) == [[2.0, 4.0, "Lake trip.", "B"]]


def test_the_app_can_ask_for_the_track_of_an_older_job(tmp_path, monkeypatch):
    from scriba import config as cfg
    from scriba import pipeline
    from scriba.cli import app
    from scriba.pipeline import Job

    # JOBS_DIR is read when the modules load, before the fixture moves SCRIBA_HOME:
    # without this the job lands in the real ~/.scriba (it did, once).
    monkeypatch.setattr(cfg, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(pipeline, "JOBS_DIR", tmp_path / "jobs")

    source = tmp_path / "Lake trip.m4a"
    source.write_bytes(b"\0" * 2048)
    job = Job(source, report=lambda m: None)
    job.dir.mkdir(parents=True, exist_ok=True)
    (job.dir / "transcript.json").write_text(json.dumps({"word_level": True, "segments": [
        {"start": 0.0, "end": 1.0, "text": "hi Rafiq",
         "words": [{"word": "hi", "start": 0.0, "end": 0.3},
                   {"word": "Rafiq", "start": 0.4, "end": 0.9}]}]}))
    (job.dir / "diarization.json").write_text(json.dumps(
        [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]))

    result = CliRunner().invoke(app, ["words", str(source)])
    assert result.exit_code == 0, result.output
    track = json.loads((job.dir / "words.json").read_text())
    assert track == [[0.0, 0.3, "hi", "SPEAKER_00"], [0.4, 0.9, "Rafiq", "SPEAKER_00"]]
    assert result.output.strip().endswith("words.json")
