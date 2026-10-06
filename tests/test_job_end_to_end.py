"""The middle of the pipeline: a speaker label turning into a name in a document.

Every other file in this suite aims at an edge. `test_voices` proves the matching
policy decides correctly, `test_export` proves the writer formats correctly, and
between the two sits `Job.run`, which carries a verdict from one to the other:
match -> profile -> names -> meta["unresolved"] -> the written file. Nothing tested
that carry. A change that promoted a borderline candidate to a confirmed name would
have left both edges green while putting somebody's name on words they may not have
said, which is the single worst thing this tool can do.

So these tests run the real `Job.run()` with only the expensive parts faked, and
then assert on the finished document rather than on the dictionaries that produced
it. A `Voce 1` in the transcript is not evidence of anything if the header
announced a name; only the file as a whole is.

Nothing here loads whisper or pyannote, decodes audio or reaches the network:
ffmpeg, the ASR and the diarizer are replaced, and the registry verdicts are
handed in by the test rather than computed from vectors. Every path lives under
tmp_path. Every personal name below is invented.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from scriba import asr, audio, config, diarize as diarize_mod, export, jobs, lang, pipeline
from scriba import voices as V
from scriba.config import Settings

# Small enough to read in a failure message, and the value carries the label so a
# stubbed match can tell which voice it was handed.
DIM = 8
DURATION = 180.0
RECORDED = "2026-03-04T10:15:00Z"

# Invented people. None of these is anybody.
CERTAIN_NAME = "Tobias Venn"
CANDIDATE_NAME = "Marisol Adeyemi"
THIRD_NAME = "Ada Nkemelu"
HAND_NAME = "Ines Kovac"


# --------------------------------------------------------------------------- #
# isolation
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Point every path the pipeline and the registry know about at tmp_path.

    Two import-time bindings have to be rebound by hand: `pipeline` did
    `from .config import JOBS_DIR`, and `voices` computed REGISTRY_PATH and
    EMB_PATH from config.VOICES_DIR. Setting SCRIBA_HOME alone would leave both
    writing into the user's real home, and these tests do enroll voices.
    """
    root = tmp_path / "scriba-home"
    jobs = root / "jobs"
    voices_dir = root / "voices"
    jobs.mkdir(parents=True)
    voices_dir.mkdir(parents=True)

    monkeypatch.setenv("SCRIBA_HOME", str(root))
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(config, "VOICES_DIR", voices_dir)
    monkeypatch.setattr(config, "JOBS_DIR", jobs)
    monkeypatch.setattr(config, "SETTINGS_PATH", root / "settings.json")
    monkeypatch.setattr(pipeline, "JOBS_DIR", jobs)
    monkeypatch.setattr(V, "VOICES_DIR", voices_dir)
    monkeypatch.setattr(V, "REGISTRY_PATH", voices_dir / "registry.json")
    monkeypatch.setattr(V, "EMB_PATH", voices_dir / "embeddings.npy")

    # Checked rather than assumed: a patch that missed would write voice prints
    # into the registry of whoever is running the suite.
    for p in (config.DATA_DIR, config.VOICES_DIR, config.JOBS_DIR,
              config.SETTINGS_PATH, pipeline.JOBS_DIR,
              V.VOICES_DIR, V.REGISTRY_PATH, V.EMB_PATH):
        assert Path(p).is_relative_to(tmp_path)
    return root


@pytest.fixture(autouse=True)
def _no_models(monkeypatch):
    """Any test here that reaches a model is a broken test, not a slow one."""
    def forbidden(*a, **kw):
        raise AssertionError("a test tried to run a real model")

    monkeypatch.setattr(asr, "transcribe", forbidden)
    monkeypatch.setattr(diarize_mod, "run", forbidden)
    monkeypatch.setattr(lang, "detect", forbidden)
    monkeypatch.setattr(pipeline, "hf_token", lambda: None)


@pytest.fixture(autouse=True)
def _no_ffmpeg(monkeypatch):
    """Probe and conversion, without a subprocess and without real audio."""
    def _probe(path):
        return audio.AudioInfo(path=Path(path), duration=DURATION, sample_rate=16_000,
                               channels=1, codec="aac", created=RECORDED)

    def _prepare(src, dest, **kw):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"RIFF" + b"\0" * 40)
        return dest

    monkeypatch.setattr(audio, "probe", _probe)
    monkeypatch.setattr(audio, "prepare", _prepare)


# --------------------------------------------------------------------------- #
# the conversations
# --------------------------------------------------------------------------- #
#
# (label, start, end, text). One diarization turn and one ASR segment per entry,
# with identical boundaries, so `assign` attributes every segment at confidence
# 1.0 and no "(uncertain)" marker appears to muddle the assertions.

TALK = [
    ("SPEAKER_00", 0.0, 40.0,
     "Allora, riprendiamo dal punto rimasto aperto la settimana scorsa."),
    ("SPEAKER_01", 40.0, 80.0,
     "Va bene, pero prima vorrei capire chi segue la raccolta dei dati."),
    ("SPEAKER_00", 80.0, 110.0,
     "Ce ne occupiamo noi, poi passiamo il file a chi deve rivederlo."),
]

TALK_THREE = TALK + [
    ("SPEAKER_02", 110.0, 150.0,
     "Intanto preparo la scaletta per il prossimo incontro e la giro a tutti."),
]

# Eight seconds exactly against five: the boundary of voice_min_speech_sec.
GATE_TALK = [
    ("SPEAKER_00", 0.0, 8.0, "Otto secondi tondi, quanto basta per confrontare la voce."),
    ("SPEAKER_01", 8.0, 13.0, "Cinque secondi soltanto, troppo poco."),
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def marker(label: str) -> float:
    """SPEAKER_00 -> 1.0. Its voice print is that number repeated."""
    return float(int(label.removeprefix("SPEAKER_")) + 1)


def label_of(embedding) -> str:
    """The inverse, so a stubbed `match` knows which voice it was handed."""
    value = int(round(float(np.asarray(embedding).reshape(-1)[0])))
    return f"SPEAKER_{value - 1:02d}"


def certain(name: str, score: float = 0.90) -> V.Match:
    """What VoiceRegistry.match returns when it is sure. `person` is set."""
    return V.Match(person=V.Person(id="p" + name[:2].lower(), name=name),
                   score=score, runner_up=None, runner_up_score=0.0,
                   reason="accepted")


def borderline(name: str, score: float = 0.62) -> V.Match:
    """Above the suggestion threshold, below the certainty one. `person` is None.

    The reason names the candidate, exactly as the real one does. That is on
    purpose: it gives the document a second way to leak the name, and the test
    below says it must not take it.
    """
    return V.Match(person=None, score=score, runner_up=None, runner_up_score=0.0,
                   reason=(f"could be {name} ({score:.3f}). That is under the certainty "
                           "threshold (0.75), so confirm it yourself"),
                   candidate=V.Person(id="c" + name[:2].lower(), name=name))


def stranger() -> V.Match:
    return V.Match(None, 0.31, None, 0.0,
                   "no useful similarity: most likely a new voice")


def stub_registry(monkeypatch, verdicts: dict[str, V.Match],
                  probed: list[str] | None = None) -> None:
    """Decide the registry verdict per speaker label, and record who was asked."""
    def _match(self, embedding, **kw):
        label = label_of(embedding)
        if probed is not None:
            probed.append(label)
        return verdicts.get(label, stranger())

    monkeypatch.setattr(V.VoiceRegistry, "match", _match)


def build_job(tmp_path, monkeypatch, spans, *, settings: Settings | None = None,
              log: list[str] | None = None,
              name: str = "riunione di progetto.m4a") -> pipeline.Job:
    """A job whose diarization is already on disk and whose ASR is a stub.

    Everything else in `run()` is the real thing.
    """
    src = tmp_path / "memos" / name
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"no decoder ever looks at these bytes")

    s = settings if settings is not None else Settings(language="it")
    job = pipeline.Job(src, s,
                       report=(log.append if log is not None else (lambda m: None)))

    turns = [{"start": float(a), "end": float(b), "speaker": label}
             for label, a, b, _ in spans]
    (job.dir / "diarization.json").write_text(json.dumps(turns))
    labels = sorted({label for label, *_ in spans})
    np.savez(job.dir / "embeddings.npz",
             **{label: np.full(DIM, marker(label), dtype=np.float32) for label in labels})
    job.state["diar_fingerprint"] = job._diar_fingerprint()

    def _transcribe(wav, s_, **kw):
        return asr.Transcript(
            segments=[{"start": float(a), "end": float(b), "text": text}
                      for _, a, b, text in spans],
            language=s_.language, word_level=False,
        )

    monkeypatch.setattr(asr, "transcribe", _transcribe)
    return job


def source_doc_of(result: pipeline.JobResult) -> str:
    """The document that matters, read back from disk."""
    hits = [p for p in result.outputs if p.name.endswith("(source).md")]
    assert len(hits) == 1, f"expected one source document, got {hits}"
    return hits[0].read_text()


def payload_of(result: pipeline.JobResult) -> dict:
    hits = [p for p in result.outputs if p.suffix == ".json"]
    assert len(hits) == 1, f"expected one json payload, got {hits}"
    return json.loads(hits[0].read_text())


def match_entry(payload: dict, label: str) -> dict:
    return next(m for m in payload["speaker_matches"] if m["speaker"] == label)


def header_line(doc: str, prefix: str) -> str:
    return next(line for line in doc.splitlines() if line.startswith(prefix))


def labels_in(result: pipeline.JobResult) -> set[str]:
    """The speaker labels the run actually produced, read from turns.json."""
    turns = json.loads((result.job_dir / "turns.json").read_text())
    return {t["speaker"] for t in turns if t.get("speaker")}


# --------------------------------------------------------------------------- #
# 1. the dangerous path: a candidate is not a name
# --------------------------------------------------------------------------- #

def test_a_borderline_candidate_never_reaches_the_document_as_a_name(
        tmp_path, monkeypatch):
    """0.62 is a guess. A guess in a source document is read as a fact.

    The registry says "could be Marisol Adeyemi, confirm it yourself". That
    sentence belongs in the briefing a human reads before deciding. It must not
    appear anywhere in the file that gets handed to a colleague or to a model,
    because whatever reads that file cannot tell a suggestion from an attribution.
    """
    job = build_job(tmp_path, monkeypatch, TALK)
    stub_registry(monkeypatch, {"SPEAKER_00": borderline(CANDIDATE_NAME, 0.62)})

    result = job.run()
    doc = source_doc_of(result)

    # The name is nowhere in the finished document, in any spelling.
    assert CANDIDATE_NAME not in doc
    assert "Marisol" not in doc
    assert "Adeyemi" not in doc

    # And the voice reads as unidentified on its own lines, not only in the header.
    assert "**Voce 1 (non identificata)** [" in doc
    assert "**Voce 1**" not in doc
    assert doc.count("**Voce 1 (non identificata)** [") == 2   # both of its turns

    assert "Non indovinare chi sia" in doc
    assert "Voce 1" in result.unresolved
    assert CANDIDATE_NAME not in result.names.values()
    assert "SPEAKER_00" not in result.names

    # Not a vacuous pass: the candidate did arrive, it was kept out on purpose.
    entry = match_entry(payload_of(result), "SPEAKER_00")
    assert entry["candidate"] == CANDIDATE_NAME
    assert entry["name"] is None
    assert entry["score"] == pytest.approx(0.62)


# --------------------------------------------------------------------------- #
# 2. the mirror: a certain match does reach the document
# --------------------------------------------------------------------------- #

def test_a_certain_match_reaches_the_document_as_a_name(tmp_path, monkeypatch):
    """The other half. Refusing every name would pass the test above and be useless."""
    job = build_job(tmp_path, monkeypatch, TALK)
    stub_registry(monkeypatch, {"SPEAKER_00": certain(CERTAIN_NAME, 0.90)})

    result = job.run()
    doc = source_doc_of(result)

    assert result.names["SPEAKER_00"] == CERTAIN_NAME
    assert doc.count(f"**{CERTAIN_NAME}** [") == 2          # both of its turns
    assert f"{CERTAIN_NAME} (non identificata)" not in doc
    assert "**Voce 1" not in doc                            # it is no longer a voice

    assert CERTAIN_NAME not in result.unresolved
    assert "Voce 1" not in result.unresolved
    assert result.unresolved == ["Voce 2"]                  # the other one still is
    assert "**Voce 2 (non identificata)** [" in doc

    entry = match_entry(payload_of(result), "SPEAKER_00")
    assert entry["name"] == CERTAIN_NAME
    assert entry["score"] == pytest.approx(0.90)


# --------------------------------------------------------------------------- #
# 3. the seam: matches -> profiles -> names -> meta -> the written files
# --------------------------------------------------------------------------- #

def test_the_finished_document_agrees_with_the_job_result(tmp_path, monkeypatch):
    """One recognised voice, one borderline, one stranger, and everything lines up.

    Each assertion here is a joint. They pass individually in every other test
    file and were never checked against each other.
    """
    settings = Settings(language="it")
    job = build_job(tmp_path, monkeypatch, TALK_THREE, settings=settings)
    stub_registry(monkeypatch, {
        "SPEAKER_00": certain(CERTAIN_NAME, 0.90),
        "SPEAKER_01": borderline(CANDIDATE_NAME, 0.62),
    })

    result = job.run()
    doc = source_doc_of(result)
    payload = payload_of(result)
    labels = labels_in(result)

    assert labels == {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02"}
    assert set(result.speakers) == labels

    # One file per configured format, all of them on disk, named as export says.
    stem = pipeline.slugify(job.source.stem)
    assert len(result.outputs) == len(settings.output_formats)
    assert {p.name for p in result.outputs} == {
        export.FILENAMES[f].format(stem=stem) for f in settings.output_formats}
    assert all(p.is_file() for p in result.outputs)

    # Named and unresolved are complementary. Neither invents a speaker.
    named = set(result.names)
    assert named == {"SPEAKER_00"}
    assert set(result.unresolved) == {export.display(l, {}, "it") for l in labels - named}

    # Nobody is both named and listed as unidentified.
    for name in result.names.values():
        assert name not in result.unresolved
        assert f"**{name} (non identificata)**" not in doc

    # Every speaker in the turns is in the participants line, under the name the
    # document uses for it everywhere else.
    participants = header_line(doc, "- **Partecipanti**:")
    for label in labels:
        assert export.display(label, result.names, "it") in participants

    # The header's caveat lists exactly the unresolved voices and nobody else.
    unidentified = header_line(doc, "- **Voci non identificate**:")
    for who in result.unresolved:
        assert who in unidentified
    assert CERTAIN_NAME not in unidentified
    assert CANDIDATE_NAME not in unidentified

    # The machine-readable half says the same thing as the JobResult.
    assert payload["names"] == result.names
    assert payload["meta"]["unresolved"] == result.unresolved
    assert {m["speaker"] for m in payload["speaker_matches"]} == labels
    assert payload["meta"]["speaker_stats"] == {
        "SPEAKER_00": 70.0, "SPEAKER_01": 40.0, "SPEAKER_02": 40.0}

    # And so do the other formats, for the one speaker that has a name.
    plain = next(p for p in result.outputs if p.suffix == ".txt").read_text()
    subtitles = next(p for p in result.outputs if p.suffix == ".srt").read_text()
    assert f"{CERTAIN_NAME}:" in plain
    assert f"[{CERTAIN_NAME}]" in subtitles
    assert CANDIDATE_NAME not in plain
    assert CANDIDATE_NAME not in subtitles


# --------------------------------------------------------------------------- #
# 4. the minimum-speech gate in identify()
# --------------------------------------------------------------------------- #

def test_a_voice_with_too_little_speech_is_never_compared_at_all(tmp_path, monkeypatch):
    """Five seconds of audio is not a voice print, whatever the score says.

    The registry here would name both speakers with high confidence. The gate has
    to stop the thin one before the comparison happens, not filter the answer
    afterwards, so the test watches which vectors reach `match`.

    The other speaker has exactly `voice_min_speech_sec` of speech and is matched,
    which pins the comparison at its boundary.
    """
    settings = Settings(language="it", voice_min_speech_sec=8.0)
    job = build_job(tmp_path, monkeypatch, GATE_TALK, settings=settings)
    probed: list[str] = []
    stub_registry(monkeypatch, {
        "SPEAKER_00": certain(CERTAIN_NAME, 0.93),
        "SPEAKER_01": certain(THIRD_NAME, 0.97),
    }, probed=probed)

    result = job.run()
    doc = source_doc_of(result)

    assert probed == ["SPEAKER_00"], \
        "the thin voice print was compared against the registry anyway"

    # 5 seconds: no name, anywhere.
    assert THIRD_NAME not in doc
    assert "Nkemelu" not in doc
    assert THIRD_NAME not in result.names.values()
    assert "**Voce 2 (non identificata)** [" in doc
    assert "Voce 2" in result.unresolved

    # 8 seconds exactly: matched and named. `<` here, not `<=`.
    assert result.names["SPEAKER_00"] == CERTAIN_NAME
    assert f"**{CERTAIN_NAME}** [" in doc
    assert "Voce 1" not in result.unresolved

    # The refusal is recorded, with its reason, instead of looking like a miss.
    entry = match_entry(payload_of(result), "SPEAKER_01")
    assert entry["name"] is None
    assert entry["candidate"] is None
    assert entry["score"] == 0.0
    assert "only 5s of speech" in entry["reason"]
    assert "too thin to trust" in entry["reason"]


# --------------------------------------------------------------------------- #
# 5. set_names: what it refuses, and what it makes permanent
# --------------------------------------------------------------------------- #

def test_set_names_refuses_to_enroll_under_an_empty_name(tmp_path, monkeypatch):
    """An empty name enrolled is a voice print filed under nobody.

    It would then match the next recording and attach that emptiness to a real
    speaker, and there is no way back from it: the registry cannot tell an
    unnamed print from a wrong one.
    """
    job = build_job(tmp_path, monkeypatch, TALK)
    stub_registry(monkeypatch, {})
    job.run()

    result = job.set_names({"SPEAKER_01": ""})
    doc = source_doc_of(result)

    reg = V.VoiceRegistry()
    assert reg.people == {}
    assert reg.summary() == []
    assert len(reg.emb) == 0

    # And the empty name never becomes a name in the document either.
    assert "SPEAKER_01" not in result.names
    assert "" not in result.names.values()
    assert "**Voce 2 (non identificata)** [" in doc
    assert "Voce 2" in result.unresolved


def test_a_hand_assigned_name_survives_into_the_document_and_the_registry(
        tmp_path, monkeypatch):
    """The point of the whole tool: name a voice once, and it stays named.

    Once here, in the rewritten document, and once for next time, in the registry.
    """
    job = build_job(tmp_path, monkeypatch, TALK)
    stub_registry(monkeypatch, {"SPEAKER_01": borderline(CANDIDATE_NAME, 0.62)})

    first = job.run()
    assert "Voce 2" in first.unresolved
    assert CANDIDATE_NAME not in source_doc_of(first)

    result = job.set_names({"SPEAKER_01": HAND_NAME})
    doc = source_doc_of(result)

    assert result.names["SPEAKER_01"] == HAND_NAME
    assert f"**{HAND_NAME}** [" in doc
    assert f"{HAND_NAME} (non identificata)" not in doc
    assert "**Voce 2" not in doc
    assert HAND_NAME not in result.unresolved
    assert "Voce 2" not in result.unresolved

    # The name outlives the job: this is what makes the next recording easier.
    reg = V.VoiceRegistry()
    person = reg.by_name(HAND_NAME)
    assert person is not None
    assert len(person.rows) == 1
    assert reg.vectors_of(person).tolist() == [[marker("SPEAKER_01")] * DIM]
    assert person.sources == [job.source.name]

    # It reached disk, not just this instance.
    stored = json.loads(V.REGISTRY_PATH.read_text())
    assert [p["name"] for p in stored["people"]] == [HAND_NAME]


# --------------------------------------------------------------------------- #
# 6. what a run was told survives it
# --------------------------------------------------------------------------- #

def run_told_two_speakers(tmp_path, monkeypatch, log=None):
    """A finished job whose caller said "two people in the room"."""
    settings = Settings(language="it", min_speakers=2, max_speakers=2)
    job = build_job(tmp_path, monkeypatch, TALK, settings=settings, log=log)
    stub_registry(monkeypatch, {})
    result = job.run()
    return job, result


def test_the_speaker_count_a_run_was_given_is_kept_with_the_job(tmp_path, monkeypatch):
    job, _ = run_told_two_speakers(tmp_path, monkeypatch)

    assert job.state["speaker_counts"] == {"min": 2, "max": 2}
    # On disk, where the next process reads it.
    assert json.loads(job.state_path.read_text())["speaker_counts"] == {"min": 2, "max": 2}


def test_a_job_built_with_no_count_falls_back_to_the_one_it_was_given(
        tmp_path, monkeypatch):
    """`scriba name` and "Put it back in the queue" never pass a count.

    The cached diarization is keyed on it, so without the fallback they looked at a
    different fingerprint and diarized the whole recording again.
    """
    job, _ = run_told_two_speakers(tmp_path, monkeypatch)

    later = pipeline.Job(job.source, Settings(language="it"))

    assert (later.s.min_speakers, later.s.max_speakers) == (2, 2)
    assert later._diar_fingerprint() == job._diar_fingerprint()


def test_putting_a_job_back_without_a_count_does_not_diarize_again(
        tmp_path, monkeypatch):
    job, _ = run_told_two_speakers(tmp_path, monkeypatch)
    log: list[str] = []

    pipeline.Job(job.source, Settings(language="it"), report=log.append).run()

    # `diarize.run` raises if it is called, so this is also a test that it was not.
    assert "diarization: reusing the cache" in log
    assert not any("diarizing again" in line for line in log)


def test_a_caller_that_gives_a_count_is_heard_and_never_mixed_with_the_old_one(
        tmp_path, monkeypatch):
    job, _ = run_told_two_speakers(tmp_path, monkeypatch)

    own = pipeline.Job(job.source, Settings(language="it", max_speakers=3))

    assert (own.s.min_speakers, own.s.max_speakers) == (None, 3)


def test_the_callers_settings_are_not_changed_by_the_fallback(tmp_path, monkeypatch):
    """One Settings is shared by every file of a batch: the next file must not inherit."""
    job, _ = run_told_two_speakers(tmp_path, monkeypatch)
    shared = Settings(language="it")

    later = pipeline.Job(job.source, shared)

    assert later.s is not shared
    assert (shared.min_speakers, shared.max_speakers) == (None, None)


def test_the_count_is_not_applied_to_a_different_recording_at_the_same_path(
        tmp_path, monkeypatch):
    job, _ = run_told_two_speakers(tmp_path, monkeypatch)
    job.source.write_bytes(b"another conversation, recorded over the first one")

    later = pipeline.Job(job.source, Settings(language="it"))

    assert (later.s.min_speakers, later.s.max_speakers) == (None, None)
    # And a run over the new audio forgets the old count as it forgets the rest.
    later._drop_stale_cache()
    assert "speaker_counts" not in later.state


def test_saving_names_never_runs_a_stage(tmp_path, monkeypatch):
    """Even with settings that would make every cached stage stale.

    The model differs, so the transcript fingerprint no longer matches, and so does
    the speaker count, so the diarization one does not either. A `run()` here
    transcribed and diarized again; the names must go onto what is there.
    """
    job, first = run_told_two_speakers(tmp_path, monkeypatch)

    def forbidden(*a, **kw):
        raise AssertionError("a stage ran while saving names")

    monkeypatch.setattr(asr, "transcribe", forbidden)
    monkeypatch.setattr(diarize_mod, "run", forbidden)
    log: list[str] = []
    later = pipeline.Job(
        job.source,
        Settings(language="it", model="small", min_speakers=3, max_speakers=3),
        report=log.append)

    result = later.set_names({"SPEAKER_01": HAND_NAME})

    doc = source_doc_of(result)
    assert f"**{HAND_NAME}** [" in doc
    assert result.names["SPEAKER_01"] == HAND_NAME
    for stage_line in ("audio:", "language:", "transcription:", "diarization:"):
        assert not any(line.startswith(stage_line) for line in log), log
    # The same speech figures as the run that made the document.
    assert (payload_of(result)["meta"]["speaker_stats"]
            == payload_of(first)["meta"]["speaker_stats"])


def test_the_speech_time_comes_from_the_diarization_on_disk(tmp_path, monkeypatch):
    job, _ = run_told_two_speakers(tmp_path, monkeypatch)

    assert job._cached_speech_time() == {"SPEAKER_00": 70.0, "SPEAKER_01": 40.0}

    (job.dir / "diarization.json").write_text("{ not json")
    assert job._cached_speech_time() is None
    (job.dir / "diarization.json").unlink()
    assert job._cached_speech_time() is None


def test_naming_a_job_that_never_got_as_far_as_documents_says_so(tmp_path, monkeypatch):
    """Nothing to rewrite from: a plain refusal, not a transcription by the back door."""
    job = build_job(tmp_path, monkeypatch, TALK)
    stub_registry(monkeypatch, {})

    with pytest.raises(ValueError, match="nothing cached to write from"):
        job.set_names({"SPEAKER_01": HAND_NAME})

    assert job.state["names"] == {"SPEAKER_01": HAND_NAME}


# --------------------------------------------------------------------------- #
# 6b. a job made before the count was stored
# --------------------------------------------------------------------------- #
#
# The folders on disk today were diarized by code that kept the count only inside
# the fingerprint. There is nothing in them to read it from, but the fingerprint is
# a hash of the count, the model and the device, and the counts a person could have
# given are few enough to hash them all.

def an_old_job(tmp_path, monkeypatch, *, min_speakers=None, max_speakers=None,
               device="auto", log=None):
    """A diarized job as the code before this one left it: a fingerprint, no count."""
    settings = Settings(language="it", min_speakers=min_speakers,
                        max_speakers=max_speakers, diarize_device=device)
    job = build_job(tmp_path, monkeypatch, TALK, settings=settings, log=log)
    job.state["source_fingerprint"] = job._source_fingerprint()
    job._save_state()
    assert "speaker_counts" not in json.loads(job.state_path.read_text())
    return job


def test_a_job_diarized_with_a_count_before_it_was_stored_gets_it_back(
        tmp_path, monkeypatch):
    job = an_old_job(tmp_path, monkeypatch, min_speakers=2, max_speakers=2)

    later = pipeline.Job(job.source, Settings(language="it"))

    assert (later.s.min_speakers, later.s.max_speakers) == (2, 2)
    assert later._diar_fingerprint() == job._diar_fingerprint()
    # Kept like any other count, on disk where the next process reads it.
    assert later.state["speaker_counts"] == {"min": 2, "max": 2}
    assert json.loads(job.state_path.read_text())["speaker_counts"] == {"min": 2, "max": 2}


@pytest.mark.parametrize("low,high", [(None, 3), (2, None), (2, 4), (1, 1), (12, 12)])
def test_either_bound_or_both_are_found_again(tmp_path, monkeypatch, low, high):
    job = an_old_job(tmp_path, monkeypatch, min_speakers=low, max_speakers=high)

    later = pipeline.Job(job.source, Settings(language="it"))

    assert (later.s.min_speakers, later.s.max_speakers) == (low, high)
    assert later.state["speaker_counts"] == {"min": low, "max": high}


def test_putting_such_a_job_back_does_not_diarize_again(tmp_path, monkeypatch):
    """The case this is for: an unfinished job, put back by the app with no count."""
    log: list[str] = []
    job = an_old_job(tmp_path, monkeypatch, min_speakers=2, max_speakers=2)
    stub_registry(monkeypatch, {})

    pipeline.Job(job.source, Settings(language="it"), report=log.append).run()

    # `diarize.run` raises if it is called, so this is also a test that it was not.
    assert "diarization: reusing the cache" in log
    assert not any("diarizing again" in line for line in log)


def test_a_job_diarized_with_no_count_stays_without_one(tmp_path, monkeypatch):
    job = an_old_job(tmp_path, monkeypatch)

    later = pipeline.Job(job.source, Settings(language="it"))

    assert (later.s.min_speakers, later.s.max_speakers) == (None, None)
    assert "speaker_counts" not in later.state
    assert "speaker_counts" not in json.loads(job.state_path.read_text())


def test_a_caller_that_gives_a_count_is_heard_before_a_recovered_one(
        tmp_path, monkeypatch):
    job = an_old_job(tmp_path, monkeypatch, min_speakers=2, max_speakers=2)

    own = pipeline.Job(job.source, Settings(language="it", max_speakers=3))

    assert (own.s.min_speakers, own.s.max_speakers) == (None, 3)
    assert "speaker_counts" not in own.state


def test_a_count_stored_already_is_used_as_it_is(tmp_path, monkeypatch):
    job = an_old_job(tmp_path, monkeypatch, min_speakers=2, max_speakers=2)
    job.state["speaker_counts"] = {"min": 3, "max": 3}
    job._save_state()

    later = pipeline.Job(job.source, Settings(language="it"))

    assert (later.s.min_speakers, later.s.max_speakers) == (3, 3)


def test_the_count_of_another_recording_at_the_same_path_is_not_recovered(
        tmp_path, monkeypatch):
    job = an_old_job(tmp_path, monkeypatch, min_speakers=2, max_speakers=2)
    job.source.write_bytes(b"another conversation, recorded over the first one")

    later = pipeline.Job(job.source, Settings(language="it"))

    assert (later.s.min_speakers, later.s.max_speakers) == (None, None)
    assert "speaker_counts" not in later.state


def test_a_fingerprint_nothing_reproduces_is_left_alone(tmp_path, monkeypatch):
    """A count beyond what the app offers, then a fingerprint of nothing at all."""
    job = an_old_job(tmp_path, monkeypatch, min_speakers=40, max_speakers=40)

    later = pipeline.Job(job.source, Settings(language="it"))
    assert (later.s.min_speakers, later.s.max_speakers) == (None, None)

    job.state["diar_fingerprint"] = "0123456789ab"
    job._save_state()
    later = pipeline.Job(job.source, Settings(language="it"))
    assert (later.s.min_speakers, later.s.max_speakers) == (None, None)
    assert "speaker_counts" not in later.state


def test_a_diarization_made_by_another_model_is_not_credited_with_a_count(
        tmp_path, monkeypatch):
    job = an_old_job(tmp_path, monkeypatch, min_speakers=2, max_speakers=2)
    monkeypatch.setattr(diarize_mod, "DIARIZATION_MODEL", "pyannote/another-model")

    later = pipeline.Job(job.source, Settings(language="it"))

    assert (later.s.min_speakers, later.s.max_speakers) == (None, None)


def test_the_count_is_found_when_the_device_was_changed_since(tmp_path, monkeypatch):
    """cpu is how a suspected Metal problem is checked: with the count it had."""
    job = an_old_job(tmp_path, monkeypatch, min_speakers=2, max_speakers=2, device="auto")

    later = pipeline.Job(job.source, Settings(language="it", diarize_device="cpu"))

    assert (later.s.min_speakers, later.s.max_speakers) == (2, 2)
    assert later.s.diarize_device == "cpu"


def test_a_job_with_no_diarization_has_nothing_to_recover_from(tmp_path, monkeypatch):
    job = build_job(tmp_path, monkeypatch, TALK)
    job.state.pop("diar_fingerprint", None)
    job._save_state()

    later = pipeline.Job(job.source, Settings(language="it"))

    assert (later.s.min_speakers, later.s.max_speakers) == (None, None)
    assert "speaker_counts" not in later.state


# --------------------------------------------------------------------------- #
# 7. a name can be taken away again
# --------------------------------------------------------------------------- #

def test_a_cleared_name_is_removed_from_the_document_and_the_state(
        tmp_path, monkeypatch):
    """The app sends `SPEAKER_01=` for a field the person emptied."""
    job = build_job(tmp_path, monkeypatch, TALK)
    stub_registry(monkeypatch, {})
    job.run()
    job.set_names({"SPEAKER_00": CERTAIN_NAME, "SPEAKER_01": THIRD_NAME})

    result = job.set_names({"SPEAKER_01": ""})
    doc = source_doc_of(result)

    assert "SPEAKER_01" not in result.names
    assert THIRD_NAME not in doc
    assert "**Voce 2 (non identificata)** [" in doc
    assert "Voce 2" in result.unresolved
    # The other voice keeps its name.
    assert result.names == {"SPEAKER_00": CERTAIN_NAME}
    assert f"**{CERTAIN_NAME}** [" in doc
    # On disk, not only in this object.
    stored = json.loads(job.state_path.read_text())
    assert stored["names"] == {"SPEAKER_00": CERTAIN_NAME}


def test_clearing_a_name_that_was_never_set_changes_nothing(tmp_path, monkeypatch):
    job = build_job(tmp_path, monkeypatch, TALK)
    stub_registry(monkeypatch, {})
    job.run()

    result = job.set_names({"SPEAKER_01": ""})

    assert result.names == {}
    assert "Voce 2" in result.unresolved


def test_the_name_command_takes_an_empty_value_as_remove(tmp_path, monkeypatch):
    """The path the app's Save button takes: `scriba name <file> SPEAKER_01=`."""
    from typer.testing import CliRunner

    from scriba import cli

    job = build_job(tmp_path, monkeypatch, TALK)
    stub_registry(monkeypatch, {})
    job.run()
    job.set_names({"SPEAKER_01": THIRD_NAME})
    assert json.loads(job.state_path.read_text())["names"] == {"SPEAKER_01": THIRD_NAME}

    outcome = CliRunner().invoke(cli.app, ["name", str(job.source), "SPEAKER_01="])

    assert outcome.exit_code == 0, outcome.output
    assert json.loads(job.state_path.read_text())["names"] == {}


# --------------------------------------------------------------------------- #
# 8. a second run of the same recording
# --------------------------------------------------------------------------- #

@pytest.fixture
def other_process():
    """A live process that is not this one, standing for another scriba."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait()


def leave_a_note(job, pid: int, **more) -> Path:
    note = job.dir / "running.json"
    note.write_text(json.dumps({"pid": pid, "started": "2026-03-04T10:15:00", **more}))
    return note


def test_a_run_is_refused_while_another_process_is_working_on_the_job(
        tmp_path, monkeypatch, other_process):
    job = build_job(tmp_path, monkeypatch, TALK)
    note = leave_a_note(job, other_process.pid)

    with pytest.raises(RuntimeError) as caught:
        job.run()

    assert str(caught.value) == ("This recording is already being transcribed by "
                                 f"another scriba (process {other_process.pid}).")
    # Nothing of the other run was touched: its note, its state, its folder.
    assert json.loads(note.read_text())["pid"] == other_process.pid
    assert "failed" not in json.loads(job.state_path.read_text())
    for name in ("transcript.json", "turns.json", "output"):
        assert not (job.dir / name).exists()


def test_saving_names_is_refused_the_same_way(tmp_path, monkeypatch, other_process):
    job, _ = run_told_two_speakers(tmp_path, monkeypatch)
    leave_a_note(job, other_process.pid)

    with pytest.raises(RuntimeError, match="already being transcribed"):
        job.set_names({"SPEAKER_01": HAND_NAME})

    assert "SPEAKER_01" not in json.loads(job.state_path.read_text()).get("names", {})


def test_a_note_left_by_a_process_that_is_gone_is_no_obstacle(tmp_path, monkeypatch):
    """What a Stop leaves behind: the engine is killed and never gets to remove it."""
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    job = build_job(tmp_path, monkeypatch, TALK)
    stub_registry(monkeypatch, {})
    note = leave_a_note(job, gone.pid)

    result = job.run()

    assert result.outputs
    assert not note.exists()


def test_a_note_naming_this_very_process_is_not_another_one(tmp_path, monkeypatch):
    job = build_job(tmp_path, monkeypatch, TALK)
    stub_registry(monkeypatch, {})
    leave_a_note(job, os.getpid())

    assert job.run().outputs


def test_the_command_line_says_it_in_a_marker_and_exits_non_zero(
        tmp_path, monkeypatch, other_process):
    from typer.testing import CliRunner

    from scriba import cli

    job = build_job(tmp_path, monkeypatch, TALK)
    leave_a_note(job, other_process.pid)
    monkeypatch.setenv("SCRIBA_MARKERS", "1")

    outcome = CliRunner().invoke(cli.app, ["run", str(job.source)])

    assert outcome.exit_code == 1
    assert ("[scriba:error] This recording is already being transcribed by another "
            f"scriba (process {other_process.pid}).") in outcome.output.splitlines()


# --------------------------------------------------------------------------- #
# 9. the words around the text are in the language of the job
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("code,overview,voice,unidentified", [
    ("it", "## Panoramica", "Voce", "non identificata"),
    ("es", "## Visión general", "Voz", "no identificada"),
    ("en", "## Overview", "Voice", "unidentified"),
    ("de", "## Overview", "Voice", "unidentified"),
])
def test_the_documents_are_written_in_the_language_of_the_job(
        tmp_path, monkeypatch, code, overview, voice, unidentified):
    """One recording, four settings. The text is the same, the frame around it is not."""
    settings = Settings(language=code)
    job = build_job(tmp_path, monkeypatch, TALK, settings=settings)
    stub_registry(monkeypatch, {"SPEAKER_00": certain(CERTAIN_NAME)})

    result = job.run()
    doc = source_doc_of(result)
    payload = payload_of(result)

    assert overview in doc
    assert f"**{voice} 2 ({unidentified})** [" in doc
    assert f"**{CERTAIN_NAME}** [" in doc
    # The list of unidentified voices is the one the document prints, and the
    # language code in the header and the JSON is the code, not a word.
    assert result.unresolved == [f"{voice} 2"]
    assert f"{voice} 2." in header_line(doc, "- **" + export.words(code)["unidentified_voices"])
    assert payload["meta"]["unresolved"] == [f"{voice} 2"]
    assert payload["meta"]["language"] == code
    assert f"- **{export.words(code)['language']}**: {code}" in doc

    # The same label in every format of the job.
    plain = next(p for p in result.outputs if p.suffix == ".txt").read_text()
    subtitles = next(p for p in result.outputs if p.suffix == ".srt").read_text()
    assert f"{voice} 2: " in plain
    assert f"[{voice} 2] " in subtitles

    # What the app and other programs look for is where it always was.
    stem = pipeline.slugify(job.source.stem)
    assert {p.name for p in result.outputs} == {
        export.FILENAMES[f].format(stem=stem) for f in settings.output_formats}
    assert any(p.name == f"{stem} (source).md" for p in result.outputs)
    assert sorted(payload) == ["meta", "names", "segments", "speaker_matches", "turns"]
    assert {t["speaker"] for t in payload["turns"]} == {"SPEAKER_00", "SPEAKER_01"}


def test_a_language_found_by_the_detector_sets_the_words_too(tmp_path, monkeypatch):
    """Nobody says `es` here: the detector hears it, and the document follows."""
    job = build_job(tmp_path, monkeypatch, TALK, settings=Settings(language="auto"))
    stub_registry(monkeypatch, {})
    monkeypatch.setattr(lang, "detect", lambda wav, **kw: lang.LanguageGuess(
        "es", 0.92, {"es": 4.6}, [(9.0, "es", 0.92)], True, "es across every window",
        agreement=1.0, strength=0.92))

    result = job.run()
    doc = source_doc_of(result)

    assert result.language == "es"
    assert "## Visión general" in doc and "## Transcripción" in doc
    assert "**Voz 1 (no identificada)** [" in doc
    assert sorted(result.unresolved) == ["Voz 1", "Voz 2"]


@pytest.mark.parametrize("code,kinds", [
    ("en", ("(from the file's creation time, which may not be when the conversation "
            "happened)", "(from the file's modification time, which may not be when "
            "the conversation happened)")),
    ("it", ("(dall'ora di creazione del file, che può non coincidere con il momento "
            "della conversazione)", "(dall'ora di modifica del file, che può non "
            "coincidere con il momento della conversazione)")),
    ("es", ("(de la hora de creación del archivo, que puede no coincidir con el "
            "momento de la conversación)", "(de la hora de modificación del archivo, "
            "que puede no coincidir con el momento de la conversación)")),
])
def test_a_date_taken_from_the_file_is_said_whole_in_the_document(
        tmp_path, monkeypatch, code, kinds):
    """The state keeps "file created" or "file modified". The page must not repeat it."""
    monkeypatch.setattr(audio, "probe", lambda path: audio.AudioInfo(
        path=Path(path), duration=DURATION, sample_rate=16_000, channels=1,
        codec="aac", created=""))
    job = build_job(tmp_path, monkeypatch, TALK, settings=Settings(language=code))
    stub_registry(monkeypatch, {})

    result = job.run()
    line = header_line(source_doc_of(result), "- **" + export.words(code)["date"])

    assert job.state["recorded_source"] in ("file created", "file modified")
    # Birth time where the filesystem keeps one, modification time where it does not.
    assert any(line.endswith(" " + kind) for kind in kinds), line
    assert "file's file" not in line and "file created" not in line
    assert "file modified" not in line


def test_naming_a_voice_afterwards_keeps_the_language_of_the_document(
        tmp_path, monkeypatch):
    """The rewrite from the cache reads the language from the job, like the run did."""
    job = build_job(tmp_path, monkeypatch, TALK, settings=Settings(language="es"))
    stub_registry(monkeypatch, {})
    job.run()

    result = job.set_names({"SPEAKER_01": HAND_NAME})
    doc = source_doc_of(result)

    assert "## Visión general" in doc
    assert f"**{HAND_NAME}** [" in doc
    assert "**Voz 1 (no identificada)** [" in doc
    assert result.unresolved == ["Voz 1"]


# --------------------------------------------------------------------------- #
# 10. a run keeps the Mac awake
# --------------------------------------------------------------------------- #

@pytest.fixture
def awake_calls(monkeypatch):
    """awake.hold, recording when it was asked and whether the job was claimed."""
    from scriba import awake

    calls: list[bool] = []
    claimed = lambda: any(pipeline.JOBS_DIR.glob("*/running.json"))   # noqa: E731
    monkeypatch.setattr(awake, "hold", lambda **kw: calls.append(claimed()) or True)
    return calls


def test_a_run_holds_the_mac_awake_once_it_has_the_job(tmp_path, monkeypatch, awake_calls):
    job = build_job(tmp_path, monkeypatch, TALK)
    stub_registry(monkeypatch, {})

    job.run()

    # Once, and after the note that says this process is working on the folder.
    assert awake_calls == [True]


def test_a_run_that_is_turned_away_holds_nothing(tmp_path, monkeypatch, awake_calls,
                                                  other_process):
    job = build_job(tmp_path, monkeypatch, TALK)
    leave_a_note(job, other_process.pid)

    with pytest.raises(RuntimeError, match="already being transcribed"):
        job.run()

    assert awake_calls == []


def test_saving_names_holds_nothing(tmp_path, monkeypatch, awake_calls):
    """It writes the documents from the cache: no model, nothing long to protect."""
    job = build_job(tmp_path, monkeypatch, TALK)
    stub_registry(monkeypatch, {})
    job.run()
    awake_calls.clear()

    job.set_names({"SPEAKER_01": HAND_NAME})

    assert awake_calls == []


def test_a_batch_from_the_command_line_holds_one_caffeinate(tmp_path, monkeypatch):
    """`scriba run a b`: both runs ask, the process starts one caffeinate."""
    from typer.testing import CliRunner

    from scriba import awake, cli

    started: list[list[str]] = []

    class Held:
        def poll(self):
            return None

    def spawn(argv, **kwargs):
        started.append(argv)
        return Held()

    real_hold = awake.hold
    monkeypatch.setattr(awake, "_held", None)
    monkeypatch.setattr(awake, "hold", lambda: real_hold(
        spawn=spawn, platform="darwin", environ={}, runnable=lambda path: True))

    first = build_job(tmp_path, monkeypatch, TALK, name="uno.m4a")
    second = build_job(tmp_path, monkeypatch, TALK, name="due.m4a")
    stub_registry(monkeypatch, {})

    outcome = CliRunner().invoke(
        cli.app, ["run", str(first.source), str(second.source), "--lang", "it",
                  "--no-diarize"])

    assert outcome.exit_code == 0, outcome.output
    assert started == [["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())]]


# --------------------------------------------------------------------------- #
# 11. a Stop, and a pid that was handed on
# --------------------------------------------------------------------------- #
#
# The app's Stop sends SIGTERM, and the engine used to die on the spot with its
# note still in the folder. A note naming a dead process is harmless until macOS
# gives that pid to another process: alive, it then refused every run and every
# Save of the job for as long as it lived.

# A start time no live process here has: the pid in the note was handed on.
LONG_AGO = "Mon Jan  1 00:00:00 2001"

needs_ps = pytest.mark.skipif(not os.access("/bin/ps", os.X_OK), reason="no /bin/ps")


@needs_ps
def test_the_note_says_when_its_process_started(tmp_path, monkeypatch):
    job = build_job(tmp_path, monkeypatch, TALK)
    seen: dict[str, object] = {}
    monkeypatch.setattr(pipeline, "model_cache_problem", lambda: None)
    monkeypatch.setattr(job, "_run", lambda *, force: seen.update(
        note=json.loads((job.dir / "running.json").read_text())) or "result")

    job.run()

    assert seen["note"]["process_started"] == jobs.process_start(os.getpid())
    assert seen["note"]["process_started"]


@needs_ps
def test_a_process_start_is_one_stable_string_that_tells_processes_apart(
        monkeypatch, other_process):
    mine = jobs.process_start(os.getpid())

    assert mine and mine == jobs.process_start(os.getpid())
    assert mine == " ".join(mine.split())
    # The same text whatever zone the Mac is in when it is asked.
    monkeypatch.setenv("TZ", "Pacific/Auckland")
    assert jobs.process_start(os.getpid()) == mine
    # Another process, started later (the fixture's), reads as another start only
    # when a second has passed, so what this pins down is that it can be read.
    assert jobs.process_start(other_process.pid)


def test_a_process_that_is_gone_or_a_tool_that_fails_gives_no_start(monkeypatch):
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    assert jobs.process_start(gone.pid) is None

    def fails(*args, **kwargs):
        raise OSError("no ps here")

    def too_slow(*args, **kwargs):
        raise subprocess.TimeoutExpired("ps", 5)

    for failing in (fails, too_slow):
        monkeypatch.setattr(jobs.subprocess, "run", failing)
        assert jobs.process_start(os.getpid()) is None


@needs_ps
def test_a_note_whose_pid_started_at_another_moment_is_a_stranger(
        tmp_path, monkeypatch, other_process):
    job = build_job(tmp_path, monkeypatch, TALK)
    leave_a_note(job, other_process.pid, process_started=LONG_AGO)

    assert jobs.running_pid(job.dir) is None


@needs_ps
def test_a_note_whose_pid_started_when_it_says_is_the_run(tmp_path, monkeypatch,
                                                           other_process):
    job = build_job(tmp_path, monkeypatch, TALK)
    leave_a_note(job, other_process.pid,
                 process_started=jobs.process_start(other_process.pid))

    assert jobs.running_pid(job.dir) == other_process.pid


@pytest.mark.parametrize("more", [{}, {"process_started": None}, {"process_started": ""},
                                  {"process_started": 12345}])
def test_a_note_without_a_usable_start_counts_by_its_pid_as_before(
        tmp_path, monkeypatch, other_process, more):
    """What the code before this wrote, and what a failed `ps` wrote."""
    job = build_job(tmp_path, monkeypatch, TALK)
    leave_a_note(job, other_process.pid, **more)

    assert jobs.running_pid(job.dir) == other_process.pid


def test_a_start_that_cannot_be_read_now_does_not_end_the_run(tmp_path, monkeypatch,
                                                              other_process):
    """`ps` failing at the wrong moment must not let a second run in."""
    job = build_job(tmp_path, monkeypatch, TALK)
    leave_a_note(job, other_process.pid, process_started=LONG_AGO)
    monkeypatch.setattr(jobs, "process_start", lambda pid: None)

    assert jobs.running_pid(job.dir) == other_process.pid


@needs_ps
def test_a_run_goes_ahead_when_the_note_names_a_pid_that_was_handed_on(
        tmp_path, monkeypatch, other_process):
    """The scenario: a Stop left the note, and a daemon now has the pid."""
    job = build_job(tmp_path, monkeypatch, TALK)
    stub_registry(monkeypatch, {})
    note = leave_a_note(job, other_process.pid, process_started=LONG_AGO)

    result = job.run()

    assert result.outputs
    assert not note.exists()


@needs_ps
def test_saving_names_goes_ahead_the_same_way(tmp_path, monkeypatch, other_process):
    job, _ = run_told_two_speakers(tmp_path, monkeypatch)
    leave_a_note(job, other_process.pid, process_started=LONG_AGO)

    job.set_names({"SPEAKER_01": HAND_NAME})

    assert json.loads(job.state_path.read_text())["names"] == {"SPEAKER_01": HAND_NAME}


@needs_ps
def test_a_note_left_by_a_real_run_is_still_a_refusal_for_a_second_one(
        tmp_path, monkeypatch, other_process):
    """The start time must not weaken V5: the process it names is the one running."""
    job = build_job(tmp_path, monkeypatch, TALK)
    leave_a_note(job, other_process.pid,
                 process_started=jobs.process_start(other_process.pid))

    with pytest.raises(RuntimeError, match="already being transcribed"):
        job.run()


# What a Stop does, in a process of its own so that it can be sent the real signal.
# The run is a stand-in that says it has started and then waits, as a long one does.
STOPPED = """
import sys, time
sys.path.insert(0, {root!r})
from pathlib import Path
from scriba import pipeline, stage
from scriba.config import Settings

pipeline.model_cache_problem = lambda: None
job = pipeline.Job(Path({source!r}), Settings(language="it"), report=lambda m: None)

def run(*, force):
    Path({ready!r}).write_text("up")
    if {with_stage!r}:
        stage.call("stage_stand_ins:sleeps", {{"pidfile": {pidfile!r}}}, report=print)
    time.sleep(120)

job._run = run
job.run()
"""


def stop_a_run(tmp_path, *, with_stage: bool):
    """Start a run in another process, wait until it holds the note, send SIGTERM."""
    source = tmp_path / "memos" / "Lake trip.m4a"
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(b"no decoder ever looks at these bytes")
    ready, pidfile = tmp_path / "ready", tmp_path / "child.pid"
    tests = str(Path(__file__).resolve().parent)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [
        tests, os.environ.get("PYTHONPATH")])))
    env.pop("SCRIBA_IN_PROCESS", None)      # a real child, where there is one
    code = STOPPED.format(root=str(Path(pipeline.__file__).resolve().parent.parent),
                          source=str(source), ready=str(ready), pidfile=str(pidfile),
                          with_stage=with_stage)
    proc = subprocess.Popen([sys.executable, "-c", code], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 90
        while not (ready.exists() and (not with_stage or (
                pidfile.exists() and pidfile.read_text().strip()))):
            assert proc.poll() is None, proc.stderr.read().decode()
            assert time.monotonic() < deadline, "the run never started"
            time.sleep(0.05)
        (job_dir,) = pipeline.JOBS_DIR.iterdir()
        assert (job_dir / "running.json").exists(), "the run holds no note"
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=60)
        child = int(pidfile.read_text()) if pidfile.exists() else None
        return proc, job_dir, child
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stderr.close()


def test_a_stop_takes_the_note_away_and_the_run_still_dies_by_the_signal(tmp_path):
    proc, job_dir, _ = stop_a_run(tmp_path, with_stage=False)

    assert not (job_dir / "running.json").exists()
    # How the app tells a Stop from a failure.
    assert proc.returncode == -signal.SIGTERM
    # Nothing failed, so nothing is written as having failed.
    assert "failed" not in json.loads((job_dir / "state.json").read_text())


def test_a_stop_during_transcription_takes_the_note_away_after_the_child(tmp_path):
    proc, job_dir, child = stop_a_run(tmp_path, with_stage=True)

    assert not (job_dir / "running.json").exists()
    assert proc.returncode == -signal.SIGTERM
    assert child is not None
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


def test_the_handler_is_there_for_the_run_only_and_the_old_one_comes_back(
        tmp_path, monkeypatch):
    job = build_job(tmp_path, monkeypatch, TALK)
    monkeypatch.setattr(pipeline, "model_cache_problem", lambda: None)
    before = signal.getsignal(signal.SIGTERM)
    seen: list[object] = []

    def run(*, force):
        seen.append(signal.getsignal(signal.SIGTERM))
        raise RuntimeError("the disk went away")

    monkeypatch.setattr(job, "_run", run)
    with pytest.raises(RuntimeError):
        job.run()

    assert seen[0] is not before and callable(seen[0])
    assert signal.getsignal(signal.SIGTERM) is before


def test_the_handler_releases_the_note_and_sends_the_signal_on(tmp_path, monkeypatch):
    """Called by hand, with the kill replaced: the real one is the test above."""
    job = build_job(tmp_path, monkeypatch, TALK)
    monkeypatch.setattr(pipeline, "model_cache_problem", lambda: None)
    before = signal.getsignal(signal.SIGTERM)
    sent: list[tuple[int, int]] = []
    seen: dict[str, object] = {}

    def run(*, force):
        handler = signal.getsignal(signal.SIGTERM)
        assert (job.dir / "running.json").exists()
        monkeypatch.setattr(os, "kill", lambda pid, sig: sent.append((pid, sig)))
        handler(signal.SIGTERM, None)
        seen["note_after"] = (job.dir / "running.json").exists()
        seen["handler_after"] = signal.getsignal(signal.SIGTERM)
        monkeypatch.undo()
        return "result"

    monkeypatch.setattr(job, "_run", run)
    job.run()

    assert seen["note_after"] is False
    assert sent == [(os.getpid(), signal.SIGTERM)]
    # Put back before the signal was sent again, so that it takes the old way out.
    assert seen["handler_after"] is before


def test_a_run_that_was_told_to_ignore_sigterm_still_ignores_it(tmp_path, monkeypatch):
    job = build_job(tmp_path, monkeypatch, TALK)
    monkeypatch.setattr(pipeline, "model_cache_problem", lambda: None)
    seen: list[object] = []
    monkeypatch.setattr(job, "_run", lambda *, force: seen.append(
        signal.getsignal(signal.SIGTERM)) or "result")
    before = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        job.run()
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGTERM, before)

    assert seen == [signal.SIG_IGN]


def test_a_run_off_the_main_thread_runs_without_the_handler(tmp_path, monkeypatch):
    """Signals cannot be set there. The note is still taken away at the end."""
    job = build_job(tmp_path, monkeypatch, TALK)
    monkeypatch.setattr(pipeline, "model_cache_problem", lambda: None)
    monkeypatch.setattr(job, "_run", lambda *, force: "result")
    outcome: dict[str, object] = {}

    def work():
        try:
            outcome["result"] = job.run()
        except BaseException as exc:  # noqa: BLE001 - the test reports it
            outcome["error"] = exc

    thread = threading.Thread(target=work)
    thread.start()
    thread.join(30)

    assert outcome == {"result": "result"}
    assert not (job.dir / "running.json").exists()


# --------------------------------------------------------------------------- #
# 12. what the command line says to the app, and tells the person to type
# --------------------------------------------------------------------------- #

def name_it(job, *assignments, markers=True):
    """`scriba name <file> ...`, as the app's Save runs it."""
    from typer.testing import CliRunner

    from scriba import cli

    if markers:
        os.environ["SCRIBA_MARKERS"] = "1"
    try:
        return CliRunner().invoke(cli.app, ["name", str(job.source), *assignments])
    finally:
        os.environ.pop("SCRIBA_MARKERS", None)


def marker_lines(outcome) -> list[str]:
    return [line for line in outcome.output.splitlines() if line.startswith("[scriba:error]")]


def test_a_refused_save_says_so_in_a_marker(tmp_path, monkeypatch, other_process):
    job, _ = run_told_two_speakers(tmp_path, monkeypatch)
    leave_a_note(job, other_process.pid)

    outcome = name_it(job, "SPEAKER_01=" + HAND_NAME)

    assert outcome.exit_code == 1
    assert marker_lines(outcome) == [
        "[scriba:error] This recording is already being transcribed by another "
        f"scriba (process {other_process.pid})."]


def test_a_voice_the_recording_does_not_have_is_a_marker_on_one_line(
        tmp_path, monkeypatch):
    job, _ = run_told_two_speakers(tmp_path, monkeypatch)

    outcome = name_it(job, "SPEAKER_07=" + HAND_NAME)

    assert outcome.exit_code == 1
    assert marker_lines(outcome) == [
        "[scriba:error] This recording has no SPEAKER_07. "
        "Voices found: SPEAKER_00, SPEAKER_01"]
    # And what a person at a terminal reads is what it always was.
    assert "This recording has no SPEAKER_07." in outcome.output.splitlines()
    assert "Voices found: SPEAKER_00, SPEAKER_01" in outcome.output.splitlines()


def test_a_job_not_diarized_yet_is_a_marker(tmp_path, monkeypatch):
    job = build_job(tmp_path, monkeypatch, TALK)
    (job.dir / "embeddings.npz").unlink()

    outcome = name_it(job, "SPEAKER_00=" + HAND_NAME)

    assert outcome.exit_code == 1
    assert marker_lines(outcome) == [
        "[scriba:error] This file has not been diarized yet. Run `scriba run` first."]


def test_a_malformed_assignment_is_a_marker(tmp_path, monkeypatch):
    job, _ = run_told_two_speakers(tmp_path, monkeypatch)

    outcome = name_it(job, "SPEAKER_00")

    assert outcome.exit_code == 1
    assert marker_lines(outcome) == [
        "[scriba:error] invalid format: SPEAKER_00 (expected SPEAKER_00=Name)"]


def test_a_job_with_nothing_to_write_from_is_a_marker(tmp_path, monkeypatch):
    job = build_job(tmp_path, monkeypatch, TALK)
    stub_registry(monkeypatch, {})

    outcome = name_it(job, "SPEAKER_01=" + HAND_NAME)

    assert outcome.exit_code == 1
    (line,) = marker_lines(outcome)
    assert line.startswith("[scriba:error] ")
    assert "has nothing cached to write from" in line


def test_a_recording_that_is_gone_is_a_marker(tmp_path, monkeypatch):
    job, _ = run_told_two_speakers(tmp_path, monkeypatch)
    job.source.unlink()

    outcome = name_it(job, "SPEAKER_01=" + HAND_NAME)

    assert outcome.exit_code == 1
    (line,) = marker_lines(outcome)
    assert line.startswith("[scriba:error] ") and "is no longer at" in line


def test_a_save_that_works_prints_no_marker_and_a_terminal_none_at_all(
        tmp_path, monkeypatch):
    job, _ = run_told_two_speakers(tmp_path, monkeypatch)

    good = name_it(job, "SPEAKER_01=" + HAND_NAME)
    assert good.exit_code == 0, good.output
    assert marker_lines(good) == []

    in_a_terminal = name_it(job, "SPEAKER_07=" + HAND_NAME, markers=False)
    assert in_a_terminal.exit_code == 1
    assert marker_lines(in_a_terminal) == []


def run_it(job, language):
    """`scriba run` on a job whose diarization is cached, the way the app starts one.

    The command builds a Job of its own, which reads the state from disk, so the
    fingerprint `build_job` only set in memory is written first.
    """
    from typer.testing import CliRunner

    from scriba import cli

    job._save_state()
    return CliRunner().invoke(cli.app, ["run", str(job.source), "--lang", language])


@pytest.mark.parametrize("code", ["en", "it", "es"])
def test_the_hint_after_a_run_is_a_command_name_accepts(tmp_path, monkeypatch, code):
    """It named the voices as the document does ("Voice 3"), which the shell split in two."""
    job = build_job(tmp_path, monkeypatch, TALK_THREE, name="uno.m4a",
                    settings=Settings(language=code))
    stub_registry(monkeypatch, {"SPEAKER_00": certain(CERTAIN_NAME)})

    outcome = run_it(job, code)

    assert outcome.exit_code == 0, outcome.output
    hint = next(line for line in outcome.output.splitlines() if "scriba name" in line)
    # The labels, whatever language the document is written in.
    assert hint.strip() == 'then: scriba name "uno.m4a" SPEAKER_01=Name SPEAKER_02=Name'
    assert "Voice" not in hint and "Voce" not in hint and "Voz" not in hint
    # Typed as it stands, it is accepted.
    argv = shlex.split(hint.strip().removeprefix("then: "))
    assert argv[:2] == ["scriba", "name"]
    typed = name_it(job, *[a.replace("=Name", "=Ines") for a in argv[3:]], markers=False)
    assert typed.exit_code == 0, typed.output


def test_the_hint_leaves_out_a_voice_that_has_its_name(tmp_path, monkeypatch):
    job = build_job(tmp_path, monkeypatch, TALK_THREE, name="uno.m4a",
                    settings=Settings(language="it"))
    stub_registry(monkeypatch, {"SPEAKER_00": certain(CERTAIN_NAME),
                                "SPEAKER_01": certain(THIRD_NAME)})

    outcome = run_it(job, "it")

    hint = next(line for line in outcome.output.splitlines() if "scriba name" in line)
    assert hint.strip() == 'then: scriba name "uno.m4a" SPEAKER_02=Name'


def test_no_hint_when_every_voice_has_a_name(tmp_path, monkeypatch):
    job = build_job(tmp_path, monkeypatch, TALK, name="uno.m4a",
                    settings=Settings(language="it"))
    stub_registry(monkeypatch, {"SPEAKER_00": certain(CERTAIN_NAME),
                                "SPEAKER_01": certain(THIRD_NAME)})

    outcome = run_it(job, "it")

    assert outcome.exit_code == 0, outcome.output
    assert "scriba name" not in outcome.output
