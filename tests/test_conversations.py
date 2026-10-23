"""Tests for splitting one long recording into its conversations.

Everything here is invented: the owner is Ada, the people she meets at the fair are
Rafiq, Lucía and Tom, and nothing comes from a real recording.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from scriba import conversations as conv


def turn(a, b, who):
    return {"start": a, "end": b, "speaker": who}


def said(a, b, who, text):
    return {"start": a, "end": b, "speaker": who, "text": text}


# --------------------------------------------------------------------- whose recording

def test_the_owner_is_the_voice_with_the_most_speech():
    assert conv.find_owner({"SPEAKER_00": 900.0, "SPEAKER_01": 120.0}, {}) == "SPEAKER_00"


def test_the_owner_can_be_named_by_label_or_by_name():
    speech = {"SPEAKER_00": 10.0, "SPEAKER_03": 5.0}
    assert conv.find_owner(speech, {}, "SPEAKER_03") == "SPEAKER_03"
    assert conv.find_owner(speech, {"SPEAKER_03": "Ada"}, "ada") == "SPEAKER_03"
    with pytest.raises(ValueError):
        conv.find_owner(speech, {}, "Nobody")


# --------------------------------------------------------------------- where they split

def test_a_new_person_after_a_pause_is_a_new_conversation():
    dia = [turn(0, 20, "A"), turn(20, 40, "B"), turn(40, 60, "A"),
           turn(70, 90, "A"), turn(90, 120, "C")]
    out = conv.split(dia, [], "A")
    assert [c.guests for c in out] == [["B"], ["C"]]
    assert out[1].start == 70


def test_the_same_people_after_a_short_pause_stay_one_conversation():
    dia = [turn(0, 20, "A"), turn(20, 40, "B"), turn(50, 70, "B"), turn(70, 80, "A")]
    out = conv.split(dia, [], "A")
    assert len(out) == 1 and out[0].guests == ["B"] and out[0].end == 80


def test_a_long_silence_ends_a_conversation_even_with_the_same_label():
    dia = [turn(0, 20, "A"), turn(20, 40, "B"), turn(140, 160, "B")]
    assert len(conv.split(dia, [], "A", long_gap=60)) == 2


def test_a_greeting_after_a_pause_starts_a_new_one_when_the_label_is_reused():
    # In a noisy hall the diarizer gives two different men the same label.
    dia = [turn(0, 20, "A"), turn(20, 40, "B"), turn(50, 70, "B")]
    words = [said(50, 55, "B", "Hola, encantado, me llamo Tom.")]
    assert len(conv.split(dia, words, "A")) == 2


def test_a_goodbye_closes_the_conversation():
    dia = [turn(0, 20, "A"), turn(20, 40, "B"), turn(50, 70, "B")]
    words = [said(35, 40, "B", "Vale, nos vemos luego.")]
    assert len(conv.split(dia, words, "A")) == 2


def test_the_owner_alone_does_not_start_a_conversation_or_count_as_one():
    dia = [turn(0, 15, "A"), turn(30, 50, "A"), turn(50, 90, "B")]
    out = conv.split(dia, [], "A")
    assert len(out) == 1
    assert out[0].guests == ["B"] and out[0].start == 0


def test_a_passer_by_is_not_a_guest():
    dia = [turn(0, 40, "A"), turn(40, 41, "X")]
    out = conv.split(dia, [], "A")
    assert out[0].guests == [] and not out[0].worth_reading


def test_each_conversation_gets_only_its_own_turns():
    dia = [turn(0, 40, "A"), turn(40, 80, "B"), turn(200, 240, "A"), turn(240, 280, "C")]
    words = [said(0, 40, "A", "first"), said(40, 80, "B", "Rafiq"),
             said(200, 240, "A", "second"), said(240, 280, "C", "Lucía")]
    out = conv.split(dia, words, "A")
    assert [t["text"] for t in out[0].turns] == ["first", "Rafiq"]
    assert [t["text"] for t in out[1].turns] == ["second", "Lucía"]


# --------------------------------------------------------------------- the right language

@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "audio16k.wav"
    sf.write(str(path), np.zeros(16000 * 300, dtype="float32"), 16000)
    return path


def test_a_conversation_in_another_language_is_found(wav, tmp_path):
    convs = [conv.Conversation(1, 0, 60, guests=["B"]),
             conv.Conversation(2, 100, 160, guests=["C"]),
             conv.Conversation(3, 200, 210, guests=[])]
    seen = []

    def detect(clip, n_samples):
        seen.append((Path(clip).name, n_samples))
        lang = "es" if Path(clip).name == "02.wav" else "en"
        return SimpleNamespace(language=lang, agreement=1.0)

    wrong = conv.check_languages(convs, wav, "en", tmp_path / "work", detect=detect)
    assert [c.index for c in wrong] == [2]
    assert convs[1].language == "es" and convs[0].language == "en"
    # The third is too short to read, so its audio is never even cut.
    assert [name for name, _ in seen] == ["01.wav", "02.wav"]
    assert sf.info(str(tmp_path / "work" / "02.wav")).duration == pytest.approx(60, abs=0.01)


def test_an_unclear_vote_keeps_the_file_language(wav, tmp_path):
    convs = [conv.Conversation(1, 0, 60, guests=["B"])]
    wrong = conv.check_languages(
        convs, wav, "en", tmp_path, detect=lambda c, n_samples: SimpleNamespace(
            language="pt", agreement=0.4))
    assert wrong == [] and convs[0].language == "en"


def test_a_conversation_transcribed_again_lands_on_the_recording_clock(tmp_path):
    c = conv.Conversation(4, 300, 360, guests=["B"], language="es")
    calls = []

    def run_stage(items):
        calls.append(items)
        Path(items[0]["out"]).write_text(json.dumps({"segments": [
            {"start": 1.0, "end": 3.0, "text": "Hola, soy Lucía.",
             "words": [{"word": "Hola", "start": 1.0, "end": 1.4}]}],
            "language": "es", "word_level": True}))

    def assign(segments, dia, word_level):
        assert word_level is True
        return [{**s, "speaker": "B"} for s in segments]

    conv.retranscribe([c], tmp_path, dia=None, run_stage=run_stage, assign=assign,
                      to_turns=lambda segs: segs)
    assert calls[0][0]["language"] == "es"
    assert c.turns[0]["start"] == 301.0 and c.turns[0]["words"][0]["start"] == 301.0
    assert c.retranscribed

    # The second time the cached transcript is used and nothing runs.
    conv.retranscribe([c], tmp_path, dia=None, run_stage=lambda items: calls.append(items),
                      assign=assign, to_turns=lambda segs: segs)
    assert len(calls) == 1


# --------------------------------------------------------------------- what came of it

def test_summaries_are_only_asked_of_this_mac():
    with pytest.raises(ValueError):
        conv.summarize("x", owner="Ada", model="m", url="https://api.example.com")


def test_a_summary_is_read_from_the_local_model():
    sent = {}

    def send(url, body, timeout):
        sent["url"], sent["body"] = url, json.loads(body)
        return json.dumps({"message": {"content": json.dumps({
            "people": [{"name": "Rafiq", "company": "Lake Boats", "role": "CTO"}, "junk"],
            "summary": "They talked about boats.",
            "agreements": ["Connect on LinkedIn", " "],
            "next_step": "Write to Rafiq on Monday."})}}).encode()

    out = conv.summarize("Ada: hi\nVoice 2: hi", owner="Ada", model="qwen3:8b", send=send)
    assert sent["url"] == "http://127.0.0.1:11434/api/chat"
    assert sent["body"]["format"] == "json" and sent["body"]["stream"] is False
    assert "Italian" in sent["body"]["messages"][0]["content"]
    assert out["people"] == [{"name": "Rafiq", "company": "Lake Boats", "role": "CTO"}]
    assert out["agreements"] == ["Connect on LinkedIn"]
    assert out["next_step"] == "Write to Rafiq on Monday."


def test_a_long_conversation_keeps_its_start_and_its_end():
    sent = {}

    def send(url, body, timeout):
        sent["text"] = json.loads(body)["messages"][0]["content"]
        return json.dumps({"message": {"content": "{}"}}).encode()

    conv.summarize("S" * 10000 + "E" * 10000, owner="Ada", model="m", send=send)
    assert "S" * 7000 in sent["text"] and "E" * 7000 in sent["text"] and "[...]" in sent["text"]


# --------------------------------------------------------------------- what is written

def test_the_index_lists_conversations_and_says_how_many_it_left_out(tmp_path):
    a = conv.Conversation(1, 0, 120, guests=["SPEAKER_01"], language="es",
                          turns=[said(5, 10, "SPEAKER_01", "Hola, soy Lucía.")],
                          summary={"people": [{"name": "Lucía", "company": "Naranja",
                                               "role": ""}],
                                   "summary": "Robots | piers.", "agreements": ["LinkedIn"],
                                   "next_step": "Send the paper."})
    b = conv.Conversation(2, 200, 210, guests=[])
    files = conv.write([a, b], tmp_path, title="Fair.m4a", recorded="2026-10-22T16:00:00",
                       names={}, language="it")
    index = (tmp_path / "index.md").read_text()
    assert "| [1](01.md) | 16:00 | 02:00 | es | Lucía (Naranja) | Robots / piers. " \
           "| Send the paper. |" in index
    assert "1 tratti senza nessun interlocutore" in index
    one = (tmp_path / "01.md").read_text()
    assert "Hola, soy Lucía." in one and "## Accordi" in one
    assert [f.name for f in files] == ["index.md", "01.md", "conversations.json"]
    data = json.loads((tmp_path / "conversations.json").read_text())
    assert [d["index"] for d in data] == [1, 2]


def test_the_whole_command_runs_on_a_cached_job(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "diarization.json").write_text(json.dumps(
        [turn(0, 40, "SPEAKER_00"), turn(40, 90, "SPEAKER_01"),
         turn(300, 340, "SPEAKER_00"), turn(340, 400, "SPEAKER_02")]))
    (job_dir / "turns.json").write_text(json.dumps(
        [said(0, 40, "SPEAKER_00", "Hi, I'm Ada."), said(40, 90, "SPEAKER_01", "Rafiq here."),
         said(300, 340, "SPEAKER_00", "Hola."), said(340, 400, "SPEAKER_02", "Soy Lucía.")]))
    held = []
    job = SimpleNamespace(
        dir=job_dir, wav=job_dir / "audio16k.wav", source=Path("Fair.m4a"),
        state={"language": "en", "source": "/x/Fair.m4a", "names": {"SPEAKER_00": "Ada"}},
        s=None, report=lambda m: None,
        _refuse_if_running=lambda: held.append("checked"),
        _claim=lambda: held.append("claimed"), _release=lambda: held.append("released"),
        _release_on_sigterm=lambda: (lambda: held.append("restored")))
    written = conv.run(job, keep_language=True, summaries=False)
    assert held == ["checked", "claimed", "released", "restored"]
    names = [p.name for p in written]
    assert names == ["index.md", "01.md", "02.md", "conversations.json"]
    assert "Rafiq here." in (job_dir / "output" / "conversations" / "01.md").read_text()


def test_a_recording_without_a_finished_run_is_refused(tmp_path):
    job = SimpleNamespace(dir=tmp_path, state={})
    with pytest.raises(RuntimeError, match="scriba run"):
        conv.run(job)


def test_a_label_heard_all_afternoon_does_not_hold_two_conversations_together():
    # Z is the hall: in every stretch, from start to end. Each stretch also has one
    # real person, a different one every time, 20 s after the last.
    dia = []
    for i in range(10):
        t = i * 80.0
        dia += [turn(t, t + 20, "A"), turn(t + 20, t + 40, f"P{i}"), turn(t + 40, t + 50, "Z")]
    out = conv.split(dia, [], "A", long_gap=60)
    assert conv.ambient(conv.stretches(dia), "A") == {"Z"}
    assert len(out) == 10
    assert all("Z" in c.guests for c in out)


def test_few_stretches_are_too_few_to_call_anyone_ambient():
    dia = [turn(0, 20, "A"), turn(20, 40, "Z"), turn(100, 120, "A"), turn(120, 140, "Z")]
    assert conv.ambient(conv.stretches(dia), "A") == set()


def test_a_language_nobody_spoke_there_is_not_believed(wav, tmp_path):
    # A minute of hall noise once came back as Norwegian Nynorsk, every window agreeing.
    convs = [conv.Conversation(1, 0, 60, guests=["B"])]
    wrong = conv.check_languages(
        convs, wav, "es", tmp_path, detect=lambda c, n_samples: SimpleNamespace(
            language="nn", agreement=1.0))
    assert wrong == [] and convs[0].language == "es"


def test_the_people_column_holds_people():
    people = [{"name": "Ada", "company": "", "role": "PhD"},
              {"name": "Rafiq", "company": "Lake Boats", "role": "CTO"},
              {"name": "", "company": "", "role": ""},
              {"name": "Voz 2", "company": "", "role": "Nosotros lo que hacemos, bueno, hacemos "
                                                     "simulaciones del mundo físico y mecánica"},
              "junk"]
    assert conv.clean_people(people, "Ada") == [
        {"name": "Rafiq", "company": "Lake Boats", "role": "CTO"},
        {"name": "Voz 2", "company": "", "role": ""}]


def test_a_conversation_with_almost_nothing_transcribed_is_not_summarized(tmp_path, monkeypatch):
    # Given a minute of noise, the model once invented two people and their companies.
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "diarization.json").write_text(json.dumps(
        [turn(0, 40, "SPEAKER_00"), turn(40, 90, "SPEAKER_01")]))
    (job_dir / "turns.json").write_text(json.dumps([said(40, 45, "SPEAKER_01", "Sí, vale.")]))
    job = SimpleNamespace(
        dir=job_dir, wav=job_dir / "a.wav", source=Path("Fair.m4a"),
        state={"language": "es", "names": {}}, s=None, report=lambda m: None,
        _refuse_if_running=lambda: None, _claim=lambda: None, _release=lambda: None,
        _release_on_sigterm=lambda: (lambda: None))

    def never(*a, **k):
        raise AssertionError("the model must not be asked")

    monkeypatch.setattr(conv, "summarize", never)
    monkeypatch.setattr(conv, "release_model", lambda m: None)
    conv.run(job, keep_language=True, summaries=True)
    index = (job_dir / "output" / "conversations" / "index.md").read_text()
    assert "Troppo poco testo trascritto" in index


def test_names_the_transcript_never_said_are_dropped_or_marked():
    transcript = "Voz 2: Trabajo en la ORVA, la Oficina Regional de Vías Acuáticas."
    summary = {"people": [{"name": "Laura", "company": "TechNova", "role": "PM"},
                          {"name": "Voce 2", "company": "ORVA", "role": ""}],
               "summary": "Hanno parlato con l'ORVA e con la KTL di Ferrybox.io e delle API.",
               "agreements": [], "next_step": ""}
    out = conv.ground(summary, transcript)
    assert out["people"] == [{"name": "", "company": "", "role": "PM"},
                             {"name": "Voce 2", "company": "ORVA", "role": ""}]
    assert out["unverified"] == ["KTL", "Ferrybox.io"]
