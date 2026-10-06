"""Tests for scriba.export.

Everything here is built by hand: no audio, no models, no network. A "turn" is
just a dict with a speaker label, a start time, some text and (sometimes) a
confidence, which is exactly what the export layer is contracted to accept.

All names and content are invented.
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from scriba import audio, export
from scriba.export import (
    FILENAMES,
    _sweep,
    display,
    hhmmss,
    markdown,
    payload_json,
    plain,
    source_doc,
    srt,
    srt_time,
    vtt,
    vtt_time,
    write_all,
)


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #

def turn(speaker, start, text, confidence=None, end=None):
    t = {"speaker": speaker, "start": start, "end": end if end is not None else start + 3.0,
         "text": text}
    if confidence is not None:
        t["confidence"] = confidence
    return t


def transcript_lines(doc: str, language: str = "en") -> list[str]:
    """The lines of the Transcript section that carry a spoken turn."""
    w = export.words(language)
    body = doc.split(f"## {w['transcript']}", 1)[1]
    body = body.split(f"## {w['to_check']}", 1)[0]
    return [ln for ln in body.splitlines() if ln.startswith("**")]


def check_lines(doc: str, language: str = "en") -> list[str]:
    """The bullet lines of the "Turns to check" section."""
    heading = f"## {export.words(language)['to_check']}"
    if heading not in doc:
        return []
    body = doc.split(heading, 1)[1]
    return [ln for ln in body.splitlines() if ln.startswith("- [")]


@pytest.fixture
def conversation():
    """Two voices, one of them never named, one shaky turn, one unattributed."""
    return [
        turn("SPEAKER_00", 0.0, "  Comincio io, se non vi dispiace.  ", 0.97),
        turn("SPEAKER_01", 5.0, "Fai pure.", 0.41),
        turn("SPEAKER_00", 12.0, "Il punto e' la scadenza di novembre."),
        turn(None, 20.0, "(rumore di fondo)", 0.99),
        turn("SPEAKER_01", 25.0, "Su quello non sono d'accordo.", 0.99),
    ]


@pytest.fixture
def segments():
    return [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.5, "text": "Comincio io."},
        {"speaker": "SPEAKER_01", "start": 2.5, "end": 6.0, "text": " Fai pure. "},
    ]


# --------------------------------------------------------------------------- #
# hhmmss
# --------------------------------------------------------------------------- #

def test_hhmmss_omits_hours_under_one_hour():
    assert hhmmss(0) == "00:00"
    assert hhmmss(9) == "00:09"
    assert hhmmss(65) == "01:05"
    assert hhmmss(3599) == "59:59"


def test_hhmmss_adds_hours_when_needed():
    assert hhmmss(3600) == "01:00:00"
    assert hhmmss(3661) == "01:01:01"
    assert hhmmss(37230) == "10:20:30"


def test_hhmmss_always_hours_pads_short_durations():
    assert hhmmss(0, always_hours=True) == "00:00:00"
    assert hhmmss(65, always_hours=True) == "00:01:05"
    assert hhmmss(3661, always_hours=True) == "01:01:01"


def test_hhmmss_truncates_the_fractional_second():
    # A timestamp you use to scrub back through audio should never round forward
    # past the moment the words start.
    assert hhmmss(59.999) == "00:59"
    assert hhmmss(3599.9) == "59:59"
    assert hhmmss(0.4) == "00:00"


def test_hhmmss_clamps_negative_input_to_zero():
    assert hhmmss(-1) == "00:00"
    assert hhmmss(-3600.5) == "00:00"
    assert hhmmss(-1, always_hours=True) == "00:00:00"


# --------------------------------------------------------------------------- #
# srt_time / vtt_time
# --------------------------------------------------------------------------- #

def test_srt_time_always_carries_hours_and_milliseconds():
    assert srt_time(0) == "00:00:00,000"
    assert srt_time(5) == "00:00:05,000"
    assert srt_time(3661.5) == "01:01:01,500"


def test_srt_time_renders_sub_second_precision():
    assert srt_time(1.25) == "00:00:01,250"
    assert srt_time(12.007) == "00:00:12,007"
    assert srt_time(90.75) == "00:01:30,750"


def test_srt_time_clamps_negative_input():
    assert srt_time(-4.2) == "00:00:00,000"


def test_vtt_time_is_the_srt_stamp_with_a_dot():
    assert vtt_time(0) == "00:00:00.000"
    assert vtt_time(3661.5) == "01:01:01.500"
    assert vtt_time(-1) == "00:00:00.000"


# Regression. srt_time rounded the milliseconds independently of the seconds,
# so 0.9996 came out as 00:00:00,1000. A four-digit millisecond field is not a
# valid SRT timestamp, and strict parsers reject the cue or the whole track.
def test_srt_time_never_emits_a_four_digit_millisecond_field():
    assert srt_time(0.9996) == "00:00:01,000"
    assert srt_time(1.9999) == "00:00:02,000"


# --------------------------------------------------------------------------- #
# display
# --------------------------------------------------------------------------- #

def test_display_counts_voices_from_one():
    assert display("SPEAKER_00", None) == "Voice 1"
    assert display("SPEAKER_01", None) == "Voice 2"
    assert display("SPEAKER_09", None) == "Voice 10"
    assert display("SPEAKER_12", None) == "Voice 13"


def test_display_prefers_a_known_name():
    names = {"SPEAKER_00": "Ada Verdolini", "SPEAKER_01": "Bruno Cassaro"}
    assert display("SPEAKER_00", names) == "Ada Verdolini"
    assert display("SPEAKER_01", names) == "Bruno Cassaro"
    # a label with no entry still falls back to the numbered voice
    assert display("SPEAKER_02", names) == "Voice 3"


def test_display_of_no_speaker_is_unattributed():
    assert display(None, None) == "Unattributed"
    assert display(None, {"SPEAKER_00": "Ada Verdolini"}) == "Unattributed"


def test_display_empty_names_mapping_behaves_like_none():
    assert display("SPEAKER_00", {}) == "Voice 1"


def test_display_returns_a_malformed_label_verbatim():
    # Better an odd-looking label in the document than a crash or a wrong number.
    assert display("SPEAKER_XY", None) == "SPEAKER_XY"
    assert display("SPEAKER_", None) == "SPEAKER_"
    assert display("SPEAKER_0a", None) == "SPEAKER_0a"


def test_display_returns_an_unrecognised_label_verbatim():
    assert display("intervistatore", None) == "intervistatore"
    assert display("intervistatore", {"SPEAKER_00": "Ada Verdolini"}) == "intervistatore"


def test_display_name_wins_even_over_a_malformed_label():
    assert display("SPEAKER_XY", {"SPEAKER_XY": "Ada Verdolini"}) == "Ada Verdolini"


# --------------------------------------------------------------------------- #
# source_doc: header
# --------------------------------------------------------------------------- #

def test_source_doc_has_a_title_and_the_two_sections(conversation):
    doc = source_doc(conversation, title="Riunione di redazione")
    assert doc.startswith("# Riunione di redazione\n")
    assert "## Overview" in doc
    assert "## Transcript" in doc
    assert doc.index("## Overview") < doc.index("## Transcript")


def test_source_doc_date_is_plain_when_it_comes_from_the_recorder(conversation):
    doc = source_doc(conversation, title="T", recorded=datetime(2026, 3, 4, 9, 30),
                     recorded_source="recorded")
    assert "- **Date**: 2026-03-04 09:30" in doc
    assert "may not be when the conversation happened" not in doc


def test_source_doc_date_is_plain_when_no_source_is_declared(conversation):
    doc = source_doc(conversation, title="T", recorded=datetime(2026, 3, 4, 9, 30))
    assert "- **Date**: 2026-03-04 09:30" in doc
    assert "may not be" not in doc


def test_source_doc_date_flags_a_filesystem_timestamp(conversation):
    # The value the engine stores (audio.recorded_at), not a word of this test's own.
    doc = source_doc(conversation, title="T", recorded=datetime(2026, 3, 4, 9, 30),
                     recorded_source="file modified")
    assert "2026-03-04 09:30" in doc
    assert "from the file's modification time" in doc
    assert "may not be when the conversation happened" in doc


def test_source_doc_date_from_a_birth_time_says_creation(conversation):
    doc = source_doc(conversation, title="T", recorded=datetime(2026, 3, 4, 9, 30),
                     recorded_source="file created")
    assert "from the file's creation time" in doc


def test_source_doc_omits_the_date_when_there_is_none(conversation):
    doc = source_doc(conversation, title="T")
    assert "**Date**" not in doc


def test_source_doc_duration_always_shows_hours(conversation):
    assert "- **Duration**: 00:01:35" in source_doc(conversation, title="T", duration=95.0)
    assert "- **Duration**: 01:02:05" in source_doc(conversation, title="T", duration=3725.0)
    assert "- **Duration**: 00:00:00" in source_doc(conversation, title="T")


def test_source_doc_reports_language(conversation):
    # The label follows the language of the document and the value never does: it is
    # the code another program reads.
    assert "- **Lingua**: it" in source_doc(conversation, title="T", language="it")
    assert "- **Idioma**: es" in source_doc(conversation, title="T", language="es")
    assert "- **Language**: de" in source_doc(conversation, title="T", language="de")


def test_source_doc_without_a_language_is_written_in_english(conversation):
    doc = source_doc(conversation, title="T")
    assert "- **Language**: en" in doc
    assert "## Overview" in doc and "## Transcript" in doc


def test_source_doc_source_file_and_device_are_optional(conversation):
    bare = source_doc(conversation, title="T")
    assert "**Source file**" not in bare
    assert "**Recorded with**" not in bare

    full = source_doc(conversation, title="T", source_file="nota vocale 12.m4a",
                      device="Registratore da tasca")
    assert "- **Source file**: nota vocale 12.m4a" in full
    assert "- **Recorded with**: Registratore da tasca" in full


def test_source_doc_lists_participants_with_their_speech_share(conversation):
    doc = source_doc(
        conversation, title="T",
        names={"SPEAKER_00": "Ada Verdolini"},
        speaker_stats={"SPEAKER_00": 620.0, "SPEAKER_01": 95.0},
    )
    assert "- **Participants**: Ada Verdolini (10:20 of speech), Voice 2 (01:35 of speech)" in doc


def test_source_doc_participants_without_stats_are_just_names(conversation):
    doc = source_doc(conversation, title="T", names={"SPEAKER_00": "Ada Verdolini"})
    assert "- **Participants**: Ada Verdolini, Voice 2" in doc


def test_source_doc_participants_ignore_unattributed_turns():
    turns = [turn(None, 0.0, "solo rumore")]
    doc = source_doc(turns, title="T")
    assert "**Participants**" not in doc


def test_source_doc_unidentified_line_singular_wording(conversation):
    doc = source_doc(conversation, title="T", names={"SPEAKER_00": "Ada Verdolini"},
                     unresolved=["Voice 2"])
    line = next(ln for ln in doc.splitlines() if ln.startswith("- **Unidentified voices**"))
    assert "Voice 2." in line
    assert "A distinct person." in line
    assert "Their name is never spoken" in line
    assert "Distinct people" not in line
    assert "Their names are" not in line
    assert "Do not guess who they are." in line


def test_source_doc_unidentified_line_plural_wording(conversation):
    doc = source_doc(conversation, title="T", unresolved=["Voice 2", "Voice 1"])
    line = next(ln for ln in doc.splitlines() if ln.startswith("- **Unidentified voices**"))
    # sorted, so the list reads in a stable order regardless of the caller's
    assert line.startswith("- **Unidentified voices**: Voice 1, Voice 2.")
    assert "Distinct people." in line
    assert "Their names are never spoken" in line
    assert "A distinct person" not in line


def test_source_doc_has_no_unidentified_line_when_everyone_is_named(conversation):
    doc = source_doc(conversation, title="T",
                     names={"SPEAKER_00": "Ada Verdolini", "SPEAKER_01": "Bruno Cassaro"})
    assert "**Unidentified voices**" not in doc


def test_source_doc_attribution_line_counts_the_shaky_turns(conversation):
    doc = source_doc(conversation, title="T")
    assert "- **Attribution**: 1 of 5 turns is marked *(uncertain)*." in doc


def test_source_doc_has_no_attribution_line_when_nothing_is_shaky():
    turns = [turn("SPEAKER_00", 0.0, "Tutto chiaro.", 0.99)]
    doc = source_doc(turns, title="T")
    assert "**Attribution**" not in doc
    assert "## Turns to check" not in doc


def test_source_doc_always_warns_that_this_is_asr(conversation):
    assert "- **Note**: this is automatic speech recognition." in source_doc(conversation, title="T")


# --------------------------------------------------------------------------- #
# source_doc: transcript body
# --------------------------------------------------------------------------- #

def test_source_doc_turn_lines_carry_name_and_timestamp(conversation):
    doc = source_doc(conversation, title="T",
                     names={"SPEAKER_00": "Ada Verdolini", "SPEAKER_01": "Bruno Cassaro"})
    lines = transcript_lines(doc)
    assert lines[0] == "**Ada Verdolini** [00:00]: Comincio io, se non vi dispiace."
    assert "**Bruno Cassaro (uncertain)** [00:05]: Fai pure." in lines
    assert "**Unattributed** [00:20]: (rumore di fondo)" in lines


def test_source_doc_marks_unidentified_on_every_line_not_only_in_the_header(conversation):
    # The requirement: a reader (or a model) quoting one line out of the middle of
    # a long document must see that the speaker was never identified.
    doc = source_doc(conversation, title="T", names={"SPEAKER_00": "Ada Verdolini"},
                     unresolved=["Voice 2"])
    voice_two = [ln for ln in transcript_lines(doc) if ln.startswith("**Voice 2")]
    assert len(voice_two) == 2, voice_two
    for line in voice_two:
        assert "(unidentified)" in line, line


def test_source_doc_never_marks_a_named_speaker_as_unidentified(conversation):
    doc = source_doc(conversation, title="T",
                     names={"SPEAKER_00": "Ada Verdolini", "SPEAKER_01": "Bruno Cassaro"})
    assert "(unidentified)" not in doc


def test_source_doc_does_not_mark_an_unattributed_turn_as_unidentified():
    # No speaker at all is a different thing from a speaker nobody could name.
    turns = [turn(None, 0.0, "colpo di tosse", 0.99)]
    doc = source_doc(turns, title="T")
    assert transcript_lines(doc) == ["**Unattributed** [00:00]: colpo di tosse"]


def test_source_doc_marks_low_confidence_turns_uncertain(conversation):
    doc = source_doc(conversation, title="T",
                     names={"SPEAKER_00": "Ada Verdolini", "SPEAKER_01": "Bruno Cassaro"})
    uncertain = [ln for ln in transcript_lines(doc) if "(uncertain)" in ln]
    assert uncertain == ["**Bruno Cassaro (uncertain)** [00:05]: Fai pure."]


def test_source_doc_treats_a_missing_confidence_as_certain():
    turns = [turn("SPEAKER_00", 0.0, "Nessun punteggio qui.")]
    doc = source_doc(turns, title="T", names={"SPEAKER_00": "Ada Verdolini"})
    assert "(uncertain)" not in doc


def test_source_doc_uncertainty_threshold_is_configurable():
    turns = [turn("SPEAKER_00", 0.0, "Forse.", 0.9)]
    names = {"SPEAKER_00": "Ada Verdolini"}
    assert "(uncertain)" not in source_doc(turns, title="T", names=names, uncertain_below=0.8)
    assert "(uncertain)" in source_doc(turns, title="T", names=names, uncertain_below=0.95)


def test_source_doc_carries_both_markers_when_they_apply():
    turns = [turn("SPEAKER_03", 61.0, "Non saprei.", 0.2)]
    doc = source_doc(turns, title="T")
    assert transcript_lines(doc) == ["**Voice 4 (unidentified) (uncertain)** [01:01]: Non saprei."]


def test_source_doc_separates_speaker_changes_with_a_blank_line(conversation):
    doc = source_doc(conversation, title="T",
                     names={"SPEAKER_00": "Ada Verdolini", "SPEAKER_01": "Bruno Cassaro"})
    body = doc.split("## Transcript", 1)[1].split("## Turns to check", 1)[0]
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("**") and i > 0:
            assert lines[i - 1] == "", f"no blank line before {line!r}"


def test_source_doc_never_leaves_three_blank_lines_in_a_row(conversation):
    doc = source_doc(conversation, title="T", recorded=datetime(2026, 3, 4, 9, 30),
                     unresolved=["Voice 2"])
    assert "\n\n\n" not in doc


def test_source_doc_survives_an_empty_transcript():
    doc = source_doc([], title="Registrazione vuota")
    assert doc.startswith("# Registrazione vuota")
    assert "## Transcript" in doc
    assert "**Participants**" not in doc
    assert transcript_lines(doc) == []


# --------------------------------------------------------------------------- #
# source_doc: "Turns to check"
# --------------------------------------------------------------------------- #

def test_source_doc_turns_to_check_lists_only_the_shaky_turns(conversation):
    doc = source_doc(conversation, title="T",
                     names={"SPEAKER_00": "Ada Verdolini", "SPEAKER_01": "Bruno Cassaro"})
    assert "## Turns to check" in doc
    assert "Speaker attribution is weakest here." in doc
    assert check_lines(doc) == ["- [00:05] Bruno Cassaro: Fai pure."]


def test_source_doc_turns_to_check_comes_after_the_transcript(conversation):
    doc = source_doc(conversation, title="T")
    assert doc.index("## Transcript") < doc.index("## Turns to check")


def test_source_doc_turns_to_check_truncates_a_long_quote():
    long_text = ("Questa e' una battuta molto lunga che serve soltanto a superare "
                 "il limite di novanta caratteri previsto dal sommario finale.")
    doc = source_doc([turn("SPEAKER_00", 0.0, long_text, 0.3)], title="T",
                     names={"SPEAKER_00": "Ada Verdolini"})
    line = check_lines(doc)[0]
    snippet = line.split(": ", 1)[1]
    assert snippet.endswith("…")
    assert len(snippet) <= 91
    assert snippet.startswith("Questa e' una battuta molto lunga")
    # the full sentence is still in the transcript, only the summary is clipped
    assert long_text in doc


def test_source_doc_turns_to_check_keeps_a_short_quote_whole():
    doc = source_doc([turn("SPEAKER_00", 0.0, "Breve.", 0.3)], title="T",
                     names={"SPEAKER_00": "Ada Verdolini"})
    assert check_lines(doc) == ["- [00:00] Ada Verdolini: Breve."]


# Regression. The "Turns to check" list called display() without the
# (unidentified) suffix that every transcript line carries, so a quote lifted
# from it read as though Voice 2 were somebody identified.
def test_source_doc_turns_to_check_also_marks_unidentified_speakers(conversation):
    doc = source_doc(conversation, title="T", names={"SPEAKER_00": "Ada Verdolini"},
                     unresolved=["Voice 2"])
    assert all("(unidentified)" in ln for ln in check_lines(doc)), check_lines(doc)


# --------------------------------------------------------------------------- #
# the tool-facing formats
# --------------------------------------------------------------------------- #

def test_markdown_puts_the_name_and_stamp_above_the_text(conversation):
    out = markdown(conversation, names={"SPEAKER_00": "Ada Verdolini"})
    assert out.startswith("**Ada Verdolini** [00:00]  \nComincio io, se non vi dispiace.")
    assert out.endswith("\n")
    assert "**Voice 2** [00:05]  \nFai pure." in out


def test_plain_is_one_line_per_turn(conversation):
    out = plain(conversation, names={"SPEAKER_00": "Ada Verdolini"})
    lines = out.rstrip("\n").split("\n")
    assert len(lines) == len(conversation)
    assert lines[0] == "Ada Verdolini: Comincio io, se non vi dispiace."
    assert lines[3] == "Unattributed: (rumore di fondo)"


def test_srt_blocks_carry_index_times_and_speaker(segments):
    out = srt(segments, names={"SPEAKER_00": "Ada Verdolini"})
    assert out.startswith("1\n00:00:00,000 --> 00:00:02,500\n[Ada Verdolini] Comincio io.\n")
    assert "[Voice 2] Fai pure.\n" in out
    assert "," in out and "-->" in out


def test_srt_skips_blank_segments(segments):
    padded = [segments[0], {"speaker": "SPEAKER_00", "start": 2.5, "end": 2.6, "text": "  "},
              segments[1]]
    out = srt(padded)
    assert "00:00:02,500 --> 00:00:02,600" not in out
    assert out.count("-->") == 2


# Regression. Cue numbers came from enumerate() over all segments while blank
# ones were skipped afterwards, leaving holes in the sequence. SRT numbering is
# meant to be consecutive, and strict parsers renumber or reject the file.
def test_srt_numbers_cues_consecutively(segments):
    padded = [segments[0], {"speaker": "SPEAKER_00", "start": 2.5, "end": 2.6, "text": ""},
              segments[1]]
    out = srt(padded)
    numbers = [b.splitlines()[0] for b in out.strip().split("\n\n")]
    assert numbers == ["1", "2"]


def test_vtt_starts_with_the_header_and_tags_the_voice(segments):
    out = vtt(segments, names={"SPEAKER_00": "Ada Verdolini"})
    assert out.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:02.500" in out
    assert "<v Ada Verdolini>Comincio io." in out
    assert "<v Voice 2>Fai pure." in out
    assert "," not in out.split("\n\n")[1].splitlines()[0]


def test_vtt_skips_blank_segments(segments):
    padded = [segments[0], {"speaker": "SPEAKER_01", "start": 2.5, "end": 2.6, "text": None},
              segments[1]]
    assert vtt(padded).count("-->") == 2


def test_payload_json_round_trips(conversation, segments):
    raw = payload_json(meta={"duration": 30.0, "language": "it"}, segments=segments,
                       turns=conversation, names={"SPEAKER_00": "Ada Verdolini"},
                       matches=[{"label": "SPEAKER_00", "score": 0.81}])
    data = json.loads(raw)
    assert set(data) == {"meta", "names", "speaker_matches", "turns", "segments"}
    assert data["meta"]["language"] == "it"
    assert data["names"]["SPEAKER_00"] == "Ada Verdolini"
    assert data["speaker_matches"][0]["score"] == 0.81
    assert len(data["turns"]) == len(conversation)
    assert len(data["segments"]) == len(segments)


def test_payload_json_keeps_non_ascii_readable(segments):
    raw = payload_json(meta={}, segments=segments,
                       turns=[turn("SPEAKER_00", 0.0, "perché però")],
                       names={}, matches=[])
    assert "perché però" in raw


# --------------------------------------------------------------------------- #
# write_all / FILENAMES / _sweep
# --------------------------------------------------------------------------- #

STEM = "Nota vocale 2026-03-04"


@pytest.fixture
def job(conversation, segments):
    return {
        "turns": conversation,
        "segments": segments,
        "names": {"SPEAKER_00": "Ada Verdolini"},
        "meta": {
            "title": "Riunione di redazione",
            "recorded": datetime(2026, 3, 4, 9, 30),
            "duration": 95.0,
            "language": "it",
            "source_file": "nota vocale.m4a",
            "speaker_stats": {"SPEAKER_00": 60.0, "SPEAKER_01": 35.0},
            "unresolved": ["Voce 2"],
            "recorded_source": "file modified",
            "device": "Registratore da tasca",
        },
        "matches": [{"label": "SPEAKER_00", "score": 0.81}],
    }


@pytest.mark.parametrize("fmt,expected", [
    ("source", f"{STEM} (source).md"),
    ("md", f"{STEM}.md"),
    ("txt", f"{STEM}.txt"),
    ("srt", f"{STEM}.srt"),
    ("vtt", f"{STEM}.vtt"),
    ("json", f"{STEM}.json"),
])
def test_write_all_names_each_format(tmp_path, job, fmt, expected):
    written = write_all(tmp_path, STEM, [fmt], **job)
    assert [p.name for p in written] == [expected]
    assert written[0].exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [expected]


def test_write_all_writes_every_requested_format(tmp_path, job):
    formats = ["source", "md", "txt", "srt", "vtt", "json"]
    written = write_all(tmp_path, STEM, formats, **job)
    assert [p.name for p in written] == [FILENAMES[f].format(stem=STEM) for f in formats]
    assert all(p.exists() for p in written)


def test_write_all_creates_the_output_directory(tmp_path, job):
    outdir = tmp_path / "esiti" / "marzo"
    written = write_all(outdir, STEM, ["txt"], **job)
    assert outdir.is_dir()
    assert written[0].read_text().startswith("Ada Verdolini: ")


def test_write_all_source_file_holds_the_source_document(tmp_path, job):
    # The job is an Italian one, so the words around the text are Italian.
    path = write_all(tmp_path, STEM, ["source"], **job)[0]
    text = path.read_text()
    assert text.startswith("# Riunione di redazione")
    assert "- **Data**: 2026-03-04 09:30 (dall'ora di modifica del file" in text
    assert "- **Durata**: 00:01:35" in text
    assert "- **Registrato con**: Registratore da tasca" in text
    assert "- **Voci non identificate**: Voce 2." in text
    assert "**Voce 2 (non identificata) (attribuzione incerta)** [00:05]: Fai pure." in text


def test_write_all_source_file_in_english_for_an_english_job(tmp_path, job):
    job["meta"].update(language="en", unresolved=["Voice 2"])
    text = write_all(tmp_path, STEM, ["source"], **job)[0].read_text()
    assert text.startswith("# Riunione di redazione")
    assert "- **Date**: 2026-03-04 09:30 (from the file's modification time" in text
    assert "- **Duration**: 00:01:35" in text
    assert "- **Recorded with**: Registratore da tasca" in text
    assert "- **Unidentified voices**: Voice 2." in text
    assert "**Voice 2 (unidentified) (uncertain)** [00:05]: Fai pure." in text


def test_write_all_source_falls_back_to_the_stem_as_title(tmp_path, job):
    job["meta"].pop("title")
    text = write_all(tmp_path, STEM, ["source"], **job)[0].read_text()
    assert text.startswith(f"# {STEM}")


def test_write_all_json_serialises_the_recorded_datetime(tmp_path, job):
    data = json.loads(write_all(tmp_path, STEM, ["json"], **job)[0].read_text())
    assert data["meta"]["recorded"] == "2026-03-04T09:30:00"
    assert data["meta"]["duration"] == 95.0
    assert data["speaker_matches"] == job["matches"]


def test_write_all_rejects_an_unknown_format(tmp_path, job):
    with pytest.raises(ValueError) as excinfo:
        write_all(tmp_path, STEM, ["txt", "tzt"], **job)
    assert "tzt" in str(excinfo.value)


def test_filenames_and_suffixes_agree():
    # _sweep can only clean up what it recognises, so every name the module writes
    # has to end in a suffix the sweep looks at.
    for pattern in FILENAMES.values():
        assert pattern.endswith((".md", ".txt", ".srt", ".vtt", ".json")), pattern


def test_write_all_removes_a_stale_output_of_a_dropped_format(tmp_path, job):
    write_all(tmp_path, STEM, ["source", "md", "txt", "json"], **job)
    write_all(tmp_path, STEM, ["md"], **job)
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{STEM}.md"]


def test_write_all_leaves_the_folder_of_another_recording_alone(tmp_path, job):
    other = tmp_path / "Riunione di aprile.md"
    other.write_text("altro verbale")
    write_all(tmp_path, STEM, ["md"], **job)
    assert other.exists()
    assert other.read_text() == "altro verbale"


def test_sweep_keeps_files_the_module_never_writes(tmp_path, job):
    audio = tmp_path / f"{STEM}.wav"
    picture = tmp_path / f"{STEM}.png"
    audio.write_bytes(b"RIFF-not-really")
    picture.write_bytes(b"\x89PNG")
    write_all(tmp_path, STEM, ["md"], **job)
    assert audio.exists() and picture.exists()
    assert audio.read_bytes() == b"RIFF-not-really"


def test_sweep_deletes_only_the_formats_that_are_gone(tmp_path):
    for name in [f"{STEM}.md", f"{STEM}.txt", f"{STEM}.srt", f"{STEM} (source).md"]:
        (tmp_path / name).write_text("vecchio")
    _sweep(tmp_path, STEM, ["md", "srt"])
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{STEM}.md", f"{STEM}.srt"]


def test_sweep_ignores_an_unknown_format_name(tmp_path):
    (tmp_path / f"{STEM}.md").write_text("vecchio")
    _sweep(tmp_path, STEM, ["md", "tzt"])
    assert (tmp_path / f"{STEM}.md").exists()


def test_sweep_does_not_touch_a_directory(tmp_path):
    (tmp_path / f"{STEM}.txt").mkdir()
    _sweep(tmp_path, STEM, ["md"])
    assert (tmp_path / f"{STEM}.txt").is_dir()


def test_sweep_on_an_empty_folder_is_a_no_op(tmp_path):
    _sweep(tmp_path, STEM, ["md", "txt"])
    assert list(tmp_path.iterdir()) == []


# Regression. _sweep globbed the stem and checked only the suffix, so it deleted
# files it could never have written: the user's own "<stem> appunti.md", or the
# outputs of a neighbouring recording called "<stem>-2".
def test_sweep_keeps_a_file_that_merely_starts_with_the_stem(tmp_path):
    notes = tmp_path / f"{STEM} appunti miei.md"
    neighbour = tmp_path / f"{STEM}-2.md"
    notes.write_text("appunti scritti a mano")
    neighbour.write_text("altra registrazione")
    _sweep(tmp_path, STEM, ["md"])
    assert notes.exists(), "a hand-written note in the output folder was deleted"
    assert neighbour.exists(), "another recording's output was deleted"


# Regression. The stem went into a glob pattern unescaped, so a bracket in a
# filename became a character class. For "Memo [2026]" the glob matched nothing,
# the sweep did nothing, and stale outputs survived beside the fresh ones.
def test_sweep_handles_a_stem_containing_glob_characters(tmp_path):
    bracketed = "Memo [2026]"
    (tmp_path / f"{bracketed}.md").write_text("fresco")
    stale = tmp_path / f"{bracketed}.txt"
    stale.write_text("vecchio")
    _sweep(tmp_path, bracketed, ["md"])
    assert not stale.exists(), "stale output survived the sweep"


# Regression. write_all swept and wrote before validating the format names, so
# one typo deleted the previous outputs of the dropped formats and left the
# folder half updated, after the transcription had already been paid for.
def test_write_all_validates_formats_before_deleting_anything(tmp_path, job):
    write_all(tmp_path, STEM, ["md", "txt"], **job)
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(ValueError):
        write_all(tmp_path, STEM, ["md", "tzt"], **job)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_module_exposes_the_expected_format_names():
    assert set(FILENAMES) == {"source", "md", "txt", "srt", "vtt", "json"}
    assert export.FILENAMES["source"].format(stem="X") == "X (source).md"


# --------------------------------------------------------------------------- #
# the words around the text follow the language of the recording
# --------------------------------------------------------------------------- #
#
# An Italian recording used to come out under "Overview" and "Participants". The
# headings, the labels, the markers and the sentences the exporter writes itself are
# now in the language of the job, for Italian, Spanish and English, and English for
# any other. What the app and other programs read stays as it was: the file names,
# the JSON keys, the SRT and VTT structure, and the language code in the header.

LANGUAGES = ["en", "it", "es"]


def shaky_conversation(language):
    """Every optional line of the header, and every kind of marker, in one document."""
    turns = [
        turn("SPEAKER_00", 0.0, "Comincio io.", 0.97),
        turn("SPEAKER_01", 5.0, "Fai pure.", 0.41),
        turn("SPEAKER_02", 9.0, "Anch'io.", 0.45),
        turn("SPEAKER_00", 12.0, "Il punto e' la scadenza."),
        turn(None, 20.0, "(rumore di fondo)", 0.99),
    ]
    names = {"SPEAKER_00": "Ada Verdolini"}
    return source_doc(
        turns, title="Lake trip", names=names, language=language,
        recorded=datetime(2026, 3, 4, 9, 30), recorded_source="file modified",
        duration=95.0, source_file="lake.m4a", device="Pocket recorder",
        speaker_stats={"SPEAKER_00": 620.0, "SPEAKER_01": 95.0, "SPEAKER_02": 40.0},
        unresolved=[display("SPEAKER_01", names, language),
                    display("SPEAKER_02", names, language)],
        untranscribed_seconds=42.0, speech_seconds=300.0,
        untranscribed_gaps=[[10.0, 25.0], [100.0, 112.0]],
    )


def line_starting(doc, prefix):
    return next(ln for ln in doc.splitlines() if ln.startswith(prefix))


@pytest.mark.parametrize("code,heading", [
    ("en", "Overview"), ("it", "Panoramica"), ("es", "Visión general"),
    # Case and region are what a person types or a tool reports.
    ("IT", "Panoramica"), ("it-IT", "Panoramica"), ("it_IT", "Panoramica"),
    ("ES-mx", "Visión general"), (" es ", "Visión general"), ("EN", "Overview"),
])
def test_words_find_the_table_for_a_language_code(code, heading):
    assert export.words(code)["overview"] == heading


@pytest.mark.parametrize("code", ["de", "fr", "pt", "zh", "auto", "", None, "xx-YY"])
def test_words_fall_back_to_english_for_anything_else(code):
    assert export.words(code) is export.words("en")


def test_every_language_has_every_string_with_the_same_blanks():
    # A string missing from one table raises KeyError at export time, after the
    # transcription was paid for. A blank that differs (a {n} in one language and
    # none in another) raises the same way or silently drops the figure.
    import re
    blanks = lambda text: sorted(re.findall(r"\{(\w+)\}", text))
    english = export._WORDS["en"]
    for code in LANGUAGES:
        table = export._WORDS[code]
        assert set(table) == set(english), code
        for key, text in table.items():
            assert text.strip(), (code, key)
            assert blanks(text) == blanks(english[key]), (code, key)
            assert "—" not in text, (code, key)


def test_only_the_three_languages_are_in_the_table():
    assert sorted(export._WORDS) == sorted(LANGUAGES)


# ---- the voice names

@pytest.mark.parametrize("code,first,none", [
    ("en", "Voice 1", "Unattributed"),
    ("it", "Voce 1", "Non attribuito"),
    ("es", "Voz 1", "Sin atribuir"),
])
def test_display_names_the_voice_in_the_language(code, first, none):
    assert display("SPEAKER_00", None, code) == first
    assert display(None, None, code) == none
    assert display(None, {"SPEAKER_00": "Ada Verdolini"}, code) == none


def test_display_counts_from_one_in_every_language():
    assert display("SPEAKER_09", None, "it") == "Voce 10"
    assert display("SPEAKER_01", {}, "es") == "Voz 2"


def test_display_a_name_or_an_odd_label_is_never_translated():
    names = {"SPEAKER_00": "Ada Verdolini"}
    for code in LANGUAGES:
        assert display("SPEAKER_00", names, code) == "Ada Verdolini"
        assert display("SPEAKER_XY", None, code) == "SPEAKER_XY"
        assert display("intervistatore", None, code) == "intervistatore"


def test_display_without_a_language_is_english():
    assert display("SPEAKER_00", None) == "Voice 1"
    assert display("SPEAKER_00", None, "de") == "Voice 1"


# ---- the source document, in each language

def test_italian_document_has_italian_headings_and_labels():
    doc = shaky_conversation("it")
    for heading in ("## Panoramica", "## Trascrizione", "## Turni da verificare"):
        assert heading in doc
    assert "- **Data**: 2026-03-04 09:30 (dall'ora di modifica del file, che può non " \
           "coincidere con il momento della conversazione)" in doc
    assert "- **Durata**: 00:01:35" in doc
    assert "- **Lingua**: it" in doc
    assert "- **File di origine**: lake.m4a" in doc
    assert "- **Registrato con**: Pocket recorder" in doc
    assert ("- **Partecipanti**: Ada Verdolini (10:20 di parlato), "
            "Voce 2 (01:35 di parlato), Voce 3 (00:40 di parlato)") in doc
    assert line_starting(doc, "- **Voci non identificate**").startswith(
        "- **Voci non identificate**: Voce 2, Voce 3. Persone distinte. I loro nomi "
        "non vengono mai pronunciati nella registrazione. Non indovinare chi siano.")
    assert line_starting(doc, "- **Attribuzione**").startswith(
        "- **Attribuzione**: 2 turni su 5 sono segnati come *(attribuzione incerta)*.")
    assert line_starting(doc, "- **Non trascritto**").startswith(
        "- **Non trascritto**: 00:42 dei 05:00 di parlato di questa registrazione")
    assert line_starting(doc, "- **Nota**").startswith("- **Nota**: questo è riconoscimento")
    assert transcript_lines(doc, "it") == [
        "**Ada Verdolini** [00:00]: Comincio io.",
        "**Voce 2 (non identificata) (attribuzione incerta)** [00:05]: Fai pure.",
        "**Voce 3 (non identificata) (attribuzione incerta)** [00:09]: Anch'io.",
        "**Ada Verdolini** [00:12]: Il punto e' la scadenza.",
        "**Non attribuito** [00:20]: (rumore di fondo)",
    ]
    assert check_lines(doc, "it") == [
        "- [00:05] Voce 2 (non identificata): Fai pure.",
        "- [00:09] Voce 3 (non identificata): Anch'io.",
    ]


def test_spanish_document_has_spanish_headings_and_labels():
    doc = shaky_conversation("es")
    for heading in ("## Visión general", "## Transcripción", "## Turnos por revisar"):
        assert heading in doc
    assert "- **Fecha**: 2026-03-04 09:30 (de la hora de modificación del archivo, que " \
           "puede no coincidir con el momento de la conversación)" in doc
    assert "- **Duración**: 00:01:35" in doc
    assert "- **Idioma**: es" in doc
    assert "- **Archivo de origen**: lake.m4a" in doc
    assert "- **Grabado con**: Pocket recorder" in doc
    assert ("- **Participantes**: Ada Verdolini (10:20 de habla), "
            "Voz 2 (01:35 de habla), Voz 3 (00:40 de habla)") in doc
    assert line_starting(doc, "- **Voces no identificadas**").startswith(
        "- **Voces no identificadas**: Voz 2, Voz 3. Personas distintas. Sus nombres "
        "no se dicen nunca en la grabación. No adivines quiénes son.")
    assert line_starting(doc, "- **Atribución**").startswith(
        "- **Atribución**: 2 turnos de 5 están marcados como *(atribución incierta)*.")
    assert line_starting(doc, "- **No transcrito**").startswith(
        "- **No transcrito**: 00:42 de los 05:00 de habla de esta grabación")
    assert line_starting(doc, "- **Nota**").startswith("- **Nota**: esto es reconocimiento")
    assert transcript_lines(doc, "es") == [
        "**Ada Verdolini** [00:00]: Comincio io.",
        "**Voz 2 (no identificada) (atribución incierta)** [00:05]: Fai pure.",
        "**Voz 3 (no identificada) (atribución incierta)** [00:09]: Anch'io.",
        "**Ada Verdolini** [00:12]: Il punto e' la scadenza.",
        "**Sin atribuir** [00:20]: (rumore di fondo)",
    ]
    assert check_lines(doc, "es") == [
        "- [00:05] Voz 2 (no identificada): Fai pure.",
        "- [00:09] Voz 3 (no identificada): Anch'io.",
    ]


def test_english_document_is_the_one_it_always_was():
    doc = shaky_conversation("en")
    assert "## Overview" in doc and "## Transcript" in doc and "## Turns to check" in doc
    assert "- **Date**: 2026-03-04 09:30 (from the file's modification time, which may " \
           "not be when the conversation happened)" in doc
    assert "- **Participants**: Ada Verdolini (10:20 of speech), Voice 2 (01:35 of " \
           "speech), Voice 3 (00:40 of speech)" in doc
    assert "- **Attribution**: 2 of 5 turns are marked *(uncertain)*." in doc
    assert "**Voice 2 (unidentified) (uncertain)** [00:05]: Fai pure." in doc
    assert "**Unattributed** [00:20]: (rumore di fondo)" in doc


@pytest.mark.parametrize("code", ["de", "auto", "", "pt-BR"])
def test_a_document_in_another_language_is_in_english_and_keeps_its_code(code):
    doc = shaky_conversation(code)
    assert "## Overview" in doc and "## Transcript" in doc
    assert "**Voice 2 (unidentified) (uncertain)** [00:05]: Fai pure." in doc
    assert f"- **Language**: {code}" in doc


@pytest.mark.parametrize("code", ["it", "es"])
def test_no_english_words_are_left_in_a_translated_document(code):
    doc = shaky_conversation(code)
    english = export._WORDS["en"]
    # The recording's own text is Italian whatever the document says, so only the
    # words the exporter adds are looked for. Each English string must be gone.
    for key in ("overview", "transcript", "to_check", "date", "duration", "language",
                "source_file", "recorded_with", "participants", "unidentified_voices",
                "attribution", "not_transcribed", "note", "unidentified", "uncertain",
                "unattributed"):
        assert english[key] not in doc, (code, key)
    for sentence in ("Do not guess", "are marked", "never spoken", "automatic speech",
                     "of speech", "going back to the audio", "The longest of these"):
        assert sentence not in doc, (code, sentence)


def test_the_document_has_the_same_shape_in_every_language():
    # Translating must not move anything: the same lines in the same order, each of
    # the same kind. A heading that lost its "## " or a bullet that lost its bold
    # label would still read fine in a language the author cannot check.
    def shape(doc):
        return [("" if not ln else "title" if ln.startswith("# ") else "h2" if ln.startswith("## ")
                 else "field" if ln.startswith("- **") else "turn" if ln.startswith("**")
                 else "check" if ln.startswith("- [") else "text") for ln in doc.splitlines()]

    english = shape(shaky_conversation("en"))
    assert "text" in english and "check" in english and "field" in english
    for code in ("it", "es", "de"):
        assert shape(shaky_conversation(code)) == english, code


@pytest.mark.parametrize("code", LANGUAGES)
def test_the_uncertain_marker_in_the_header_is_the_one_on_the_lines(code):
    # The header tells whatever reads the file what the marker means, so the two
    # have to be spelled the same or the explanation points at nothing.
    import re
    doc = shaky_conversation(code)
    header = line_starting(doc, "- **" + export.words(code)["attribution"] + "**")
    marker = re.search(r"\*(\(.+?\))\*", header).group(1)
    assert marker == export.words(code)["uncertain"]
    marked = [ln for ln in transcript_lines(doc, code) if marker in ln]
    assert len(marked) == 2
    assert all(ln.startswith("**") and ln.split("**")[1].endswith(marker) for ln in marked)


@pytest.mark.parametrize("code,one,many", [
    ("en", "1 of 3 turns is marked", "2 of 3 turns are marked"),
    ("it", "1 turno su 3 è segnato come", "2 turni su 3 sono segnati come"),
    ("es", "1 turno de 3 está marcado como", "2 turnos de 3 están marcados como"),
])
def test_the_attribution_line_agrees_with_the_count(code, one, many):
    word = export.words(code)["attribution"]
    one_shaky = [turn("SPEAKER_00", 0, "Uno.", 0.3), turn("SPEAKER_00", 5, "Due."),
                 turn("SPEAKER_00", 9, "Tre.")]
    two_shaky = [turn("SPEAKER_00", 0, "Uno.", 0.3), turn("SPEAKER_00", 5, "Due.", 0.2),
                 turn("SPEAKER_00", 9, "Tre.")]
    assert one in line_starting(source_doc(one_shaky, title="T", language=code), f"- **{word}**")
    assert many in line_starting(source_doc(two_shaky, title="T", language=code), f"- **{word}**")


@pytest.mark.parametrize("code,one,many", [
    ("en", "A distinct person. Their name is never spoken",
     "Distinct people. Their names are never spoken"),
    ("it", "Una persona distinta. Il suo nome non viene mai pronunciato",
     "Persone distinte. I loro nomi non vengono mai pronunciati"),
    ("es", "Una persona distinta. Su nombre no se dice nunca",
     "Personas distintas. Sus nombres no se dicen nunca"),
])
def test_the_unidentified_line_agrees_with_the_count(conversation, code, one, many):
    label = export.words(code)["unidentified_voices"]
    single = source_doc(conversation, title="T", language=code,
                        unresolved=[display("SPEAKER_01", None, code)])
    pair = source_doc(conversation, title="T", language=code,
                      unresolved=[display("SPEAKER_01", None, code),
                                  display("SPEAKER_00", None, code)])
    assert one in line_starting(single, f"- **{label}**")
    assert many in line_starting(pair, f"- **{label}**")


@pytest.mark.parametrize("code,total,where", [
    ("en", "00:42 of the 05:00 of speech", "The longest of these start at 00:10, 01:40."),
    ("it", "00:42 dei 05:00 di parlato", "I più lunghi iniziano a 00:10, 01:40."),
    ("es", "00:42 de los 05:00 de habla", "Los más largos empiezan en 00:10, 01:40."),
])
def test_the_not_transcribed_line_says_how_much_and_where(code, total, where):
    label = export.words(code)["not_transcribed"]
    gaps = [[10.0, 25.0], [100.0, 112.0]]
    with_gaps = source_doc([turn("SPEAKER_00", 0, "Ciao.")], title="T", language=code,
                           untranscribed_seconds=42.0, speech_seconds=300.0,
                           untranscribed_gaps=gaps)
    line = line_starting(with_gaps, f"- **{label}**")
    assert total in line and line.endswith(where)

    no_gaps = source_doc([turn("SPEAKER_00", 0, "Ciao.")], title="T", language=code,
                         untranscribed_seconds=42.0, speech_seconds=300.0)
    assert total in line_starting(no_gaps, f"- **{label}**")
    assert where.split(" at ")[0].split(" a ")[0] not in line_starting(no_gaps, f"- **{label}**")

    short = source_doc([turn("SPEAKER_00", 0, "Ciao.")], title="T", language=code,
                       untranscribed_seconds=4.9, speech_seconds=300.0)
    assert f"**{label}**" not in short


@pytest.mark.parametrize("code,plain_date", [
    ("en", "- **Date**: 2026-03-04 09:30"),
    ("it", "- **Data**: 2026-03-04 09:30"),
    ("es", "- **Fecha**: 2026-03-04 09:30"),
])
def test_the_date_says_where_it_came_from_in_every_language(code, plain_date):
    when = datetime(2026, 3, 4, 9, 30)
    kinds = {"en": ("creation", "modification"), "it": ("creazione", "modifica"),
             "es": ("creación", "modificación")}[code]
    # The recorder's own stamp is stated plainly.
    for given in ("recorded", ""):
        doc = source_doc([turn("SPEAKER_00", 0, "Ciao.")], title="T", language=code,
                         recorded=when, recorded_source=given)
        assert line_starting(doc, "- **").startswith(plain_date)
        assert plain_date + " (" not in doc
    # What the filesystem guessed is flagged, with the words for how it was guessed.
    for source, kind in zip(("file created", "file modified"), kinds):
        doc = source_doc([turn("SPEAKER_00", 0, "Ciao.")], title="T", language=code,
                         recorded=when, recorded_source=source)
        line = line_starting(doc, plain_date)
        assert plain_date + " (" in line and kind in line
        assert "file created" not in line and "file modified" not in line
    # A source this module has no words for is written as it came, not lost.
    doc = source_doc([turn("SPEAKER_00", 0, "Ciao.")], title="T", language=code,
                     recorded=when, recorded_source="the scanner")
    assert "the scanner" in line_starting(doc, plain_date)


# The whole line, for every source the engine can store, in every language. A table
# and not a loop over the translations: what has to be true is how each of them
# reads, and that "file" is said once.
DATE_LINES = [
    ("en", "file created", "- **Date**: 2026-03-04 09:30 (from the file's creation "
                           "time, which may not be when the conversation happened)"),
    ("en", "file modified", "- **Date**: 2026-03-04 09:30 (from the file's modification "
                            "time, which may not be when the conversation happened)"),
    ("en", "the scanner", "- **Date**: 2026-03-04 09:30 (date taken from: the scanner; "
                          "it may not be when the conversation happened)"),
    ("it", "file created", "- **Data**: 2026-03-04 09:30 (dall'ora di creazione del "
                           "file, che può non coincidere con il momento della "
                           "conversazione)"),
    ("it", "file modified", "- **Data**: 2026-03-04 09:30 (dall'ora di modifica del "
                            "file, che può non coincidere con il momento della "
                            "conversazione)"),
    ("it", "the scanner", "- **Data**: 2026-03-04 09:30 (data ricavata da: the "
                          "scanner; può non coincidere con il momento della "
                          "conversazione)"),
    ("es", "file created", "- **Fecha**: 2026-03-04 09:30 (de la hora de creación del "
                           "archivo, que puede no coincidir con el momento de la "
                           "conversación)"),
    ("es", "file modified", "- **Fecha**: 2026-03-04 09:30 (de la hora de modificación "
                            "del archivo, que puede no coincidir con el momento de la "
                            "conversación)"),
    ("es", "the scanner", "- **Fecha**: 2026-03-04 09:30 (fecha obtenida de: the "
                          "scanner; puede no coincidir con el momento de la "
                          "conversación)"),
]


@pytest.mark.parametrize("code,source,expected", DATE_LINES)
def test_the_date_line_reads_whole_for_every_source_in_every_language(
        code, source, expected):
    doc = source_doc([turn("SPEAKER_00", 0, "Ciao.")], title="T", language=code,
                     recorded=datetime(2026, 3, 4, 9, 30), recorded_source=source)
    line = line_starting(doc, "- **" + export.words(code)["date"])
    assert line == expected
    # The value the state keeps never reaches the page, and "file" is not said twice:
    # the old sentence read "from the file's file created time".
    if source in export._FILE_TIME_KINDS:
        assert source not in line
    assert "file's file" not in line


@pytest.mark.parametrize("code", ["en", "it", "es", "de"])
def test_every_source_the_engine_can_store_has_its_own_words(code):
    """Tied to audio.recorded_at itself, so a third source added there fails here.

    Both filesystem branches are run for real, and what each returns goes into the
    document. A source nobody wrote a word for would come out in the sentence meant
    for the unknown, which names it as it came: a state value such as "file
    created" on the page is exactly that.
    """
    when = datetime(2026, 3, 4, 9, 30)
    born = SimpleNamespace(st_birthtime=when.timestamp(), st_mtime=when.timestamp())
    only_modified = SimpleNamespace(st_mtime=when.timestamp())
    seen = set()
    for stat in (born, only_modified):
        info = SimpleNamespace(created="", path=SimpleNamespace(stat=lambda s=stat: s))
        recorded, source = audio.recorded_at(info)
        seen.add(source)
        doc = source_doc([turn("SPEAKER_00", 0, "Ciao.")], title="T", language=code,
                         recorded=recorded, recorded_source=source)
        line = line_starting(doc, "- **" + export.words(code)["date"])
        assert source in export._FILE_TIME_KINDS
        assert source not in line
        assert export.words(code)["date_from_other"].split("{")[0] not in line
    assert seen == {"file created", "file modified"}


def test_a_translated_document_with_nothing_to_flag_is_still_whole():
    turns = [turn("SPEAKER_00", 0.0, "Tutto chiaro.", 0.99)]
    for code in ("it", "es"):
        doc = source_doc(turns, title="T", language=code,
                         names={"SPEAKER_00": "Ada Verdolini"})
        assert doc.startswith("# T\n\n## " + export.words(code)["overview"])
        assert transcript_lines(doc, code) == ["**Ada Verdolini** [00:00]: Tutto chiaro."]
        assert export.words(code)["to_check"] not in doc
        assert export.words(code)["attribution"] not in doc
        assert "\n\n\n" not in doc


# ---- the other formats

@pytest.mark.parametrize("code,two,none", [
    ("en", "Voice 2", "Unattributed"),
    ("it", "Voce 2", "Non attribuito"),
    ("es", "Voz 2", "Sin atribuir"),
])
def test_the_tool_formats_label_the_voices_in_the_language(conversation, segments,
                                                            code, two, none):
    names = {"SPEAKER_00": "Ada Verdolini"}
    assert f"**{two}** [00:05]  \nFai pure." in markdown(conversation, names=names, language=code)
    assert f"{none}: (rumore di fondo)" in plain(conversation, names=names, language=code)
    assert f"[{two}] Fai pure.\n" in srt(segments, names=names, language=code)
    assert f"<v {two}>Fai pure." in vtt(segments, names=names, language=code)


def test_subtitles_keep_their_structure_in_every_language(segments):
    # Only the label inside the brackets may differ. The numbering, the timecodes and
    # the arrows are what a video editor parses.
    def skeleton(text):
        return [ln for ln in text.splitlines()
                if "-->" in ln or ln.strip().isdigit() or ln in ("WEBVTT", "")]

    for code in ("it", "es", "de"):
        assert skeleton(srt(segments, language=code)) == skeleton(srt(segments))
        assert skeleton(vtt(segments, language=code)) == skeleton(vtt(segments))
    assert srt(segments, language="it").startswith("1\n00:00:00,000 --> 00:00:02,500\n")
    assert vtt(segments, language="es").startswith("WEBVTT\n\n00:00:00.000 --> 00:00:02.500\n")


# ---- write_all

@pytest.mark.parametrize("code", ["en", "it", "es", "de"])
def test_write_all_names_the_files_the_same_in_every_language(tmp_path, job, code):
    # The app finds the document by "(source).md" and the folder is swept by name.
    job["meta"]["language"] = code
    written = write_all(tmp_path, STEM, ["source", "md", "txt", "srt", "vtt", "json"], **job)
    assert [p.name for p in written] == [
        f"{STEM} (source).md", f"{STEM}.md", f"{STEM}.txt", f"{STEM}.srt",
        f"{STEM}.vtt", f"{STEM}.json"]


@pytest.mark.parametrize("code,voice", [("en", "Voice 2"), ("it", "Voce 2"),
                                        ("es", "Voz 2"), ("de", "Voice 2")])
def test_write_all_gives_every_file_the_language_of_the_job(tmp_path, job, code, voice):
    job["meta"]["language"] = code
    job["meta"]["unresolved"] = [voice]
    write_all(tmp_path, STEM, ["source", "md", "txt", "srt", "vtt"], **job)
    for name in (f"{STEM} (source).md", f"{STEM}.md", f"{STEM}.txt",
                 f"{STEM}.srt", f"{STEM}.vtt"):
        text = (tmp_path / name).read_text()
        assert voice in text, name
        other = {"Voice 2", "Voce 2", "Voz 2"} - {voice}
        assert not any(o in text for o in other), name


def test_write_all_json_is_the_same_whatever_the_language(tmp_path, job):
    # The keys, the raw speaker labels and the language code are for scripts.
    payloads = {}
    for code in LANGUAGES:
        job["meta"]["language"] = code
        folder = tmp_path / code
        data = json.loads(write_all(folder, STEM, ["json"], **job)[0].read_text())
        assert data["meta"]["language"] == code
        assert sorted(data) == ["meta", "names", "segments", "speaker_matches", "turns"]
        data["meta"].pop("language")
        payloads[code] = data
    assert payloads["it"] == payloads["en"] == payloads["es"]
    assert {t["speaker"] for t in payloads["it"]["turns"]} == {"SPEAKER_00", "SPEAKER_01", None}


def test_write_all_without_a_language_keeps_assuming_italian(tmp_path, job):
    # The pipeline's own default when a job has no language recorded.
    job["meta"].pop("language")
    text = write_all(tmp_path, STEM, ["source"], **job)[0].read_text()
    assert "## Panoramica" in text and "- **Lingua**: it" in text
