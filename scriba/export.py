"""Output formats. The one that really matters is `source`.

A source is the document you hand to something else: a colleague, a notes app, a
model with a context window. Whatever reads it reads prose, not a data structure,
so anything that is not readable text is noise. No JSON, no millisecond timestamps,
no one block per sentence. The header carries the metadata (who is there, when, how
long it runs) because that is what anyone asks about first. Then come real
conversation turns, with the name up front.

The other formats exist for tools that want them: subtitles for a video editor,
JSON for a script, plain text for grep.

The words scriba adds around what was said (headings, labels, the voice names) are in
the language of the recording, for the three languages it was written for.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# helper
# --------------------------------------------------------------------------- #

def hhmmss(seconds: float, *, always_hours: bool = False) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h or always_hours:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def srt_time(seconds: float) -> str:
    # Round to milliseconds first, then split. Rounding the fraction on its own
    # lets 0.9996 come out as "00:00:00,1000", a four-digit field that strict
    # subtitle parsers reject outright.
    total_ms = int(round(max(0.0, float(seconds)) * 1000))
    ms = total_ms % 1000
    h, rem = divmod(total_ms // 1000, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def vtt_time(seconds: float) -> str:
    return srt_time(seconds).replace(",", ".")


# --------------------------------------------------------------------------- #
# the words scriba writes itself
# --------------------------------------------------------------------------- #

# Headings, labels, markers and the few sentences of the source document, per
# language. What the recording said stays as it was said, so an Italian recording
# came out under "Overview" and "Participants", which read as though the document
# were in another language than its own text. Italian, Spanish and English are the
# three this was written and checked for; any other language gets English, because
# a table nobody has read in that language would be worse than a plain fallback.
#
# What is not in here on purpose: the file names (the app looks for
# "<stem> (source).md"), the JSON keys, the SRT and VTT structure, and the language
# code in the header, which is the value another program reads.
#
# A sentence is written whole in each language rather than assembled from English
# fragments, because the order of the clauses and the agreement of "is" and "are"
# differ between them. The {braces} are the only parts that vary.
_WORDS: dict[str, dict[str, str]] = {
    "en": {
        "unattributed": "Unattributed",
        "voice": "Voice {n}",
        "unidentified": "(unidentified)",
        "uncertain": "(uncertain)",
        "overview": "Overview",
        "transcript": "Transcript",
        "to_check": "Turns to check",
        "date": "Date",
        "duration": "Duration",
        "language": "Language",
        "source_file": "Source file",
        "recorded_with": "Recorded with",
        "participants": "Participants",
        "unidentified_voices": "Unidentified voices",
        "attribution": "Attribution",
        "not_transcribed": "Not transcribed",
        "note": "Note",
        "of_speech": "{name} ({share} of speech)",
        "date_from": "(from the file's {kind} time, which may not be when the "
                     "conversation happened)",
        "kind_created": "creation",
        "kind_modified": "modification",
        "date_from_other": "(date taken from: {source}; it may not be when the "
                           "conversation happened)",
        "unresolved_one": "A distinct person. Their name is never spoken in the "
                          "recording. Do not guess who they are.",
        "unresolved_many": "Distinct people. Their names are never spoken in the "
                           "recording. Do not guess who they are.",
        "attribution_one": "{n} of {total} turns is marked *{marker}*.",
        "attribution_many": "{n} of {total} turns are marked *{marker}*.",
        "attribution_rest": "The words are transcribed; which of the speakers said "
                            "them is a guess, usually where two people talk over each "
                            "other. Do not attribute a quote from those turns to a "
                            "named person.",
        "missing": "{missing} of the {speech} of speech in this recording produced "
                   "no text: the diarizer heard somebody there and the transcriber "
                   "wrote nothing. Do not read the absence of a topic as evidence it "
                   "was never raised.",
        "missing_where": "The longest of these start at {where}.",
        "note_text": "this is automatic speech recognition. Figures, dates and proper "
                     "names are the parts it gets wrong most often, and they are "
                     "worth checking against the audio before relying on them.",
        "to_check_intro": "Speaker attribution is weakest here. The timestamps are "
                          "for going back to the audio.",
    },
    "it": {
        "unattributed": "Non attribuito",
        "voice": "Voce {n}",
        "unidentified": "(non identificata)",
        "uncertain": "(attribuzione incerta)",
        "overview": "Panoramica",
        "transcript": "Trascrizione",
        "to_check": "Turni da verificare",
        "date": "Data",
        "duration": "Durata",
        "language": "Lingua",
        "source_file": "File di origine",
        "recorded_with": "Registrato con",
        "participants": "Partecipanti",
        "unidentified_voices": "Voci non identificate",
        "attribution": "Attribuzione",
        "not_transcribed": "Non trascritto",
        "note": "Nota",
        "of_speech": "{name} ({share} di parlato)",
        "date_from": "(dall'ora di {kind} del file, che può non coincidere con il "
                     "momento della conversazione)",
        "kind_created": "creazione",
        "kind_modified": "modifica",
        "date_from_other": "(data ricavata da: {source}; può non coincidere con il "
                           "momento della conversazione)",
        "unresolved_one": "Una persona distinta. Il suo nome non viene mai pronunciato "
                          "nella registrazione. Non indovinare chi sia.",
        "unresolved_many": "Persone distinte. I loro nomi non vengono mai pronunciati "
                           "nella registrazione. Non indovinare chi siano.",
        "attribution_one": "{n} turno su {total} è segnato come *{marker}*.",
        "attribution_many": "{n} turni su {total} sono segnati come *{marker}*.",
        "attribution_rest": "Le parole sono trascritte; chi le abbia dette è un'ipotesi, "
                            "di solito dove due persone parlano una sopra l'altra. Non "
                            "attribuire una citazione di quei turni a una persona con "
                            "nome.",
        "missing": "{missing} dei {speech} di parlato di questa registrazione non "
                   "hanno prodotto testo: la separazione delle voci ha rilevato "
                   "qualcuno che parlava e la trascrizione non ha scritto nulla. Non "
                   "leggere l'assenza di un argomento come prova che non sia mai "
                   "stato affrontato.",
        "missing_where": "I più lunghi iniziano a {where}.",
        "note_text": "questo è riconoscimento vocale automatico. Cifre, date e nomi "
                     "propri sono le parti che sbaglia più spesso, e conviene "
                     "controllarli sull'audio prima di fidarsene.",
        "to_check_intro": "L'attribuzione dei parlanti è più debole qui. I timestamp "
                          "servono per tornare all'audio.",
    },
    "es": {
        "unattributed": "Sin atribuir",
        "voice": "Voz {n}",
        "unidentified": "(no identificada)",
        "uncertain": "(atribución incierta)",
        "overview": "Visión general",
        "transcript": "Transcripción",
        "to_check": "Turnos por revisar",
        "date": "Fecha",
        "duration": "Duración",
        "language": "Idioma",
        "source_file": "Archivo de origen",
        "recorded_with": "Grabado con",
        "participants": "Participantes",
        "unidentified_voices": "Voces no identificadas",
        "attribution": "Atribución",
        "not_transcribed": "No transcrito",
        "note": "Nota",
        "of_speech": "{name} ({share} de habla)",
        "date_from": "(de la hora de {kind} del archivo, que puede no coincidir con "
                     "el momento de la conversación)",
        "kind_created": "creación",
        "kind_modified": "modificación",
        "date_from_other": "(fecha obtenida de: {source}; puede no coincidir con el "
                           "momento de la conversación)",
        "unresolved_one": "Una persona distinta. Su nombre no se dice nunca en la "
                          "grabación. No adivines quién es.",
        "unresolved_many": "Personas distintas. Sus nombres no se dicen nunca en la "
                           "grabación. No adivines quiénes son.",
        "attribution_one": "{n} turno de {total} está marcado como *{marker}*.",
        "attribution_many": "{n} turnos de {total} están marcados como *{marker}*.",
        "attribution_rest": "Las palabras están transcritas; quién las dijo es una "
                            "conjetura, normalmente donde dos personas hablan a la "
                            "vez. No atribuyas una cita de esos turnos a una persona "
                            "con nombre.",
        "missing": "{missing} de los {speech} de habla de esta grabación no "
                   "produjeron texto: la separación de voces detectó a alguien "
                   "hablando y la transcripción no escribió nada. No interpretes la "
                   "ausencia de un tema como prueba de que nunca se planteó.",
        "missing_where": "Los más largos empiezan en {where}.",
        "note_text": "esto es reconocimiento automático del habla. Las cifras, las "
                     "fechas y los nombres propios son lo que más suele fallar, y "
                     "conviene comprobarlos con el audio antes de fiarse de ellos.",
        "to_check_intro": "La atribución de hablantes es más débil aquí. Las marcas "
                          "de tiempo sirven para volver al audio.",
    },
}

# The values the job state keeps for where the date came from (audio.recorded_at),
# mapped to the words that say it. "recorded" is the recorder's own stamp and never
# reaches this: only the filesystem's guesses are explained.
_FILE_TIME_KINDS = {"file created": "kind_created", "file modified": "kind_modified"}


def words(language: str | None) -> dict[str, str]:
    """The table for a language: Italian, Spanish or English, English for any other.

    Reads what the job state holds and what a person types: case is ignored and a
    region is dropped, so "IT" and "es-MX" find their table. "auto" and an empty
    string are not languages and fall to English like any unknown code.
    """
    code = (language or "").strip().lower().replace("_", "-").split("-")[0]
    return _WORDS.get(code, _WORDS["en"])


def display(speaker: str | None, names: dict[str, str] | None,
            language: str = "en") -> str:
    w = words(language)
    if speaker is None:
        return w["unattributed"]
    if names and speaker in names:
        return names[speaker]
    # SPEAKER_00 -> "Voice 1". pyannote counts from zero and pads with a leading
    # zero. Nobody reading a document needs either, so we count from one.
    if speaker.startswith("SPEAKER_"):
        try:
            return w["voice"].format(n=int(speaker.removeprefix("SPEAKER_")) + 1)
        except ValueError:
            pass
    return speaker


# --------------------------------------------------------------------------- #
# source
# --------------------------------------------------------------------------- #

def source_doc(
    turns: list[dict],
    *,
    title: str,
    names: dict[str, str] | None = None,
    recorded: datetime | None = None,
    duration: float = 0.0,
    language: str = "en",
    source_file: str = "",
    speaker_stats: dict[str, float] | None = None,
    unresolved: list[str] | None = None,
    recorded_source: str = "",
    device: str = "",
    untranscribed_seconds: float = 0.0,
    speech_seconds: float = 0.0,
    untranscribed_gaps: list[list[float]] | None = None,
    uncertain_below: float = 0.8,
) -> str:
    names = names or {}
    # `language` is both the value the header shows and the language the document
    # is written in, and one nobody gave is English like any other the table does not
    # know. `unresolved` arrives already rendered by display(), in the same language,
    # so the voices it lists read the way the transcript below names them.
    w = words(language)
    lines: list[str] = []

    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"## {w['overview']}")
    lines.append("")
    if recorded:
        stamp = recorded.strftime("%Y-%m-%d %H:%M")
        # Say where the date came from. Only "recorded" is the time the recorder
        # wrote down; the others are the filesystem's guess, and an exported voice
        # memo carries the timestamp of the export rather than of the conversation.
        if recorded_source and recorded_source != "recorded":
            # The state keeps "file created" and "file modified", which already say
            # "file": put into "the file's {kind} time" they read "the file's file
            # created time". The table has the one word that fits the sentence. A
            # source this module has no word for is written as it came, in a
            # sentence of its own, since no kind of "time" can be assumed for it.
            if recorded_source in _FILE_TIME_KINDS:
                sentence = w["date_from"].format(
                    kind=w[_FILE_TIME_KINDS[recorded_source]])
            else:
                sentence = w["date_from_other"].format(source=recorded_source)
            lines.append(f"- **{w['date']}**: {stamp} " + sentence)
        else:
            lines.append(f"- **{w['date']}**: {stamp}")
    lines.append(f"- **{w['duration']}**: {hhmmss(duration, always_hours=True)}")
    lines.append(f"- **{w['language']}**: {language}")
    if source_file:
        lines.append(f"- **{w['source_file']}**: {source_file}")
    if device:
        lines.append(f"- **{w['recorded_with']}**: {device}")

    partecipanti = []
    for spk in sorted({t["speaker"] for t in turns if t.get("speaker")}):
        nome = display(spk, names, language)
        if speaker_stats and spk in speaker_stats:
            share = speaker_stats[spk]
            partecipanti.append(w["of_speech"].format(name=nome, share=hhmmss(share)))
        else:
            partecipanti.append(nome)
    if partecipanti:
        lines.append(f"- **{w['participants']}**: {', '.join(partecipanti)}")
    if unresolved:
        # Stating the uncertainty inside the source is the whole point: leave it out
        # and whatever reads the file treats "Voice 2" as an identified person.
        chi = ", ".join(sorted(unresolved))
        sentence = w["unresolved_many" if len(unresolved) > 1 else "unresolved_one"]
        lines.append(f"- **{w['unidentified_voices']}**: {chi}. {sentence}")
    shaky = [t for t in turns if float(t.get("confidence", 1.0)) < uncertain_below]
    if shaky:
        count = w["attribution_many" if len(shaky) != 1 else "attribution_one"].format(
            n=len(shaky), total=len(turns), marker=w["uncertain"])
        lines.append(f"- **{w['attribution']}**: {count} {w['attribution_rest']}")
    missing, speech = float(untranscribed_seconds or 0.0), float(speech_seconds or 0.0)
    if missing >= 5.0:
        # Say what is not here, and say where. A transcript reads as complete
        # whatever it leaves out, and somebody reading it a month later has no way
        # to tell. The timestamps are the point: they turn "something is missing"
        # into a place to go and listen.
        where = ", ".join(hhmmss(a) for a, _ in
                          sorted(untranscribed_gaps or [], key=lambda g: g[0] - g[1])[:5])
        text = w["missing"].format(missing=hhmmss(missing), speech=hhmmss(speech))
        if where:
            text += " " + w["missing_where"].format(where=where)
        lines.append(f"- **{w['not_transcribed']}**: {text}")
    lines.append(f"- **{w['note']}**: {w['note_text']}")
    lines.append("")
    lines.append(f"## {w['transcript']}")
    lines.append("")

    last_speaker = None
    for t in turns:
        nome = display(t.get("speaker"), names, language)
        stamp = hhmmss(t["start"])
        # Repeat the caveat on the line itself. The header says which voices went
        # unidentified, and by the middle of a long document that header is far away.
        # A model quoting a line reads the line, not the preamble.
        if t.get("speaker") and t["speaker"] not in names:
            nome = f"{nome} {w['unidentified']}"
        if float(t.get("confidence", 1.0)) < uncertain_below:
            nome = f"{nome} {w['uncertain']}"
        if nome != last_speaker:
            lines.append("")
        lines.append(f"**{nome}** [{stamp}]: {t['text'].strip()}")
        last_speaker = nome

    if shaky:
        lines.append("")
        lines.append(f"## {w['to_check']}")
        lines.append("")
        lines.append(w["to_check_intro"])
        lines.append("")
        for t in shaky:
            who = display(t.get("speaker"), names, language)
            # Same rule as the transcript. A quote lifted from this list without
            # the marker reads as though "Voice 2" were somebody identified.
            if t.get("speaker") and t["speaker"] not in names:
                who = f"{who} {w['unidentified']}"
            snippet = t["text"].strip()
            if len(snippet) > 90:
                snippet = snippet[:90].rstrip() + "…"
            lines.append(f"- [{hhmmss(t['start'])}] {who}: {snippet}")

    lines.append("")
    return "\n".join(lines).replace("\n\n\n", "\n\n")


# --------------------------------------------------------------------------- #
# other formats
# --------------------------------------------------------------------------- #

def markdown(turns: list[dict], *, names: dict[str, str] | None = None,
             language: str = "en") -> str:
    return "\n\n".join(
        f"**{display(t.get('speaker'), names, language)}** [{hhmmss(t['start'])}]  \n"
        f"{t['text'].strip()}"
        for t in turns
    ) + "\n"


def plain(turns: list[dict], *, names: dict[str, str] | None = None,
          language: str = "en") -> str:
    return "\n".join(
        f"{display(t.get('speaker'), names, language)}: {t['text'].strip()}"
        for t in turns
    ) + "\n"


def srt(segments: list[dict], *, names: dict[str, str] | None = None,
        language: str = "en") -> str:
    out = []
    # Counted over what is written, not over what was offered. Numbering the
    # source segments and then skipping the empty ones leaves holes in the
    # sequence, which strict parsers treat as a damaged file.
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        who = display(seg.get("speaker"), names, language)
        out.append(f"{len(out) + 1}\n{srt_time(seg['start'])} --> "
                   f"{srt_time(seg['end'])}\n[{who}] {text}\n")
    return "\n".join(out)


def vtt(segments: list[dict], *, names: dict[str, str] | None = None,
        language: str = "en") -> str:
    out = ["WEBVTT", ""]
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        who = display(seg.get("speaker"), names, language)
        out.append(f"{vtt_time(seg['start'])} --> {vtt_time(seg['end'])}")
        out.append(f"<v {who}>{text}")
        out.append("")
    return "\n".join(out)


def payload_json(
    *,
    meta: dict[str, Any],
    segments: list[dict],
    turns: list[dict],
    names: dict[str, str],
    matches: list[dict],
) -> str:
    return json.dumps(
        {"meta": meta, "names": names, "speaker_matches": matches,
         "turns": turns, "segments": segments},
        indent=2, ensure_ascii=False,
    )


# Every file this module writes, in one place. _sweep needs to know what a stale
# output looks like, and a second list would drift from the first.
FILENAMES = {
    "source": "{stem} (source).md",
    "md": "{stem}.md",
    "txt": "{stem}.txt",
    "srt": "{stem}.srt",
    "vtt": "{stem}.vtt",
    "json": "{stem}.json",
}


def _sweep(outdir: Path, stem: str, formats: Iterable[str]) -> None:
    """Delete outputs for this recording that no longer correspond to a format.

    Only files this module could have written are considered, and only for this
    stem: the output folder belongs to one job, but people do drop things in it.
    """
    wanted = {FILENAMES[f].format(stem=stem) for f in formats if f in FILENAMES}
    # Every name this module could have written for this stem, minus the ones it
    # is about to write. Matching on the suffix instead was close enough to work
    # and wrong: with a stem like "Nota vocale 2026-03-04" it also matched the
    # user's own "Nota vocale 2026-03-04 appunti.md" and deleted it. Globbing was
    # the same mistake from the other side, since a stem containing [ or ] reads
    # as a character class and quietly matches nothing at all.
    for pattern in FILENAMES.values():
        name = pattern.format(stem=stem)
        if name in wanted:
            continue
        stale = outdir / name
        if stale.is_file():
            stale.unlink()


def write_all(
    outdir: Path,
    stem: str,
    formats: Iterable[str],
    *,
    turns: list[dict],
    segments: list[dict],
    names: dict[str, str],
    meta: dict[str, Any],
    matches: list[dict],
) -> list[Path]:
    # Check the names before touching the folder. This used to sweep, write what
    # it understood and raise afterwards, which left a half-updated folder and
    # deleted the previous file for the format that was misspelled, after the
    # transcription had already been paid for.
    formats = list(formats)
    unknown = [f for f in formats if f not in FILENAMES]
    if unknown:
        raise ValueError(
            f"unknown output format {unknown[0]!r}. A typo here used to cost one "
            "file with no warning, after the transcription had already run.")

    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    # One language for every file of the job, so the same voice carries the same
    # label in the document, the subtitles and the plain text.
    language = meta.get("language", "it")

    # write_all only ever writes. If the name of a format ever changes again, the
    # old file stays in the folder next to the new one and there is no telling the
    # stale copy from the fresh one. Sweep anything matching this stem that we are
    # not about to write.
    _sweep(outdir, stem, formats)

    for fmt in formats:
        if fmt == "source":
            path = outdir / FILENAMES["source"].format(stem=stem)
            path.write_text(source_doc(
                turns,
                title=meta.get("title", stem),
                names=names,
                recorded=meta.get("recorded"),
                duration=meta.get("duration", 0.0),
                language=language,
                source_file=meta.get("source_file", ""),
                speaker_stats=meta.get("speaker_stats"),
                unresolved=meta.get("unresolved"),
                recorded_source=meta.get("recorded_source", ""),
                device=meta.get("device", ""),
                untranscribed_seconds=meta.get("untranscribed_seconds", 0.0),
                speech_seconds=meta.get("speech_seconds", 0.0),
                untranscribed_gaps=meta.get("untranscribed_gaps"),
            ))
        elif fmt == "md":
            path = outdir / FILENAMES["md"].format(stem=stem)
            path.write_text(markdown(turns, names=names, language=language))
        elif fmt == "txt":
            path = outdir / FILENAMES["txt"].format(stem=stem)
            path.write_text(plain(turns, names=names, language=language))
        elif fmt == "srt":
            path = outdir / FILENAMES["srt"].format(stem=stem)
            path.write_text(srt(segments, names=names, language=language))
        elif fmt == "vtt":
            path = outdir / FILENAMES["vtt"].format(stem=stem)
            path.write_text(vtt(segments, names=names, language=language))
        elif fmt == "json":
            path = outdir / FILENAMES["json"].format(stem=stem)
            path.write_text(payload_json(meta={k: (v.isoformat() if isinstance(v, datetime) else v)
                                                for k, v in meta.items()},
                                         segments=segments, turns=turns,
                                         names=names, matches=matches))
        written.append(path)
    return written
