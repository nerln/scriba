"""One long recording, many conversations: where each starts, who it was with, what came of it.

A day at a fair, a dinner, an afternoon of meetings in the same room: the phone
records for hours, and what comes out is one document in which twenty
conversations with twenty people run into each other. Nobody reads that the next
morning to remember whom to write to.

This works on what a run has already cached and transcribes nothing it does not
have to. The boundaries come from the voices, not the words: the owner of the
recording is the voice that is there from the first conversation to the last, and
a conversation ends when the people talking with them change, or when nobody
talks for long enough. Greetings and goodbyes help where the diarizer has given
two different people the same label, which in a noisy hall it does.

Languages move from one conversation to the next, and the run transcribed the
whole file in the one language that won a vote over a few windows. A conversation
in another language comes out of that as fluent, punctuated, invented text. Each
one is checked on its own audio and transcribed again in its own language when
that differs.

The summaries come from a model running on this Mac through Ollama, and nothing
else is asked for: the address is checked to be this machine before a word is
sent.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from . import export

# How long nobody may speak before whatever comes next is another conversation.
# Walking from one stand to the next at a fair takes about that.
LONG_GAP = 60.0
# The pause that separates two stretches of talk. Shorter than this is the same
# exchange; longer, and the next stretch has to show it is the same people.
SHORT_GAP = 6.0
# Speech a voice needs in a stretch to count as a person in it rather than a
# passer-by, a laugh or the hall leaking into the microphone.
MIN_GUEST = 3.0
# Below this nobody gets a summary: a hello in a corridor is not a conversation.
MIN_CONVERSATION = 30.0
# Below this many transcribed words the model is not asked at all. Given a minute
# of hall noise with nothing transcribed in it, it once answered with two people,
# their jobs, their companies and what they agreed, none of which existed.
MIN_WORDS = 30

GREETING = re.compile(
    r"\b(hola|buenas tardes|buenos d[ií]as|encantad[oa]|mucho gusto|me llamo|"
    r"nice to meet|pleasure to meet|my name is|hello|piacere|mi chiamo)\b",
    re.IGNORECASE)
FAREWELL = re.compile(
    r"\b(adi[oó]s|hasta luego|hasta pronto|nos vemos|chao|goodbye|bye bye|"
    r"see you|nice talking|a presto|arrivederci)\b",
    re.IGNORECASE)


@dataclass
class Stretch:
    """Talk with no pause longer than SHORT_GAP in it."""
    start: float
    end: float
    seconds: dict[str, float] = field(default_factory=dict)


@dataclass
class Conversation:
    index: int
    start: float
    end: float
    speakers: dict[str, float] = field(default_factory=dict)
    guests: list[str] = field(default_factory=list)
    language: str = ""
    retranscribed: bool = False
    turns: list[dict] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def worth_reading(self) -> bool:
        """Somebody other than the owner, for long enough to be a conversation."""
        return bool(self.guests) and self.duration >= MIN_CONVERSATION


# --------------------------------------------------------------------- whose recording

def find_owner(speech: dict[str, float], names: dict[str, str],
               owner: str | None = None) -> str | None:
    """The label of the person whose recording this is.

    Asked by label or by name, it is that one. Otherwise the voice with the most
    speech: on a recording somebody carried around all day, they are in every
    conversation and nobody else is.
    """
    if owner:
        if owner in speech:
            return owner
        for label, name in names.items():
            if name and name.casefold() == owner.casefold():
                return label
        raise ValueError(f"No voice in this recording is labelled or named {owner!r}.")
    if not speech:
        return None
    return max(speech, key=speech.get)


# --------------------------------------------------------------------- where they split

def stretches(dia_turns: list[dict], short_gap: float = SHORT_GAP) -> list[Stretch]:
    out: list[Stretch] = []
    for t in sorted(dia_turns, key=lambda t: float(t["start"])):
        a, b, who = float(t["start"]), float(t["end"]), t["speaker"]
        if out and a - out[-1].end < short_gap:
            cur = out[-1]
            cur.end = max(cur.end, b)
        else:
            cur = Stretch(a, b)
            out.append(cur)
        cur.seconds[who] = cur.seconds.get(who, 0.0) + (b - a)
    return out


def ambient(all_stretches: list[Stretch], owner: str | None,
            min_guest: float = MIN_GUEST, share: float = 0.25, span: float = 0.5) -> set[str]:
    """Labels that are in too many stretches, over too much of the day, to be one person.

    In a noisy hall the diarizer puts several people, and the hall itself, under
    one label, and that label turns up from the first conversation to the last.
    Counted only on recordings with enough stretches for "too many" to mean
    something.
    """
    with_people = [st for st in all_stretches
                   if any(w != owner and s >= min_guest for w, s in st.seconds.items())]
    if len(with_people) < 8:
        return set()
    total = all_stretches[-1].end - all_stretches[0].start
    seen: dict[str, list[Stretch]] = {}
    for st in with_people:
        for who, sec in st.seconds.items():
            if who != owner and sec >= min_guest:
                seen.setdefault(who, []).append(st)
    return {who for who, sts in seen.items()
            if len(sts) >= share * len(with_people)
            and sts[-1].end - sts[0].start >= span * total}


def _text_between(turns: list[dict], a: float, b: float) -> str:
    return " ".join(t.get("text", "") for t in turns
                    if float(t["end"]) > a and float(t["start"]) < b)


def split(dia_turns: list[dict], turns: list[dict], owner: str | None, *,
          long_gap: float = LONG_GAP, short_gap: float = SHORT_GAP,
          min_guest: float = MIN_GUEST) -> list[Conversation]:
    """Cut the recording where the people talking with the owner change.

    A stretch joins the conversation before it unless nobody spoke for
    `long_gap`, or the people in it other than the owner are none of the people
    already in the conversation, or it opens with a greeting after a pause, or
    the conversation before it ended with a goodbye. A stretch with nobody but the
    owner in it (thinking aloud, a phone held up in a queue) never starts one.
    """
    convs: list[Conversation] = []
    all_stretches = stretches(dia_turns, short_gap)
    everywhere = ambient(all_stretches, owner, min_guest)
    for st in all_stretches:
        guests = {who for who, sec in st.seconds.items() if who != owner and sec >= min_guest}
        if convs:
            cur = convs[-1]
            gap = st.start - cur.end
            # Who is in the conversation, as far as telling one from the next goes.
            # A label heard all afternoon is a crowd, not a person, and sharing it
            # says nothing about whether the person across the table changed.
            known = set(cur.guests) - everywhere
            guests_seen = guests - everywhere
            greets = bool(GREETING.search(_text_between(turns, st.start, st.start + 20)))
            left = bool(FAREWELL.search(_text_between(turns, cur.end - 20, cur.end)))
            new = (gap >= long_gap
                   or (guests_seen and known and not guests_seen & known)
                   or (guests and cur.guests and gap >= short_gap and (greets or left)))
            if not new:
                cur.end = max(cur.end, st.end)
                for who, sec in st.seconds.items():
                    cur.speakers[who] = cur.speakers.get(who, 0.0) + sec
                cur.guests = sorted(set(cur.guests) | guests)
                continue
        convs.append(Conversation(index=len(convs) + 1, start=st.start, end=st.end,
                                  speakers=dict(st.seconds), guests=sorted(guests)))
    for c in convs:
        c.turns = [t for t in turns
                   if float(t["start"]) >= c.start - 1 and float(t["end"]) <= c.end + 1]
    return convs


# --------------------------------------------------------------------- the right language

def slice_wav(wav: Path, a: float, b: float, out: Path) -> Path:
    import soundfile as sf

    info = sf.info(str(wav))
    rate = info.samplerate
    data, _ = sf.read(str(wav), start=int(max(0.0, a) * rate), stop=int(b * rate),
                      dtype="float32")
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), data, rate)
    return out


LANGUAGES = ("it", "es", "en", "fr", "de", "pt", "ca")


def check_languages(convs: list[Conversation], wav: Path, file_language: str, work: Path, *,
                    detect: Callable[..., Any], min_agreement: float = 0.6,
                    allowed: tuple[str, ...] = LANGUAGES) -> list[Conversation]:
    """Detect each conversation's language on its own audio. Return the ones that differ.

    Only where the vote is clear, and only into a language somebody could have
    spoken there: a minute of hall noise once came back as Norwegian Nynorsk, all
    windows agreeing, and transcribing it as that produced nothing anyone said.
    Two close languages half-agreeing are better left in the language the whole
    file was transcribed in than switched on a coin.
    """
    wrong: list[Conversation] = []
    for c in convs:
        c.language = file_language
        if not c.worth_reading:
            continue
        clip = slice_wav(wav, c.start, c.end, work / f"{c.index:02d}.wav")
        guess = detect(clip, n_samples=max(1, min(5, math.ceil(c.duration / 30))))
        if (guess.language != file_language and guess.agreement >= min_agreement
                and guess.language in allowed):
            c.language = guess.language
            wrong.append(c)
    return wrong


def retranscribe(wrong: list[Conversation], work: Path, dia: Any, *,
                 run_stage: Callable[[list[dict]], None], assign: Callable[..., list[dict]],
                 to_turns: Callable[[list[dict]], list[dict]]) -> None:
    """Transcribe the conversations in another language again, and put their turns back.

    The words come back timed from the start of the clip; they are moved to the
    recording's clock before the voices are assigned, so the diarization of the
    whole file decides who said them, as it does everywhere else.
    """
    items = []
    for c in wrong:
        out = work / f"{c.index:02d}-{c.language}.json"
        if not out.exists():
            items.append({"wav": str(work / f"{c.index:02d}.wav"), "language": c.language,
                          "out": str(out), "label": f"conversation {c.index}"})
    if items:
        run_stage(items)
    for c in wrong:
        data = json.loads((work / f"{c.index:02d}-{c.language}.json").read_text())
        segments = data["segments"]
        for seg in segments:
            seg["start"] = float(seg.get("start", 0.0)) + c.start
            seg["end"] = float(seg.get("end", 0.0)) + c.start
            for w in seg.get("words", []) or []:
                if "start" in w:
                    w["start"] = float(w["start"]) + c.start
                if "end" in w:
                    w["end"] = float(w["end"]) + c.start
        c.turns = to_turns(assign(segments, dia, word_level=data.get("word_level", False)))
        c.retranscribed = True


# --------------------------------------------------------------------- what came of it

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

SUMMARY_LANGUAGE = {"it": "Italian", "es": "Spanish", "en": "English"}

PROMPT = """This is the transcript of one conversation recorded by {owner} at an event. \
Lines start with who is speaking. Transcription is automatic: names and figures may \
be misheard.

Answer as JSON with exactly these keys:
- "people": list of {{"name", "company", "role"}} for the people {owner} talked with, \
not {owner}, as they introduce themselves. A few words each; empty strings where not \
said. Never invent a name, and never copy sentences of the transcript into these fields.
- "summary": two or three sentences on what was talked about.
- "agreements": list of what was agreed: to connect on LinkedIn, to send an email, \
an introduction, a meeting, an offer. Empty list if nothing.
- "next_step": the one thing {owner} should do next about this conversation, or "".
If the transcript is too short or unclear to tell, leave the fields empty: an empty \
field is right, an invented one is a mistake.

Transcript:
{transcript}

Write "summary", "agreements" and "next_step" in {language}, whatever language the \
transcript is in."""


def summarize(text: str, *, owner: str, model: str, language: str = "it",
              url: str = "http://127.0.0.1:11434", timeout: float = 600.0,
              send: Callable[[str, bytes, float], bytes] | None = None) -> dict[str, Any]:
    """Ask the local model for who, what and what next. Refuses any other host."""
    host = urllib.parse.urlparse(url).hostname or ""
    if host not in LOCAL_HOSTS:
        raise ValueError(f"Summaries are only asked of a model on this Mac, not {host!r}.")
    # A long conversation is cut, not skipped: the start says who and why, and the
    # end says what was agreed, so the middle is what goes.
    if len(text) > 14000:
        text = text[:7000] + "\n[...]\n" + text[-7000:]
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(
            owner=owner, language=SUMMARY_LANGUAGE.get(language, "English"),
            transcript=text)}],
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0.2},
    }).encode()
    raw = (send or _post)(url.rstrip("/") + "/api/chat", body, timeout)
    content = json.loads(raw)["message"]["content"]
    data = json.loads(content)
    return {
        "people": clean_people(data.get("people", []), owner),
        "summary": str(data.get("summary", "")).strip(),
        "agreements": [str(a) for a in data.get("agreements", []) if str(a).strip()],
        "next_step": str(data.get("next_step", "")).strip(),
    }


def clean_people(people: Any, owner: str, limit: int = 60) -> list[dict[str, str]]:
    """The people the model named, without the owner and without what is not a name.

    A small model asked who was there lists the owner too, fills a role with a
    sentence copied from the transcript, and returns entries with nothing in them.
    The index has a column, not a page, for this.
    """
    out = []
    for p in people if isinstance(people, list) else []:
        if not isinstance(p, dict):
            continue
        entry = {k: " ".join(str(p.get(k) or "").split()) for k in ("name", "company", "role")}
        if entry["name"].casefold() == owner.casefold():
            continue
        if any(len(v) > limit for v in entry.values()):
            entry = {k: (v if len(v) <= limit else "") for k, v in entry.items()}
        if any(entry.values()):
            out.append(entry)
    return out


# Names a summary can get wrong in a way that matters: acronyms (ORVA, KTL) and
# brand-shaped words (LakeWorks, Ferrybox.io). Generic technical acronyms are left out:
# they are what the conversation was about, not who it was with.
NAMEISH = re.compile(r"\b(?:[A-Z][A-Z0-9]{2,}|[A-Za-z]+[A-Z][a-z0-9]+\w*|[A-Z][a-z0-9]+\.[a-z]{2,})\b")
GENERIC = {"API", "APIS", "LLM", "LLMS", "SDK", "MCP", "MCPS", "SAAS", "CEO", "CTO", "CFO",
           "PHD", "GPU", "CPU", "B2B", "B2C", "KPI", "ROI", "URL", "PDF", "VR", "AR", "XR"}


def _fold(text: str) -> str:
    import unicodedata
    plain = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in plain if ch.isalnum()).casefold()


def ground(summary: dict[str, Any], transcript: str) -> dict[str, Any]:
    """Keep what the transcript supports; mark what it does not.

    A name in the people column is somebody to write to the next day, so a name or
    a company that was never said is dropped. In the summary the sentence stays,
    and the names in it that were never said are listed to be checked: asked
    about a conversation with someone from one UN agency, the model once wrote two
    others into the summary.
    """
    said = _fold(transcript)
    out = dict(summary)
    people = []
    for p in summary.get("people", []):
        entry = dict(p)
        for k in ("name", "company"):
            value = entry.get(k, "")
            if value and not re.fullmatch(r"(Voce|Voice|Voz) \d+", value) \
                    and _fold(value) not in said:
                entry[k] = ""
        if any(entry.values()):
            people.append(entry)
    out["people"] = people
    # The next step is advice and may name a channel nobody mentioned; what was
    # agreed and what was said are claims about the conversation, and checked.
    prose = " ".join([summary.get("summary", ""), *summary.get("agreements", [])])
    unchecked = []
    for token in NAMEISH.findall(prose):
        if token.upper() in GENERIC or _fold(token) in said or token in unchecked:
            continue
        unchecked.append(token)
    out["unverified"] = unchecked
    return out


def _post(url: str, body: bytes, timeout: float) -> bytes:
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def release_model(model: str, url: str = "http://127.0.0.1:11434") -> None:
    """Tell Ollama to drop the model now rather than in five minutes. Best effort."""
    try:
        _post(url.rstrip("/") + "/api/generate",
              json.dumps({"model": model, "keep_alive": 0}).encode(), 30)
    except Exception:
        pass


# --------------------------------------------------------------------- what is written

HEADINGS = {
    "it": {"title": "Conversazioni", "time": "Ora", "length": "Durata", "lang": "Lingua",
           "with": "Con chi", "about": "Di cosa", "next": "Prossimo passo",
           "agreed": "Accordi", "transcript": "Trascrizione",
           "owner_only": "{n} tratti senza nessun interlocutore o troppo brevi, non elencati.",
           "unknown": "non detto",
           "too_little": "Troppo poco testo trascritto per un riassunto: da riascoltare.",
           "check": "da verificare, non detto nella trascrizione",
           "retranscribed": "ritrascritta in {lang}"},
    "en": {"title": "Conversations", "time": "Time", "length": "Length", "lang": "Language",
           "with": "With", "about": "About", "next": "Next step",
           "agreed": "Agreed", "transcript": "Transcript",
           "owner_only": "{n} stretches with nobody else in them or too short, not listed.",
           "unknown": "not said",
           "too_little": "Too little was transcribed for a summary: worth listening to again.",
           "check": "to check, not said in the transcript",
           "retranscribed": "transcribed again in {lang}"},
    "es": {"title": "Conversaciones", "time": "Hora", "length": "Duración", "lang": "Idioma",
           "with": "Con quién", "about": "De qué", "next": "Próximo paso",
           "agreed": "Acuerdos", "transcript": "Transcripción",
           "owner_only": "{n} tramos sin interlocutor o demasiado breves, no listados.",
           "unknown": "no dicho",
           "too_little": "Demasiado poco texto transcrito para resumir: conviene escucharla.",
           "check": "por verificar, no dicho en la transcripción",
           "retranscribed": "transcrita de nuevo en {lang}"},
}


def clock(recorded: str | None, offset: float) -> str:
    """The time of day a moment of the recording happened, when the start is known."""
    if recorded:
        try:
            return (datetime.fromisoformat(recorded) + timedelta(seconds=offset)).strftime("%H:%M")
        except ValueError:
            pass
    return export.hhmmss(offset)


def people_line(c: Conversation, unknown: str) -> str:
    people = c.summary.get("people") or []
    out = []
    for p in people:
        name = p.get("name") or unknown
        extra = ", ".join(x for x in (p.get("role"), p.get("company")) if x)
        out.append(f"{name} ({extra})" if extra else name)
    return "; ".join(out) or unknown


def render_turns(c: Conversation, names: dict[str, str], language: str) -> str:
    return "\n\n".join(
        f"**{export.display(t.get('speaker'), names, language)}** "
        f"[{export.hhmmss(float(t['start']))}]: {t.get('text', '').strip()}"
        for t in c.turns if t.get("text", "").strip())


def write(convs: list[Conversation], out_dir: Path, *, title: str, recorded: str | None,
          names: dict[str, str], language: str) -> list[Path]:
    h = HEADINGS.get(language, HEADINGS["en"])
    out_dir.mkdir(parents=True, exist_ok=True)
    listed = [c for c in convs if c.worth_reading]
    written: list[Path] = []

    rows = [f"# {h['title']}: {title}", "",
            f"| # | {h['time']} | {h['length']} | {h['lang']} | {h['with']} | {h['about']} "
            f"| {h['next']} |", "|---|---|---|---|---|---|---|"]
    for c in listed:
        about = c.summary.get("summary") or (h["too_little"] if c.summary.get("note") else "")
        if c.summary.get("unverified"):
            about += f" ({h['check']}: {', '.join(c.summary['unverified'])})"
        about = about.replace("|", "/").replace("\n", " ")
        nxt = (c.summary.get("next_step") or "").replace("|", "/").replace("\n", " ")
        rows.append(f"| [{c.index}]({c.index:02d}.md) | {clock(recorded, c.start)} "
                    f"| {export.hhmmss(c.duration)} | {c.language} "
                    f"| {people_line(c, h['unknown']).replace('|', '/')} | {about} | {nxt} |")
    skipped = len(convs) - len(listed)
    if skipped:
        rows += ["", h["owner_only"].format(n=skipped)]
    index = out_dir / "index.md"
    index.write_text("\n".join(rows) + "\n")
    written.append(index)

    for c in listed:
        lines = [f"# {c.index}. {people_line(c, h['unknown'])}", "",
                 f"- **{h['time']}**: {clock(recorded, c.start)} to {clock(recorded, c.end)} "
                 f"({export.hhmmss(c.start)} to {export.hhmmss(c.end)} in the recording)",
                 f"- **{h['length']}**: {export.hhmmss(c.duration)}",
                 f"- **{h['lang']}**: {c.language}"
                 + (f" ({h['retranscribed'].format(lang=c.language)})" if c.retranscribed else "")]
        if c.summary.get("summary"):
            lines += ["", c.summary["summary"]]
        if c.summary.get("unverified"):
            lines += ["", f"_{h['check']}: {', '.join(c.summary['unverified'])}_"]
        if c.summary.get("agreements"):
            lines += ["", f"## {h['agreed']}", ""] + [f"- {a}" for a in c.summary["agreements"]]
        if c.summary.get("next_step"):
            lines += ["", f"**{h['next']}**: {c.summary['next_step']}"]
        lines += ["", f"## {h['transcript']}", "", render_turns(c, names, language)]
        path = out_dir / f"{c.index:02d}.md"
        path.write_text("\n".join(lines) + "\n")
        written.append(path)

    data = out_dir / "conversations.json"
    data.write_text(json.dumps([{**asdict(c), "duration": round(c.duration, 1),
                                 "clock": clock(recorded, c.start)} for c in convs],
                               ensure_ascii=False, indent=1))
    written.append(data)
    return written


# --------------------------------------------------------------------- all of it, on a job

def run(job: Any, *, owner: str | None = None, model: str = "qwen3:8b",
        language: str = "it", keep_language: bool = False, summaries: bool = True,
        long_gap: float = LONG_GAP, languages: tuple[str, ...] = LANGUAGES) -> list[Path]:
    """Split a transcribed recording into conversations and write them out.

    Works on the job's cache: the diarization and the turns of a finished run.
    Holds the job like a run does, because re-transcribing a conversation in its
    own language loads whisper, and summarizing loads the local model: one heavy
    thing at a time on this machine, and never two on the same recording.
    """
    from . import awake, diarize, lang, stage

    dia_path, turns_path = job.dir / "diarization.json", job.dir / "turns.json"
    if not dia_path.exists() or not turns_path.exists():
        raise RuntimeError("This recording has no finished run to split: transcribe it "
                           "first with `scriba run` (with diarization on).")
    job._refuse_if_running()
    job._claim()
    restore = job._release_on_sigterm()
    awake.hold()
    try:
        dia_turns = json.loads(dia_path.read_text())
        turns = json.loads(turns_path.read_text())
        state = job.state
        names = {**{lbl: m["name"] for lbl, m in (state.get("matches") or {}).items()
                    if isinstance(m, dict) and m.get("name")},
                 **(state.get("names") or {})}
        speech: dict[str, float] = {}
        for t in dia_turns:
            speech[t["speaker"]] = speech.get(t["speaker"], 0.0) + float(t["end"]) - float(t["start"])
        me = find_owner(speech, names, owner)
        convs = split(dia_turns, turns, me, long_gap=long_gap)
        listed = [c for c in convs if c.worth_reading]
        job.report(f"conversations: {len(listed)} with somebody in them, "
                   f"{len(convs) - len(listed)} other stretches")

        work = job.dir / "conversations"
        file_language = state.get("language") or "en"
        if not keep_language:
            wrong = check_languages(convs, job.wav, file_language, work, detect=lang.detect,
                                    allowed=languages)
            if wrong:
                job.report(f"conversations: {len(wrong)} not in {file_language}, "
                           f"transcribing those again in their own language")
                dia = diarize.Diarization(turns=[diarize.SpeakerTurn(**{k: t[k] for k in
                                                 ("start", "end", "speaker")}) for t in dia_turns])
                settings = asdict(job.s)
                retranscribe(
                    wrong, work, dia,
                    run_stage=lambda items: stage.call(
                        "scriba.asr:transcribe_many_to_files",
                        {"items": items, "settings": settings},
                        report=job.report, what="transcription"),
                    assign=diarize.assign, to_turns=diarize.to_turns)

        if summaries and listed:
            who = export.display(me, names, language) if me else "the owner"
            for i, c in enumerate(listed, 1):
                job.report(f"summary: conversation {c.index} ({i} of {len(listed)})")
                text = "\n".join(f"{export.display(t.get('speaker'), names, language)}: "
                                 f"{t.get('text', '').strip()}" for t in c.turns)
                if sum(len(t.get("text", "").split()) for t in c.turns) < MIN_WORDS:
                    c.summary = {"summary": "", "note": "too little transcribed"}
                    continue
                # Kept per conversation and per everything the answer depends on, so
                # running the command again (another split, a fixed language) only
                # asks the model about what changed.
                key = hashlib.sha1(json.dumps([PROMPT, model, language, who, text])
                                   .encode()).hexdigest()[:12]
                cached = work / f"summary-{c.index:02d}-{key}.json"
                if cached.exists():
                    c.summary = ground(json.loads(cached.read_text()), text)
                    continue
                try:
                    answer = summarize(text, owner=who, model=model, language=language)
                    work.mkdir(parents=True, exist_ok=True)
                    cached.write_text(json.dumps(answer, ensure_ascii=False))
                    c.summary = ground(answer, text)
                except Exception as exc:
                    # One conversation the model could not read is a gap in the
                    # index, not a reason to lose the other nineteen.
                    c.summary = {"summary": "", "error": str(exc)[:200]}
            release_model(model)

        title = Path(state.get("source") or job.source).name
        return write(convs, job.dir / "output" / "conversations", title=title,
                     recorded=state.get("recorded"), names=names, language=language)
    finally:
        job._release()
        restore()
