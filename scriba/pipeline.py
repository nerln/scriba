"""The orchestrator: from an audio file to a readable source document.

A job is a folder under ~/.scriba/jobs/<slug>/ holding all intermediate state.
That is deliberate: the phases cost minutes of CPU, and redoing only the name
attribution without re-transcribing is the normal case, not the exception.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
import re
import signal
import unicodedata
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import audio, awake, diarize, export, lang, naming, stage
from .config import JOBS_DIR, Settings, ensure_dirs, hf_token, model_cache_problem, write_atomic
from .jobs import RUNNING_NOTE, process_start, running_pid
from .voices import VoiceRegistry

Reporter = Callable[[str], None]



# Machine-readable error markers.
#
# The macOS app has to tell "you never configured a token" apart from any other
# failure, so it can say something useful instead of dumping a stack trace. It used
# to do that by matching a phrase from the human message. That broke silently the
# moment the sentence was reworded, on an error path no test exercises. A marker is
# ugly in a terminal and impossible to break by accident; prose is the opposite.
ERR_NO_TOKEN = "[scriba:error:hf-token]"
# The general one: printed once per failed file, on its own line, when the
# environment says an application is reading the output (SCRIBA_MARKERS=1). In a
# terminal it is not printed at all. The app used to have nothing to quote but
# the exit code, because the readable message above it was six lines up in a log
# it only kept the tail of.
ERR_MARK = "[scriba:error]"

# The transcription stage, as the module:function stage.call runs in a child
# process (stage.py says why it is not run here).
ASR_STAGE = "scriba.asr:transcribe_to_file"


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s-]", "", text).strip()
    text = re.sub(r"[\s_-]+", "-", text)
    return text.lower() or "recording"


def job_slug(source: Path) -> str:
    """Working folder name: readable, and still unique per path.

    With the file name alone, a `Recording 38.m4a` on the Desktop and another one
    under ~/Library/CloudStorage would land in the same job folder, and the second
    would silently reuse the transcript of the first. The suffix fingerprints the
    containing folder, so two files with the same name in different folders stay
    apart.

    Move the source file to another folder and the slug changes with it: the old
    job is orphaned and the work is redone. That is the honest trade for having
    the folder in the name at all.
    """
    digest = hashlib.sha1(str(source.parent).encode()).hexdigest()[:6]
    return f"{slugify(source.stem)}-{digest}"



@dataclass
class JobResult:
    job_dir: Path
    outputs: list[Path]
    language: str
    speakers: list[str]
    names: dict[str, str]
    unresolved: list[str]
    dossier_path: Path
    duration: float


def _diar_fingerprint_of(min_speakers: int | None, max_speakers: int | None,
                         device: str) -> str:
    """What decides the diarization: a function of its inputs, so they can be guessed back."""
    # The device belongs here. config.py offers cpu as the way to check a
    # suspected Metal problem, and without this the check reused the Metal
    # result and proved nothing.
    payload = {"model": diarize.DIARIZATION_MODEL,
               "min": min_speakers, "max": max_speakers, "device": device}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _savez_atomic(path: Path, **arrays) -> None:
    """np.savez through the same rename as everything else.

    An interrupted run left a truncated archive, and the next `scriba name` died
    inside np.load with a BadZipFile instead of saying the job has no speakers
    yet. The sibling turns.json two lines away was already atomic.
    """
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    write_atomic(path, buf.getvalue())


class Job:
    def __init__(self, source: Path, settings: Settings | None = None,
                 *, report: Reporter | None = None):
        ensure_dirs()
        self.source = Path(source).expanduser().resolve()
        if not self.source.exists():
            # Worded, because the bare path is what reached the screen: a queued
            # recording moved or unmounted before its turn failed with nothing
            # but its old location as the explanation.
            raise FileNotFoundError(
                f"{self.source.name} is no longer at {self.source.parent}. "
                "It was moved, renamed or its disk is not connected.")
        # Refuse before creating anything. The folder used to be made first, so
        # pointing this at a README left a job directory named after it, listed
        # for ever as state "nothing" until somebody ran prune.
        if self.source.is_dir() or not audio.is_audio(self.source):
            raise ValueError(
                f"{self.source.name} is not an audio or video file. "
                f"Extensions it knows: {', '.join(sorted(audio.AUDIO_EXTS))}")
        self.s = settings or Settings.load()
        self.report: Reporter = report or (lambda m: print(f"[scriba] {m}"))
        self.dir = JOBS_DIR / job_slug(self.source)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.wav = self.dir / "audio16k.wav"
        self.state_path = self.dir / "state.json"
        self.state: dict[str, Any] = (
            json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        )
        # Say which file this is before anything can fail. The source used to be
        # written only after the audio was converted, so a run that stopped
        # before that, on a video with no audio track or a missing ffmpeg, left
        # a folder the list could only name by its slug.
        if self.state.get("source") != str(self.source):
            self.state["source"] = str(self.source)
            self._save_state()
        self.s = self._with_remembered_speaker_counts(self.s)

    # ------------------------------------------------------------------ util
    def _save_state(self) -> None:
        write_atomic(self.state_path,
                     json.dumps(self.state, indent=2, ensure_ascii=False, default=str))

    def _with_remembered_speaker_counts(self, s: Settings) -> Settings:
        """The speaker counts an earlier run was given, when this caller gives none.

        "People in the room: 2" is the one thing the app asks for before a run, and
        it was kept only inside the diarization fingerprint, where nothing can read
        it back. `scriba name` (the app's Save) and "Put it back in the queue" both
        build a Job with no count, so the fingerprint no longer matched and a
        finished recording was diarized again from scratch, minutes of work, with a
        different number of voices possible and the typed names then applied to
        whoever got the labels.

        A caller that gives either bound is heard as it is, and the bounds are never
        mixed with the remembered ones. The Settings is copied, not changed: one is
        shared by every file of a batch.

        A folder made before the count was stored has none to read. The fingerprint
        still holds it, as a hash of the count with the model and the device, so the
        counts a person could have given can be tried against it
        (`_counts_behind_the_diarization`), and the one that matches is kept like any
        other.
        """
        counts = self.state.get("speaker_counts")
        if s.min_speakers is not None or s.max_speakers is not None:
            return s
        if not counts and not self.state.get("diar_fingerprint"):
            return s
        recorded = self.state.get("source_fingerprint")
        if recorded is not None and recorded != self._source_fingerprint():
            return s  # different audio at the same path: the count was about the old one
        if not counts:
            counts = self._counts_behind_the_diarization(s)
            if not counts:
                return s
            self.state["speaker_counts"] = counts
            self._save_state()
        return replace(s, min_speakers=counts.get("min"), max_speakers=counts.get("max"))

    def _counts_behind_the_diarization(self, s: Settings) -> dict[str, int | None] | None:
        """The speaker count the cached diarization was made with, or None for none.

        Hashes every count the app or a command line could have given, 1 to 12 for
        each bound or neither (the app's stepper stops at 12), against the
        fingerprint on disk. The device is tried too: it is part of the hash, and
        somebody who switched it to cpu to check a Metal problem is the one asking
        for the same diarization again, with the count it had. A fingerprint nothing
        here reproduces (another model, a count of 40) is left alone, and the
        diarization is made again as it always was.
        """
        stored = self.state.get("diar_fingerprint")
        counts = (None, *range(1, 13))
        for device in dict.fromkeys((s.diarize_device, "auto", "cpu", "mps")):
            for low in counts:
                for high in counts:
                    if low is not None and high is not None and low > high:
                        continue
                    if _diar_fingerprint_of(low, high, device) == stored:
                        return None if low is None and high is None else {
                            "min": low, "max": high}
        return None

    # A run in progress leaves a note in its folder saying so, with its process
    # id. Without it a job being transcribed from a terminal was listed by the
    # app as "started and produced nothing", which is the description of a
    # failure, for as long as the run lasted. The note is removed when the run
    # ends, by success, failure or a Stop (`_release_on_sigterm`); one left behind
    # by a crash names a process that no longer exists, and `jobs.running_pid`
    # treats that as no note at all.
    #
    # It also says when the process started. A note that outlives its run (a
    # SIGKILL, a power cut) names a pid that macOS will give to some other process
    # in time, and that process, alive, made every later run of the job refuse.
    # The start time tells the two apart. None when `ps` could not say, which
    # reads the same as a note from before this field.
    def _claim(self) -> None:
        write_atomic(self.dir / RUNNING_NOTE, json.dumps(
            {"pid": os.getpid(), "started": datetime.now().isoformat(timespec="seconds"),
             "process_started": process_start(os.getpid())}))

    def _release(self) -> None:
        try:
            (self.dir / RUNNING_NOTE).unlink()
        except FileNotFoundError:
            pass

    def _release_on_sigterm(self) -> Callable[[], None]:
        """Take the note away when the app's Stop arrives; return the function that undoes it.

        A Stop sends SIGTERM, and Python's answer to that is to die on the spot,
        past the `finally` that removes the note. Every Stop left one behind. In
        a stage the handler in stage.py runs first, stops the child, and hands the
        signal on to this one by restoring it and sending it again.

        Then the process dies as the signal would have killed it, so the app still
        tells a Stop from a failure by how the process ended. No `failed` is
        written: nothing failed.
        """
        if signal.getsignal(signal.SIGTERM) is signal.SIG_IGN:
            return lambda: None  # somebody chose to ignore it, and so does the run

        def on_term(signum: int, frame: Any) -> None:
            self._release()
            signal.signal(signum, previous if previous is not None else signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        try:
            previous = signal.signal(signal.SIGTERM, on_term)
        except ValueError:
            return lambda: None  # not the main thread, where signals cannot be set
        return lambda: signal.signal(signal.SIGTERM,
                                     previous if previous is not None else signal.SIG_DFL)

    # ------------------------------------------------------- cache validity
    #
    # Every stage here caches to disk, and until now nothing checked that the cache
    # still described the file in front of it. Three ways that went wrong, all of
    # them producing a finished document that looked right:
    #
    #   Record over a memo, keep the name. The job folder is keyed on the path, so
    #   the second run said "reusing the cache" and wrote out the first recording's
    #   words under the second recording's date.
    #
    #   Change the model, the language, the hotwords. Nothing re-ran, because the
    #   cached transcript carried no record of what produced it.
    #
    #   Ask the app to redo a job in a different language. The state flipped, the
    #   text did not, and the header then announced a language the words were not in.
    #
    # Both fingerprints below are cheap and are checked on every run.

    def _source_fingerprint(self) -> str:
        """Identifies the source audio well enough to notice it was replaced.

        Size and modification time, not a content hash: hashing a 200 MB file on
        every run to catch a rare mistake is the wrong trade. This misses an edit
        that preserves both, which no recording app does.
        """
        st = self.source.stat()
        return f"{st.st_size}:{st.st_mtime_ns}"

    def _asr_fingerprint(self) -> str:
        """The settings that decide what the transcript says."""
        s = self.s
        payload = {
            "model": s.model, "compute_type": s.compute_type,
            # Where it ran belongs in the fingerprint. float16 on the GPU and int8
            # on the CPU do not produce the same numbers, and a cache that ignores
            # that hands back the other one while reporting the one you asked for.
            "asr_device": s.asr_device,
            "language": self.state.get("language", s.language),
            "beam_size": s.beam_size, "align": s.align,
            "vad_onset": s.vad_onset, "vad_offset": s.vad_offset,
            "initial_prompt": s.initial_prompt, "hotwords": sorted(s.hotwords),
            "temperature_fallback": s.temperature_fallback,
            "condition_on_previous_text": s.condition_on_previous_text,
            # These three decide which segments survive decoding and when the
            # temperature fallback fires, so they change the words. Leaving them
            # out meant that changing one and rerunning printed "reusing the
            # cache" and handed back the old transcript.
            "no_speech_threshold": s.no_speech_threshold,
            "log_prob_threshold": s.log_prob_threshold,
            "compression_ratio_threshold": s.compression_ratio_threshold,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def _diar_fingerprint(self) -> str:
        return _diar_fingerprint_of(self.s.min_speakers, self.s.max_speakers,
                                    self.s.diarize_device)

    def _drop_stale_cache(self) -> None:
        """If the source changed, everything derived from it is wrong. Say so, then go."""
        current = self._source_fingerprint()
        recorded = self.state.get("source_fingerprint")
        if recorded is None or recorded == current:
            self.state["source_fingerprint"] = current
            return

        self.report("the source file changed since this job was last run, "
                    "so the cached transcript belongs to different audio. Redoing it.")
        for name in ("audio16k.wav", "transcript.json", "diarization.json",
                     "embeddings.npz", "turns.json"):
            (self.dir / name).unlink(missing_ok=True)
        for stale in ("language", "language_note", "language_confidence",
                      "language_agreement", "language_strength",
                      "language_samples", "word_level", "duration",
                      "asr_fingerprint", "diar_fingerprint", "speaker_counts",
                      # Speaker labels are positional. Keeping the old mapping
                      # over new audio meant run() reapplied the previous
                      # conversation's names to whoever happens to be SPEAKER_00
                      # this time, and produced a finished document saying so.
                      "names", "matches"):
            self.state.pop(stale, None)
        self.state["source_fingerprint"] = current
        self._save_state()

    def _cache(self, key: str, path: Path, produce: Callable[[], Any],
               load: Callable[[Path], Any], dump: Callable[[Any, Path], None],
               *, force: bool) -> Any:
        if path.exists() and not force:
            self.report(f"{key}: reusing the cached result")
            return load(path)
        value = produce()
        dump(value, path)
        return value

    # ---------------------------------------------------------------- phases
    def prepare_audio(self, *, force: bool = False) -> Path:
        info = audio.probe(self.source)

        # A video with no audio track is a plausible mistake, and ffmpeg reports it as
        # "Error opening output files: Invalid argument", which says nothing about the
        # actual problem.
        if info.sample_rate == 0 and info.channels == 0:
            raise RuntimeError(f"{self.source.name} has no audio track.")
        if info.duration <= 0:
            raise RuntimeError(f"{self.source.name} contains no audio.")

        if self.wav.exists() and not force:
            # A WAV from a run that was interrupted opens and decodes like any other,
            # it is simply shorter. Comparing it against the source is the only way to
            # tell, and everything downstream believes whatever this file says.
            cached = audio.probe(self.wav).duration
            if info.duration > 0 and abs(cached - info.duration) > info.duration * 0.01:
                self.report(f"audio: the prepared copy is {cached:.0f}s against "
                            f"{info.duration:.0f}s of source, so it is a leftover from "
                            "an interrupted run. Converting again.")
                force = True
            else:
                self.report("audio: already prepared")
                return self.wav

        self.report("audio: converting to 16 kHz mono WAV, normalizing the level")
        audio.prepare(self.source, self.wav)

        when, when_source = audio.recorded_at(info)
        self.state["duration"] = info.duration
        self.state["source"] = str(self.source)
        self.state["codec"] = info.codec
        self.state["recorded"] = when.isoformat() if when else None
        self.state["recorded_source"] = when_source
        if info.memo_uuid:
            self.state["memo_uuid"] = info.memo_uuid
        if info.device:
            self.state["device"] = info.device
        self._save_state()
        return self.wav

    def detect_language(self, *, force: bool = False) -> str:
        if self.s.language and self.s.language != "auto":
            self.state["language"] = self.s.language
            self.state["language_note"] = "set by hand"
            self._save_state()
            return self.s.language
        if self.state.get("language") and not force:
            return self.state["language"]

        self.report("language: sampling the file at 5 points")
        guess = lang.detect(self.wav)
        self.state["language"] = guess.language
        self.state["language_confidence"] = guess.confidence
        self.state["language_agreement"] = guess.agreement
        self.state["language_strength"] = guess.strength
        self.state["language_note"] = guess.note
        self.state["language_samples"] = guess.samples
        self._save_state()
        flag = "" if guess.reliable else "  ⚠️  "
        # The note can itself contain commas, so it goes in brackets. A second colon
        # would read badly against the "language:" prefix the app matches on.
        # Both halves, because one number cannot say which of them is low. A file
        # every window agreed on, weakly, and a file the windows split on read the
        # same when the two are multiplied together first.
        self.report(f"language:{flag} {guess.language} "
                    f"({guess.agreement:.0%} of the windows, "
                    f"model {guess.strength:.0%} sure) [{guess.note}]")
        return guess.language

    def transcribe(self, *, force: bool = False) -> list[dict]:
        path = self.dir / "transcript.json"
        want = self._asr_fingerprint()
        if path.exists() and not force and self.state.get("asr_fingerprint") == want:
            self.report("transcription: reusing the cache")
            data = json.loads(path.read_text())
            self.state["word_level"] = data.get("word_level", False)
            return data["segments"]
        if path.exists() and not force:
            self.report("transcription: the settings changed since the cached run, "
                        "transcribing again")

        s = Settings(**{**asdict(self.s), "language": self.state.get("language", self.s.language)})
        # In a process of its own, which writes transcript.json and exits. Whisper
        # and the aligner leave gigabytes of Metal memory that only the end of the
        # process gives back, and diarization, next, is the stage that ran out of it
        # on a long recording. A cached transcript never gets here, so it never
        # starts one. The line saying where it runs is printed by the stage.
        stage.call(ASR_STAGE,
                   {"wav": str(self.wav), "settings": asdict(s), "out": str(path)},
                   report=self.report, what="transcription")
        data = json.loads(path.read_text())
        self.state["word_level"] = data.get("word_level", False)
        self.state["asr_fingerprint"] = want
        self._save_state()
        return data["segments"]

    def diarize(self, *, force: bool = False) -> diarize.Diarization:
        turns_path = self.dir / "diarization.json"
        emb_path = self.dir / "embeddings.npz"
        want = self._diar_fingerprint()
        if (turns_path.exists() and emb_path.exists() and not force
                and self.state.get("diar_fingerprint") != want):
            self.report("diarization: the model or the speaker count changed, "
                        "diarizing again")
            force = True
        if turns_path.exists() and emb_path.exists() and not force:
            self.report("diarization: reusing the cache")
            raw = json.loads(turns_path.read_text())
            emb = dict(np.load(emb_path))
            return diarize.Diarization(
                turns=[diarize.SpeakerTurn(**t) for t in raw],
                embeddings={k: np.asarray(v) for k, v in emb.items()},
            )

        token = hf_token()
        if not token:
            raise RuntimeError(
                f"{ERR_NO_TOKEN} The Hugging Face token for pyannote is missing.\n"
                "  scriba token hf_xxxxxxxx   # stores it in the Keychain\n"
                "and accept the terms at hf.co/pyannote/speaker-diarization-3.1"
            )
        # Naming the model and the device on every run is deliberate. Which of the two
        # models is in use depends on the installed pyannote, and Metal is picked
        # automatically. Both belong in the log rather than in someone's memory.
        dev = diarize._pick_device(self.s.diarize_device)
        self.report(f"diarization: {diarize.DIARIZATION_MODEL.split('/')[-1]} "
                    f"on {dev}, with voice print extraction")
        dia = diarize.run(
            self.wav, hf_token=token,
            min_speakers=self.s.min_speakers, max_speakers=self.s.max_speakers,
            device=self.s.diarize_device,
        )
        write_atomic(turns_path, json.dumps([asdict(t) for t in dia.turns], ensure_ascii=False))
        _savez_atomic(emb_path, **dia.embeddings)
        self.state["diar_fingerprint"] = want
        self._save_state()
        self.report(f"diarization: {len(dia.speakers())} distinct voices, "
                    f"{len(dia.embeddings)} usable voice prints")
        return dia

    def identify(self, dia: diarize.Diarization) -> dict[str, dict]:
        """Match the voice prints against the voice registry."""
        reg = VoiceRegistry()
        speech = dia.speech_time()
        matches: dict[str, dict] = {}
        for label, vec in dia.embeddings.items():
            if speech.get(label, 0.0) < self.s.voice_min_speech_sec:
                # "candidate" has to be present even here. The reporting loop below
                # reads that field on every entry, and a dict built without it raised
                # KeyError on exactly the speakers this branch exists to handle.
                matches[label] = {
                    "name": None, "candidate": None, "score": 0.0,
                    "reason": f"only {speech.get(label, 0):.0f}s of speech: "
                              "voice print too thin to trust",
                }
                continue
            m = reg.match(vec,
                          threshold=self.s.voice_match_threshold,
                          suggest_threshold=self.s.voice_suggest_threshold,
                          margin=self.s.voice_match_margin)
            matches[label] = {
                "name": m.person.name if m.person else None,
                "candidate": m.candidate.name if m.candidate else None,
                "score": m.score, "reason": m.reason,
            }
        # Say the stage happened even when it found nothing. The app reads these
        # lines to tick its checklist, and on an empty registry, which is every
        # first run, no line was printed at all: the stage the whole tool is
        # about was the one that never ticked.
        named = sum(1 for i in matches.values() if i.get("name"))
        self.report(f"registry: compared {len(matches)} voices against "
                    f"{len(reg.people)} on file, {named} recognised")
        for label, info in matches.items():
            if info.get("name"):
                self.report(f"voice {label} → {info['name']} ({info['score']:.3f})")
            elif info.get("candidate"):
                self.report(f"voice {label} ≈ {info['candidate']}? "
                            f"({info['score']:.3f}), to be confirmed")
        return matches

    # ------------------------------------------------------------------- run
    def _refuse_if_running(self) -> None:
        """Say no to a second run of a job another process is working on.

        Two runs wrote the same folder (the audio copy, transcript.json, state.json)
        and held the GPU at the same time, and nothing told the person. The second
        took over the first one's note and whichever ended first deleted it, so the
        other was listed as not running while it still was. A note left by a
        process that is gone does not count (`running_pid`). Called before
        anything is written, because the note and the state in this folder belong
        to the run that is going on.
        """
        other = running_pid(self.dir)
        if other and other != os.getpid():
            raise RuntimeError("This recording is already being transcribed by "
                               f"another scriba (process {other}).")

    def run(self, *, force: str | None = None) -> JobResult:
        self._refuse_if_running()
        # Before the audio is even converted. Converting takes seconds and the
        # model load that follows is where a missing disk shows up, worded by
        # whichever library got there first.
        if problem := model_cache_problem():
            raise RuntimeError(problem)
        self._claim()
        restore = self._release_on_sigterm()
        # From here the machine is working for this recording, whoever started it.
        # After the refusals above, so a run that was turned away holds nothing.
        awake.hold()
        try:
            result = self._run(force=force)
        except Exception as exc:
            # Written down, so the folder can say why it stopped. Until now the
            # reason lived in the terminal that printed it and nowhere else, and
            # the app's "Not finished" panel could list what was missing but
            # never what went wrong.
            self.state["failed"] = str(exc).replace(ERR_NO_TOKEN, "").strip()[:600]
            self._save_state()
            raise
        finally:
            self._release()
            restore()
        self.state.pop("failed", None)
        self._save_state()
        return result

    def _run(self, *, force: str | None) -> JobResult:
        f_all = force == "all"
        self._drop_stale_cache()
        if self.s.min_speakers is not None or self.s.max_speakers is not None:
            self.state["speaker_counts"] = {"min": self.s.min_speakers,
                                            "max": self.s.max_speakers}
        self.prepare_audio(force=f_all)
        self.detect_language(force=f_all or force == "lang")
        segments = self.transcribe(force=f_all or force == "asr")

        dia = None
        matches: dict[str, dict] = {}
        if self.s.diarize:
            dia = self.diarize(force=f_all or force == "diar")
            segments = diarize.assign(segments, dia,
                                      word_level=self.state.get("word_level", False))
            matches = self.identify(dia)

            # What the diarizer heard and the transcriber never wrote down.
            #
            # These two look at the same audio and disagree about where the speech
            # is, because the transcriber has a gate in front of it deciding what
            # is worth decoding. When that gate is wrong nothing says so: the
            # document simply lacks what was said and reads perfectly without it.
            # The gaps are kept with their timestamps, because the only way to
            # find out whether one of them held a sentence is to go and listen.
            gaps = dia.gaps(segments)
            missing = sum(b - a for a, b in gaps)
            total = sum(t.duration for t in dia.turns)
            self.state["untranscribed_seconds"] = round(missing, 1)
            self.state["speech_seconds"] = round(total, 1)
            self.state["untranscribed_gaps"] = [[round(a, 1), round(b, 1)] for a, b in gaps]
            if missing >= 5.0:
                worst = ", ".join(export.hhmmss(a) for a, _ in
                                  sorted(gaps, key=lambda g: g[0] - g[1])[:3])
                self.report(
                    f"coverage: {missing:.0f}s of the {total:.0f}s of speech found by the "
                    f"diarizer never reached the transcript, in {len(gaps)} stretches of "
                    f"two seconds or more. The longest start at {worst}.")

        turns = diarize.to_turns(segments)
        write_atomic(self.dir / "turns.json",
                     json.dumps(turns, ensure_ascii=False, indent=1))
        # The words with their times and speakers, for the app's player: the one
        # place where the speaker of each word survives the run.
        write_atomic(self.dir / "words.json",
                     json.dumps(diarize.word_track(segments), ensure_ascii=False))

        return self.write_outputs(turns, segments, matches,
                                  speech=dia.speech_time() if dia else {})

    def write_outputs(self, turns: list[dict], segments: list[dict],
                      matches: dict[str, dict], *, speech: dict[str, float]) -> JobResult:
        """Everything after the words and the voices: names, briefing, documents.

        Separate from run() so the documents can be rewritten from what is already
        cached. Rewriting used to mean running the whole pipeline again, and after
        any change to the settings that the cache keys on, that meant transcribing
        an hour of audio to produce a file the engine could have written in a
        second from data it already had.
        """
        language = self.state.get("language", "it")
        profiles = naming.build_profiles(turns, speech, language, matches)

        dossier_path = self.dir / "who-is-who.md"
        dossier_path.write_text(naming.dossier(
            profiles, language=language, title=self.source.stem))

        # Names already settled in an earlier pass stay put.
        saved: dict[str, str] = self.state.get("names", {})
        names = naming.apply_names(saved, profiles)
        self.state["names"] = names
        self.state["matches"] = matches
        self._save_state()

        # Rendered in the language of the documents, because this list is printed into
        # them: "Voce 2" in the header over "Voce 2" in the transcript, not "Voice 2".
        unresolved = [export.display(p.label, names, language) for p in profiles
                      if p.label not in names]

        recorded = None
        if self.state.get("recorded"):
            try:
                recorded = datetime.fromisoformat(self.state["recorded"])
            except ValueError:
                recorded = None

        meta = {
            "title": self.state.get("title") or self.source.stem,
            "recorded": recorded,
            # Where the date came from travels with it. "recorded" is the timestamp
            # the recorder wrote; anything else is the filesystem's opinion, and for
            # an exported voice memo the filesystem thinks the conversation happened
            # at the moment it was copied.
            "recorded_source": self.state.get("recorded_source", ""),
            "duration": self.state.get("duration", 0.0),
            "language": language,
            "source_file": self.source.name,
            "device": self.state.get("device", ""),
            "speaker_stats": speech,
            "untranscribed_seconds": self.state.get("untranscribed_seconds", 0.0),
            "speech_seconds": self.state.get("speech_seconds", 0.0),
            "untranscribed_gaps": self.state.get("untranscribed_gaps", []),
            "unresolved": unresolved,
        }
        outputs = export.write_all(
            self.dir / "output", slugify(self.source.stem), self.s.output_formats,
            turns=turns, segments=segments, names=names, meta=meta,
            matches=[{"speaker": k, **v} for k, v in matches.items()],
        )
        self.report(f"wrote {len(outputs)} files to {self.dir / 'output'}")

        return JobResult(
            job_dir=self.dir, outputs=outputs, language=language,
            speakers=[p.label for p in profiles], names=names,
            unresolved=unresolved, dossier_path=dossier_path,
            duration=self.state.get("duration", 0.0),
        )

    def rewrite(self, *, speech: dict[str, float] | None = None) -> JobResult:
        """Write the documents again from the cache, transcribing nothing.

        Needed whenever the shape of a document changes, or a format is added, or
        an output file is lost. Without it the only way to get a document back was
        to run the whole pipeline, and a change to any setting the ASR cache keys
        on turns that into an hour of transcription for a file the engine already
        has the words for.

        `speech` is the seconds each voice spoke, when the caller has the
        diarizer's own count. Without it the turns are summed, which also counts
        the pauses inside them.
        """
        turns_path = self.dir / "turns.json"
        transcript = self.dir / "transcript.json"
        if not turns_path.exists() or not transcript.exists():
            raise ValueError(
                f"{self.source.name} has nothing cached to write from. "
                "Transcribe it first with `scriba run`.")

        turns = json.loads(turns_path.read_text())
        segments = json.loads(transcript.read_text()).get("segments", [])
        matches = self.state.get("matches") or {}

        if speech is None:
            speech = {}
            for t in turns:
                if t.get("speaker"):
                    speech[t["speaker"]] = speech.get(t["speaker"], 0.0) + (t["end"] - t["start"])

        return self.write_outputs(turns, segments, matches, speech=speech)

    def _cached_speech_time(self) -> dict[str, float] | None:
        """Seconds of speech per voice as the diarizer counted them, read from disk.

        The figure `run` gives the briefing and the document header. Read as it
        is, with no check that it still matches the settings: this is for putting
        names on a result that exists, not for deciding whether to make it again.
        """
        try:
            raw = json.loads((self.dir / "diarization.json").read_text())
            return diarize.Diarization(
                turns=[diarize.SpeakerTurn(**t) for t in raw], embeddings={},
            ).speech_time()
        except (OSError, ValueError, TypeError):
            return None

    def speaker_labels(self) -> set[str]:
        """The speaker labels this job actually produced, read from the embeddings.

        Used to reject a name for a speaker that does not exist. Without this, a typo
        in the label was stored in the job state, reported as a successful enrollment,
        and exited zero.
        """
        emb = self.dir / "embeddings.npz"
        if not emb.exists():
            return set()
        try:
            with np.load(emb) as data:
                return set(data.files)
        except (zipfile.BadZipFile, ValueError, OSError):
            # A run killed mid-write leaves a truncated archive. Writes go
            # through a rename now, so this is history rather than a live
            # failure, but the file is already out there on somebody's disk and
            # "no speakers yet" is a better answer than a BadZipFile traceback.
            self.report("the voice prints for this job are unreadable: "
                        "rerun with --force diar to rebuild them")
            return set()

    # ---------------------------------------------------------------- rename
    def set_names(self, mapping: dict[str, str], *, enroll: bool = True) -> JobResult:
        """Assign the names, then learn them for next time. That is the whole point.

        The documents are written again from what is cached and no stage runs. This
        used to call `run()`, and `run()` diarized the recording again whenever the
        speaker count it was given differed from the one the cached diarization was
        made with, which `scriba name` never has a count for. The new diarization
        could find another number of voices and give the labels to other people,
        while the names just typed were applied by label to whoever held it.
        """
        self._refuse_if_running()
        names = {**self.state.get("names", {}), **mapping}
        # An empty value takes the name away. The app sends `SPEAKER_01=` for a
        # field the person cleared, and merging that in as it was left the old
        # name in the document with no way to remove it. A voice the registry
        # recognises gets its name back when the documents are written, which is
        # what recognising it means.
        self.state["names"] = {label: name for label, name in names.items() if name}
        self._save_state()

        if enroll:
            emb_path = self.dir / "embeddings.npz"
            if emb_path.exists() and self.speaker_labels():
                reg = VoiceRegistry()
                data = np.load(emb_path)
                added = 0
                for label, name in mapping.items():
                    if label in data and name:
                        reg.enroll(name, data[label], source=self.source.name)
                        added += 1
                reg.save()
                self.report(f"voice registry: {added} voice prints enrolled. "
                            "On the next recording these names attach on their own")
        return self.rewrite(speech=self._cached_speech_time())
