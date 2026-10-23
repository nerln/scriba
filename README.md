<div align="center">

# scriba

**Recorded conversations become a document that says who said what.**

Transcribes, separates the voices, and remembers a voice once you have named it.
Everything runs on your own machine.

[![macOS](https://img.shields.io/badge/macOS-14%2B-black?logo=apple&logoColor=white)](#the-app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#install)
[![Licence](https://img.shields.io/badge/licence-MIT-green)](#licence)
[![Built on](https://img.shields.io/badge/built%20on-whisperX%20%2B%20pyannote-orange)](#speed-and-why-it-is-what-it-is)

<img src="docs/img/04-who-is-speaking.png" width="860" alt="Scriba showing two speakers, both recognised from the voice registry, with confidence scores and a quote from each">

</div>

---

Diarization tells you there were two people. It will not tell you that the second one
is Dana, and the next recording will number them differently anyway. scriba keeps the
voice prints that pyannote computes and throws away, files them under the name you
give, and matches them the next time.

```bash
scriba run meeting.m4a
scriba name meeting.m4a SPEAKER_00=Ada SPEAKER_01=Rafiq
scriba run tomorrow.m4a          # Ada and Rafiq are picked out on their own
```

The screenshots below use two recordings that are entirely synthetic: the dialogue is
invented and the voices are macOS speech synthesis. `python tools/make-demo.py`
rebuilds them, so you can reproduce the whole flow without a real conversation.

## What it looks like

Drop a file in and nothing happens until you say so. Transcription runs for about as
long as the recording lasts, which is not something to start by accident.

<img src="docs/img/02-before-you-start.png" width="800" alt="A queued recording with the language and speaker-count options, and a Transcribe button">

The two settings above the button are the ones worth touching. Language is left to
detection unless you know better, and the number of people in the room changes the
result more than anything else here.

<img src="docs/img/03-running.png" width="800" alt="The seven stages of a run, with the finished ones ticked and the engine's own log underneath">

Seven stages, each ticked as it finishes, with the engine's own output underneath. The
window can be closed and the work carries on.

<img src="docs/img/01-empty.png" width="800" alt="The empty window listing previously processed recordings in the sidebar">

## Why this exists

Record a conversation, transcribe it, hand the text to whatever you use for reading
and searching. The flow is obvious. Doing it for real breaks in four places.

### 1. The wrong language raises no error, it invents a text

The bug this project was born from. A script with a hardcoded `--language it`, handed a
recording that was actually in Spanish. Whisper did not fail. It guessed, and produced
fluent, punctuated, entirely made-up Italian.

The two languages are close enough that guessing produces real words. Spanish and
Italian are full of pairs that sound alike and mean different things, so the output
lands on the wrong one and reads perfectly:

| Spanish | Heard as Italian | Which means |
|---|---|---|
| `salir` (to leave) | `salire` | to go up |
| `carta` (letter) | `carta` | paper |
| `burro` (donkey) | `burro` | butter |
| `éxito` (success) | `esito` | outcome |

Those four are the textbook pairs, put here to show the shape of the failure. What it
does to a real recording is worse, because it does it to every sentence at once and
each one comes out plausible.

Nothing on screen says anything is wrong. The text reads as correct, and you then
reason on top of it for weeks.

Left alone, Whisper picks the language from the first 30 seconds, which in a real
conversation is small talk. [`lang.py`](scriba/lang.py) samples five windows across the
whole file and votes, weighting each window by confidence, and says so when the vote is
close instead of choosing in silence.

It also says so when the windows agree and none of them was sure. Agreement on its own
used to be the whole confidence, which is a share: five windows that all say Galician
at 30% agree perfectly, and the file came out reported at 100% confidence. Now the
strength of the winning windows is part of the number, and a unanimous shrug is called
one.

### 2. Speaker labels restart from zero every time

Diarization hands you `SPEAKER_00`, `SPEAKER_01`. Those labels are arbitrary, and they
are reassigned on every file: the same person is `SPEAKER_00` on Monday and
`SPEAKER_02` on Tuesday. So you rename them by hand, every time, forever.

pyannote computes a 256-dimensional embedding per speaker anyway, since that is how it
decides who is who. whisperX discards it by default; since 3.8.6, the version pinned
here, `return_embeddings=True` hands it back. So the embeddings on their own no longer
argue for going around whisperX.

[`diarize.py`](scriba/diarize.py) calls pyannote directly all the same, and now for a
different reason: whisperX's diarization wrapper runs against pyannote 4 and nothing
else, and it drops the overlap-free segmentation pyannote 4 also returns, with no flag
to ask for it back. Calling pyannote directly keeps one file working on pyannote 3 and 4
alike. The vectors survive either way, and [`voices.py`](scriba/voices.py) files them
under the name you assign. From then on the name attaches itself.

There are three zones rather than two:

| Cosine similarity | What happens |
|---|---|
| 0.75 and above | the name is applied |
| 0.55 to 0.75 | "Maybe *Ada*?" A suggestion. Confirming stays your job. |
| below 0.55 | new voice |

There is also a margin of 0.05 over the runner-up: when two people on file come out
almost equally close, nothing is chosen. A wrong name raises no error either, and it
quietly poisons every transcript downstream, so an unanswered question is the better
failure.

The thresholds are a starting point to calibrate, not a published constant. To set them
properly: hand-label three or four conversations, find where the same-speaker and
different-speaker similarity distributions cross, put the threshold there. Twenty
minutes of annotated audio tells you more than any benchmark, and for diarization in
Italian there is no benchmark to argue with.

### 3. Whisper's "segments" are not conversational turns

Whisper cuts roughly every 30 seconds no matter who is speaking. On a seven-minute test
file that gave 16 blocks, each one holding more than one speaker. scriba realigns at
word level and reassembles real turns: same file, 41 turns. That is the difference
between a list of fragments and something a person, or a model, can read.

### 4. What reads the transcript reads prose

SRT and VTT shatter every sentence into two-second blocks with sequence numbers and
timecodes. All of that is noise that eats context and makes retrieval worse. JSON is
worse still.

The default output is one Markdown document: a header with participants, duration and
language, then turns with the name in bold and a timestamp at the start of each.

Voices that could not be identified are marked as unidentified inside the document
itself, on every line and not only in the header. Leave that out and "Voice 2" gets
read as a person somebody knows, and confident answers get built on top of it.

```markdown
# Team sync, 18 July

## Overview
- **Duration**: 00:47:12
- **Participants**: Ada (18:22 of speech), Rafiq (14:05 of speech)
- **Unidentified voices**: Voice 3. A distinct person, but their name is
  never spoken in the recording. Do not guess who they are.

## Transcript

**Ada** [00:12]: Right, so where did we land on the migration?

**Rafiq** [00:31]: Staging is done. Production waits for the backup window.
```

Five other formats come out of the same run for the tools that want them: plain
Markdown, text, SRT, VTT and JSON.

The words scriba writes around the text follow the language of the recording. An
Italian one has `## Panoramica`, `- **Partecipanti**:` and `Voce 3 (non identificata)`
where the document above has Overview, Participants and Voice 3 (unidentified);
Spanish does the same, and any other language gets English. Only what scriba adds
changes. The transcript is as it was said, and the file names, the JSON keys, the
SRT and VTT structure and the language code in the header are the same in every
language, because other programs read them.

## Install

macOS with Apple Silicon, Python 3.10 or newer, `ffmpeg`, and a Hugging Face account
for the pyannote models. Set aside a few gigabytes and a few minutes: the
dependencies pull in torch, and the first run downloads the `large-v3` weights.

```bash
brew install ffmpeg
```

The Python side wants an environment of its own, because whisperX and pyannote pin a
torch that has no business on the system interpreter. Conda if you have it:

```bash
conda create -n scriba python=3.11 -y && conda activate scriba
pip install git+https://github.com/nerln/scriba
```

A plain virtual environment does the same job:

```bash
python3 -m venv ~/.venvs/scriba && source ~/.venvs/scriba/bin/activate
pip install git+https://github.com/nerln/scriba
```

Either way the `scriba` command lives inside that environment and nowhere else, so
every use starts by activating it. Forget that step and the shell says
`command not found`, which is the least informative thing it could say. A launcher on
the PATH is worth the thirty seconds:

```bash
mkdir -p ~/.local/bin
cat > ~/.local/bin/scriba <<'EOF'
#!/bin/zsh
# Set SCRIBA_BIN to the scriba inside your environment if none of these is right.
ENV_NAME="${SCRIBA_ENV:-scriba}"
for candidate in "$SCRIBA_BIN" \
                 "$HOME/.venvs/$ENV_NAME/bin/scriba" \
                 "$HOME/miniforge3/envs/$ENV_NAME/bin/scriba" \
                 "$HOME/miniconda3/envs/$ENV_NAME/bin/scriba" \
                 "$HOME/anaconda3/envs/$ENV_NAME/bin/scriba" \
                 "/opt/homebrew/Caskroom/miniforge/base/envs/$ENV_NAME/bin/scriba"; do
    [[ -x "$candidate" ]] && exec "$candidate" "$@"
done
print -u2 "scriba: nothing called '$ENV_NAME' in any of the usual places."
print -u2 "        Set SCRIBA_BIN to the scriba inside your environment."
exit 1
EOF
chmod +x ~/.local/bin/scriba
```

Check that `~/.local/bin` is on your `PATH`, and `scriba` works from any shell. The
loop matters more than it looks: the path to a conda environment depends on how conda
was installed, and a launcher that hardcodes one of them fails with a message that
never mentions scriba.

The pyannote token goes into the Keychain rather than into a file:

```bash
scriba token hf_xxxxxxxx
```

Then accept the conditions on
[speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1),
[segmentation-3.0](https://hf.co/pyannote/segmentation-3.0) and
[speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1),
using the same account as the token. Skip that and the download fails with a bare 403
that explains nothing. The third one matters even if you never ask for community-1:
under pyannote 4, loading 3.1 pulls one of its files from that repository, so a single
unaccepted licence blocks both models.

### A note on telemetry

pyannote 4 ships a telemetry module. Its `config.yaml` sets `metrics_enabled: true`,
and it reports to an endpoint at pyannote.ai on every model load and every file
processed. scriba turns it off in `scriba/__init__.py`, before any import path can
reach pyannote, because the flag is read once at import time and setting it afterwards
does nothing. If you use pyannote directly for recordings of private conversations, set
`PYANNOTE_METRICS_ENABLED=false` yourself.

## Use

```bash
# the language is worked out on its own
scriba run recording.m4a

# if you know how many people are in the room, saying so helps a lot
scriba run meeting.m4a --lang es --min-speakers 2 --max-speakers 2

# who is who: how much each one talks, their longest turns,
# and the moments where somebody says a name out loud
scriba dossier meeting.m4a

# assign the names, and from here on they are remembered
scriba name meeting.m4a SPEAKER_00=Ada SPEAKER_01=Rafiq

scriba show meeting.m4a               # where the document went, and what else was written
scriba voices list                    # who is on file
scriba watch ~/Memos                  # transcribes every audio file that lands there
scriba jobs list                      # what has been processed, and how far it got
```

`show --reveal` opens the folder in the Finder. It exists because a run prints its
paths and then scrolls away, and a week later the job folder is a slug with a hash
on the end.

`watch` keeps its bookkeeping in a `.scriba-done` folder inside the folder it is
watching, and a file that failed is recorded there too, so it is not retried on the
next pass. Delete its entry to try it again.

`jobs list` exists because the job folder is not somewhere anybody browses. The names
are slugs with a hash on the end, and "did I ever transcribe that one" otherwise has no
answer short of opening files. It also shows what the disk is holding: each job keeps a
full 16 kHz copy of the audio, about 2 MB a minute, which `jobs prune --audio` gives
back. Transcripts and voice prints cost CPU to make and are never touched by it.

Every stage is cached under `~/.scriba/jobs/<name>/`, so redoing just the names
re-transcribes nothing. Pass `--force asr|diar|lang|all` to rerun the stage you want.
The number of people a run was given is remembered by the job, so a recording put
back in the queue is heard with the same count, and `scriba name` rewrites the
documents from what is cached without diarizing again.

Transcription runs in a child process of its own. The Metal memory a model takes is
not all given back when the model is deleted, so the diarizer used to start with
whatever transcription had left, and on a 16 GB Mac a long recording ran out of it
there. The child ends and the system takes everything back. `SCRIBA_IN_PROCESS=1`
runs it in the calling process instead, which is what a debugger and the tests want.

A run leaves `running.json` in its job folder with its process id and the moment that
process started. A Stop removes it, and a pid that macOS later gives to another process
is told from the run by that moment, so a leftover note never blocks the job.

A run started from a terminal keeps the Mac from going to sleep while it works. The
first recording that starts holds `/usr/bin/caffeinate -i -w <pid>`, one per process,
which lets go by itself when scriba ends, by a clean exit or a kill. It refuses idle
sleep only: closing the lid and the Sleep menu still work. `scriba watch` and
`scriba memos` hold it from the first recording until they are stopped. The hold is in
the engine, so a run the app starts has it too, and a second assertion taken by the
app beside it changes nothing. `SCRIBA_KEEP_AWAKE=0` turns it off, for a laptop that
should sleep on battery and carry on when it wakes.

### One recording, many conversations

A day at a fair or an evening of introductions is one recording with twenty
conversations in it, and nobody reads that the next morning to remember whom to write
to. After a run:

```bash
scriba conversations "fair.m4a"            # summaries in Italian
scriba conversations "fair.m4a" --language en --owner Ada
```

It writes `output/conversations/` in the job folder: an `index.md` with one line per
conversation (time of day, length, language, who, what about, next step), one file per
conversation with its own transcript, and `conversations.json` for anything else that
wants to read them.

The boundaries come from the voices. The owner of the recording is the voice with the
most speech, or the one named with `--owner`; a conversation ends when the people
talking with them change, or after a minute of silence (`--gap`). Greetings and
goodbyes help where a noisy hall made the diarizer give two different people the same
label. A stretch with nobody but the owner in it, or shorter than half a minute, is
counted and not listed.

Languages move between conversations, and the run transcribed the whole file in the
one that won the vote. Each conversation is checked on its own audio, and the ones in
another language are transcribed again in theirs (`--keep-language` skips that). The
summaries come from a model on this Mac through Ollama (`--model`, default
`qwen3:8b`); the address is checked to be this machine, and `--no-summary` splits
without them.

### Finding your own voice

Most of your own recordings have you in them, and that fact does two jobs at once.

```bash
scriba whoami ~/Recordings
```

This diarizes every recording in the folder and transcribes none of them, which is what
makes it practical: working out who is present needs embeddings rather than words, and
on Metal that runs at about a tenth of realtime. Hours of audio take minutes.

It then compares speakers *across* recordings, ignoring pairs from inside the same one.
Two people in the same room were separated by the diarizer because they sound
different, so those pairs say nothing about whether one person sounds like themselves
on another day, which is the question that matters here.

The voice present in the most recordings is whoever owns the microphone. You get told
which one that is, with a timestamp in each recording to listen to, and nothing is
written until you confirm:

```bash
scriba whoami ~/Recordings --name "Your name"
```

That enrolls every print of that voice at once, from different days and different
rooms, which is what makes later matching hold up. A registry built from one recording
only ever proves that a file matches itself.

The same scan answers the calibration question from earlier. Comparing two recordings
of the same person made on different days is the honest way to set a threshold, and a
folder of recordings contains hundreds of those comparisons already. `whoami` prints
where the two populations separate in your own material, so the number is measured
rather than assumed.

## Voice memos, without moving them

The recordings are already on the machine. `scriba memos` watches the Voice Memos
library and transcribes each new one as it appears:

```bash
scriba memos                 # waits for new recordings
scriba memos --once          # does what is already there, then stops
```

macOS keeps that folder closed, and the first run will say so. There are two ways
past it and they are not equivalent.

The blunt one is Full Disk Access for whatever runs scriba. In practice that is a
Python interpreter, and Full Disk Access on a Python interpreter is Full Disk
Access for every package ever installed beside it. Not recommended.

The other is [paranco](https://github.com/nerln/paranco): a separate application
whose only job is to hold that one permission and lift new files out of a short,
compiled-in list of protected folders into ordinary ones, along routes you define.
Add a route from Voice Memos to `~/.scriba/inbox` and install its agent. scriba
then reads an ordinary folder and needs nothing at all. The app looks in that
folder whenever the window comes to the front and lists what it finds as waiting,
so a memo that came off the phone is in the queue before anybody asks; nothing
starts until the button is pressed. A row swiped away stays away.

It has to be an application started by launchd or by a person, and that is not
ceremony. macOS charges an access to the process that asked for it, so a
permission cannot be borrowed: driving Finder or System Events from a terminal
gets the terminal's permission tested, not theirs. Worth knowing because the
refusal does not look like one. Asked through System Events, the library reports
itself as empty rather than protected, which is the most misleading answer
available. paranco was written around that finding.

`scriba memos` uses the library directly when it can, and the inbox when it
cannot, and says which.

Nothing is written into Apple's folder. The list of what has been done goes under
`~/.scriba/watch/`, and the recordings are read and never touched. That applies to
any watched folder that will not take a file: the ledger normally lives inside the
folder so it travels with it, and moves out of the way when it cannot.

To have it running without being asked, a launchd agent at
`~/Library/LaunchAgents/dev.nerelli.scriba.memos.plist` pointing at `scriba memos`
does it. Worth knowing before you do: transcription uses the whole machine for
up to half the length of the recording, so an unattended watcher will make itself
felt on a laptop.

## The app

```bash
cd macapp && ./build.sh && open Scriba.app
```

SwiftUI with no outside dependencies and no `.xcodeproj`. SwiftPM builds the executable
and the script assembles the bundle around it, so the whole thing stays in git as text
and rebuilds from a terminal.

The app implements none of the work itself. It runs the same `scriba` and reads its
JSON state, which keeps the engine usable on its own and leaves no second copy of the
logic to drift apart a day later.

One rule for the interface: every row in the speaker table has a play button next to
the name field. If assigning a name costs less than checking it, people will assign
without checking. Voices with very little speech are flagged as probable artefacts,
which is usually what they are, somebody else's murmured agreement promoted to a
person.

The same goes for the text. Play and read, at the top of a finished transcript, plays
the recording with the transcript under it, the word being heard lit up and kept in
view. A click on a word plays from there, a drag still selects text to copy, and space
plays and pauses. A transcript read on its own is where a misheard name or figure goes
unchecked, and finding the moment again by ear in a waveform takes longer than the
reading did. It reads `words.json`, the time and the speaker of every word, which runs
write from now on; for an older run the app has `scriba words` make it from what the
run left in its folder, without loading a model.

On first run, open Settings. The app needs the Python of the environment where you
installed `scriba`, and both fields say whether what you typed is there. The second
field, the package folder, only matters when you are running from a clone; a pip
install can leave it alone. The pyannote token can be saved there too. Settings hands
it to `/usr/bin/security` on its standard input, so it is the same Keychain item
`scriba token` makes, the engine reads it without a dialog in the middle of a run, and
the token is in no command line and no environment. It has to be `hf_` and then
letters and digits, which is what Hugging Face issues; anything else is refused with a
message instead of being stored as a different token.

The queue is the list at the top of the sidebar. Its header counts what is
waiting, says when one is running and how many did not finish; the button under
the sidebar starts everything with an estimate of the time next to it, and the
Transcribe button on one recording starts that one alone. While one runs, the same
button on another puts it next, and the queue stops behind it: only Transcribe all
carries on through the list. Language and the number of people belong to each
recording, with a shortcut to give the same settings to every one still waiting.
A recording made in the app is queued with the language chosen on the welcome
screen, beside the switch for live text; left on automatic, the live text listens for
the language of the Mac and says that it is guessing. A file that is small, or held by
a cloud service, is queued with a note and not refused: the engine opens it and says
if it is empty.
The list survives a quit: what was waiting is waiting again at the next launch.

A run can be stopped from the strip under the sidebar or from the Transcribe menu.
Quitting the app stops the engine with it, and closes a recording in progress and
puts it in the queue. A second Stop, or a quit, while a stop is still finishing waits
for it, so the file is closed and queued before the app goes. Closing the window
stops nothing: the queue, the run and the recorder belong to the app, and the next
window shows them as they were. There is only ever one window, and the Dock icon or
Window > Show Scriba brings it back. The keys are the ones a Mac application puts
them on: `cmd O` to add recordings, `cmd T` to transcribe all, `cmd .` to stop, `cmd R`
to re-read the list.

The window stays usable while a transcription runs. That took saying twice, because
the first version handed the whole detail pane to the progress display, and a run
takes up to half the length of the recording. The strip says what is happening
from wherever you are in the app; the pane is yours. A run started from a
terminal shows in the list as running, not as a failure, for as long as it lasts.

While Scriba records or transcribes it asks macOS not to put the Mac to sleep when it
is idle, and lets go the moment the work ends, whichever way it ends (finished,
failed, stopped, or the app quit). It does not keep a closed lid awake: closing the lid of a
laptop puts it to sleep whatever an application asks, unless it is on power with an
external display. For a long run, leave the lid open or use that setup. The screen can
still turn off.

When Transcribe all finishes, or a single recording finishes while Scriba is in the
background, Scriba shows one notification with numbers only, such as "3
transcribed, 1 did not go through". It never contains a file name, a person's name or
anything from a recording, because notifications appear on the lock screen. The first
time you start a run, macOS asks whether Scriba may notify you; if you say no, nothing
else changes. When recordings were put next and one of them fails, the failure is held
until the last of them has run, and told once with the others, as Transcribe all does.

If the microphone changes while you record (AirPods connect, a USB microphone is
plugged in or out), Scriba carries on into the same file with the new input; a fraction
of a second around the switch is lost. If it cannot restart the input, it stops the
recording, puts what it has in the queue as it would after Stop, and says why at the
top of the window. Live text follows the new microphone if it can, and stops alone if
it cannot.

## Speed, and why it is what it is

`ctranslate2`, the engine under faster-whisper, has no Metal backend in any released
version, so on Apple Silicon it runs entirely on the CPU. That is the single reason
a seven-minute recording used to cost seven minutes on a machine whose GPU sat idle.
Measured on an M4 with 16 GB, the same 6:45 of Spanish audio through both versions,
same settings, warm model cache:

| Stage | whisperX 3.3.1 | whisperX 3.8.6 |
|---|---|---|
| Load `large-v3` int8 | 9.1s | 34.7s |
| Transcription, 8 threads | 601.8s | 443.1s |
| Load the aligner | 0.5s | 2.8s |
| Forced alignment, word level | 13.3s | 14.1s |
| Diarization, pyannote 3.1 on CPU | 372.3s | 211.0s |
| Diarization, community-1 on CPU | | 225.9s |
| Diarization, community-1 on Metal | | **35.7s** |
| **Total, best of each** | **16m 37s** | **8m 50s** |

### Metal, if you build for it

There is an open pull request adding a Metal backend to ctranslate2
([OpenNMT/CTranslate2#2077](https://github.com/OpenNMT/CTranslate2/pull/2077)). It is
not in any wheel: it has to be built from that branch and installed on purpose. With
it in place, the same 6:45 recording:

| | Transcription | Words |
|---|---|---|
| CPU, int8 | 246s | 706 |
| Metal, float16 | 91s | 700 |

Two and a half times faster for the same words, and the difference is only where
the matrices are multiplied. Both rows were taken in one sitting on 31 August 2026
through scriba's own pipeline, so they are comparable to each other; the pair that
stood here before was measured months apart on a build that has changed since, and
comparing across those is how you get a number nobody can reproduce.

`asr_device` decides: `auto` uses Metal when the installed ctranslate2 has it and
the CPU otherwise, so a stock install behaves exactly as it did. `mps` insists and fails with an explanation rather than quietly running slowly.

Build it with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWITH_MPS=ON -DWITH_ACCELERATE=ON \
      -DWITH_MKL=OFF -DWITH_RUY=ON -DOPENMP_RUNTIME=NONE -DCMAKE_POLICY_VERSION_MINIMUM=3.5
```

`WITH_RUY` is not optional: without it the build has no int8 on the CPU at all, so
falling back to the CPU stops working. `CMAKE_POLICY_VERSION_MINIMUM` is needed on
CMake 4 because a vendored dependency declares an ancient minimum. Both were missing
from the branch's own instructions when this was written; they are in its
documentation now, after they cost an hour here and got reported.

Rebuilt on 14 October 2026, and that is the build installed here now: the pull
request as it stood on 27 September (c9600d57), merged with master as it stood on
12 October (51526bf), plus three fixes found while doing it (branch
`apple-mps-backend-update` on nerln/CTranslate2, head 35d1ecd5).

- A test from master caught Metal's TopK returning 0 in the empty slots for k=16,
  which a beam of 8 would read as the best score.
- Several workers on one model crashed inside Apple's matrix multiplication, which
  shared one kernel object between threads.
- An error reported by the GPU could end the process from an idle worker, or from a
  model being unloaded, instead of reaching the next call as an error.

scriba runs one worker and, by default, a beam of 5, so neither of the first two was on
its path. The third matters only when the GPU reports an error, which it has not done
here. The words and the speed are the same as before. The September protocol calls
ctranslate2 directly, beam 5, on three 30-second windows of a public FLEURS Spanish
sample. Six runs of each build, interleaved in one session against the 27 September
build, give 157 words every time and medians of 21.4s and 21.0s, inside the spread of
either.

The build now warns on every model load that the backend is experimental. scriba
quiets ctranslate2's warnings while its own model loads on Metal, because the window
showed that line under the progress bar for the whole transcription.

Re-measured again at head dcf9306f on 7 September 2026, which was the build
installed here until 14 October. That head makes a request for `int8` on Metal resolve to
`float16` instead of running a native INT8 path, and for scriba it changes
nothing: Metal came out at 91s again, CPU at 246s against 227s the week before,
which is the machine and not the library. scriba never asked for int8 on the GPU,
so the path that got faster is not one it was on.

Rebuilt and re-measured at head 74b510a0 on 31 August 2026. The documented line
configures and builds as written, and the C++ suite from that same MPS-enabled
build gives 390 passed and 1 skipped, against 370 passed and 2 skipped at 2f6a066.
That head resolves a generic MPS int8 to int8_float16 and expands the weights to
FP16 once, which was meant to fix int8 being the slowest path on the GPU. On this
machine the resolution happens and the speed does not follow: three 30-second
windows give MPS int8 39.9s against MPS float16 16.2s and CPU int8 35.1s, and
switching the weight cache off changes that by 1.5%. So scriba keeps mapping int8
to float16 itself, now for a measured reason rather than an old one.

Live text while recording does not go through any of this. It runs on Apple's own
model on the Neural Engine, which is neither the CPU nor the GPU, and costs about
ten megabytes.

Transcription and diarization are where the time goes, and no setting changes that: it
is the library. The job cache exists for this reason, so you pay it once. Alignment,
the stage that sounds expensive, costs 14 seconds.

The newer whisperX is 26% faster on the transcription, which is not the reason to move
to it. In 3.3.1 every token containing digits came out of the aligner with no timestamp
at all, 5 out of 5 on the test file, three of them amounts of money. In 3.8.6 it is 0
out of 7, and two words the old version dropped from the transcript altogether now
appear. On a recording where the numbers are the content, those are the words you least
want to lose.

For contrast, Apple's own on-device recogniser via
[yap](https://github.com/finnvoor/yap) does the same file in 24 seconds, about 17 times
realtime. It picks up the opening small talk that Whisper's VAD skips over. It garbles
surnames more, splitting one of them into two words that are not words. And it has no
diarization at all, so on its own it does not cover this job.

What helps, in order:

| Move | Effect | Risk |
|---|---|---|
| `--min-speakers` / `--max-speakers` when you know them | No speed-up. It is the one parameter that moves correctness the most | none |
| `large-v3-turbo` in place of `large-v3` | About twice as fast | loses a little on noisy audio |
| Metal for the diarization, which is now the default | 226s to 36s, so 6.3x | Verified identical to the CPU output before the default changed: same 165 turns, same labels, every boundary matching to the millisecond. [#1337](https://github.com/pyannote/pyannote-audio/issues/1337) reports wrong timestamps under Metal and was closed with no fix, so it is worth knowing the shape of that failure. If one speaker ever owns the whole recording, set `diarize_device=cpu` and compare |

Things that are not worth it, checked rather than assumed: `mlx-whisper` implements no
beam search, so moving to it trades quality away against `beam_size=5`.
`lightning-whisper-mlx` and `stable-ts` are both stalled. Apple's `SpeechAnalyzer` API
is fast and exposes no diarization whatsoever, so it solves half the problem at best.

## Layout

```
scriba/
  lang.py       language detection over several samples
  asr.py        whisper, tuned for conversational speech
  diarize.py    pyannote called directly, on either major version
  voices.py     the voice-print registry
  naming.py     the "who is who" briefing: textual cues plus registry matches
  export.py     output formats, the readable document first
  pipeline.py   the orchestrator, cached per stage
  stage.py      runs a model stage in a child process and hands back its result
  awake.py      keeps the Mac from idle sleep while a run works
  conversations.py  one long recording split into its conversations, with summaries
  watch.py      watched folder
macapp/         the SwiftUI app
tools/          stylecheck.py, make-demo.py, and the writing rules this repo is held to
```

## Notes

Comments in the code say why, not what. Where there is a magic number or a choice that
reads oddly, the reason sits next to it, usually a bug already paid for.

The tests live in `tests/` and run in about eight seconds, because none of them
load a model:

```bash
pytest
```

They were written by a set of agents pointed at one module each and told to report
defects rather than fix them, which turned up around twenty. The interesting ones
are in the commit log: a registry save that could hand one person's voice print to
another, a cache that kept the previous recording's speaker names, a language vote
that reported a file nobody could identify as its most confident result.

`tools/stylecheck.py` holds the writing rules for this repo, prose and comments alike.
Run both checks before a commit:

```bash
python tools/stylecheck.py --code README.md scriba/*.py macapp/Sources/Scriba/*.swift
python tools/check-output-style.py                   # what the source produces
```

`--code` restricts the check to comments, docstrings and strings shown to a person.
Without it the banned word list is applied to identifiers as well, and every sort
in the codebase comes back as a finding for the argument it sorts by.

The second one exists because the first is not enough. Every document scriba writes is
assembled from string fragments, so each line can pass on its own and the finished file
still be wrong. `check-output-style.py` renders all six output formats plus the
briefing from stand-in data and checks those instead.

Written in one sitting, paired with Claude, starting from a real problem: a Spanish
recording transcribed into Italian that looked flawless and was not. The code was read,
run against real audio, and corrected. The numbers here come from measurements. It is
still an afternoon's worth of code, so take it for what it is.

## Licence

MIT.
