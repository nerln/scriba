"""Apple's Audio Mix, for recordings an iPhone made with Spatial Audio.

An iPhone that records Spatial Audio (Voice Memos saves a `.qta`, the Camera a
`.mov`) writes the recording twice: a stereo track any player can use, and four
channels of Apple's positional audio, codec tag `apac`, with the analysis the phone
made while recording. Voice Memos plays it back with Studio Voice, which pushes the
room and its echo down and keeps the voices near the phone; Photos calls the same
thing Audio Mix. ffmpeg cannot decode the positional track, so without this the
stereo one is all there is to read.

The Cinematic framework renders the mix on a Mac with macOS 26 or later. A small
Swift helper does it, compiled the first time it is needed, as apple.py does with
its speech helper.

Measured on a minute of conversation from a Voice Memos `.qta` recorded on an
iPhone: with Studio at full intensity the quietest tenth of the audio went from
-48 dB to -75 dB, and the speech stayed within 1 dB of where it was. The transcript
did not improve with it: on two stretches of the same recording whisper heard fewer
words through Studio, short replies among them, and aligned them with less
confidence. That is why config.py leaves it off.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path

from .apple import compile_helper
from .config import AUDIO_MIX_STYLES, DATA_DIR

SOURCE = Path(__file__).resolve().parent.parent / "tools" / "apple-audio-mix.swift"
BINARY = DATA_DIR / "bin" / "apple-audio-mix"

# The containers an iPhone writes Spatial Audio into. Asking the helper about any
# other file only costs a second to learn that it has none.
SUFFIXES = {".qta", ".mov"}

# Voice Memos and Photos name the first four. "voices" is Studio with the
# background left out altogether, the stem Apple's framework exposes for editing.
STYLES = AUDIO_MIX_STYLES


def available() -> tuple[bool, str]:
    """Whether the mix can be rendered here, and why not when it cannot."""
    if platform.system() != "Darwin":
        return False, "Apple's Audio Mix exists only on macOS"
    major = int(platform.mac_ver()[0].split(".")[0] or 0)
    if major < 26:
        return False, (f"needs macOS 26 for Audio Mix, this is "
                       f"macOS {platform.mac_ver()[0]}")
    if not BINARY.exists() and shutil.which("swiftc") is None:
        return False, ("needs the Swift compiler to build the helper once: install "
                       "the Xcode command line tools with `xcode-select --install`")
    if not SOURCE.exists() and not BINARY.exists():
        return False, f"the helper source is missing at {SOURCE}"
    return True, ""


def info(path: Path) -> dict:
    """What the file holds: {"spatial", "supported", "default_style", "default_intensity"}.

    A file the helper cannot open, or a Mac that cannot run it, answers as a file
    without Spatial Audio: the stereo track is still there to read.
    """
    ok, why = available()
    if not ok:
        return {"spatial": False, "why": why}
    proc = subprocess.run([str(compile_helper(SOURCE, BINARY, "the Audio Mix helper")),
                           "info", str(path)], capture_output=True, text=True)
    try:
        return json.loads(proc.stdout) if proc.returncode == 0 else {"spatial": False}
    except ValueError:
        return {"spatial": False}


def render(src: Path, dest: Path, style: str, intensity: float) -> Path:
    """Write the mix of `src` to `dest` as 16 kHz stereo WAV, through a scratch name."""
    if style not in STYLES:
        raise ValueError(f"unknown Audio Mix style {style!r} (one of {', '.join(STYLES)})")
    partial = dest.with_suffix(dest.suffix + ".part")
    proc = subprocess.run(
        [str(compile_helper(SOURCE, BINARY, "the Audio Mix helper")), "render",
         str(src), str(partial), style, f"{intensity:.2f}"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not partial.exists():
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Apple's Audio Mix could not render {src.name}: "
            + (proc.stderr.strip() or f"exit code {proc.returncode}"))
    partial.replace(dest)
    return dest
