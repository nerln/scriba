"""What the command line prints for an application that is reading it.

The macOS app sets SCRIBA_MARKERS and reads the output through a pipe. It takes the
reason a recording failed from one line, `[scriba:error] <reason>`, and from the last
lines of the log when that line is missing. Two things went wrong on a real failure
(an out-of-memory error from torch): the marker carried only the first line of the
message, and rich, which has no terminal width on a pipe, cut the message and the
"did not go through" line under it at 80 columns, so the app's fallback found the
last fragment of a sentence and showed that as the reason.

Every name and path here is invented.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from scriba import cli
from scriba.pipeline import ERR_NO_TOKEN

ROOT = Path(__file__).resolve().parent.parent


def test_the_marker_carries_the_whole_message_on_one_line(monkeypatch, capsys):
    monkeypatch.setenv("SCRIBA_MARKERS", "1")

    cli._mark_error(RuntimeError(
        "Could not load the diarization model:\n"
        "  the token was rejected\n\n"
        "and the second attempt failed too.  "))

    assert capsys.readouterr().out == (
        "[scriba:error] Could not load the diarization model: the token was rejected "
        "and the second attempt failed too.\n")


def test_the_token_marker_inside_the_message_survives_the_joining(monkeypatch, capsys):
    monkeypatch.setenv("SCRIBA_MARKERS", "1")

    cli._mark_error(RuntimeError(
        f"{ERR_NO_TOKEN} The Hugging Face token for pyannote is missing.\n"
        "  scriba token hf_xxxxxxxx   # stores it in the Keychain"))

    out = capsys.readouterr().out
    assert out.count("\n") == 1
    assert out.startswith(f"[scriba:error] {ERR_NO_TOKEN} The Hugging Face token")
    assert out.endswith("scriba token hf_xxxxxxxx   # stores it in the Keychain\n")


def test_an_exception_with_no_text_is_named_by_its_type(monkeypatch, capsys):
    monkeypatch.setenv("SCRIBA_MARKERS", "1")

    cli._mark_error(RuntimeError())

    assert capsys.readouterr().out == "[scriba:error] RuntimeError\n"


def test_nothing_is_printed_in_a_terminal(monkeypatch, capsys):
    monkeypatch.delenv("SCRIBA_MARKERS", raising=False)

    cli._mark_error(RuntimeError("anything"))

    assert capsys.readouterr().out == ""


def run_a_missing_file(tmp_path, *, markers: bool, command: tuple[str, ...] = ("run",),
                       after: tuple[str, ...] = ()) -> list[str]:
    """`scriba run` on a file that is not there, through a pipe 40 columns wide.

    A real process, because the console is built when the module is imported and
    the width is whatever a pipe reports. The failure is the engine's own message
    for a recording that was moved, so no model and no audio are involved.
    """
    env = dict(os.environ, COLUMNS="40", SCRIBA_HOME=str(tmp_path / "home"),
               PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])))
    env.pop("SCRIBA_MARKERS", None)
    if markers:
        env["SCRIBA_MARKERS"] = "1"
    missing = tmp_path / "gone" / "Lake trip.m4a"
    done = subprocess.run([sys.executable, "-m", "scriba.cli", *command, str(missing), *after],
                          env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 1, done.stderr
    return done.stdout.splitlines()


def test_rich_does_not_cut_the_failure_into_fragments_for_the_app(tmp_path):
    lines = run_a_missing_file(tmp_path, markers=True)

    reason = (f"Lake trip.m4a is no longer at {tmp_path / 'gone'}. "
              "It was moved, renamed or its disk is not connected.")
    # The message, the marker and the line in the tally, each whole.
    assert f"  {reason}" in lines
    assert f"[scriba:error] {reason}" in lines
    assert f"  Lake trip.m4a: {reason}" in lines


def test_a_terminal_still_gets_wrapped_text(tmp_path):
    """The other half: the change is for pipes read by the app, not for people."""
    lines = run_a_missing_file(tmp_path, markers=False)

    assert not any(line.startswith("[scriba:error]") for line in lines)
    assert max(len(line) for line in lines) <= 40


def test_the_tally_under_the_marker_is_one_line_too(monkeypatch, tmp_path, capsys):
    """The app falls back to the tail of the log when it finds no marker.

    The tally is that tail. A message with line breaks of its own, such as the
    missing-token one, left "and accept the terms at hf.co/..." as the last line,
    which is not a reason anybody can read.
    """
    from typer.testing import CliRunner

    monkeypatch.setenv("SCRIBA_MARKERS", "1")

    def fails(*args, **kwargs):
        raise RuntimeError("Could not load the model:\n  the token was rejected\n\n"
                           "and again.")

    monkeypatch.setattr(cli, "_run_one", fails)

    outcome = CliRunner().invoke(cli.app, ["run", str(tmp_path / "Lake trip.m4a")])

    lines = outcome.output.splitlines()
    # Short enough for the 80 columns rich assumes on a pipe here, so that it is the
    # line breaks of the message that are tested and not the wrapping.
    one_line = "Could not load the model: the token was rejected and again."
    assert outcome.exit_code == 1
    assert f"[scriba:error] {one_line}" in lines
    assert lines[-1] == f"  Lake trip.m4a: {one_line}"
    assert "1 of 1 did not go through:" in lines[-2]


def test_a_failure_with_no_message_is_named_by_its_type_in_the_tally(
        monkeypatch, tmp_path):
    from typer.testing import CliRunner

    monkeypatch.setattr(cli, "_run_one", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))

    outcome = CliRunner().invoke(cli.app, ["run", str(tmp_path / "Lake trip.m4a")])

    assert outcome.output.splitlines()[-1] == "  Lake trip.m4a: RuntimeError"


def test_name_fails_with_a_marker_through_the_real_entry_point(tmp_path):
    """`scriba name` on a recording that is gone: the marker, the sentence, exit 1."""
    lines = run_a_missing_file(tmp_path, markers=True, command=("name",),
                               after=("SPEAKER_00=Ada",))

    reason = (f"Lake trip.m4a is no longer at {tmp_path / 'gone'}. "
              "It was moved, renamed or its disk is not connected.")
    assert f"[scriba:error] {reason}" in lines
    assert any(line.startswith("File not found:") for line in lines)


def test_name_in_a_terminal_prints_no_marker(tmp_path):
    lines = run_a_missing_file(tmp_path, markers=False, command=("name",),
                               after=("SPEAKER_00=Ada",))

    assert not any(line.startswith("[scriba:error]") for line in lines)
