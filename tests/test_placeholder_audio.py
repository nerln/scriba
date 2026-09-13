"""Tests for the recording that is only a name.

A sync service that keeps this Mac's copy small leaves the file in the folder
and takes its bytes away. Copy it and you get a header with nothing behind it.
ffprobe then says "moov atom not found", which is true, and which sent one
person looking for a corrupt file when the file was simply not here yet.

None of these checks know which application owns the recording.
"""

from __future__ import annotations

import os

import pytest

from scriba import audio


def test_a_real_recording_is_not_a_placeholder(tmp_path):
    f = tmp_path / "talk.m4a"
    f.write_bytes(b"\0" * (200 * 1024))
    assert audio.placeholder(f) is None


def test_a_header_only_copy_is_named_for_what_it_is(tmp_path):
    f = tmp_path / "talk.m4a"
    f.write_bytes(b"\0" * (9 * 1024))
    why = audio.placeholder(f)
    assert why is not None
    assert "9 KB" in why
    assert "download" in why.lower()
    assert "moov" not in why


def test_an_empty_file_is_not_called_a_placeholder(tmp_path):
    # Zero bytes is an empty file. Calling it a download problem would send
    # somebody to iCloud to fetch a recording that never existed.
    f = tmp_path / "talk.m4a"
    f.write_bytes(b"")
    assert audio.placeholder(f) is None


def test_a_size_with_no_blocks_is_a_placeholder(tmp_path):
    f = tmp_path / "talk.m4a"
    fd = os.open(f, os.O_WRONLY | os.O_CREAT, 0o600)
    os.ftruncate(fd, 10_000_000)
    os.close(fd)
    st = os.stat(f)
    if getattr(st, "st_blocks", 1) != 0:
        pytest.skip("this filesystem allocated blocks for a hole")
    why = audio.placeholder(f)
    assert why is not None and "cloud" in why.lower()


def test_a_missing_file_is_nobodys_placeholder(tmp_path):
    assert audio.placeholder(tmp_path / "gone.m4a") is None


def test_probe_says_the_human_thing_before_ffprobes(tmp_path):
    """The failure this file exists for.

    The header-only copy is what ffprobe sees as a missing moov atom. Both
    statements are true; only one of them tells the reader what to do.
    """
    if audio._ffprobe() is None:
        pytest.skip("no ffprobe here")
    f = tmp_path / "talk.m4a"
    f.write_bytes(b"\0\0\0\x18ftypM4A " + b"\0" * 6000)
    with pytest.raises(RuntimeError) as caught:
        audio.probe(f)
    message = str(caught.value)
    assert "header" in message
    assert "moov" not in message


def test_probe_still_reports_ffprobes_reason_for_other_garbage(tmp_path):
    if audio._ffprobe() is None:
        pytest.skip("no ffprobe here")
    f = tmp_path / "talk.m4a"
    f.write_bytes(os.urandom(300 * 1024))       # big enough not to be a header
    with pytest.raises(RuntimeError) as caught:
        audio.probe(f)
    assert "cannot read talk.m4a as audio" in str(caught.value)
