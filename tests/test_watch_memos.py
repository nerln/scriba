"""Tests for watching a folder that is not ours.

The Voice Memos library is where the recordings already are, which makes it the
folder worth watching and the one folder scriba must be most careful with. It
belongs to another application, macOS protects it, and the error it returns when
it says no is the same one it returns for a folder that does not exist. Reporting
that as an empty library to somebody with two hundred memos in it is the failure
these tests exist to prevent.
"""

from __future__ import annotations

import os
import stat

import pytest

from scriba import watch as w


def test_a_normal_folder_keeps_its_ledger_inside_itself(tmp_path):
    # So the record travels with the recordings: move them somewhere else and you
    # do not get two hundred transcriptions of things already done.
    ledger = w.ledger_for(tmp_path)
    assert ledger == tmp_path / w.DONE_MARK
    assert ledger.is_dir()


def test_a_folder_we_cannot_write_to_keeps_its_ledger_elsewhere(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "DATA_DIR", tmp_path / "home")
    theirs = tmp_path / "someone-elses"
    theirs.mkdir()
    theirs.chmod(stat.S_IRUSR | stat.S_IXUSR)      # readable, not writable
    try:
        ledger = w.ledger_for(theirs)
        assert (tmp_path / "home") in ledger.parents
        assert ledger.is_dir()
        # And the name says which folder it belongs to, because a directory of
        # hashes is unreadable when you come back to it.
        assert "someone-elses" in ledger.name
    finally:
        theirs.chmod(stat.S_IRWXU)


def test_two_folders_with_the_same_name_do_not_share_a_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "DATA_DIR", tmp_path / "home")
    first, second = tmp_path / "a" / "Recordings", tmp_path / "b" / "Recordings"
    for folder in (first, second):
        folder.mkdir(parents=True)
        folder.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        assert w.ledger_for(first) != w.ledger_for(second)
    finally:
        for folder in (first, second):
            folder.chmod(stat.S_IRWXU)


def test_a_missing_folder_is_reported_as_missing(tmp_path):
    ok, why = w.readable(tmp_path / "nowhere")
    assert ok is False
    assert "no folder" in why


def test_a_protected_folder_says_what_to_do_about_it(tmp_path):
    """The whole point. macOS gives back the same error as for a missing folder.

    Without this, `scriba memos` on a machine that has not granted Full Disk
    Access reports nothing to do, and the person concludes the tool does not see
    their recordings rather than that the system is refusing.
    """
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        ok, why = w.readable(locked)
        assert ok is False
        assert "Full Disk Access" in why
        assert str(locked) in why
    finally:
        locked.chmod(stat.S_IRWXU)


def test_a_readable_folder_is_readable(tmp_path):
    assert w.readable(tmp_path) == (True, "")


def test_watching_a_folder_we_cannot_read_stops_instead_of_looping(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        with pytest.raises(PermissionError, match="Full Disk Access"):
            w.watch(locked, create=False, report=lambda _: None)
    finally:
        locked.chmod(stat.S_IRWXU)


def test_watching_does_not_create_somebody_elses_library(tmp_path):
    """A watcher pointed at a library that is not mounted must not invent it.

    Creating it empty is worse than failing: the watcher then sits happily on a
    folder the recordings are not in, and says nothing for ever.
    """
    missing = tmp_path / "not-here"
    with pytest.raises(PermissionError):
        w.watch(missing, create=False, report=lambda _: None)
    assert not missing.exists()


def test_the_voice_memos_path_is_where_macos_keeps_them():
    assert w.VOICE_MEMOS.name == "Recordings"
    assert "group.com.apple.VoiceMemos.shared" in str(w.VOICE_MEMOS)


def test_the_inbox_is_an_ordinary_folder_under_our_own_data(monkeypatch, tmp_path):
    """Which is the whole point of the mirror.

    Reading Apple's library needs Full Disk Access. Rather than granting that to
    whatever runs scriba, which in practice means a Python interpreter and every
    package ever installed beside it, a separate application holds the permission
    and copies here. This folder needs nothing.
    """
    assert w.INBOX.name == "inbox"
    assert w.readable(tmp_path) == (True, "")


def test_the_inbox_ledger_stays_inside_the_inbox(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "DATA_DIR", tmp_path)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    assert w.ledger_for(inbox) == inbox / w.DONE_MARK
