"""Tests for the sentence that replaces "please log in to Hugging Face".

Every model scriba loads lives in one cache, and on a Mac with a small disk
that cache is a link to an external drive. Unplug the drive and the link points
at nothing. What a person then sees depends on which library reaches the cache
first: faster-whisper complains about a folder nobody asked for, pyannote asks
for a login. Neither is the problem, and the second sends somebody to check a
token that was fine. This check runs before both and names the disk.
"""

from __future__ import annotations

import os

from scriba import config


def test_a_real_cache_is_no_problem(tmp_path, monkeypatch):
    (tmp_path / "hub").mkdir()
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    assert config.model_cache_problem() is None


def test_a_cache_folder_without_a_hub_yet_is_no_problem(tmp_path, monkeypatch):
    # First run on a fresh machine: the folder exists and the hub inside it will
    # be made by the first download. Nothing to say.
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    assert config.model_cache_problem() is None


def test_a_link_to_an_unmounted_volume_names_the_disk(tmp_path, monkeypatch):
    link = tmp_path / "huggingface"
    os.symlink("/Volumes/NotHere/models/huggingface", link)
    monkeypatch.setenv("HF_HOME", str(link))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    why = config.model_cache_problem()
    assert why is not None
    assert "/Volumes/NotHere" in why
    assert "not connected" in why
    assert "log in" not in why.lower()


def test_a_dangling_link_elsewhere_is_still_reported(tmp_path, monkeypatch):
    link = tmp_path / "huggingface"
    os.symlink(str(tmp_path / "gone"), link)
    monkeypatch.setenv("HF_HOME", str(link))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    why = config.model_cache_problem()
    assert why is not None and "does not exist" in why


def test_an_explicit_hub_cache_that_exists_wins(tmp_path, monkeypatch):
    # HF_HUB_CACHE points somewhere real even though HF_HOME dangles: the
    # libraries will use the hub cache, so there is nothing to report.
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("HF_HOME", "/Volumes/NotHere/x")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    assert config.model_cache_problem() is None


def test_a_rejected_token_is_named_as_rejected_not_as_a_login():
    """pyannote says "Please log in" to a revoked token.

    The person has a token in the Keychain and reads that as scriba having lost
    it. The Hub says 401 for a token it does not recognise and 403 for one it
    knows whose owner never accepted the model's conditions; the sentence has to
    keep those apart, because the fix is different.
    """
    from scriba import diarize
    why = diarize._explain_download_failure(
        Exception("401 Client Error ... Invalid user token. Please log in."), "hf_x")
    assert "rejected" in why and "Keychain" in why
    assert "log in" not in why.lower()
    assert "settings/tokens" in why

    why = diarize._explain_download_failure(Exception("403 Client Error"), "hf_x")
    assert "accepted the conditions" in why

    why = diarize._explain_download_failure(Exception("401"), None)
    assert "No Hugging Face token" in why
