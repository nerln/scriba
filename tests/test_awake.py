"""Keeping the Mac awake while a run works (scriba/awake.py).

The decision and the command are checked with every input made up: the platform, the
environment and the spawner are all handed in, so nothing here starts a process and
the tests mean the same on a Mac and on a Linux box. The suite as a whole runs with
the switch off (conftest), so a Job.run in some other file starts nothing either.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from scriba import awake

PRESENT = lambda path: True       # noqa: E731
ABSENT = lambda path: False       # noqa: E731


class Process:
    """What `Popen` gives back, as far as hold() looks: is it still running."""

    def __init__(self):
        self.code = None

    def poll(self):
        return self.code


class Spawner:
    """Stands in for `subprocess.Popen`: records what was asked, starts nothing."""

    def __init__(self, *, fail: Exception | None = None):
        self.calls: list[tuple[list[str], dict]] = []
        self.processes: list[Process] = []
        self.fail = fail

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.fail:
            raise self.fail
        self.processes.append(Process())
        return self.processes[-1]


@pytest.fixture(autouse=True)
def nothing_held(monkeypatch):
    """Each test starts as a process that has not held anything yet."""
    monkeypatch.setattr(awake, "_held", None)


@pytest.fixture
def mac(monkeypatch):
    """The decision on a Mac that has the tool and has not been told to refrain."""
    monkeypatch.delenv(awake.SWITCH, raising=False)
    return dict(platform="darwin", runnable=PRESENT)


# --------------------------------------------------------------------------- #
# the decision
# --------------------------------------------------------------------------- #

def test_a_mac_with_the_tool_and_no_switch_holds():
    assert awake.wanted(platform="darwin", environ={}, runnable=PRESENT)


@pytest.mark.parametrize("platform", ["linux", "win32", "freebsd14", "cygwin", ""])
def test_only_macos_holds(platform):
    # Even with the tool "present": the path is a macOS path and means nothing there.
    assert not awake.wanted(platform=platform, environ={}, runnable=PRESENT)


@pytest.mark.parametrize("value", ["0", "off", "OFF", "no", "false", "False", " 0 "])
def test_the_switch_turns_it_off(value):
    assert not awake.wanted(platform="darwin", environ={awake.SWITCH: value},
                            runnable=PRESENT)


@pytest.mark.parametrize("value", ["", "1", "on", "yes", "true", "anything"])
def test_any_other_value_of_the_switch_leaves_it_on(value):
    assert awake.wanted(platform="darwin", environ={awake.SWITCH: value},
                        runnable=PRESENT)


def test_a_mac_without_the_tool_does_not_hold():
    asked: list[str] = []

    def runnable(path):
        asked.append(path)
        return False

    assert not awake.wanted(platform="darwin", environ={}, runnable=runnable)
    assert asked == ["/usr/bin/caffeinate"]


def test_the_tool_is_not_looked_for_where_it_cannot_be_used():
    asked: list[str] = []
    awake.wanted(platform="linux", environ={}, runnable=lambda p: asked.append(p) or True)
    awake.wanted(platform="darwin", environ={awake.SWITCH: "0"},
                 runnable=lambda p: asked.append(p) or True)
    assert asked == []


def test_the_defaults_read_the_real_environment(monkeypatch):
    # The suite sets the switch to 0 for every test, and that is what the default
    # sees. This is the line between "the tests start nothing" and a promise.
    assert os.environ[awake.SWITCH] == "0"
    assert not awake.wanted()
    monkeypatch.delenv(awake.SWITCH)
    assert awake.wanted(platform="darwin", runnable=PRESENT)
    assert not awake.wanted(platform="linux", runnable=PRESENT)


@pytest.mark.skipif(sys.platform != "darwin", reason="caffeinate is part of macOS")
def test_the_tool_is_where_the_module_says_it_is():
    # The one check against the real machine, and it only looks.
    assert awake.CAFFEINATE == "/usr/bin/caffeinate"
    assert os.access(awake.CAFFEINATE, os.X_OK)


# --------------------------------------------------------------------------- #
# the command
# --------------------------------------------------------------------------- #

def test_the_command_refuses_idle_sleep_for_as_long_as_the_process_lives():
    assert awake.command(4242) == ["/usr/bin/caffeinate", "-i", "-w", "4242"]


def test_the_command_is_for_this_process_unless_told_otherwise():
    assert awake.command() == ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())]


def test_the_command_holds_idle_sleep_and_nothing_else():
    # -d would keep the display on, -s would hold against the lid and the Sleep menu,
    # -u would wake the screen. None of them is wanted for a background run.
    flags = [a for a in awake.command(1) if a.startswith("-")]
    assert flags == ["-i", "-w"]


# --------------------------------------------------------------------------- #
# holding
# --------------------------------------------------------------------------- #

def test_hold_starts_caffeinate_tied_to_this_process(mac):
    spawn = Spawner()

    assert awake.hold(spawn=spawn, **mac) is True

    assert len(spawn.calls) == 1
    argv, kwargs = spawn.calls[0]
    assert argv == ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())]
    # Nothing it might print reaches the terminal the transcription is printing to.
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["stdout"] == subprocess.DEVNULL
    assert kwargs["stderr"] == subprocess.DEVNULL


def test_hold_starts_only_one_for_a_whole_process(mac):
    spawn = Spawner()

    # A batch of recordings, a watcher that meets a hundred: each run asks again.
    results = [awake.hold(spawn=spawn, **mac) for _ in range(5)]

    assert results == [True] * 5
    assert len(spawn.calls) == 1


def test_hold_starts_another_when_the_first_one_is_gone(mac):
    spawn = Spawner()
    awake.hold(spawn=spawn, **mac)
    spawn.processes[0].code = -15            # somebody killed it

    assert awake.hold(spawn=spawn, **mac) is True

    assert len(spawn.calls) == 2
    # And that one is the one that now counts.
    awake.hold(spawn=spawn, **mac)
    assert len(spawn.calls) == 2


def test_hold_starts_nothing_off_a_mac():
    spawn = Spawner()

    assert awake.hold(spawn=spawn, platform="linux", runnable=PRESENT) is False

    assert spawn.calls == []


def test_hold_starts_nothing_when_switched_off(monkeypatch):
    monkeypatch.setenv(awake.SWITCH, "0")
    spawn = Spawner()

    assert awake.hold(spawn=spawn, platform="darwin", runnable=PRESENT) is False

    assert spawn.calls == []


def test_hold_starts_nothing_when_the_tool_is_missing(mac):
    spawn = Spawner()

    assert awake.hold(spawn=spawn, platform="darwin", runnable=ABSENT) is False

    assert spawn.calls == []


@pytest.mark.parametrize("failure", [FileNotFoundError("caffeinate"),
                                     PermissionError("caffeinate"), OSError("no fork")])
def test_hold_never_fails_the_run(mac, failure):
    # A transcription about to start must not die because a helper would not start.
    spawn = Spawner(fail=failure)

    assert awake.hold(spawn=spawn, **mac) is False

    # And it is tried again at the next run, not remembered as held.
    assert awake.hold(spawn=spawn, **mac) is False
    assert len(spawn.calls) == 2


def test_hold_under_the_test_switch_spawns_nothing_real(monkeypatch):
    """What every Job.run in this suite goes through: the default arguments."""
    def forbidden(*args, **kwargs):
        raise AssertionError("a test started a real process through awake.hold")

    monkeypatch.setattr(subprocess, "Popen", forbidden)

    assert awake.hold() is False
    assert awake._held is None
