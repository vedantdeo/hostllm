"""Specification for deploy/launch.py: a server watched until its port opens, failing at once if it
dies first."""

from __future__ import annotations

import socket
import subprocess
import sys
import time

import pytest

from hostllm.deploy.launch import start

LISTEN = (
    "import socket, time; s = socket.socket(); s.bind(('127.0.0.1', {port})); s.listen(); "
    "time.sleep(30)"
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_start_returns_the_process_once_its_port_opens() -> None:
    port = free_port()

    process = start([sys.executable, "-c", LISTEN.format(port=port)], port, timeout=20, poll=0.05)

    assert process.poll() is None, "the server is left running"
    process.kill()


def test_a_server_that_dies_before_listening_fails_at_once() -> None:
    with pytest.raises(RuntimeError, match="exited with code 3"):
        start([sys.executable, "-c", "import sys; sys.exit(3)"], free_port(), timeout=20, poll=0.05)


def test_a_server_that_never_listens_is_killed_at_the_timeout() -> None:
    sleeper = [sys.executable, "-c", "import time; time.sleep(30)"]
    started: list[subprocess.Popen[bytes]] = []

    began = time.monotonic()

    with pytest.raises(RuntimeError, match="not listening"):
        start(sleeper, free_port(), timeout=0.5, poll=0.05, spawned=started.append)

    assert time.monotonic() - began < 10, "killed at the timeout, not left to run out"
    assert started[0].poll() is not None, "a stuck server is not left behind"
