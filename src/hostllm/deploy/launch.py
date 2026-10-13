"""Start a server process and wait for its port, failing at once if it dies first."""

from __future__ import annotations

import socket
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence


def _listening(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def start(
    command: Sequence[str],
    port: int,
    timeout: float,
    poll: float = 1.0,
    env: Mapping[str, str] | None = None,
    spawned: Callable[[subprocess.Popen[bytes]], None] = lambda _: None,
) -> subprocess.Popen[bytes]:
    """The running server once it accepts connections on `port`; RuntimeError if it exits first, or
    if it is still not listening after `timeout` seconds, when it is killed."""
    process = subprocess.Popen(list(command), env=None if env is None else dict(env))
    spawned(process)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"{command[0]} exited with code {code} before listening on {port}")
        if _listening(port):
            return process
        time.sleep(poll)
    process.kill()
    process.wait()
    raise RuntimeError(f"{command[0]} not listening on {port} after {timeout:.0f}s")
