"""Specification for scheduler.py: v1 serves requests one at a time, in the order they arrive."""

from __future__ import annotations

import threading
import time

import pytest

from hostllm.protocol import ChatRequest, Completion
from hostllm.scheduler import Sequential


def request(n: int) -> ChatRequest:
    return ChatRequest(f"m{n}", [{"role": "user", "content": str(n)}], [], 4, 0.0, None)


class SlowEngine:
    """Takes a moment per request and records how many ran at once, and in what order."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = 0
        self.most = 0
        self.order: list[str] = []

    def __call__(self, req: ChatRequest) -> Completion:
        with self.lock:
            self.running += 1
            self.most = max(self.most, self.running)
            self.order.append(req.model)
        time.sleep(0.01)
        with self.lock:
            self.running -= 1
        return Completion(req.model, [], 1, 1, "stop")


def test_requests_run_one_at_a_time_in_arrival_order() -> None:
    engine = SlowEngine()
    scheduler = Sequential(engine)

    futures = [scheduler.submit(request(n)) for n in range(6)]
    answers = [f.result(timeout=5).content for f in futures]
    scheduler.close()

    assert engine.most == 1, "two requests ran at once"
    assert engine.order == answers == [f"m{n}" for n in range(6)]


def test_an_engine_error_reaches_only_its_own_request() -> None:
    def engine(req: ChatRequest) -> Completion:
        if req.model == "m1":
            raise RuntimeError("CUDA out of memory")
        return Completion(req.model, [], 1, 1, "stop")

    scheduler = Sequential(engine)
    first, broken, last = (scheduler.submit(request(n)) for n in range(3))

    with pytest.raises(RuntimeError, match="out of memory"):
        broken.result(timeout=5)
    assert first.result(timeout=5).content == "m0" and last.result(timeout=5).content == "m2"
    scheduler.close()
