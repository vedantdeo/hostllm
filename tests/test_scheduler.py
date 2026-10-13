"""Specification for scheduler.py: v1 serves requests one at a time, in the order they arrive; v2
answers whatever is waiting in one batch, and logs each batch."""

from __future__ import annotations

import logging
import re
import threading
import time

import pytest

from hostllm.protocol import ChatRequest, Completion
from hostllm.scheduler import Batching, Sequential


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


class BatchRecorder:
    """Answers a batch after a pause, recording each batch's models."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def __call__(self, requests: list[ChatRequest]) -> list[Completion]:
        self.batches.append([r.model for r in requests])
        time.sleep(0.05)
        return [Completion(r.model, [], 1, 1, "stop") for r in requests]


def test_waiting_requests_are_answered_together_each_with_its_own_result() -> None:
    engine = BatchRecorder()
    scheduler = Batching(engine, max_batch=8, window=0.02)

    futures = [scheduler.submit(request(n)) for n in range(6)]
    answers = [str(f.result(timeout=5).content) for f in futures]
    scheduler.close()

    assert answers == [f"m{n}" for n in range(6)], "each request gets its own answer"
    assert max(len(b) for b in engine.batches) > 1, "requests waiting together share a batch"
    assert sorted(m for b in engine.batches for m in b) == sorted(answers)


def test_a_batch_never_exceeds_its_cap() -> None:
    engine = BatchRecorder()
    scheduler = Batching(engine, max_batch=3, window=0.05)

    futures = [scheduler.submit(request(n)) for n in range(7)]
    for f in futures:
        f.result(timeout=5)
    scheduler.close()

    assert max(len(b) for b in engine.batches) == 3 and sum(map(len, engine.batches)) == 7


def test_each_answered_batch_is_logged_with_its_size_and_longest_answer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    batches: list[tuple[int, int]] = []

    def engine(requests: list[ChatRequest]) -> list[Completion]:
        answers = [Completion(r.model, [], 1, int(r.model[1:]) + 1, "stop") for r in requests]
        batches.append((len(requests), max(a.completion_tokens for a in answers)))
        return answers

    with caplog.at_level(logging.INFO, logger="hostllm.scheduler"):
        scheduler = Batching(engine, max_batch=3, window=0.05)
        for future in [scheduler.submit(request(n)) for n in range(7)]:
            future.result(timeout=5)
        scheduler.close()

    line = re.compile(r"batch of (\d+), longest answer (\d+) tokens, \d+\.\d\d s$")
    logged = [(int(m[1]), int(m[2])) for r in caplog.records if (m := line.match(r.getMessage()))]
    assert logged == batches and len(batches) >= 3, caplog.text


def test_an_engine_error_fails_its_batch_and_the_next_batch_still_runs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = 0

    def engine(requests: list[ChatRequest]) -> list[Completion]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("CUDA out of memory")
        return [Completion(r.model, [], 1, 1, "stop") for r in requests]

    scheduler = Batching(engine, max_batch=1, window=0.0)
    broken, fine = scheduler.submit(request(0)), scheduler.submit(request(1))

    with pytest.raises(RuntimeError, match="out of memory"):
        broken.result(timeout=5)
    assert fine.result(timeout=5).content == "m1"
    scheduler.close()
    assert "batch of 1 failed: RuntimeError: CUDA out of memory" in caplog.messages


@pytest.mark.parametrize(
    "window", [pytest.param(0.0, id="no window"), pytest.param(0.05, id="a short window")]
)
def test_a_lone_request_is_answered_when_the_window_closes(window: float) -> None:
    engine = BatchRecorder()
    scheduler = Batching(engine, max_batch=8, window=window)

    began = time.monotonic()
    answer = scheduler.submit(request(0)).result(timeout=5)
    waited = time.monotonic() - began
    scheduler.close()

    assert answer.content == "m0" and waited < 2, "the window closes; it does not wait for more"
