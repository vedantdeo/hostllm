"""Schedulers turn concurrent requests into work for the engine: v1, `Sequential`, serves one at a
time in arrival order; v2, `Batching`, answers whatever is waiting in one batch."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Protocol

from hostllm.protocol import ChatRequest, Completion

Complete = Callable[[ChatRequest], Completion]


class Scheduler(Protocol):
    """What the server needs of a scheduler."""

    def submit(self, request: ChatRequest) -> Future[Completion]: ...

    def close(self) -> None: ...


class Sequential:
    """One worker thread, so the engine never sees two requests at once."""

    def __init__(self, complete: Complete) -> None:
        self._complete = complete
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine")

    def submit(self, request: ChatRequest) -> Future[Completion]:
        return self._pool.submit(self._complete, request)

    def close(self) -> None:
        self._pool.shutdown(wait=True)


CompleteBatch = Callable[[list[ChatRequest]], list[Completion]]
_Waiting = tuple[ChatRequest, Future[Completion]]


class Batching:
    """v2: one worker takes the first waiting request, gathers whatever else arrives within
    `window` seconds up to `max_batch`, and answers them in one batch; an engine error fails only
    that batch."""

    def __init__(self, complete_batch: CompleteBatch, max_batch: int, window: float) -> None:
        self._complete = complete_batch
        self._max = max_batch
        self._window = window
        self._queue: queue.Queue[_Waiting | None] = queue.Queue()
        self._worker = threading.Thread(target=self._run, name="engine", daemon=True)
        self._worker.start()

    def submit(self, request: ChatRequest) -> Future[Completion]:
        future: Future[Completion] = Future()
        self._queue.put((request, future))
        return future

    def close(self) -> None:
        self._queue.put(None)
        self._worker.join()

    def _gather(self, first: _Waiting) -> tuple[list[_Waiting], bool]:
        batch, deadline = [first], time.monotonic() + self._window
        while len(batch) < self._max:
            remaining = deadline - time.monotonic()
            try:
                item = (
                    self._queue.get(timeout=remaining)
                    if remaining > 0
                    else self._queue.get_nowait()
                )
            except queue.Empty:
                break
            if item is None:
                return batch, True
            batch.append(item)
        return batch, False

    def _run(self) -> None:
        closing = False
        while not closing:
            first = self._queue.get()
            if first is None:
                return
            batch, closing = self._gather(first)
            try:
                answers = self._complete([request for request, _ in batch])
            except Exception as failed:  # the batch's failure, not the worker's
                for _, future in batch:
                    future.set_exception(failed)
                continue
            for (_, future), answer in zip(batch, answers, strict=True):
                future.set_result(answer)
