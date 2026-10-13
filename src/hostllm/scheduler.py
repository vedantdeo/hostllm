"""Schedulers turn concurrent requests into work for the engine. v1, `Sequential`, serves one
request at a time in arrival order."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from hostllm.protocol import ChatRequest, Completion

Complete = Callable[[ChatRequest], Completion]


class Sequential:
    """One worker thread, so the engine never sees two requests at once."""

    def __init__(self, complete: Complete) -> None:
        self._complete = complete
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine")

    def submit(self, request: ChatRequest) -> Future[Completion]:
        return self._pool.submit(self._complete, request)

    def close(self) -> None:
        self._pool.shutdown(wait=True)
