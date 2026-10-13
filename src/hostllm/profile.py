"""Where a decode step's time goes: one prefill and the steps after it, timed on the bare base, on
one adapter, and with an adapter per row, at each batch size."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from peft import PeftModel
from torch import Tensor

PATHS = ("base", "adapter", "per-row")


@dataclass(frozen=True)
class Timing:
    """One path at one batch size: the prefill, and the mean decode step, in milliseconds."""

    path: str
    batch: int
    prefill_ms: float
    step_ms: float


def _clock(model: PeftModel) -> float:
    if model.device.type == "cuda":
        torch.cuda.synchronize()  # CUDA runs behind the Python that queued it
    return time.perf_counter()


def _run(model: PeftModel, path: str, adapter: str, ids: Tensor, steps: int) -> tuple[float, float]:
    names = {"adapter_names": [adapter] * ids.shape[0]} if path == "per-row" else {}
    model(input_ids=ids, use_cache=True, **names)  # warm-up, untimed
    began = _clock(model)
    out = model(input_ids=ids, use_cache=True, **names)
    prefilled = _clock(model)
    for _ in range(steps):
        fed = out.logits[:, -1].argmax(-1, keepdim=True)
        out = model(input_ids=fed, past_key_values=out.past_key_values, use_cache=True, **names)
    ended = _clock(model)
    return (prefilled - began) * 1000, (ended - prefilled) * 1000 / steps


@torch.no_grad()
def time_steps(
    model: PeftModel, adapter: str, prompt: Sequence[int], batches: Sequence[int], steps: int
) -> list[Timing]:
    """A Timing for each path in PATHS at each batch size, every row decoding `prompt` greedily."""
    timings: list[Timing] = []
    for path in PATHS:
        for batch in batches:
            ids = torch.tensor([list(prompt)] * batch, device=model.device)
            if path == "base":
                with model.disable_adapter():
                    prefill, step = _run(model, path, adapter, ids, steps)
            else:
                model.set_adapter(adapter)
                prefill, step = _run(model, path, adapter, ids, steps)
            timings.append(Timing(path, batch, prefill, step))
    return timings


def table(timings: Sequence[Timing]) -> str:
    """The timings as aligned text, a line each under a header."""
    lines = [f"{'path':<8} {'batch':>5} {'prefill_ms':>10} {'step_ms':>8}"]
    lines += [f"{t.path:<8} {t.batch:>5} {t.prefill_ms:>10.1f} {t.step_ms:>8.1f}" for t in timings]
    return "\n".join(lines)
