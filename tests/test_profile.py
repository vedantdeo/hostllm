"""Specification for profile.py: a prefill and the decode steps after it, timed on each adapter path
at each batch size."""

from __future__ import annotations

import torch
from peft import PeftModel
from peft.tuners.lora import LoraLayer

from hostllm.profile import PATHS, table, time_steps
from tests.conftest import ADAPTERS

PROMPT = [3, 4, 5, 6, 7]
Seen = tuple[bool, bool, tuple[int, ...]]  # adapters off, an adapter per row, the ids' shape


def watch(model: PeftModel) -> list[Seen]:
    """Every forward pass from here on: whether adapters were off, per row, and the ids' shape."""
    lora = next(m for m in model.modules() if isinstance(m, LoraLayer))
    seen: list[Seen] = []

    def record(
        module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> None:
        ids = kwargs["input_ids"]
        assert isinstance(ids, torch.Tensor)
        names = kwargs.get("adapter_names")
        assert names is None or names == [ADAPTERS[0]] * ids.shape[0], names
        seen.append((bool(lora.disable_adapters), names is not None, tuple(ids.shape)))

    model.register_forward_pre_hook(record, with_kwargs=True)
    return seen


def test_each_path_is_timed_at_each_batch_size(adapted: PeftModel) -> None:
    timings = time_steps(adapted, ADAPTERS[0], PROMPT, batches=(1, 3), steps=2)

    assert [(t.path, t.batch) for t in timings] == [(p, b) for p in PATHS for b in (1, 3)]
    assert all(t.prefill_ms > 0 and t.step_ms > 0 for t in timings), timings


def test_each_path_runs_a_warm_up_a_prefill_and_its_steps_on_its_own_weights(
    adapted: PeftModel,
) -> None:
    seen = watch(adapted)

    time_steps(adapted, ADAPTERS[0], PROMPT, batches=(2,), steps=3)

    prefill, step = (2, len(PROMPT)), (2, 1)
    shapes = [prefill, prefill, step, step, step]  # the first prefill is the untimed warm-up
    assert seen == [
        *[(True, False, s) for s in shapes],  # base
        *[(False, False, s) for s in shapes],  # one adapter
        *[(False, True, s) for s in shapes],  # per-row adapters
    ]


def test_the_table_has_a_line_per_timing(adapted: PeftModel) -> None:
    timings = time_steps(adapted, ADAPTERS[0], PROMPT, batches=(1, 2), steps=1)

    lines = table(timings).splitlines()

    assert len(lines) == 1 + len(timings) and lines[0].split() == [
        "path",
        "batch",
        "prefill_ms",
        "step_ms",
    ]
    assert lines[1].split()[:2] == ["base", "1"] and lines[-1].split()[:2] == ["per-row", "2"]
