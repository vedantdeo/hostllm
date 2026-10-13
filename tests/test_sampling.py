"""Specification for sampling.py: the next token, greedy at temperature zero, else a draw from the
nucleus."""

from __future__ import annotations

import pytest
import torch

from hostllm.sampling import choose, nucleus

LOGITS = torch.tensor([0.0, 3.0, 1.0, 2.9])


def test_temperature_zero_is_argmax() -> None:
    assert choose(LOGITS, temperature=0.0, top_p=None) == 1


@pytest.mark.parametrize(
    ("top_p", "kept"),
    [
        pytest.param(1.0, [0, 1, 2, 3], id="top_p of one keeps every token"),
        pytest.param(0.5, [1, 3], id="the token that crosses p is kept"),
        pytest.param(0.01, [1], id="a tiny top_p keeps only the likeliest"),
    ],
)
def test_nucleus_keeps_the_smallest_set_reaching_p(top_p: float, kept: list[int]) -> None:
    filtered = nucleus(LOGITS, top_p)

    assert torch.isfinite(filtered).nonzero().flatten().tolist() == kept, filtered


def test_a_draw_comes_from_the_nucleus_and_repeats_with_its_generator() -> None:
    draws = [choose(LOGITS, 1.0, 0.5, torch.Generator().manual_seed(seed)) for seed in range(40)]
    again = [choose(LOGITS, 1.0, 0.5, torch.Generator().manual_seed(seed)) for seed in range(40)]

    assert set(draws) == {1, 3} and draws == again


def test_a_high_temperature_reaches_tokens_a_low_one_does_not() -> None:
    cold = {choose(LOGITS, 0.01, None, torch.Generator().manual_seed(s)) for s in range(60)}
    hot = {choose(LOGITS, 5.0, None, torch.Generator().manual_seed(s)) for s in range(60)}

    assert cold == {1} and hot == {0, 1, 2, 3}
