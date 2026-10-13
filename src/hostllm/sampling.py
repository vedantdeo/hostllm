"""The next token from a row of logits: greedy at temperature zero, else a draw from the nucleus."""

from __future__ import annotations

import torch
from torch import Tensor


def nucleus(logits: Tensor, top_p: float) -> Tensor:
    """`logits` with every token outside the top-`top_p` nucleus set to -inf."""
    ordered, index = logits.sort(descending=True)
    probs = ordered.softmax(dim=-1)
    # the total before each token decides, so the token that crosses p is kept
    keep = probs.cumsum(dim=-1) - probs < top_p
    return logits.scatter(-1, index, ordered.masked_fill(~keep, float("-inf")))


def choose(
    logits: Tensor,
    temperature: float,
    top_p: float | None,
    generator: torch.Generator | None = None,
) -> int:
    """One token id from a (vocab,) row; `generator` makes a draw repeatable."""
    if temperature == 0:
        return int(logits.argmax())
    scaled = logits.float() / temperature
    if top_p is not None:
        scaled = nucleus(scaled, top_p)
    return int(torch.multinomial(scaled.softmax(dim=-1), num_samples=1, generator=generator))
