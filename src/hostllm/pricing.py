"""What a served token costs: the GPU's hourly rate over the time the engine spends on it, for
hostllm one request at a time and for vLLM at its measured rate. Standard library only, so a
consumer prices a call without torch."""

from __future__ import annotations

from dataclasses import dataclass

from hostllm import config


@dataclass(frozen=True)
class TokenPrice:
    """USD per million tokens."""

    input: float
    output: float


def token_price(gpu_usd_per_hour: float, prefill_ms_per_token: float, step_ms: float) -> TokenPrice:
    """The price of a prompt token prefilled in `prefill_ms_per_token` and of an answer token
    decoded in `step_ms`, on a GPU billed at `gpu_usd_per_hour`."""
    usd_per_ms = gpu_usd_per_hour / 3_600_000
    return TokenPrice(usd_per_ms * prefill_ms_per_token * 1e6, usd_per_ms * step_ms * 1e6)


SERVED = token_price(
    config.DEPLOY_GPU_USD_PER_HOUR,
    config.PRICE_PREFILL_MS / config.PRICE_PREFILL_TOKENS,
    config.PRICE_STEP_MS,
)
VLLM_SERVED = token_price(
    config.DEPLOY_GPU_USD_PER_HOUR,
    config.PRICE_PREFILL_MS / config.PRICE_PREFILL_TOKENS,  # vLLM's own prefill is faster
    1000 / config.PRICE_VLLM_TOKENS_PER_SECOND,
)
