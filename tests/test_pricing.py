"""Specification for pricing.py: a token's price from the GPU's rate and the engine's time, and
that pricing needs none of the engine's libraries."""

from __future__ import annotations

import subprocess
import sys

import pytest

from hostllm.pricing import SERVED, VLLM_SERVED, TokenPrice, token_price


@pytest.mark.parametrize(
    ("usd_per_hour", "prefill_ms", "step_ms", "expected"),
    [
        pytest.param(
            3.6, 1.0, 100.0, TokenPrice(1.0, 100.0), id="$3.60 an hour is $1 a million ms"
        ),
        pytest.param(0.9, 0.4, 40.0, TokenPrice(0.1, 10.0), id="a quarter the rate, 0.4 the time"),
        pytest.param(0.0, 1.0, 100.0, TokenPrice(0.0, 0.0), id="a free GPU"),
    ],
)
def test_token_price(
    usd_per_hour: float, prefill_ms: float, step_ms: float, expected: TokenPrice
) -> None:
    price = token_price(usd_per_hour, prefill_ms, step_ms)

    assert (price.input, price.output) == pytest.approx((expected.input, expected.output)), price


def test_the_served_price_is_the_l4_over_the_one_at_a_time_profile() -> None:
    assert SERVED.input == pytest.approx(0.0657, abs=1e-4), SERVED
    assert SERVED.output == pytest.approx(26.07, abs=1e-2), SERVED


def test_vllm_s_price_is_the_l4_over_its_measured_rate_four_requests_at_a_time() -> None:
    assert VLLM_SERVED.output == pytest.approx(3.70, abs=1e-2), VLLM_SERVED
    assert VLLM_SERVED.input == SERVED.input, "its prefill is unmeasured: ours bounds it"


@pytest.mark.parametrize("module", ["hostllm", "hostllm.config", "hostllm.pricing"])
def test_pricing_imports_without_the_engine_s_libraries(module: str) -> None:
    probe = (
        f"import importlib, sys; importlib.import_module({module!r}); "
        "print(','.join(h for h in ('torch', 'peft', 'transformers') if h in sys.modules))"
    )

    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert done.stdout.strip() == "", f"{module} loaded {done.stdout.strip()}"
