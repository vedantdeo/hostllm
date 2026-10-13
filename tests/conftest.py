"""a tiny Qwen3 built from a config, and two LoRA adapters that change its output."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

VOCAB = 64
ADAPTERS = ("headline-reader", "report-analyst")


def tiny_qwen() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
    )
    return Qwen3ForCausalLM(config).eval()


def _lora(seed: int) -> LoraConfig:
    torch.manual_seed(seed)
    # random B as well as A, so an adapter moves the output from the first step
    return LoraConfig(
        r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], init_lora_weights=False
    )


@pytest.fixture
def adapted() -> PeftModel:
    """The tiny model with both adapters attached, the first active."""
    model = cast(PeftModel, get_peft_model(tiny_qwen(), _lora(1), adapter_name=ADAPTERS[0]))
    model.add_adapter(ADAPTERS[1], _lora(2))
    return model.eval()
