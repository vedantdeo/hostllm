"""Specification for deploy/spec.py: what to host, read from JSON and refused when malformed, and
the command each server runs — ours and the vLLM reference — built from it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hostllm import config
from hostllm.deploy.spec import SPEC_ENV, SPEC_JSON_ENV, Spec, load_spec, spec_from_env

ADAPTERS = {"headline-reader": "vedantdeo/hl", "report-analyst": "vedantdeo/ra"}
RAW = {"base_model": "Qwen/Qwen3-4B", "adapters": ADAPTERS, "max_lora_rank": 16}
SPEC = Spec("Qwen/Qwen3-4B", ADAPTERS, 16)


def test_a_spec_is_read_from_json() -> None:
    assert load_spec(json.dumps(RAW)) == SPEC


@pytest.mark.parametrize(
    ("raw", "refusal"),
    [
        pytest.param([], "object", id="not an object"),
        pytest.param({**RAW, "base_model": ""}, "base_model", id="no base model"),
        pytest.param({**RAW, "adapters": {}}, "adapters", id="no adapters"),
        pytest.param({**RAW, "adapters": ["a"]}, "adapters", id="adapters not a mapping"),
        pytest.param({**RAW, "adapters": {"a": ""}}, "adapter 'a'", id="an adapter with no source"),
        pytest.param({**RAW, "adapters": {"base": "x"}}, "base", id="the base's own name"),
        pytest.param(
            {**RAW, "adapters": {"Bad Name": "x"}}, "name", id="a name a URL cannot carry"
        ),
        pytest.param({**RAW, "max_lora_rank": 0}, "max_lora_rank", id="a rank of zero"),
        pytest.param({**RAW, "max_lora_rank": "16"}, "max_lora_rank", id="a rank as text"),
        pytest.param({**RAW, "max_lora_rank": True}, "max_lora_rank", id="a rank of true"),
        pytest.param({**RAW, "extra": 1}, "extra", id="an unknown field"),
    ],
)
def test_a_bad_spec_is_refused_by_name(raw: object, refusal: str) -> None:
    with pytest.raises(ValueError, match=refusal):
        load_spec(json.dumps(raw))


def test_our_command_names_the_base_and_every_adapter_in_order() -> None:
    assert SPEC.ours_command(8000) == [
        "python", "-m", "hostllm.server",
        "--base-model", "Qwen/Qwen3-4B",
        "--port", "8000",
        "--adapter", "headline-reader=vedantdeo/hl",
        "--adapter", "report-analyst=vedantdeo/ra",
    ]  # fmt: skip


def test_the_vllm_command_serves_the_same_models_with_lora_and_tools_on() -> None:
    command = SPEC.vllm_command(8000)

    assert command[:3] == ["vllm", "serve", "Qwen/Qwen3-4B"]
    flags = " ".join(command[3:])
    assert "--lora-modules headline-reader=vedantdeo/hl report-analyst=vedantdeo/ra" in flags
    assert "--enable-lora" in flags and "--max-loras 2" in flags and "--max-lora-rank 16" in flags
    assert "--enable-auto-tool-choice --tool-call-parser hermes" in flags
    assert f"--max-model-len {config.DEPLOY_MAX_MODEL_LEN}" in flags and "--port 8000" in flags
    assert "--served-model-name base" in flags, "the base answers to the name ours gives it"
    assert "api-key" not in flags, "the key travels in the environment, never on a command line"


def test_a_deployed_container_reads_the_spec_its_image_carries(tmp_path: Path) -> None:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(RAW))

    assert spec_from_env({SPEC_ENV: str(path)}) == SPEC
    assert spec_from_env({SPEC_JSON_ENV: json.dumps(RAW), SPEC_ENV: "/nowhere"}) == SPEC


def test_no_spec_in_the_environment_is_refused() -> None:
    with pytest.raises(ValueError, match=SPEC_ENV):
        spec_from_env({})
