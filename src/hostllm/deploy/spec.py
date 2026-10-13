"""What a deployment hosts — a base model and its named adapters — read from JSON, and the command
each server runs to host it: ours, and vLLM as the reference."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from hostllm import config

BASE = "base"
SPEC_ENV = "HOSTLLM_SPEC"  # a path to the spec, where it is deployed from
SPEC_JSON_ENV = "HOSTLLM_SPEC_JSON"  # the spec itself, as a deployed image carries it
_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")
_FIELDS = frozenset({"base_model", "adapters", "max_lora_rank"})


@dataclass(frozen=True)
class Spec:
    """A base model, each adapter by name and source (a Hub repo or path), and the largest rank."""

    base_model: str
    adapters: dict[str, str]
    max_lora_rank: int

    def ours_command(self, port: int) -> list[str]:
        command = ["python", "-m", "hostllm.server", "--base-model", self.base_model]
        command += ["--port", str(port)]
        for name, source in sorted(self.adapters.items()):
            command += ["--adapter", f"{name}={source}"]
        return command

    def vllm_command(self, port: int) -> list[str]:
        """`vllm serve` for the same models; the API key goes in VLLM_API_KEY, not here."""
        modules = [f"{name}={source}" for name, source in sorted(self.adapters.items())]
        return [
            "vllm", "serve", self.base_model,
            "--served-model-name", BASE,
            "--port", str(port),
            "--max-model-len", str(config.DEPLOY_MAX_MODEL_LEN),
            "--enable-lora",
            "--max-loras", str(len(self.adapters)),
            "--max-lora-rank", str(self.max_lora_rank),
            "--lora-modules", *modules,
            "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
        ]  # fmt: skip


def load_spec(text: str) -> Spec:
    """The spec in `text`, or ValueError naming what is wrong with it."""
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("a spec is a JSON object")
    fields = cast(dict[str, object], raw)
    unknown = sorted(set(fields) - _FIELDS)
    if unknown:
        raise ValueError(f"unknown spec fields: {', '.join(unknown)}")
    base_model = fields.get("base_model")
    if not isinstance(base_model, str) or not base_model:
        raise ValueError("base_model must name a model")
    adapters = fields.get("adapters")
    if not isinstance(adapters, dict) or not adapters:
        raise ValueError("adapters must map at least one name to its source")
    named: dict[str, str] = {}
    for name, source in cast(dict[str, object], adapters).items():
        if name == BASE:
            raise ValueError(f"an adapter cannot be called {BASE!r}: that name is the base's")
        if not _NAME.fullmatch(name):
            raise ValueError(f"adapter name {name!r}: lower-case letters, digits and dashes")
        if not isinstance(source, str) or not source:
            raise ValueError(f"adapter {name!r} has no source")
        named[name] = source
    rank = fields.get("max_lora_rank")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError("max_lora_rank must be a positive integer")
    return Spec(base_model, named, rank)


def spec_from_env(environ: Mapping[str, str]) -> Spec:
    """The spec an image carries, else the one at the path the deployer named."""
    if SPEC_JSON_ENV in environ:
        return load_spec(environ[SPEC_JSON_ENV])
    if SPEC_ENV in environ:
        return load_spec(Path(environ[SPEC_ENV]).read_text(encoding="utf-8"))
    raise ValueError(f"no spec: set {SPEC_ENV} to the path of one")
