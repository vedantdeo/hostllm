"""The engine: one base model loaded once with its LoRA adapters attached, each request decoded on
the adapter it names, or on the bare base, by our own loop over a KV cache."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from peft import PeftModel
from torch import Tensor

from hostllm.protocol import ChatRequest, Completion, FinishReason, Message
from hostllm.sampling import choose

Render = Callable[[Sequence[Message], Sequence[Message]], list[int]]
Detokenize = Callable[[Sequence[int]], str]
_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


class Step(Protocol):
    """One forward pass: logits for `ids` given the cache so far, and the cache after them."""

    def __call__(self, ids: Tensor, cache: object | None) -> tuple[Tensor, object]: ...


def decode(
    step: Step,
    prompt: Sequence[int],
    max_new: int,
    stop: Collection[int],
    choose: Callable[[Tensor], int],
) -> tuple[list[int], FinishReason]:
    """Prefill `prompt`, then feed back one chosen token at a time until a stop token, which is not
    kept, or `max_new` tokens."""
    ids = torch.tensor([list(prompt)])
    cache: object | None = None
    out: list[int] = []
    for _ in range(max_new):
        logits, cache = step(ids, cache)
        token = choose(logits[0, -1])
        if token in stop:
            return out, "stop"
        out.append(token)
        ids = torch.tensor([[token]])
    return out, "length"


def split_reply(text: str) -> tuple[str | None, list[dict[str, object]]]:
    """The reply's text with each well-formed, named <tool_call> taken out, and those calls; a block
    that does not parse stays in the text."""
    calls: list[dict[str, object]] = []

    def take(match: re.Match[str]) -> str:
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            return match.group(0)
        if not isinstance(parsed, dict) or "name" not in parsed:
            return match.group(0)
        calls.append(cast(dict[str, object], parsed))
        return ""

    content = _CALL.sub(take, text).strip()
    return content or None, calls


@dataclass
class Engine:
    """`model` carries every adapter; `base` is the name that asks for none."""

    model: PeftModel
    base: str
    render: Render
    detokenize: Detokenize
    stop: frozenset[int]

    def models(self) -> list[str]:
        return [self.base, *sorted(self.model.peft_config)]

    @contextmanager
    def selected(self, name: str) -> Iterator[None]:
        """Run the body on adapter `name`, or with adapters off for the base."""
        if name == self.base:
            with self.model.disable_adapter():
                yield
            return
        self.model.set_adapter(name)
        yield

    def _step(self, ids: Tensor, cache: object | None) -> tuple[Tensor, object]:
        out = self.model(input_ids=ids.to(self.model.device), past_key_values=cache, use_cache=True)
        return out.logits, out.past_key_values

    @torch.no_grad()
    def complete(
        self, request: ChatRequest, generator: torch.Generator | None = None
    ) -> Completion:
        """Answer one request on the weights it names."""
        prompt = self.render(request.messages, request.tools)

        def pick(logits: Tensor) -> int:
            return choose(logits, request.temperature, request.top_p, generator)

        with self.selected(request.model):
            tokens, finish = decode(self._step, prompt, request.max_tokens, self.stop, pick)
        content, calls = split_reply(self.detokenize(tokens))
        return Completion(
            content=content,
            tool_calls=calls,
            prompt_tokens=len(prompt),
            completion_tokens=len(tokens),
            finish_reason="tool_calls" if calls else finish,
        )


def load(base_model: str, adapters: Mapping[str, str], base: str = "base") -> Engine:
    """An Engine over `base_model` from the Hub with each adapter, named, from its repo or path; the
    chat template renders with thinking off."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=dtype if cuda else torch.float32, device_map="cuda" if cuda else "cpu"
    )
    peft = attach(model, adapters)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def render(messages: Sequence[Message], tools: Sequence[Message]) -> list[int]:
        ids = tokenizer.apply_chat_template(
            list(messages),
            tools=list(tools) or None,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
        )
        return list(cast(list[int], ids))

    def detokenize(tokens: Sequence[int]) -> str:
        return str(tokenizer.decode(list(tokens), skip_special_tokens=False))

    stop = {tid for tid in (tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id)}
    return Engine(peft, base, render, detokenize, frozenset(int(t) for t in stop if t is not None))


def attach(model: object, adapters: Mapping[str, str]) -> PeftModel:
    """`model` with each adapter loaded under its name; ValueError for none."""
    if not adapters:
        raise ValueError("no adapters to host")
    (first, path), *rest = sorted(adapters.items())
    peft = PeftModel.from_pretrained(cast(torch.nn.Module, model), path, adapter_name=first)
    for name, more in rest:
        peft.load_adapter(more, adapter_name=name)
    return peft.eval()
