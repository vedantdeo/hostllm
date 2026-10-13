"""Specification for engine.py: the decode loop over a KV cache, each request served by the adapter
it names or the bare base, and a reply split into text and tool calls."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from peft import PeftModel
from torch import Tensor

from hostllm.engine import Engine, attach, decode, split_reply, text_io
from hostllm.protocol import ChatRequest
from tests.conftest import ADAPTERS, VOCAB, tiny_qwen


class ScriptedStep:
    """A model that answers each step with the next scripted token, recording what it was fed."""

    def __init__(self, tokens: Sequence[int]) -> None:
        self.tokens = list(tokens)
        self.fed: list[list[int]] = []
        self.caches: list[object] = []

    def __call__(self, ids: Tensor, cache: object | None) -> tuple[Tensor, object]:
        self.fed.append(ids[0].tolist())
        self.caches.append(cache)
        logits = torch.full((1, ids.shape[1], VOCAB), -1e9)
        logits[0, -1, self.tokens.pop(0)] = 0.0
        return logits, f"cache-{len(self.fed)}"


def greedy(logits: Tensor) -> int:
    return int(logits.argmax())


def test_decode_feeds_the_prompt_once_then_each_new_token_with_its_cache() -> None:
    step = ScriptedStep([5, 6, 2])

    tokens, finish = decode(step, [10, 11, 12], max_new=8, stop={2}, choose=greedy)

    assert (tokens, finish) == ([5, 6], "stop"), "the stop token ends the reply and is not in it"
    assert step.fed == [[10, 11, 12], [5], [6]]
    assert step.caches == [None, "cache-1", "cache-2"]


def test_decode_stops_at_max_new_tokens() -> None:
    tokens, finish = decode(ScriptedStep([5, 6, 7, 2]), [1], max_new=3, stop={2}, choose=greedy)

    assert (tokens, finish) == ([5, 6, 7], "length")


@pytest.mark.parametrize(
    ("text", "content", "calls"),
    [
        pytest.param('{"company": "Infosys"}', '{"company": "Infosys"}', [], id="plain text"),
        pytest.param(
            '<tool_call>\n{"name": "calculate", "arguments": {"expression": "2*3"}}\n</tool_call>',
            None,
            [{"name": "calculate", "arguments": {"expression": "2*3"}}],
            id="a call alone",
        ),
        pytest.param(
            'Let me compute.\n<tool_call>{"name": "calculate", "arguments": {}}</tool_call>',
            "Let me compute.",
            [{"name": "calculate", "arguments": {}}],
            id="text beside a call",
        ),
        pytest.param(
            "<tool_call>{not json}</tool_call>",
            "<tool_call>{not json}</tool_call>",
            [],
            id="a malformed call stays text",
        ),
        pytest.param(
            '<tool_call>{"arguments": {}}</tool_call>',
            '<tool_call>{"arguments": {}}</tool_call>',
            [],
            id="a call with no name stays text",
        ),
    ],
)
def test_split_reply(text: str, content: str | None, calls: list[dict[str, object]]) -> None:
    assert split_reply(text) == (content, calls)


def engine(model: PeftModel, reply: str = "") -> Engine:
    return Engine(
        model=model,
        base="base",
        render=lambda messages, tools: [3, 4, 5, 6][: 2 + len(messages) + len(tools)],
        detokenize=lambda tokens: reply or " ".join(map(str, tokens)),
        stop=frozenset({VOCAB}),  # out of range, so only max_tokens ends a reply
    )


def request(model: str, **overrides: object) -> ChatRequest:
    plain = ChatRequest(model, [{"role": "user", "content": "hi"}], [], 4, 0.0, None)
    return replace(plain, **overrides)


def first_logits(server: Engine, name: str) -> Tensor:
    with server.selected(name):
        return server.model(input_ids=torch.tensor([[3, 4, 5]])).logits[0, -1]


def test_the_engine_hosts_the_base_and_each_adapter(adapted: PeftModel) -> None:
    assert engine(adapted).models() == ["base", *sorted(ADAPTERS)]


def test_each_name_selects_its_own_weights_and_leaves_none_behind(adapted: PeftModel) -> None:
    server = engine(adapted)

    one, base, two = (first_logits(server, n) for n in (ADAPTERS[0], "base", ADAPTERS[1]))

    assert not torch.allclose(one, base) and not torch.allclose(two, base)
    assert not torch.allclose(one, two)
    assert torch.allclose(first_logits(server, ADAPTERS[0]), one), "base left the adapter off"
    with adapted.disable_adapter():
        bare = adapted(input_ids=torch.tensor([[3, 4, 5]])).logits[0, -1]
    assert torch.allclose(base, bare)


def test_a_greedy_completion_counts_its_tokens(adapted: PeftModel) -> None:
    done = engine(adapted).complete(request(ADAPTERS[0], max_tokens=4))

    assert done.prompt_tokens == 3 and done.completion_tokens == 4
    assert done.finish_reason == "length" and done.tool_calls == []
    assert done.content is not None and len(done.content.split()) == 4


def test_greedy_is_repeatable_and_each_adapter_answers_differently(adapted: PeftModel) -> None:
    server = engine(adapted)

    replies = {n: server.complete(request(n, max_tokens=6)).content for n in (*ADAPTERS, "base")}

    assert server.complete(request(ADAPTERS[1], max_tokens=6)).content == replies[ADAPTERS[1]]
    assert len(set(replies.values())) == 3, replies


def test_a_reply_with_a_call_finishes_as_tool_calls(adapted: PeftModel) -> None:
    call = '<tool_call>{"name": "calculate", "arguments": {"expression": "1+1"}}</tool_call>'

    done = engine(adapted, reply=call).complete(request(ADAPTERS[1]))

    assert done.finish_reason == "tool_calls" and done.content is None
    assert done.tool_calls == [{"name": "calculate", "arguments": {"expression": "1+1"}}]


def test_sampling_follows_the_request_and_repeats_with_a_seed(adapted: PeftModel) -> None:
    server = engine(adapted)
    hot = request(ADAPTERS[0], max_tokens=8, temperature=5.0)

    a = server.complete(hot, generator=torch.Generator().manual_seed(1)).content
    b = server.complete(hot, generator=torch.Generator().manual_seed(1)).content
    draws = {
        server.complete(hot, generator=torch.Generator().manual_seed(s)).content for s in range(5)
    }

    assert a == b and len(draws) > 1


def test_attach_loads_each_saved_adapter_under_its_name(adapted: PeftModel, tmp_path: Path) -> None:
    saved: dict[str, str] = {}
    for name in ADAPTERS:
        adapted.save_pretrained(str(tmp_path / name), selected_adapters=[name])
        config = next((tmp_path / name).rglob("adapter_config.json"))  # PEFT may nest by name
        saved[name] = str(config.parent)

    loaded = attach(tiny_qwen(), saved)

    assert sorted(loaded.peft_config) == sorted(ADAPTERS)
    ids = torch.tensor([[3, 4, 5]])
    for name in ADAPTERS:
        adapted.set_adapter(name)
        loaded.set_adapter(name)
        assert torch.allclose(loaded(input_ids=ids).logits, adapted(input_ids=ids).logits), name


def test_attach_refuses_no_adapters() -> None:
    with pytest.raises(ValueError, match="no adapters"):
        attach(tiny_qwen(), {})


class CharTokenizer:
    """One id per character; the chat template is a visible stand-in that records its arguments."""

    eos_token_id: int | None = 1

    def __init__(self) -> None:
        self.template_calls: list[dict[str, object]] = []

    def apply_chat_template(
        self, conversation: list[dict[str, object]], **kwargs: object
    ) -> object:
        self.template_calls.append(kwargs)
        return "".join(str(m["content"]) for m in conversation) + ">"

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in text], "attention_mask": [1] * len(text)}

    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        return "".join(chr(i) for i in ids)

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return 2 if token == "<|im_end|>" else None


def test_text_io_renders_the_template_as_text_then_tokenizes_it() -> None:
    tokenizer = CharTokenizer()
    render, detokenize, stop = text_io(tokenizer)

    ids = render([{"role": "user", "content": "hi"}], [])

    assert ids == [ord("h"), ord("i"), ord(">")], "token ids, not the dict's keys"
    assert tokenizer.template_calls == [
        {"tools": None, "add_generation_prompt": True, "enable_thinking": False, "tokenize": False}
    ]
    assert detokenize(ids) == "hi>" and stop == frozenset({1, 2})


def test_text_io_passes_tools_through() -> None:
    tokenizer = CharTokenizer()
    render, _, _ = text_io(tokenizer)
    tool = {"type": "function", "function": {"name": "calculate"}}

    render([{"role": "user", "content": "x"}], [tool])

    assert tokenizer.template_calls[0]["tools"] == [tool]
