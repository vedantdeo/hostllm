"""OpenAI's chat completions API, the part hostllm speaks: a request read and refused as the API
would, and a completion or an error written back in its shape."""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal, cast

from hostllm import config

Message = dict[str, object]
FinishReason = Literal["stop", "length", "tool_calls"]
ROLES = frozenset({"system", "user", "assistant", "tool"})


@dataclass(frozen=True)
class ChatRequest:
    """One request, validated: `model` names the adapter, or the base."""

    model: str
    messages: list[Message]
    tools: list[Message]
    max_tokens: int
    temperature: float
    top_p: float | None


@dataclass(frozen=True)
class Completion:
    """What the engine made of a request: text, or calls with any text beside them stripped."""

    content: str | None
    tool_calls: list[dict[str, object]]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: FinishReason


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def parse_request(body: object, models: Collection[str]) -> ChatRequest:
    """The request in `body`, or ValueError naming the first field the API would refuse."""
    if not isinstance(body, dict):
        raise ValueError("the body must be a JSON object")
    fields = cast(dict[str, object], body)
    model = fields.get("model")
    if not isinstance(model, str) or model not in models:
        raise ValueError(f"no model {model!r}; hosted: {', '.join(models)}")
    messages = fields.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    for message in cast(list[object], messages):
        if not isinstance(message, dict):
            raise ValueError("messages must be objects")
        role = cast(dict[str, object], message).get("role")
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}; one of {', '.join(sorted(ROLES))}")
    tools = fields.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")
    max_tokens = fields.get("max_tokens", config.SERVE_DEFAULT_MAX_TOKENS)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
        raise ValueError("max_tokens must be an integer")
    if not 1 <= max_tokens <= config.SERVE_MAX_TOKENS:
        raise ValueError(f"max_tokens must be from 1 to {config.SERVE_MAX_TOKENS}")
    temperature = fields.get("temperature", config.SERVE_DEFAULT_TEMPERATURE)
    if not _number(temperature) or cast(float, temperature) < 0:
        raise ValueError("temperature must be a number, zero or more")
    top_p = fields.get("top_p")
    if top_p is not None and (not _number(top_p) or not 0 < cast(float, top_p) <= 1):
        raise ValueError("top_p must be above 0 and at most 1")
    if fields.get("stream"):
        raise ValueError("stream is not supported; send stream: false")
    return ChatRequest(
        model=model,
        messages=cast(list[Message], messages),
        tools=cast(list[Message], tools),
        max_tokens=max_tokens,
        temperature=float(cast(float, temperature)),
        top_p=None if top_p is None else float(cast(float, top_p)),
    )


def completion_body(
    request_id: str, model: str, done: Completion, created: int
) -> dict[str, object]:
    """A `chat.completion` object; each call gets an id and its arguments as a JSON string."""
    message: dict[str, object] = {"role": "assistant", "content": done.content}
    if done.tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": call.get("name"),
                    "arguments": json.dumps(call.get("arguments", {})),
                },
            }
            for i, call in enumerate(done.tool_calls)
        ]
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": done.finish_reason}],
        "usage": {
            "prompt_tokens": done.prompt_tokens,
            "completion_tokens": done.completion_tokens,
            "total_tokens": done.prompt_tokens + done.completion_tokens,
        },
    }


def error_body(message: str, kind: str) -> dict[str, object]:
    """An error object, as the API returns one."""
    return {"error": {"message": message, "type": kind}}
