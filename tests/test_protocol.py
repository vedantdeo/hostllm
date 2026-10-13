"""Specification for protocol.py: a chat completion request read and refused as OpenAI's API would,
and a completion written back in its shape."""

from __future__ import annotations

import json
from typing import cast

import pytest

from hostllm import config
from hostllm.protocol import ChatRequest, Completion, completion_body, error_body, parse_request

MODELS = ("base", "headline-reader")
USER: dict[str, object] = {"role": "user", "content": "headline: Infosys Q3 profit rises 11%"}


def body(**overrides: object) -> dict[str, object]:
    return {"model": "headline-reader", "messages": [USER], **overrides}


def test_a_request_takes_the_defaults_for_what_it_leaves_out() -> None:
    request = parse_request(body(), MODELS)

    assert request == ChatRequest(
        model="headline-reader",
        messages=[USER],
        tools=[],
        max_tokens=config.SERVE_DEFAULT_MAX_TOKENS,
        temperature=config.SERVE_DEFAULT_TEMPERATURE,
        top_p=None,
    )


def test_a_request_keeps_what_it_sets() -> None:
    tool = {"type": "function", "function": {"name": "calculate"}}

    request = parse_request(
        body(tools=[tool], max_tokens=64, temperature=0, top_p=0.9, stream=False), MODELS
    )

    assert (request.tools, request.max_tokens, request.temperature, request.top_p) == (
        [tool],
        64,
        0.0,
        0.9,
    )


@pytest.mark.parametrize(
    ("sent", "refusal"),
    [
        pytest.param([], "JSON object", id="not an object"),
        pytest.param(body(model="gpt-9"), "no model 'gpt-9'", id="a model not hosted"),
        pytest.param({"messages": [USER]}, "no model", id="no model at all"),
        pytest.param(body(messages=[]), "messages", id="no messages"),
        pytest.param(body(messages=["hi"]), "messages", id="a message that is not an object"),
        pytest.param(
            body(messages=[{"role": "robot", "content": "x"}]), "role", id="an unknown role"
        ),
        pytest.param(body(tools={"name": "calculate"}), "tools", id="tools not a list"),
        pytest.param(body(max_tokens=0), "max_tokens", id="no tokens to generate"),
        pytest.param(
            body(max_tokens=config.SERVE_MAX_TOKENS + 1), "max_tokens", id="more than the cap"
        ),
        pytest.param(body(max_tokens=True), "max_tokens", id="a bool for max_tokens"),
        pytest.param(body(temperature=-0.1), "temperature", id="a negative temperature"),
        pytest.param(body(temperature="hot"), "temperature", id="a temperature not a number"),
        pytest.param(body(top_p=0), "top_p", id="a top_p of zero"),
        pytest.param(body(top_p=1.5), "top_p", id="a top_p above one"),
        pytest.param(body(stream=True), "stream", id="streaming asked for"),
    ],
)
def test_a_bad_request_is_refused_by_name(sent: object, refusal: str) -> None:
    with pytest.raises(ValueError, match=refusal):
        parse_request(sent, MODELS)


def test_a_completion_is_written_in_openai_shape() -> None:
    done = Completion(
        content='{"company": "Infosys"}',
        tool_calls=[],
        prompt_tokens=40,
        completion_tokens=9,
        finish_reason="stop",
    )

    out = completion_body("chatcmpl-1", "headline-reader", done, created=1700000000)

    assert out == {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "headline-reader",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": '{"company": "Infosys"}'},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 9, "total_tokens": 49},
    }


def test_tool_calls_carry_ids_and_arguments_as_a_json_string() -> None:
    call = {"name": "calculate", "arguments": {"expression": "2 * 10**7"}}
    done = Completion(None, [call], 40, 20, "tool_calls")

    out = completion_body("chatcmpl-2", "report-analyst", done, created=1)

    (choice,) = cast(list[dict[str, object]], out["choices"])
    message = cast(dict[str, object], choice["message"])
    (sent,) = cast(list[dict[str, object]], message["tool_calls"])
    function = cast(dict[str, str], sent["function"])
    assert message["content"] is None and choice["finish_reason"] == "tool_calls"
    assert sent["id"] == "call_0" and sent["type"] == "function"
    assert function["name"] == "calculate"
    assert json.loads(function["arguments"]) == {"expression": "2 * 10**7"}


def test_an_error_is_written_in_openai_shape() -> None:
    assert error_body("no model 'gpt-9'", "invalid_request_error") == {
        "error": {"message": "no model 'gpt-9'", "type": "invalid_request_error"}
    }
