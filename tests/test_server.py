"""Specification for server.py: the OpenAI-compatible HTTP surface over a scheduler — health, the
model list, chat completions, the bearer key, and every refusal as an OpenAI error."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import cast

import pytest

from hostllm.protocol import ChatRequest, Completion
from hostllm.scheduler import Sequential
from hostllm.server import make_server, parse_adapters

MODELS = ["base", "headline-reader"]
KEY = "sk-test"


def engine(req: ChatRequest) -> Completion:
    if req.messages[-1].get("content") == "explode":
        raise RuntimeError("CUDA out of memory")
    return Completion(f"echo {req.model}", [], 7, 3, "stop")


@pytest.fixture
def url() -> Iterator[str]:
    scheduler = Sequential(engine)
    server = make_server(scheduler, MODELS, KEY, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    scheduler.close()


def call(
    url: str, path: str, body: object = None, key: str | None = KEY, raw: bytes | None = None
) -> tuple[int, dict[str, object]]:
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    sent = urllib.request.Request(url + path, data=data, method="POST" if data else "GET")
    if key is not None:
        sent.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(sent, timeout=5) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as refused:
        return refused.code, json.loads(refused.read())


CHAT = {"model": "headline-reader", "messages": [{"role": "user", "content": "hi"}]}


def test_health_needs_no_key(url: str) -> None:
    assert call(url, "/health", key=None) == (200, {"status": "ok"})


def test_the_model_list_names_each_hosted_model(url: str) -> None:
    status, out = call(url, "/v1/models")

    assert status == 200 and out["object"] == "list"
    assert [m["id"] for m in cast(list[dict[str, object]], out["data"])] == MODELS


def test_a_chat_completion_comes_back_in_openai_shape(url: str) -> None:
    status, out = call(url, "/v1/chat/completions", CHAT)

    assert status == 200 and out["object"] == "chat.completion"
    assert str(out["id"]).startswith("chatcmpl-") and isinstance(out["created"], int)
    assert out["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "echo headline-reader"},
            "finish_reason": "stop",
        }
    ]
    assert out["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


@pytest.mark.parametrize(
    ("path", "body", "key", "raw", "status", "kind"),
    [
        pytest.param("/v1/models", None, None, None, 401, "authentication_error", id="no key"),
        pytest.param(
            "/v1/chat/completions",
            CHAT,
            "sk-wrong",
            None,
            401,
            "authentication_error",
            id="the wrong key",
        ),
        pytest.param("/v1/nothing", None, KEY, None, 404, "not_found_error", id="an unknown path"),
        pytest.param(
            "/v1/chat/completions",
            None,
            KEY,
            b"{not json",
            400,
            "invalid_request_error",
            id="a body that is not JSON",
        ),
        pytest.param(
            "/v1/chat/completions",
            {**CHAT, "model": "gpt-9"},
            KEY,
            None,
            400,
            "invalid_request_error",
            id="a model not hosted",
        ),
        pytest.param(
            "/v1/chat/completions",
            {**CHAT, "messages": [{"role": "user", "content": "explode"}]},
            KEY,
            None,
            500,
            "server_error",
            id="the engine failing",
        ),
    ],
)
def test_a_refusal_is_an_openai_error(
    url: str, path: str, body: object, key: str | None, raw: bytes | None, status: int, kind: str
) -> None:
    got, out = call(url, path, body, key=key, raw=raw)

    error = out["error"]
    assert got == status and isinstance(error, dict) and error["type"] == kind, out


def test_no_key_configured_means_no_auth() -> None:
    scheduler = Sequential(engine)
    server = make_server(scheduler, MODELS, None, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        status, _ = call(f"http://127.0.0.1:{server.server_address[1]}", "/v1/models", key=None)
    finally:
        server.shutdown()
        scheduler.close()

    assert status == 200


@pytest.mark.parametrize(
    ("given", "parsed"),
    [
        pytest.param([], {}, id="none"),
        pytest.param(
            ["headline-reader=vedantdeo/hl", "report-analyst=/data/v3"],
            {"headline-reader": "vedantdeo/hl", "report-analyst": "/data/v3"},
            id="a repo and a path",
        ),
    ],
)
def test_parse_adapters(given: list[str], parsed: dict[str, str]) -> None:
    assert parse_adapters(given) == parsed


@pytest.mark.parametrize(
    "given",
    [
        pytest.param(["headline-reader"], id="no source"),
        pytest.param(["=vedantdeo/hl"], id="no name"),
        pytest.param(["a=x", "a=y"], id="a name twice"),
        pytest.param(["base=x"], id="the base's own name"),
    ],
)
def test_parse_adapters_refuses(given: list[str]) -> None:
    with pytest.raises(ValueError):
        parse_adapters(given)
