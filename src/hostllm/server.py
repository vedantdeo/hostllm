"""The OpenAI-compatible HTTP server: health, the model list and chat completions over a scheduler,
each refusal an OpenAI error.

uv run hostllm-serve --base-model Qwen/Qwen3-4B --adapter headline-reader=<repo or path> ...
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from hostllm import config
from hostllm.protocol import completion_body, error_body, parse_request
from hostllm.scheduler import Sequential

BASE = "base"


def make_server(
    scheduler: Sequential, models: Sequence[str], api_key: str | None, host: str, port: int
) -> ThreadingHTTPServer:
    """A server answering on host:port; with `api_key`, every route but /health wants it as a
    bearer token."""
    hosted = list(models)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

        def _send(self, status: int, body: dict[str, object]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _refuse(self, status: int, message: str, kind: str) -> None:
            self._send(status, error_body(message, kind))

        def _authorised(self) -> bool:
            if api_key is None or self.headers.get("Authorization") == f"Bearer {api_key}":
                return True
            self._refuse(401, "a valid bearer API key is required", "authentication_error")
            return False

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif not self._authorised():
                return
            elif self.path == "/v1/models":
                data = [{"id": m, "object": "model", "owned_by": "hostllm"} for m in hosted]
                self._send(200, {"object": "list", "data": data})
            else:
                self._refuse(404, f"no route {self.path}", "not_found_error")

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorised():
                return
            if self.path != "/v1/chat/completions":
                self._refuse(404, f"no route {self.path}", "not_found_error")
                return
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                request = parse_request(json.loads(raw), hosted)
            except json.JSONDecodeError:
                self._refuse(400, "the body is not JSON", "invalid_request_error")
                return
            except ValueError as refused:
                self._refuse(400, str(refused), "invalid_request_error")
                return
            try:
                done = scheduler.submit(request).result()
            except Exception as failed:  # the engine's failure is this request's, not the server's
                self._refuse(500, f"{type(failed).__name__}: {failed}", "server_error")
                return
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            self._send(200, completion_body(request_id, request.model, done, int(time.time())))

    return ThreadingHTTPServer((host, port), Handler)


def parse_adapters(given: Sequence[str]) -> dict[str, str]:
    """`name=source` pairs as a dict; ValueError for a pair missing either side, a repeated name, or
    the base's own name."""
    adapters: dict[str, str] = {}
    for pair in given:
        name, _, source = pair.partition("=")
        if not name or not source:
            raise ValueError(f"adapter {pair!r}: want name=repo-or-path")
        if name in adapters or name == BASE:
            raise ValueError(f"adapter name {name!r} is taken")
        adapters[name] = source
    return adapters


def main() -> None:
    from hostllm.engine import load

    parser = argparse.ArgumentParser(description="Serve a causal LM and its LoRA adapters.")
    parser.add_argument("--base-model", required=True, help="a Hub repo or a local path")
    parser.add_argument("--adapter", action="append", default=[], help="name=repo-or-path")
    parser.add_argument("--host", default=config.SERVE_HOST)
    parser.add_argument("--port", type=int, default=config.SERVE_PORT)
    args = parser.parse_args()
    engine = load(args.base_model, parse_adapters(args.adapter), base=BASE)
    scheduler = Sequential(engine.complete)
    server = make_server(
        scheduler, engine.models(), os.environ.get(config.SERVE_API_KEY_ENV), args.host, args.port
    )
    print(f"serving {', '.join(engine.models())} on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        scheduler.close()


if __name__ == "__main__":
    main()
