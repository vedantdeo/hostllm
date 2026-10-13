"""Benchmark an OpenAI-compatible endpoint: one cold request, then each workload at each
concurrency level, reporting latency percentiles, throughput and cost; with --stop-modal-app, the
app's containers are stopped afterwards.

uv run hostllm-bench --url <endpoint> --workload headlines=<test.jsonl>:headline-flavour \\
    --api-key-file ~/.config/hostllm-api-key --modal-proxy ~/.config/hostllm-modal-proxy.json \\
    --stop-modal-app hostllm --out bench.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from itertools import cycle, islice
from pathlib import Path
from typing import cast

from hostllm import config

Body = dict[str, object]
Send = Callable[[Body], tuple[dict[str, object], str]]
Cli = Callable[[Sequence[str]], str]


@dataclass(frozen=True)
class Workload:
    name: str
    bodies: list[Body]


@dataclass(frozen=True)
class Sample:
    latency: float
    tokens: int
    error: str | None


@dataclass(frozen=True)
class Level:
    workload: str
    concurrency: int
    requests: int
    errors: int
    p50: float
    p95: float
    requests_per_second: float
    tokens_per_second: float
    gpu_seconds_per_request: float
    usd_per_1k: float


def load_workload(name: str, path: Path, model: str) -> Workload:
    """One greedy request body per exported row, with its tools if it has any."""
    bodies: list[Body] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = cast(dict[str, object], json.loads(line))
        body: Body = {
            "model": model,
            "messages": row["messages"],
            "temperature": 0,
            "max_tokens": config.BENCH_MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if row.get("tools"):
            body["tools"] = row["tools"]
        bodies.append(body)
    return Workload(name, bodies)


def percentile(values: Sequence[float], q: float) -> float:
    """The nearest-rank q-th percentile of a non-empty sequence."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def output_tokens(usage: Mapping[str, object], finish_reason: str, counts_stop: bool) -> int:
    """Completion tokens without the stop token, which some engines count and others do not."""
    tokens = int(cast(int, usage.get("completion_tokens", 0)))
    return tokens - 1 if counts_stop and finish_reason in ("stop", "tool_calls") else tokens


def run_level(
    send: Send, bodies: Sequence[Body], concurrency: int, requests: int, counts_stop: bool
) -> tuple[list[Sample], float]:
    """`requests` requests cycled from `bodies`, `concurrency` in flight at once; the samples and
    the wall time they took."""

    def one(body: Body) -> Sample:
        started = time.perf_counter()
        try:
            usage, finish = send(body)
        except Exception as failed:  # a failed request is data, not a crash
            return Sample(time.perf_counter() - started, 0, str(failed))
        return Sample(
            time.perf_counter() - started, output_tokens(usage, finish, counts_stop), None
        )

    began = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        samples = list(pool.map(one, islice(cycle(bodies), requests)))
    return samples, time.perf_counter() - began


def summarise(
    workload: str, concurrency: int, samples: Sequence[Sample], wall: float, usd_per_hour: float
) -> Level:
    """One level's numbers; failed requests count as errors and stay out of the latencies."""
    good = [s for s in samples if s.error is None]
    latencies = [s.latency for s in good] or [math.nan]
    per_request = wall / len(samples)
    return Level(
        workload=workload,
        concurrency=concurrency,
        requests=len(samples),
        errors=len(samples) - len(good),
        p50=percentile(latencies, 50),
        p95=percentile(latencies, 95),
        requests_per_second=len(good) / wall,
        tokens_per_second=sum(s.tokens for s in good) / wall,
        gpu_seconds_per_request=per_request,
        usd_per_1k=per_request * 1000 * usd_per_hour / 3600,
    )


def estimate_usd(
    requests: int, seconds_per_request: float, cold_seconds: float, usd_per_hour: float
) -> float:
    """An upper bound: one cold start, then every request one after another."""
    return (cold_seconds + requests * seconds_per_request) * usd_per_hour / 3600


def http_send(url: str, headers: dict[str, str], timeout: float) -> Send:
    """A Send posting to `url`'s chat completions, raising RuntimeError on an HTTP error and
    TimeoutError when no answer comes within `timeout` seconds."""

    def send(body: Body) -> tuple[dict[str, object], str]:
        request = urllib.request.Request(
            url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **headers},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as reply:
                out = cast(dict[str, object], json.loads(reply.read()))
        except urllib.error.HTTPError as refused:
            raise RuntimeError(f"status {refused.code}: {refused.read().decode()[:200]}") from None
        choice = cast(list[dict[str, object]], out["choices"])[0]
        return cast(dict[str, object], out["usage"]), str(choice["finish_reason"])

    return send


def stop_modal_app(app: str, cli: Cli, wait: Callable[[float], None] = time.sleep) -> list[str]:
    """Stop every running container of `app` and wait until none is left; the ids stopped."""

    def running() -> list[str]:
        listed = cast(
            list[dict[str, str]], json.loads(cli(["modal", "container", "list", "--json"]))
        )
        return [c["container_id"] for c in listed if c.get("app_name") == app]

    stopped = running()
    for container in stopped:
        cli(["modal", "container", "stop", "--yes", container])
    for _ in range(config.BENCH_STOP_POLLS):
        if not running():
            return stopped
        wait(config.BENCH_STOP_POLL_SECONDS)
    raise RuntimeError(f"{app} containers still running after stopping: {', '.join(running())}")


def _cli(argv: Sequence[str]) -> str:
    return subprocess.run(list(argv), capture_output=True, text=True, check=True).stdout


def _headers(api_key_file: Path | None, modal_proxy: Path | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if modal_proxy is not None:
        proxy = cast(dict[str, str], json.loads(modal_proxy.read_text()))
        headers |= {"Modal-Key": proxy["Modal-Key"], "Modal-Secret": proxy["Modal-Secret"]}
    if api_key_file is not None:
        headers["Authorization"] = f"Bearer {api_key_file.read_text().strip()}"
    return headers


def _workload(spec: str) -> Workload:
    name, _, rest = spec.partition("=")
    path, _, model = rest.rpartition(":")
    if not name or not path or not model:
        raise ValueError(f"workload {spec!r}: want name=path:model")
    return load_workload(name, Path(path), model)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible endpoint.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--workload", action="append", required=True, help="name=path:model")
    parser.add_argument("--levels", default=",".join(map(str, config.BENCH_LEVELS)))
    parser.add_argument("--requests", type=int, default=config.BENCH_REQUESTS, help="per level")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--modal-proxy", type=Path, help="a proxy token's JSON")
    parser.add_argument("--counts-stop-token", action="store_true", help="vLLM: yes; hostllm: no")
    parser.add_argument("--stop-modal-app", help="stop this Modal app's containers afterwards")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and its cost only")
    args = parser.parse_args(argv)
    workloads = [_workload(w) for w in args.workload]
    levels = [int(n) for n in args.levels.split(",")]
    total = 1 + len(workloads) * len(levels) * args.requests
    usd = estimate_usd(
        total, config.BENCH_ASSUMED_SECONDS, config.BENCH_ASSUMED_COLD_SECONDS,
        config.DEPLOY_GPU_USD_PER_HOUR,
    )  # fmt: skip
    print(f"{total} requests ({len(workloads)} workloads x {len(levels)} levels x {args.requests},")
    print(f"plus one cold); at most ${usd:.2f} of GPU at ${config.DEPLOY_GPU_USD_PER_HOUR}/h")
    if args.dry_run:
        print("nothing sent. re-run without --dry-run to send these requests.")
        return
    headers = _headers(args.api_key_file, args.modal_proxy)
    send = http_send(args.url, headers, config.BENCH_TIMEOUT_SECONDS)
    first = http_send(args.url, headers, config.BENCH_COLD_TIMEOUT_SECONDS)
    try:
        cold, _ = run_level(first, workloads[0].bodies[:1], 1, 1, args.counts_stop_token)
        if cold[0].error is not None:
            raise RuntimeError(f"the cold request failed: {cold[0].error}")
        results: list[Level] = []
        for workload in workloads:
            for concurrency in levels:
                samples, wall = run_level(
                    send, workload.bodies, concurrency, args.requests, args.counts_stop_token
                )
                level = summarise(
                    workload.name, concurrency, samples, wall, config.DEPLOY_GPU_USD_PER_HOUR
                )
                results.append(level)
                print(
                    f"{level.workload:10} c={level.concurrency:<3} p50 {level.p50:6.2f}s  "
                    f"p95 {level.p95:6.2f}s  {level.requests_per_second:5.2f} req/s  "
                    f"{level.tokens_per_second:7.1f} tok/s  ${level.usd_per_1k:.3f}/1k  "
                    f"errors {level.errors}",
                    flush=True,
                )
    finally:
        if args.stop_modal_app:
            print(f"stopped {stop_modal_app(args.stop_modal_app, _cli)}")
    print(f"cold start {cold[0].latency:.1f}s")
    if args.out is not None:
        result = {
            "url": args.url,
            "cold_seconds": cold[0].latency,
            "levels": [asdict(level) for level in results],
        }
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
