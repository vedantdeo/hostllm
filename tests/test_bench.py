"""Specification for bench.py: workloads read from exported eval rows, requests kept in flight at a
concurrency level, latencies and throughput summarised, a dry run priced, and containers stopped."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from hostllm import config
from hostllm.bench import (
    Sample,
    estimate_usd,
    load_workload,
    main,
    output_tokens,
    percentile,
    run_level,
    stop_modal_app,
    summarise,
)
from hostllm.protocol import ChatRequest, Completion
from hostllm.scheduler import Sequential
from hostllm.server import make_server

TOOL = {"type": "function", "function": {"name": "calculate"}}
MESSAGES = [{"role": "system", "content": "rules"}, {"role": "user", "content": "q"}]


def write_rows(path: Path, tools: bool) -> Path:
    rows = [
        {"id": f"r{i}", "messages": MESSAGES, **({"tools": [TOOL]} if tools else {})}
        for i in range(3)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.mark.parametrize(
    "tools", [pytest.param(False, id="headlines"), pytest.param(True, id="figures")]
)
def test_a_workload_is_one_greedy_body_per_row(tools: bool, tmp_path: Path) -> None:
    workload = load_workload("w", write_rows(tmp_path / "w.jsonl", tools), "report-analyst")

    assert workload.name == "w" and len(workload.bodies) == 3
    body = workload.bodies[0]
    assert body["model"] == "report-analyst" and body["messages"] == MESSAGES
    assert body["temperature"] == 0 and body["max_tokens"] == config.BENCH_MAX_TOKENS
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert ("tools" in body) is tools


@pytest.mark.parametrize(
    ("values", "q", "expected"),
    [
        pytest.param([3.0, 1.0, 2.0], 50, 2.0, id="the median of three"),
        pytest.param([float(v) for v in range(1, 101)], 95, 95.0, id="p95 of a hundred"),
        pytest.param([7.0], 95, 7.0, id="one value is every percentile"),
        pytest.param([1.0, 2.0], 100, 2.0, id="p100 is the largest"),
    ],
)
def test_percentile_is_nearest_rank(values: list[float], q: float, expected: float) -> None:
    assert percentile(values, q) == expected


@pytest.mark.parametrize(
    ("usage", "finish", "counts_stop", "expected"),
    [
        pytest.param({"completion_tokens": 37}, "stop", True, 36, id="vLLM counts the stop token"),
        pytest.param({"completion_tokens": 36}, "stop", False, 36, id="ours does not"),
        pytest.param(
            {"completion_tokens": 512}, "length", True, 512, id="no stop token at the cap"
        ),
        pytest.param(
            {"completion_tokens": 29}, "tool_calls", True, 28, id="a call ends on one too"
        ),
    ],
)
def test_output_tokens_count_the_stop_token_the_same_way(
    usage: dict[str, int], finish: str, counts_stop: bool, expected: int
) -> None:
    assert output_tokens(usage, finish, counts_stop) == expected


def test_summarise_reports_latency_throughput_and_cost() -> None:
    samples = [Sample(latency=s, tokens=10, error=None) for s in (1.0, 2.0, 3.0, 4.0)]

    level = summarise("w", 2, samples, wall=4.0, usd_per_hour=3.6)

    assert (level.p50, level.p95) == (2.0, 4.0)
    assert level.requests_per_second == 1.0 and level.tokens_per_second == 10.0
    assert level.gpu_seconds_per_request == 1.0 and level.usd_per_1k == 1.0
    assert level.errors == 0


def test_summarise_leaves_failed_requests_out_of_the_latencies() -> None:
    samples = [Sample(1.0, 10, None), Sample(9.0, 0, "status 500")]

    level = summarise("w", 1, samples, wall=10.0, usd_per_hour=0.8)

    assert level.errors == 1 and level.p95 == 1.0 and level.tokens_per_second == 1.0


def test_run_level_keeps_the_concurrency_in_flight_and_sends_n_requests() -> None:
    lock = threading.Lock()
    running = most = 0
    sent: list[dict[str, object]] = []

    def send(body: dict[str, object]) -> tuple[dict[str, object], str]:
        nonlocal running, most
        with lock:
            running += 1
            most = max(most, running)
            sent.append(body)
        time.sleep(0.02)
        with lock:
            running -= 1
        return {"completion_tokens": 5}, "length"

    bodies: list[dict[str, object]] = [{"n": i} for i in range(3)]
    samples, wall = run_level(send, bodies, concurrency=4, requests=10, counts_stop=False)

    assert len(samples) == 10 and most == 4 and wall > 0
    assert [b["n"] for b in sent].count(0) >= 3, "bodies are cycled"
    assert all(s.tokens == 5 and s.error is None for s in samples)


def test_a_failed_request_is_a_sample_with_its_error() -> None:
    def send(body: dict[str, object]) -> tuple[dict[str, object], str]:
        raise RuntimeError("status 500: boom")

    samples, _ = run_level(send, [{}], concurrency=1, requests=2, counts_stop=False)

    assert [s.error for s in samples] == ["status 500: boom"] * 2


def test_the_dry_run_prices_a_cold_start_and_every_request_unbatched() -> None:
    usd = estimate_usd(requests=100, seconds_per_request=9.0, cold_seconds=180.0, usd_per_hour=3.6)

    assert usd == pytest.approx((180 + 100 * 9) / 1000)


class FakeCli:
    """`modal container list/stop`: the app's containers go once stopped, the others stay."""

    def __init__(self) -> None:
        self.containers = [
            {"container_id": "ta-1", "app_name": "hostllm"},
            {"container_id": "ta-2", "app_name": "other"},
        ]
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> str:
        self.calls.append(list(argv))
        if "list" in argv:
            return json.dumps(self.containers)
        self.containers = [c for c in self.containers if c["container_id"] != argv[-1]]
        return ""


def test_stop_modal_app_stops_only_that_apps_containers_until_none_run() -> None:
    cli = FakeCli()

    stopped = stop_modal_app("hostllm", cli, wait=lambda _: None)

    assert stopped == ["ta-1"]
    assert [c["container_id"] for c in cli.containers] == ["ta-2"]
    assert ["modal", "container", "stop", "--yes", "ta-1"] in cli.calls


def test_stop_modal_app_gives_up_loudly_if_a_container_will_not_go() -> None:
    def stuck(argv: Sequence[str]) -> str:
        return (
            json.dumps([{"container_id": "ta-1", "app_name": "hostllm"}]) if "list" in argv else ""
        )

    with pytest.raises(RuntimeError, match="still running"):
        stop_modal_app("hostllm", stuck, wait=lambda _: None)


@pytest.fixture
def url() -> Iterator[str]:
    def engine(req: ChatRequest) -> Completion:
        return Completion("{}", [], 10, 4, "stop")

    scheduler = Sequential(engine)
    server = make_server(scheduler, ["base", "report-analyst"], "sk-test", "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    scheduler.close()


def test_main_benches_each_workload_at_each_level_and_writes_json(url: str, tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_text("sk-test\n")
    rows = write_rows(tmp_path / "f.jsonl", tools=True)
    out = tmp_path / "bench.json"

    main([
        "--url", url, "--api-key-file", str(key), "--workload", f"figures={rows}:report-analyst",
        "--levels", "1,2", "--requests", "3", "--out", str(out),
    ])  # fmt: skip

    result = json.loads(out.read_text())
    assert result["url"] == url and result["cold_seconds"] > 0
    levels = [(lv["workload"], lv["concurrency"]) for lv in result["levels"]]
    assert levels == [("figures", 1), ("figures", 2)]
    assert all(lv["errors"] == 0 and lv["tokens_per_second"] > 0 for lv in result["levels"])


def test_main_with_dry_run_sends_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = write_rows(tmp_path / "h.jsonl", tools=False)

    main([
        "--url", "http://127.0.0.1:9", "--workload", f"headlines={rows}:headline-flavour",
        "--levels", "1,4,16", "--requests", "32", "--dry-run",
    ])  # fmt: skip

    printed = capsys.readouterr().out
    assert "97 requests" in printed and "$" in printed, printed
