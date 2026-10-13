# Results

*Each bench run, newest last: what was measured, on what, and what it showed. Raw numbers in
`docs/bench/`. The plan behind them is `docs/plan.md`.*

## 2026-10-13: hostllm v1 against vLLM

**Setup.** One Modal L4 per server, at most one container, Qwen3-4B in bf16 with both adapters
(`headline-flavour`, `report-analyst`, rank 16 on all seven projections). `hostllm-bench`: one cold
request, then 32 requests per level at concurrency 1, 4 and 16, greedy, for two workloads. Headlines
are a ~574-token prompt and a ~36-token answer; figures are a ~1,600-token prompt with `calculate`
offered, and a first turn of ~30–50 tokens. Token rates count no stop token for either engine. vLLM
0.11.0 on transformers 4.57.1; hostllm v1, one request at a time. $0.32 for both runs.

| | hostllm p50 | vLLM p50 | hostllm tok/s | vLLM tok/s | hostllm $/1k | vLLM $/1k |
|---|---|---|---|---|---|---|
| headlines, c=1 | 5.1 s | 2.3 s | 6.7 | 15.6 | 1.15 | 0.50 |
| headlines, c=4 | 17.7 s | 2.2 s | 7.9 | 62.6 | 0.99 | 0.12 |
| headlines, c=16 | 70.3 s | 2.5 s | 7.8 | 210.0 | 1.00 | 0.037 |
| figures, c=1 | 5.8 s | 2.3 s | 6.0 | 15.4 | 1.31 | 0.50 |
| figures, c=4 | 19.1 s | 2.3 s | 7.3 | 60.0 | 1.08 | 0.13 |
| figures, c=16 | 77.8 s | 2.6 s | 7.0 | 176.8 | 1.12 | 0.044 |
| cold start | 33.5 s | 349.4 s | | | | |

**What it showed.**

- **One request at a time, vLLM is 2.3× faster:** fused kernels, CUDA graphs and batched LoRA
  kernels, against a Python decode loop and PEFT's unmerged LoRA matmuls on every projection. Even
  vLLM decodes only ~15 tok/s here, well below what an L4 manages for a plain 4B model.
- **At 16 in flight, vLLM is 27× faster and 27× cheaper per request.** Its latency barely moves
  (2.2 → 2.5 s) while throughput climbs almost linearly (15 → 63 → 210 tok/s): decoding is bound by
  reading the weights, so sixteen sequences in one forward pass cost little more than one. hostllm's
  throughput is flat at ~7 tok/s and its latency grows with the queue, as a sequential scheduler must.
- **The cold start runs the other way: hostllm is 10× faster** (34 s against 349 s). vLLM profiles
  memory and compiles CUDA graphs before it serves; an earlier start the same day took 183 s.
- **So the engine depends on the traffic.** Rare requests to a server that scales to zero pay the cold
  start each time, and hostllm wins (~40 GPU-seconds against ~350). Sustained load is vLLM's by more
  than an order of magnitude.
- **Both engines answered the smoke cases identically:** the same records, and the same prompt token
  counts, for a headline and for a two-turn `calculate` loop.

**Next:** v2, static batching per adapter, to show how much of the 27× is batching alone; and a
profile of hostllm's ~150 ms per token.
