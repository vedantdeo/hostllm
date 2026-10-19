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

## 2026-10-13: hostllm v2, batching

**Setup.** As above, with the `Batching` scheduler: a 10 ms window, up to 16 a batch, each row on its
own adapter through PEFT's `adapter_names`. The first attempt finished only the headline levels: **Modal
preempted the GPU container** after 10 minutes (14 of the figure level's 32 requests answered, then
it vanished without a log line; GPU functions cannot be made non-preemptible), and the bench waited out
the 1,200 s it gave every request on the one the preemption dropped. The bench now gives the cold
request 900 s and every later one 300 s, so a dropped request is an error and the run goes on. The
rerun below finished with no errors. Any client of a preemptible endpoint, aigent's included, needs a
timeout sized to warm latency and a retry. Both v2 runs: ~$0.40.

| tok/s (p50) | v1 | v2 | vLLM | v2 ÷ v1 |
|---|---|---|---|---|
| headlines, c=1 | 6.7 (5.1 s) | 5.2 (6.7 s) | 15.6 (2.3 s) | 0.8× |
| headlines, c=4 | 7.9 (17.7 s) | 11.0 (12.6 s) | 62.6 (2.2 s) | 1.4× |
| headlines, c=16 | 7.8 (70.3 s) | 29.9 (17.7 s) | 210.0 (2.5 s) | 3.8× |
| figures, c=1 | 6.0 (5.8 s) | 4.9 (7.2 s) | 15.4 (2.3 s) | 0.8× |
| figures, c=4 | 7.3 (19.1 s) | 8.5 (16.1 s) | 60.0 (2.3 s) | 1.2× |
| figures, c=16 | 7.0 (77.8 s) | 15.1 (32.9 s) | 176.8 (2.6 s) | 2.2× |
| cold start | 33.5 s | 109.0 s | 349.4 s | |

**What it showed.**

- **Static batching is worth 2–4× at 16 in flight**: 3.8× on headlines, 2.2× on figures, with p50
  falling from 70 s to 18 s and from 78 s to 33 s. That is the share of vLLM's lead batching alone
  buys; vLLM is still 7× ahead on headlines and 12× on figures.
- **Figures gain less than headlines.** Their prompts are ~3× longer, so a padded batch spends more on
  prefill, and a static batch runs until its longest answer ends while finished rows idle — the waste
  continuous batching removes by refilling a slot the moment a row finishes.
- **One at a time, v2 is ~20% slower than v1** (6.7 s against 5.1 s). Unmeasured, but the likely cause
  is PEFT's per-row `adapter_names` path, which splits each layer's LoRA by adapter on every step even
  for a batch of one. If a profile confirms it, a batch of one should take the v1 path.
- **Cold starts vary:** 109 s here against v1's 34 s on the same image and cached weights. Nothing in
  hostllm changed what loads at start, so the spread is likely Modal's (which machine, what is cached
  on it); cold-start numbers need several samples before they are compared.
- **Run to run:** the two v2 runs' headline levels differ by up to ~25% (24.0 against 29.9 tok/s at 16
  in flight); a gap smaller than that is not a finding.

**Next:** log batch sizes; send a batch of one down the single-request path; profile the ~150 ms per
token; then the configure rung (JarvisLabs Serverless) and the buy rung (per-token LoRA hosting).

## 2026-10-13: where a decode step's time goes

**Setup.** `hostllm.deploy.modal_profile` on one L4: the engine loaded as the server loads it, then a
574-token prompt (a headline's length) prefilled and 32 greedy steps timed, on three paths — the bare
base, `headline-flavour` set as the one active adapter, and the same adapter named per row through
`adapter_names` — at batch 1, 4 and 16. One run, ~$0.05. Raw table in
`docs/bench/2026-10-13-profile.txt`.

| ms per step (prefill) | batch 1 | batch 4 | batch 16 |
|---|---|---|---|
| base | 66 (120) | 64 (427) | 63 (1,981) |
| one adapter | 117 (170) | 116 (752) | 118 (3,466) |
| an adapter per row | 226 (308) | 221 (838) | 225 (3,752) |

**What it showed.**

- **A step costs the same at 1 row as at 16.** On every path the step time is flat across batch
  sizes, so a step is bound by per-layer overhead, not by arithmetic: sixteen rows are nearly free.
  Prefill is the opposite — it grows in step with the tokens fed, 2–4 s for sixteen headline prompts.
- **Unmerged LoRA nearly doubles a step** (66 → 117 ms): PEFT runs two extra small matmuls on each
  of seven projections in each layer. The bare base's 66 ms is vLLM's one-at-a-time rate (15 tok/s).
- **The per-row path doubles it again** (117 → 226 ms), at any batch size. That is the cost v2 paid
  on every step, and the reason a batch of one now takes the single-request path.
- **So two changes are known to be worth ~2× each, and are not made yet:** a batch whose rows all
  name one adapter can set it once instead of naming it per row (226 → 117 ms), and merged adapter
  weights would decode at the base's speed (117 → 66 ms) at the price of a copy of the base each.

## 2026-10-13: v2 again, a batch of one on the single-request path, batches logged

**Setup.** The v2 bench as before, after two changes: a lone request takes v1's path, and the
scheduler logs each batch's size, longest answer and seconds. One run, no errors, ~14 minutes of L4
(~$0.19). Raw numbers in `docs/bench/2026-10-13-ours-v2-single-path.json`; batch sizes read from
Modal's logs (`modal app logs hostllm --search "batch of"`).

| tok/s (p50) | v1 | v2, first run | v2, this run | batches formed |
|---|---|---|---|---|
| headlines, c=1 | 6.7 (5.1 s) | 5.2 (6.7 s) | 6.9 (5.1 s) | 32 of 1 |
| headlines, c=4 | 7.9 (17.7 s) | 11.0 (12.6 s) | 10.4 (13.3 s) | 8 of 1, 8 of 3 |
| headlines, c=16 | 7.8 (70.3 s) | 29.9 (17.7 s) | 28.9 (18.3 s) | 2 of 1, 2 of 15 |
| figures, c=1 | 6.0 (5.8 s) | 4.9 (7.2 s) | 6.2 (5.5 s) | 32 of 1 |
| figures, c=4 | 7.3 (19.1 s) | 8.5 (16.1 s) | 7.9 (17.2 s) | 8 of 1, 8 of 3 |
| figures, c=16 | 7.0 (77.8 s) | 15.1 (32.9 s) | 14.2 (33.9 s) | 2 of 1, 2 of 15 |
| cold start | 33.5 s | 109.0 s | 90.4 s | |

**What it showed.**

- **One at a time, v2 now matches v1** (5.1 s, 6.9 tok/s on headlines; 5.5 s on figures): the ~20%
  it lost was the per-row adapter path, as the profile above measured.
- **Under load nothing moved**, within the ~25% run-to-run spread, as expected: those batches still
  take the per-row path.
- **No level ever formed a full batch.** Every loaded level alternates a batch of one with a batch
  of the rest (1 and 3, 1 and 15). When a batch ends, the one request left waiting is picked up and
  the 10 ms window closes before the just-answered clients' next requests arrive, so it decodes
  alone while they queue behind it. At 16 in flight about a quarter of the time (4.3 s of each ~19 s
  cycle) serves one request. A window longer than a client's round trip would end the convoy.
- **A batch of 15 headlines takes ~15 s**: ~3.7 s of prefill, then ~40 steps at ~270 ms, in line
  with the profile's 225 ms per-row step.

**Parked (2026-10-13, Vedant's call):** no further inference optimisation until the capstone is
built on what is served now. Where to start afterwards, cheapest first: a longer batch window (a
setting); setting the adapter once for a batch whose rows share it (~2× a step); merged adapter
weights (~2× again).

## 2026-10-19: what a served token costs, per engine

`hostllm.pricing` prices a token as the L4's $0.80 an hour over the engine time it takes. No new
run: both rates come from the measurements above.

| | prompt, $/M | answer, $/M | from |
|---|---|---|---|
| `SERVED`, hostllm | 0.066 | 26.07 | the step profile, one request at a time (169.6 ms per 574 prompt tokens, 117.3 ms a step) |
| `VLLM_SERVED`, vLLM | 0.066 | 3.70 | the v1-vs-vLLM bench: figures at 60.0 tok/s with four in flight, the capstone's `ANALYST_WORKERS` |

vLLM's prefill was not measured apart from its decode, so its prompt rate is ours, an upper bound.
Both rates assume a busy GPU: the seconds a container idles between calls, and its cold start, are
billed by Modal and in neither.
