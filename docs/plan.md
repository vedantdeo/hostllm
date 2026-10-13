# The serving comparison

*Decided 2026-10-13. hostllm's decision record: why it exists, what it is measured against, and what
each choice was.*

## What is served

One base, Qwen3-4B, with two LoRA adapters trained in underhood: the **headline reader**, served as
`headline-flavour` (a market headline → one structured record), and the **report analyst**, served as
`report-analyst` (filing passages → figures, calling `calculate`). Each is a private Hub repo of the
same name under `vistack/` on the Hub. aigent's capstone routes narrow jobs to them and measures what routing saves.

## Three rungs: build, configure, buy

Serverless serving comes in three kinds, split by who runs the inference engine. The comparison takes
one of each, on the same base and adapters:

| Rung | What | Kind | Teaches |
|---|---|---|---|
| **build** | hostllm, our own server, on Modal | serverless GPU container: we bring the server | how a server works, and why ours is slow |
| **configure** | vLLM on Modal; then JarvisLabs Serverless | the same container with a real engine; a managed engine we only configure | what continuous batching, PagedAttention and batched multi-LoRA buy; what managed hosting adds |
| **buy** | per-token LoRA hosting (Fireworks or Together, whichever takes Qwen3-4B adapters) | someone else's GPU, shared base, billed per token | what it costs to own no GPU at all |

Modal was chosen over Beam, Cerebrium and Cloud Run for the container rung: they are close, and Modal
has the most documentation. Only a container platform can run our own server.

## hostllm's own iterations

1. **v1, sequential:** one request at a time. The baseline.
2. **v2, static batching:** whatever is waiting goes through one left-padded decode, each row on its
   own adapter through PEFT's per-row `adapter_names` (decided 2026-10-13 over one adapter per batch,
   which underuses the GPU on mixed traffic). Each row's output equals that request answered alone.
   Each row's `position_ids` count from its first real token; no test can tell them from the shifted
   positions HF would assume, because Qwen3's RoPE scores depend only on relative position, so they
   are kept for numerical closeness, untested. A finished row's input and mask are never read.
   A batch of one takes v1's single-request path (2026-10-13): the first v2 bench ran ~20% slower
   than v1 one at a time, and the per-row path is the only difference. The scheduler logs each
   batch's size, longest answer and seconds, so a bench run shows what batches a level really formed.
3. Continuous batching is out of scope (a Week 7 stretch at most); vLLM is the reference for it.

Each version is benchmarked, so the gap to vLLM is split into what batching alone is worth and what
remains.

## What a served token costs

*Decided 2026-10-13, Vedant's calls: the price lives here, underhood imports it, and it is the
one-at-a-time rate.*

`hostllm.pricing.SERVED` is the L4's $0.80 an hour over the engine's time for one request alone, from
the step profile: 169.6 ms to prefill 574 tokens and 117.3 ms a step on one adapter, so **$0.066 per
million prompt tokens and $26.07 per million answer tokens**. One at a time is how a single agent
loop calls the endpoint, and it is the dearest rate: at 16 in flight a headline call cost a quarter
as much. It is engine time only. The bench's wall-clock figure, which adds HTTP and Modal's proxy, is
~15% higher ($1.13 against $0.98 per thousand headline calls), and neither counts cold starts or the
idle minutes before the endpoint scales down.

So that underhood, and aigent through it, can import the price without torch, the package's
`dependencies` are empty and the engine's libraries are the `serve` group, installed by default
here. Re-measure and edit the three `PRICE_` settings when the engine or the GPU changes.

## What is measured

- **Speed and cost** (`bench`): cold start, p50 and p95 latency, throughput at concurrency 1, 4 and
  16, and cost per thousand requests.
- **Correctness:** the 50 headlines and 43 figure cases graded through each endpoint by aigent, which
  must match the offline scores. This is the train/serve skew check.

Paid steps — each deploy and each bench run against a live endpoint — need their own go-ahead, with
the dry-run estimate first.
