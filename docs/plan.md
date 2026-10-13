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
2. **v2, static batching:** waiting requests that share an adapter go through one padded generate.
3. Continuous batching is out of scope (a Week 7 stretch at most); vLLM is the reference for it.

Each version is benchmarked, so the gap to vLLM is split into what batching alone is worth and what
remains.

## What is measured

- **Speed and cost** (`bench`): cold start, p50 and p95 latency, throughput at concurrency 1, 4 and
  16, and cost per thousand requests.
- **Correctness:** the 50 headlines and 43 figure cases graded through each endpoint by aigent, which
  must match the offline scores. This is the train/serve skew check.

Paid steps — each deploy and each bench run against a live endpoint — need their own go-ahead, with
the dry-run estimate first.
