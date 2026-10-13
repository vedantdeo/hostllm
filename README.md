# hostllm

A from-scratch, OpenAI-compatible server for a causal LM and its LoRA adapters: one base model loaded
once, each request naming the adapter it wants. Built to be measured against vLLM, and to serve the
fine-tuned specialists from [underhood](https://github.com/vedantdeo/underhood) to the agent in
[aigent](https://github.com/vedantdeo/aigent).

Work in progress.
