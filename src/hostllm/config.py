"""hostllm's settings, in one place."""

from __future__ import annotations

# The request (protocol): the defaults for what it leaves out, and the most it may ask for
SERVE_DEFAULT_MAX_TOKENS = 512
SERVE_MAX_TOKENS = 4096
SERVE_DEFAULT_TEMPERATURE = 1.0  # OpenAI's default; evals send 0

# The HTTP server
SERVE_HOST = "0.0.0.0"
SERVE_PORT = 8000
SERVE_API_KEY_ENV = "HOSTLLM_API_KEY"  # the bearer token clients send; unset means no auth

# Deployments (deploy/): one L4 per server, stopped after five idle minutes, never more than one
DEPLOY_GPU = "L4"
DEPLOY_SCALEDOWN_SECONDS = 300
DEPLOY_MAX_CONTAINERS = 1  # so a benchmark measures the scheduler, not the autoscaler
DEPLOY_STARTUP_SECONDS = 900  # the first start downloads the base into the cache volume
DEPLOY_MAX_MODEL_LEN = 8192  # bounds vLLM's KV cache to what one request here needs
DEPLOY_PORT = 8000
DEPLOY_CONCURRENT_INPUTS = 32  # requests one container accepts at once; the scheduler queues them
DEPLOY_PACKAGES = ("peft==0.21.2", "torch==2.14.1", "transformers==5.19.0")  # as uv.lock has them
DEPLOY_VLLM_PACKAGE = "vllm==0.11.0"
DEPLOY_CACHE_VOLUME = "hostllm-hf-cache"
DEPLOY_SECRET = "hostllm"  # holds HF_TOKEN and HOSTLLM_API_KEY
DEPLOY_PROXY_AUTH = True  # Modal refuses callers without a proxy token before a GPU starts
