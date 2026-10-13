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
