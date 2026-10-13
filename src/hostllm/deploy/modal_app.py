"""Both servers on Modal, each an L4 web endpoint that scales to zero: `ours`, hostllm itself, and
`vllm`, the reference, hosting the spec at HOSTLLM_SPEC. Deploying costs nothing; requests do.

HOSTLLM_SPEC=../underhood/specialists.json uv run modal deploy -m hostllm.deploy.modal_app
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict

import modal

from hostllm import config
from hostllm.deploy.spec import SPEC_JSON_ENV, spec_from_env

SPEC = spec_from_env(os.environ)
CARRIED = {
    SPEC_JSON_ENV: json.dumps(asdict(SPEC))
}  # the image carries the spec it was deployed with

# each server: one L4, stopped after DEPLOY_SCALEDOWN_SECONDS idle, the Hub cache on a volume
app = modal.App("hostllm")
cache = modal.Volume.from_name(config.DEPLOY_CACHE_VOLUME, create_if_missing=True)
secret = modal.Secret.from_name(config.DEPLOY_SECRET)


def image(*packages: str) -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(*packages)
        .env(CARRIED)
        .add_local_python_source("hostllm")
    )


@app.function(
    image=image(*config.DEPLOY_PACKAGES),
    gpu=config.DEPLOY_GPU,
    volumes={"/root/.cache/huggingface": cache},
    secrets=[secret],
    scaledown_window=config.DEPLOY_SCALEDOWN_SECONDS,
    timeout=config.DEPLOY_STARTUP_SECONDS,
    max_containers=config.DEPLOY_MAX_CONTAINERS,
)
@modal.concurrent(max_inputs=config.DEPLOY_CONCURRENT_INPUTS)
@modal.web_server(
    config.DEPLOY_PORT,
    startup_timeout=config.DEPLOY_STARTUP_SECONDS,
    requires_proxy_auth=config.DEPLOY_PROXY_AUTH,
)
def ours() -> None:
    subprocess.Popen(SPEC.ours_command(config.DEPLOY_PORT))


@app.function(
    image=image(config.DEPLOY_VLLM_PACKAGE),
    gpu=config.DEPLOY_GPU,
    volumes={"/root/.cache/huggingface": cache},
    secrets=[secret],
    scaledown_window=config.DEPLOY_SCALEDOWN_SECONDS,
    timeout=config.DEPLOY_STARTUP_SECONDS,
    max_containers=config.DEPLOY_MAX_CONTAINERS,
)
@modal.concurrent(max_inputs=config.DEPLOY_CONCURRENT_INPUTS)
@modal.web_server(
    config.DEPLOY_PORT,
    startup_timeout=config.DEPLOY_STARTUP_SECONDS,
    requires_proxy_auth=config.DEPLOY_PROXY_AUTH,
)
def vllm() -> None:
    key = os.environ.get(config.SERVE_API_KEY_ENV, "")
    subprocess.Popen(SPEC.vllm_command(config.DEPLOY_PORT), env={**os.environ, "VLLM_API_KEY": key})
