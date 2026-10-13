"""The step profile on one Modal L4: the spec's base and adapters loaded as the server loads them,
then `profile.time_steps`, printed as a table. A run costs a few GPU-minutes.

HOSTLLM_SPEC=../underhood/hosting/specialists.json uv run modal run -m hostllm.deploy.modal_profile
"""

from __future__ import annotations

import modal

from hostllm import config
from hostllm.deploy.modal_app import SPEC, cache, image, secret

app = modal.App("hostllm-profile")


@app.function(
    image=image(*config.DEPLOY_PACKAGES),
    gpu=config.DEPLOY_GPU,
    volumes={"/root/.cache/huggingface": cache},
    secrets=[secret],
    timeout=config.PROFILE_TIMEOUT_SECONDS,
)
def steps() -> str:
    from hostllm.engine import load
    from hostllm.profile import table, time_steps

    engine = load(SPEC.base_model, SPEC.adapters)
    adapter = sorted(SPEC.adapters)[0]
    prompt = [engine.pad] * config.PROFILE_PROMPT_TOKENS
    timings = time_steps(
        engine.model, adapter, prompt, config.PROFILE_BATCHES, config.PROFILE_STEPS
    )
    return f"{SPEC.base_model}, adapter {adapter}, {config.DEPLOY_GPU}\n{table(timings)}"


@app.local_entrypoint()
def main() -> None:
    print(steps.remote())
