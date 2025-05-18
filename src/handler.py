import os
import runpod
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine

vllm_engine = vLLMEngine()
OpenAIvLLMEngine = OpenAIvLLMEngine(vllm_engine)

async def handler(job):
    job_input = JobInput(job["input"])
    engine = OpenAIvLLMEngine if job_input.openai_route else vllm_engine
    results_generator = engine.generate(job_input)

    # ⬇️ On récupère toutes les batches dans une liste
    collected = []
    async for batch in results_generator:
        collected.append(batch)

    # ⬇️ On retourne l'output final avec status="COMPLETED"
    return {
        "output": collected[-1] if len(collected) == 1 else collected,
        "status": "COMPLETED"
    }

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)
