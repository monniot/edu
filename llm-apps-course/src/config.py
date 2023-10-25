"""Configuration for the LLM Apps Course"""
from types import SimpleNamespace

TEAM = None
PROJECT = "llmapps_cacib"
JOB_TYPE = "production"

default_config = SimpleNamespace(
    project=PROJECT,
    entity=TEAM,
    job_type=JOB_TYPE,
    vector_store_artifact="pmon/llmapps_cacib/vector_store:latest",
    chat_prompt_artifact="pmon/llmapps_cacib/chat_prompt:latest",
    chat_temperature=0.3,
    max_fallback_retries=1,
    model_name="gpt-4",
    eval_model="gpt-3.5-turbo",
    eval_artifact="pmon/llmapps/generated_examples:v0",
)