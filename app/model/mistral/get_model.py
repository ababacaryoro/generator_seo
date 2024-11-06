import os
from dotenv import load_dotenv

# from langchain.llms.vllm import VLLMOpenAI
from langchain_community.llms import VLLMOpenAI
import datetime
import logging


def instantiate_mixtral() -> VLLMOpenAI:
    """Instantiates the Mixtral LLM."""

    llm = VLLMOpenAI(
        openai_api_key=os.environ["MIXTRAL_API_KEY"],
        openai_api_base=os.environ["MIXTRAL_API_BASE"],
        model_name="mistralai/Mixtral-8X7B-Instruct-v0.1",
        temperature=0.1,
        max_tokens=5000,
        verbose=True,
    )
    return llm
