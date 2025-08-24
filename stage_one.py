#!/usr/bin/env python3

import json
from argparse import ArgumentParser
from functools import cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_core.rate_limiters import InMemoryRateLimiter

from solar.knowledge_acquisition import predict


@cache
def chat_model(model: str, no_cache: bool) -> BaseChatModel:
    kwargs, provider = {}, "openai"
    if not no_cache:
        set_llm_cache(SQLiteCache(database_path="./.cache_acquisition"))

    if model in ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]:
        kwargs["temperature"] = 0.0
    if model == "llama-3.3-70B":
        model = "accounts/fireworks/models/llama-v3p3-70b-instruct"
        provider = "fireworks"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.1,
        )
    if model == "llama-4-maverick":
        model = "accounts/fireworks/models/llama4-maverick-instruct-basic"
        provider = "fireworks"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.1,
        )
    if model == "deepseek-v3":
        model = "accounts/fireworks/models/deepseek-v3"
        provider = "fireworks"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.1,
        )
    if model == "qwen-2.5-72B":
        model = "accounts/fireworks/models/qwen2p5-72b-instruct"
        provider = "fireworks"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.1,
        )
    if model == "claude-3.7":
        model = "claude-3-7-sonnet-20250219"
        provider = "anthropic"
        kwargs["temperature"] = 0.0
    if model == "gemini-2.5-flash":
        model = "gemini-2.5-flash-preview-04-17"
        provider = "google_genai"
        kwargs["temperature"] = 0.0
    return init_chat_model(model, model_provider=provider, **kwargs)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--statute")
    parser.add_argument("--output-tbox")
    parser.add_argument("--output-py")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    with open(args.statute, "r") as f:
        statute = f.read()

    tbox, tbox_eval_code = predict(
        chat_model(args.model, args.no_cache),
        statute,
        args.debug,
    )
    print(f"Writing TBox to {args.output_tbox}.")
    with open(args.output_tbox, "w") as f:
        json.dump(tbox.model_dump(), f)

    print(f"Writing TBox evaluation code to {args.output_py}.")
    with open(args.output_py, "w") as f:
        f.write(tbox_eval_code)


if __name__ == "__main__":
    main()
