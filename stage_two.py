#!/usr/bin/env python3

import time
import json
from argparse import ArgumentParser
from functools import cache, partial
from pathlib import Path
from typing import Literal, Callable, Tuple
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_core.globals import set_llm_cache
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_community.cache import SQLiteCache
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.utilities import PythonREPL
from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel, Field
from tqdm import tqdm

from solar.model import TBox
from solar.knowledge_application import predict as predict_solar_core


@dataclass
class QueryMetrics:
    result: int | None
    duration: float
    tokens: int
    error: str | None = None


@dataclass
class SummaryStats:
    total_queries: int
    successful_predictions: int
    accuracy: float
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    total_tokens: int
    avg_tokens: float
    min_tokens: int
    max_tokens: int
    tokens_per_second: float
    error_count: int


class MonetaryResult(BaseModel):
    currency: Literal["USD", "GBP"] = Field(
        default="USD", description="The currency code"
    )
    amount: int = Field(
        ...,
        description="The monetary amount with no decimal places - rounded to the nearest integer",
    )


class TokenCountingCallback(BaseCallbackHandler):
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_tokens = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.current_tokens = 0

    def on_llm_end(self, response, **kwargs):
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            if token_usage:
                self.current_tokens = token_usage.get("total_tokens", 0)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--mode", default="zero-shot")
    parser.add_argument("--dataset", default="test")
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=10.0,
        help="threshold percentage for considering a prediction correct",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument(
        "--tbox-path",
        default="./sara_tbox_o1.json",
        help="path to the TBox JSON file for solar prediction",
    )
    parser.add_argument(
        "--tbox-interpreter-path",
        default="./sara_eval_o1.py",
        help="path to the TBox interpreter Python file for solar prediction",
    )
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


@cache
def chat_model(model: str, no_cache: bool) -> BaseChatModel:
    kwargs, provider = {}, "openai"
    if not no_cache:
        set_llm_cache(SQLiteCache(database_path="./.cache"))

    if model in ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]:
        kwargs["temperature"] = 0.0
    if model == "llama-3.3-70B":
        model = "accounts/fireworks/models/llama-v3p3-70b-instruct"
        provider = "fireworks"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(requests_per_second=0.1)
    if model == "llama-4-maverick":
        model = "accounts/fireworks/models/llama4-maverick-instruct-basic"
        provider = "fireworks"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(requests_per_second=0.1)
    if model == "deepseek-v3":
        model = "accounts/fireworks/models/deepseek-v3"
        provider = "fireworks"
        kwargs["temperature"] = 0.0
    if model == "qwen-2.5-72B":
        model = "accounts/fireworks/models/qwen2p5-72b-instruct"
        provider = "fireworks"
        kwargs["temperature"] = 0.0
    if model == "claude-3.7":
        model = "claude-3-7-sonnet-20250219"
        provider = "anthropic"
        kwargs["temperature"] = 0.0
    if model == "gemini-2.5-flash":
        model = "gemini-2.5-flash-preview-04-17"
        provider = "google_genai"
        kwargs["temperature"] = 0.0

    return init_chat_model(model, model_provider=provider, **kwargs)


def load_data(task: str, filename: str):
    return pd.read_csv(
        Path("./legalbench/data") / task / filename,
        sep="\t",
        index_col="index",
    )


def predict_with_metrics(
    prediction_func: Callable[[str, str, bool, bool], int | None],
    model_name: str,
    task_data: str,
    debug: bool,
    no_cache: bool,
) -> QueryMetrics:
    start_time, tokens_used = time.time(), 0
    result, error = None, None

    try:
        if "openai" in model_name.lower() or "gpt" in model_name.lower():
            with get_openai_callback() as cb:
                result = prediction_func(model_name, task_data, debug, no_cache)
                tokens_used = cb.total_tokens
        else:
            token_callback = TokenCountingCallback()
            model = chat_model(model_name, no_cache)

            if hasattr(model, "callbacks"):
                model.callbacks = [token_callback]

            result = prediction_func(model_name, task_data, debug, no_cache)
            tokens_used = token_callback.current_tokens

    except Exception as ex:
        error = str(ex)
        if debug:
            print(f"Error in prediction: {ex}")
        result = None

    duration = time.time() - start_time

    return QueryMetrics(
        result=result, duration=duration, tokens=tokens_used, error=error
    )


def predict_zero_shot(model: str, task: str, debug: bool, no_cache: bool) -> int | None:
    res = (
        chat_model(model, no_cache)
        .with_structured_output(MonetaryResult, method="json_schema")
        .invoke(task)
    )
    if debug:
        print(f"RES: {res}")
    if isinstance(res, MonetaryResult):
        return res.amount
    else:
        print("err: failed to parse evaluation result")
        return None


def predict_chain_of_code(
    model: str, task: str, debug: bool, no_cache: bool
) -> int | None:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful assistant that solves problems using Chain-of-Code methodology. "
                    "Break down the problem into logical steps and write Python code to solve each step. "
                    "For each step:\n"
                    "1. Explain what you're doing (as a Python comment)\n"
                    "2. Write the code to solve that step\n"
                    "3. Show intermediate results\n\n"
                    "Your response should be a complete Python program that:\n"
                    "- Shows step-by-step reasoning through comments\n"
                    "- Breaks down the problem into manageable parts\n"
                    "- Calculates intermediate values\n"
                    "- Ends with exactly one `print(result)` statement where result is a number\n\n"
                    "Example structure:\n"
                    "# Step 1: Parse the input and identify key values\n"
                    "# Step 2: Apply relevant calculations\n"
                    "# Step 3: Combine results to get final answer\n"
                    "print(result)"
                ),
            ),
            ("human", "Problem to solve: {input}"),
        ]
    )
    python_repl = PythonREPL()
    res = chat_model(model, no_cache).invoke(prompt.format(input=task))
    if debug:
        print(f"Chain-of-Code Response: {res.content}")
    if not res.content or not isinstance(res.content, str):
        print(f"err: empty response {res.content}")
        return None

    eval_res = python_repl.run(res.content)
    if debug:
        print(f"Code Execution Result: {eval_res}")
    if not eval_res:
        print(f"err: empty evaluation result {eval_res}")
        return None

    try:
        return int(float(eval_res.strip()))
    except Exception as ex:
        print(f"err when evaluating chain-of-code python: {ex}")
        return None


def predict_solar(
    model: str,
    task: str,
    debug: bool,
    no_cache: bool,
    tbox_path: str,
    tbox_interpreter_path: str,
) -> int | None:
    with open(tbox_path, "r") as f:
        tbox = TBox.model_validate(json.load(f))
    with open(tbox_interpreter_path, "r") as f:
        tbox_interpreter = f.read()

    try:
        task_data = (
            json.loads(task)
            if isinstance(task, str) and task.startswith("{")
            else {"text": task}
        )
        description = task_data.get("description", "")
        question = task_data.get("question", task_data.get("text", ""))
    except json.JSONDecodeError:
        description, question = "", task

    if debug:
        print(f"Description: {description}")
        print(f"Question: {question}")

    result, _ = predict_solar_core(
        chat_model(model, no_cache),
        description,
        question,
        tbox,
        tbox_interpreter,
        debug,
    )
    return int(result)


def get_prediction_method(
    mode: str, tbox_path: str | None = None, tbox_interpreter_path: str | None = None
) -> Callable[[str, str, bool, bool], int | None]:
    methods = {
        "zero-shot": predict_zero_shot,
        "chain-of-code": predict_chain_of_code,
    }

    if mode == "solar":
        if not tbox_path or not tbox_interpreter_path:
            raise ValueError(
                "tbox_path and tbox_interpreter_path are required for solar mode"
            )
        return partial(
            predict_solar,
            tbox_path=tbox_path,
            tbox_interpreter_path=tbox_interpreter_path,
        )
    elif mode in methods:
        return methods[mode]
    else:
        raise ValueError(
            f"Unsupported prediction mode: {mode}. Available: {list(methods.keys()) + ['solar']}"
        )


def calculate_success_metric(
    true_vals: list[int], predictions: list[int | None], threshold: float
) -> Tuple[float, int]:
    if len(true_vals) != len(predictions):
        raise ValueError("length of true_vals and predictions must be the same")

    valid_pairs = [
        (true, pred) for true, pred in zip(true_vals, predictions) if pred is not None
    ]

    if not valid_pairs:
        return (0.0, 0)

    correct_count = 0
    for true_val, pred_val in valid_pairs:
        err_margin = abs(true_val) * (threshold / 100.0)
        if abs(true_val - pred_val) <= err_margin:
            correct_count += 1

    return (correct_count / len(valid_pairs), len(valid_pairs))


def calculate_summary_stats(
    metrics: list[QueryMetrics], true_vals: list[int], threshold: float
) -> SummaryStats:
    predictions = [m.result for m in metrics]
    durations = [m.duration for m in metrics]
    tokens = [m.tokens for m in metrics if m.tokens > 0]
    errors = [m for m in metrics if m.error is not None]

    accuracy, successful = calculate_success_metric(true_vals, predictions, threshold)

    return SummaryStats(
        total_queries=len(metrics),
        successful_predictions=successful,
        accuracy=accuracy,
        total_time=sum(durations),
        avg_time=sum(durations) / len(durations) if durations else 0,
        min_time=min(durations) if durations else 0,
        max_time=max(durations) if durations else 0,
        total_tokens=sum(tokens),
        avg_tokens=sum(tokens) / len(tokens) if tokens else 0,
        min_tokens=min(tokens) if tokens else 0,
        max_tokens=max(tokens) if tokens else 0,
        tokens_per_second=sum(tokens) / sum(durations)
        if durations and sum(durations) > 0
        else 0,
        error_count=len(errors),
    )


def print_summary_stats(stats: SummaryStats, total_data_points: int):
    print(f"\n--- Performance Summary ---")
    print(f"Total predictions: {stats.successful_predictions} / {total_data_points}")
    print(f"Accuracy: {stats.accuracy:.3f}")
    print(f"Errors: {stats.error_count}")

    print(f"\n--- Timing Statistics ---")
    print(f"Total queries processed: {stats.total_queries}")
    print(f"Total time: {stats.total_time:.3f}s")
    print(f"Average time per query: {stats.avg_time:.3f}s")
    print(f"Min time per query: {stats.min_time:.3f}s")
    print(f"Max time per query: {stats.max_time:.3f}s")

    if stats.total_tokens > 0:
        print(f"\n--- Token Usage Statistics ---")
        print(f"Total tokens used: {stats.total_tokens}")
        print(f"Average tokens per query: {stats.avg_tokens:.1f}")
        print(f"Min tokens per query: {stats.min_tokens}")
        print(f"Max tokens per query: {stats.max_tokens}")
        print(f"Tokens per second: {stats.tokens_per_second:.1f}")
    else:
        print(f"\n--- Token Usage Statistics ---")
        print("No token usage data available (may need to configure callbacks)")


def main():
    args = parse_args()
    data = load_data("sara_numeric", f"{args.dataset}.tsv")
    load_dotenv()

    prediction_method = get_prediction_method(
        args.mode, args.tbox_path, args.tbox_interpreter_path
    )
    metrics: list[QueryMetrics] = []
    true_vals: list[int] = []

    for _, row in tqdm(data.iterrows(), total=len(data)):
        if args.limit >= 0 and len(metrics) >= args.limit:
            print(f"Limit reached: {args.limit}")
            break

        if args.mode == "solar":
            task_data = json.dumps(
                {
                    "description": row.get("description", ""),
                    "question": row.get("question", row.get("text", "")),
                }
            )
        else:
            task_data = row["text"]

        query_metrics = predict_with_metrics(
            prediction_method, args.model, task_data, args.debug, args.no_cache
        )

        metrics.append(query_metrics)
        true_vals.append(int(row["answer"].replace("$", "")))

        if args.debug:
            print(f"Expected answer: {row['answer']}")
            print(f"Predicted: {query_metrics.result}")
            print(f"Query time: {query_metrics.duration:.3f}s")
            print(f"Tokens used: {query_metrics.tokens}")
            if query_metrics.error:
                print(f"Error: {query_metrics.error}")

    stats = calculate_summary_stats(metrics, true_vals, args.success_threshold)

    predictions = [m.result for m in metrics]
    print(f"\nPredictions: {predictions}")
    print(f"True values: {true_vals}")

    print_summary_stats(stats, len(data))


if __name__ == "__main__":
    main()
