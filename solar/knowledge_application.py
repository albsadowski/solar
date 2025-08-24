from functools import cache, partial
from typing import Callable

import nltk.sem.logic as logic
from langchain.chat_models.base import BaseChatModel
from langgraph.graph import StateGraph, END, START
from nltk.inference import ResolutionProver
from pydantic import BaseModel, Field

from .model import ABox, TBox, ExtractionEnvelope, ABoxAssertion
from .prompts import FACT_EXTRACTION_PROMPT, ANSWER_GENERATION_PROMPT
from .validation import validate_rule


class State(BaseModel):
    description: str
    query: str
    tbox: TBox
    tbox_callable: Callable[..., object]
    abox: ABox | None = None
    inferred_facts: list[ExtractionEnvelope[ABoxAssertion]] | None = None
    answer: float | None = None


class FactExtractionResult(BaseModel):
    abox: ABox = Field(..., description="Final assertion box.")


class QueryItem(BaseModel):
    taxpayer_id: str = Field(..., description="Taxpayer identifier")
    year: int = Field(..., description="Year of calculation")


class QueryAnalysisResult(BaseModel):
    query: QueryItem


def normalize_abox(tbox: TBox, abox: ABox) -> ABox:
    numeric_properties = set()
    for prop in tbox.properties:
        if prop.type == "datatype" and len(prop.arguments) > 1:
            data_type = prop.arguments[1].lower()
            if prop.arguments[1].lower() in ["decimal", "integer", "float", "number"]:
                numeric_properties.add(prop.name)
    normalized_assertions = []
    for assertion_env in abox.assertions:
        assertion = assertion_env.object
        if (
            assertion.property_name in numeric_properties
            and len(assertion.arguments) > 1
        ):
            args = assertion.arguments.copy()
            value_str = args[1]
            if isinstance(value_str, str):
                cleaned = (
                    value_str.replace("$", "")
                    .replace("USD", "")
                    .replace(",", "")
                    .strip()
                )
                try:
                    float_value = float(cleaned)
                    args[1] = str(float_value)
                    normalized_assertions.append(
                        ExtractionEnvelope(
                            object=ABoxAssertion(
                                property_name=assertion.property_name,
                                arguments=args,
                            ),
                            confidence=assertion_env.confidence,
                            explanation=assertion_env.explanation
                            + " (numeric value normalized)",
                        )
                    )
                except ValueError as err:
                    print(
                        f"err: conversion failed {value_str} for property {assertion} :: {err}"
                    )
            else:
                normalized_assertions.append(assertion_env)
        else:
            normalized_assertions.append(assertion_env)
    return ABox(
        individuals=abox.individuals,
        assertions=normalized_assertions,
    )


@cache
def prover():
    return ResolutionProver()


def fact_extraction_node(state: State, llm: BaseChatModel, debug: bool):
    if debug:
        print("E fact_extraction_node")

    properties_str = ""
    for p in state.tbox.properties:
        if p.type == "unary":
            properties_str += f"- {p.name}({p.arguments[0]}): {p.description}\n"
        else:
            properties_str += (
                f"- {p.name}({p.arguments[0]}, {p.arguments[1]}): {p.description}\n"
            )

    res = llm.with_structured_output(FactExtractionResult).invoke(
        FACT_EXTRACTION_PROMPT.format(
            description=state.description,
            classes="\n".join(
                [f"- {c.name}: {c.description}" for c in state.tbox.classes]
            ),
            properties=properties_str,
            tbox_interpreter_docstring=state.tbox_callable.__doc__,
        )
    )
    if isinstance(res, FactExtractionResult):
        if debug:
            print("X fact_extraction_node")
        return {"abox": normalize_abox(state.tbox, res.abox)}
    else:
        print("err: fact extraction failed")
        return {}


def symbolic_inference_node(state: State, debug: bool):
    def create_predicate(property_name, args):
        args_str = ", ".join(
            [f"'{arg}'" if isinstance(arg, str) else str(arg) for arg in args]
        )
        expr_str = f"{property_name}({args_str})"
        try:
            return logic.Expression.fromstring(expr_str)
        except Exception as e:
            print(
                f"err when building expression for '{property_name}', args: '{args}' :: {e}"
            )
            return None

    kb_expressions, inferred_facts = [], []

    if state.abox is not None:
        for env_ind in state.abox.individuals:
            ind = env_ind.object
            if predicate := create_predicate(ind.class_name, [ind.name]):
                kb_expressions.append(predicate)
                if debug:
                    print(f"Added class assertion: {ind.class_name}({ind.name})")

        for env_assert in state.abox.assertions:
            assert_obj = env_assert.object
            if predicate := create_predicate(
                assert_obj.property_name, assert_obj.arguments
            ):
                kb_expressions.append(predicate)
                if debug:
                    print(
                        f"Added property assertion: {assert_obj.property_name}{assert_obj.arguments}"
                    )

    class_names = {cls.name for cls in state.tbox.classes}
    prop_names = {prop.name for prop in state.tbox.properties}
    for ix, rule in enumerate(state.tbox.rules):
        issues = validate_rule(ix, rule, class_names, prop_names)
        if issues:
            print(f"warn: not a valid rule '{rule.fol_expression}'")
            continue

        premises = logic.Expression.fromstring(rule.fol_expression)

        try:
            if prover().prove(premises, kb_expressions):
                if debug:
                    print(f"rule: '{rule.fol_expression}' evaluated to true")
                inferred = ExtractionEnvelope(
                    object=ABoxAssertion(
                        property_name=rule.implication_property_name,
                        arguments=rule.implication_property_arguments,
                    ),
                    confidence=1.0,
                    explanation=f"inferred using rule: {rule.description}. The premises '{rule.fol_expression}' were satisfied.",
                )
                inferred_facts.append(inferred)
        except Exception as e:
            print(
                f"err when evaluating premises '{premises}' against {kb_expressions} :: {e}"
            )
    return {"inferred_facts": inferred_facts}


def answer_generation_node(state: State, llm: BaseChatModel, debug: bool):
    if debug:
        print("E answer_generation_node")
    if not state.abox:
        return {}

    res = llm.with_structured_output(QueryAnalysisResult).invoke(
        ANSWER_GENERATION_PROMPT.format(
            query=state.query,
            individuals="\n".join(
                [
                    f"- {ind.object.name} (type: {ind.object.class_name})"
                    for ind in state.abox.individuals
                ]
            ),
        )
    )
    if isinstance(res, QueryAnalysisResult):
        print(f"ABox {state.abox.as_dict()}")
        answer = state.tbox_callable(
            state.abox.as_dict(), res.query.taxpayer_id, res.query.year
        )
        return {"answer": answer}
    else:
        print("err: answer generation call failed")
        return {}


def graph(llm: BaseChatModel, debug: bool):
    builder = StateGraph(State)

    builder.add_node(
        "fact_extraction", partial(fact_extraction_node, llm=llm, debug=debug)
    )
    builder.add_node(
        "symbolic_inference", partial(symbolic_inference_node, debug=debug)
    )
    builder.add_node(
        "answer_generation", partial(answer_generation_node, llm=llm, debug=debug)
    )

    builder.add_edge(START, "fact_extraction")
    builder.add_edge("fact_extraction", "symbolic_inference")
    builder.add_edge("symbolic_inference", "answer_generation")
    builder.add_edge("answer_generation", END)

    return builder.compile()


def predict(
    llm: BaseChatModel,
    description: str,
    query: str,
    tbox: TBox,
    tbox_interpreter: str,
    debug: bool,
) -> tuple[float, dict]:
    app = graph(llm, debug)

    local_namespace = {}
    exec(tbox_interpreter, {}, local_namespace)
    tbox_callable = local_namespace.get("calculate")
    if not callable(tbox_callable):
        raise Exception("'calculate' function not found in TBox interpreter")

    final_state = app.invoke(
        State(
            description=description,
            query=query,
            tbox=tbox,
            tbox_callable=tbox_callable,
        )
    )
    if debug:
        print(f"Answer: {final_state['answer']}")
    return final_state["answer"], final_state["abox"].as_dict()
