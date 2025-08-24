import json
import traceback
from functools import partial
from typing import Final

import pandas as pd
from langchain.chat_models.base import BaseChatModel
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field

from .model import ExtractionEnvelope, TBox, TBoxClass, TBoxProperty, TBoxRule
from .knowledge_application import predict as knowledge_application_predict
from .prompts import (
    CODE_GENERATION_PROMPT,
    CONCEPT_EXTRACTION_PROMPT,
    RULE_FORMULATION_PROMPT,
    RULE_INTEGRATION_PROMPT,
)
from .validation import validate_tbox


MAX_FIX_ATTEMPTS: Final[int] = 3


class TrainingProblem(BaseModel):
    description: str
    query: str
    extracted_abox: dict | None = None
    answer: float | None = None
    error: str | None = None


class State(BaseModel):
    raw_statute_text: str
    candidate_classes: list[ExtractionEnvelope[TBoxClass]] | None = None
    candidate_properties: list[ExtractionEnvelope[TBoxProperty]] | None = None
    candidate_rules: list[ExtractionEnvelope[TBoxRule]] | None = None
    tbox: TBox | None = None
    tbox_validation_issues: list[str] | None = None
    tbox_eval_code: str | None = None
    fix_attempts: int = 0
    human_feedback: str | None = None
    training_evaluation_error: TrainingProblem | None = None


class ConceptExtractionResult(BaseModel):
    classes: list[ExtractionEnvelope[TBoxClass]] = Field(
        ..., description="List of extracted classes."
    )
    properties: list[ExtractionEnvelope[TBoxProperty]] = Field(
        ..., description="List of extracted properties."
    )


class RuleFormulationResult(BaseModel):
    rules: list[ExtractionEnvelope[TBoxRule]] = Field(
        ...,
        description=("List of formulated rules, where each 'object' is a TBoxRule."),
    )


class RuleIntegrationResult(BaseModel):
    tbox: TBox = Field(..., description="Final TBox.")


def concept_extraction_node(state: State, llm: BaseChatModel, debug: bool):
    if debug:
        print("E concept_extraction_node")

    statute = state.raw_statute_text.strip() if state.raw_statute_text else None
    if not statute:
        return {
            "candidate_classes": [],
            "candidate_properties": [],
        }

    res = llm.with_structured_output(ConceptExtractionResult).invoke(
        CONCEPT_EXTRACTION_PROMPT.format(
            statute=statute,
        )
    )
    if isinstance(res, ConceptExtractionResult):
        if debug:
            print("X concept_extraction_node")
        return {
            "candidate_classes": res.classes,
            "candidate_properties": res.properties,
        }
    else:
        print("err: concept extraction failed")
        return {
            "candidate_classes": [],
            "candidate_properties": [],
        }


def rule_formulation_node(state: State, llm: BaseChatModel, debug: bool):
    if debug:
        print("E rule_formulation_node")

    statute = state.raw_statute_text.strip() if state.raw_statute_text else None
    if not statute:
        return {"candidate_rules": None}

    res = llm.with_structured_output(RuleFormulationResult).invoke(
        RULE_FORMULATION_PROMPT.format(
            statute=statute,
        ),
    )
    if isinstance(res, RuleFormulationResult):
        if debug:
            print("X rule_formulation_node")
        return {"candidate_rules": res.rules}
    else:
        print("err: rule extraction failed")
        return {"candidate_rules": None}


def rule_integration_node(state: State, llm: BaseChatModel, debug: bool):
    if debug:
        print(
            "E rule_integration_node"
            + (
                f" (fix attempt #{state.fix_attempts})"
                if state.fix_attempts > 0
                else ""
            )
        )

        if state.human_feedback:
            print(f"\tHuman feedback: {state.human_feedback}")
            print("")
        if state.tbox_validation_issues:
            print("\tTBox Validation Issues:")
            for issue in state.tbox_validation_issues:
                print(f"\t  -  {issue}")
            print("")

    res = llm.with_structured_output(RuleIntegrationResult).invoke(
        RULE_INTEGRATION_PROMPT.format(
            candidate_classes=json.dumps(
                [cls.model_dump() for cls in (state.candidate_classes or [])]
            ),
            candidate_properties=json.dumps(
                [prop.model_dump() for prop in (state.candidate_properties or [])]
            ),
            candidate_rules=json.dumps(
                [rule.model_dump() for rule in (state.candidate_rules or [])]
            ),
            last_tbox=json.dumps(state.tbox.model_dump()) if state.tbox else None,
            tbox_validation_issues="\n".join(
                [
                    f"  - {issue.strip()}"
                    for issue in (state.tbox_validation_issues or [])
                ]
            )
            if state.tbox_validation_issues
            else None,
            human_feedback=state.human_feedback,
        )
    )
    if isinstance(res, RuleIntegrationResult):
        if debug:
            print("X rule_integration_node")
        return {"tbox": res.tbox}
    else:
        print("err: rule integration failed")
        return {"tbox": None}


def code_generation_node(state: State, llm: BaseChatModel, debug: bool):
    if debug:
        print(f"E code_generation_node")

    if not state.tbox:
        print(f"err: empty TBox")
        return {}

    res = llm.invoke(
        CODE_GENERATION_PROMPT.format(
            statute=state.raw_statute_text,
            classes=json.dumps([cls.model_dump() for cls in state.tbox.classes]),
            properties=json.dumps(
                [prop.model_dump() for prop in state.tbox.properties]
                + [
                    TBoxProperty(
                        type=rule.implication_property_type,
                        name=rule.implication_property_name,
                        arguments=rule.implication_property_arguments,
                        description=rule.description,
                    ).model_dump()
                    for rule in state.tbox.rules
                ]
            ),
            last_interpreter=state.tbox_eval_code,
            last_error=(
                "\nTest case that failed:\n"
                f"Description: {state.training_evaluation_error.description}\n"
                f"Question: {state.training_evaluation_error.query}\n\n"
                "For which, given your instructions, the following ABox was constructed:\n"
                f"{state.training_evaluation_error.extracted_abox}\n\n"
                f"For which the code failed with the following error:\n"
                f"Error: {state.training_evaluation_error.error}"
            )
            if state.training_evaluation_error
            else None,
        )
    )
    if isinstance(res.content, str):
        if debug:
            print("X code_generation_node")
        return {"tbox_eval_code": res.content}
    else:
        print("err: code generation failed")
        return {"tbox_eval_code": None}


def human_review_node(state: State):
    print(f"\n{'=' * 50}\nREVIEW\n{'=' * 50}")
    if state.tbox:
        issues = validate_tbox(state.tbox)
        if not issues:
            print("\nValid TBox.")
        else:
            print("\nValidation issues:")
            for issue in issues:
                print(f"  - {issue}")

        print("\nClasses:")
        for cls in state.tbox.classes:
            print(f"  - {cls.name}: {cls.description}")

        print("\nProperties:")
        for prop in state.tbox.properties:
            print(f"  - {prop.name} ({prop.type}): {prop.description}")
            print(f"    arguments: {prop.arguments}")

        print("\nRules:")
        for rule in state.tbox.rules:
            print(f"  - {rule.description}")
            print(f"    FOL: {rule.fol_expression}")
            print(
                f"    Implies: {rule.implication_property_name}({', '.join(rule.implication_property_arguments)})"
            )

    print("\nOptions:")
    print("1. Approve (type 'approve')")
    print("2. Provide feedback for improvements (type your feedback)")

    while True:
        human_input = input("\nYour response: ").strip()
        if not human_input:
            print("Please provider a valid response.")
            continue
        return {"human_feedback": human_input, "tbox_validation_issues": None}


def training_evaluation_node(
    state: State, llm: BaseChatModel, debug: bool, err_tolerance=0.1
):
    assert state.tbox != None, "tbox expected"
    assert state.tbox_eval_code != None, "tbox_eval_code expected"

    fix_attempts = (
        1 if state.training_evaluation_error == None else state.fix_attempts + 1
    )
    if fix_attempts > MAX_FIX_ATTEMPTS:
        print(f"err max allowed number of fix attempts has been reached {fix_attempts}")
        return {"training_evaluation_error": None}

    df = pd.read_csv(
        "./legalbench/data/sara_numeric/train.tsv", sep="\t", index_col="index"
    )
    for ix, row in df.iterrows():
        if debug:
            print(f"Testing case #{ix}")
            print(f"Description: {row['description']}")
            print(f"Question: {row['question']}")

        exp_answer = float(row["answer"][1:])
        lbound = exp_answer * (1 - err_tolerance)
        ubound = exp_answer * (1 + err_tolerance)

        try:
            res, abox = knowledge_application_predict(
                llm=llm,
                debug=debug,
                tbox=state.tbox,
                tbox_interpreter=state.tbox_eval_code,
                description=row["description"],
                query=row["question"],
            )
        except Exception as ex:
            if debug:
                print(f"err when evaluating test case #{ix} :: {ex}")
            stack_trace = traceback.format_exc().splitlines()[-10:]
            return {
                "training_evaluation_error": TrainingProblem(
                    description=row["description"],
                    query=row["question"],
                    error=f"{ex}\n{'\n'.join(stack_trace)}",
                ),
                "fix_attempts": fix_attempts,
            }

        if res < lbound or res > ubound:
            return {
                "training_evaluation_error": TrainingProblem(
                    description=row["description"],
                    query=row["question"],
                    extracted_abox=abox,
                    error=(
                        f"code produced {res} but expected was {exp_answer} and that's "
                        "outside of allowed error tolerance {err_tolerance}"
                    ),
                ),
                "fix_attempts": fix_attempts,
            }

    return {"training_evaluation_error": None, "fix_attempts": None}


def decide_after_human_review(state: State):
    match (state.human_feedback or "").lower().strip():
        case "approve":
            return END
        case _:
            return "rule_integration"


def tbox_validation_node(state: State):
    if not state.tbox:
        return []

    if state.fix_attempts >= MAX_FIX_ATTEMPTS:
        return {
            "tbox_validation_issues": None,
            "fix_attempts": 0,
            "human_feedback": None,
        }

    return {
        "tbox_validation_issues": validate_tbox(state.tbox),
        "human_feedback": None,
        "fix_attempts": state.fix_attempts + 1,
    }


def decide_after_tbox_validation(state: State):
    if state.tbox_validation_issues:
        return "rule_integration"
    return END


def decide_after_training_evaluation(state: State):
    if state.training_evaluation_error:
        return "code_generation"
    return END


def graph(llm: BaseChatModel, debug: bool):
    builder = StateGraph(State)

    builder.add_node(
        "concept_extraction", partial(concept_extraction_node, llm=llm, debug=debug)
    )
    builder.add_node(
        "rule_formulation", partial(rule_formulation_node, llm=llm, debug=debug)
    )
    builder.add_node(
        "rule_integration", partial(rule_integration_node, llm=llm, debug=debug)
    )
    builder.add_node("tbox_validation", tbox_validation_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node(
        "code_generation", partial(code_generation_node, llm=llm, debug=debug)
    )
    builder.add_node(
        "training_evaluation", partial(training_evaluation_node, llm=llm, debug=debug)
    )

    builder.add_edge(START, "concept_extraction")
    builder.add_edge(START, "rule_formulation")
    builder.add_edge("concept_extraction", "rule_integration")
    builder.add_edge("rule_formulation", "rule_integration")
    builder.add_edge("rule_integration", "tbox_validation")
    builder.add_edge("code_generation", "training_evaluation")

    builder.add_conditional_edges(
        "tbox_validation",
        decide_after_tbox_validation,
        {
            "rule_integration": "rule_integration",
            END: "human_review",
        },
    )
    builder.add_conditional_edges(
        "human_review",
        decide_after_human_review,
        {
            "rule_integration": "rule_integration",
            END: "code_generation",
        },
    )
    builder.add_conditional_edges(
        "training_evaluation",
        decide_after_training_evaluation,
        {
            "code_generation": "code_generation",
            END: END,
        },
    )

    return builder.compile()


def predict(llm: BaseChatModel, input: str, debug: bool) -> tuple[TBox, str]:
    app = graph(llm, debug)
    if debug:
        print(app.get_graph().draw_ascii())

    final_state = app.invoke(State(raw_statute_text=input))
    tbox = final_state["tbox"]
    if debug:
        print(json.dumps(tbox.model_dump(), indent=2))
        print("\n\nTBox Evaluation Code\n\n")
        print(final_state["tbox_eval_code"])
    return tbox, final_state["tbox_eval_code"]
