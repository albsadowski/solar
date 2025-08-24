from functools import cache
from typing import Final

from nltk.sem.logic import Expression, LogicParser

from .model import TBox, TBoxClass, TBoxRule, TBoxProperty


SUPPORTED_DATA_TYPES: Final[set[str]] = set(
    [
        "string",
        "integer",
        "decimal",
        "boolean",
    ]
)


def preds_from_fol(expr: Expression) -> set[str]:
    preds = set()
    if pred := getattr(expr, "pred", None):
        preds.add(str(pred))
    if func := getattr(expr, "function", None):
        if pred := getattr(func, "pred", None):
            preds.add(str(pred))
    if first := getattr(expr, "first", None):
        preds.update(preds_from_fol(first))
    if second := getattr(expr, "second", None):
        preds.update(preds_from_fol(second))
    if term := getattr(expr, "term", None):
        preds.update(preds_from_fol(term))
    return preds


def validate_classes(classes: list[TBoxClass]) -> tuple[set[str], list[str]]:
    issues, class_names = [], set()

    if not classes:
        issues.append("no classes found")

    for ix, cls in enumerate(classes):
        if not cls.name:
            issues.append(f"class at {ix}: name is empty")
            continue
        if cls.name in class_names:
            issues.append(f"class at {ix}: name '{cls.name}' is not unique")
        else:
            class_names.add(cls.name)

    return class_names, issues


def validate_property(
    class_names: set[str], prop_type: str, args: list[str]
) -> list[str]:
    issues = []
    match prop_type:
        case "unary":
            if len(args) != 1:
                issues.append("exactly one argument expected")
            else:
                if args[0] not in class_names:
                    issues.append(f"reference to an undefined class {args[0]}")
        case "object":
            if len(args) != 2:
                issues.append("exactly two arguments expected")
            else:
                for arg in args:
                    if arg not in class_names:
                        issues.append(f"reference to an undefined class {arg}")
        case "datatype":
            if len(args) != 2:
                issues.append("exactly two arguments expected")
            else:
                if args[0] not in class_names:
                    issues.append(f"reference to an undefined class {args[0]}")
                if args[1] not in SUPPORTED_DATA_TYPES:
                    issues.append(f"reference to an unknown data type {args[1]}")
        case other:
            issues.append(f"unsupported property type '{other}'")
    return issues


def validate_properties(props: list[TBoxProperty], class_names: set[str]) -> list[str]:
    issues, prop_names = [], set()

    for ix, prop in enumerate(props):
        if not prop.name:
            issues.append(f"property at {ix}: name is empty")
            continue
        if prop.name in prop_names:
            issues.append(f"property at {ix}: name '{prop.name}' is not unique")
        else:
            prop_names.add(prop.name)

        prefix = f"property at {ix} ({prop.name}):"
        for issue in validate_property(class_names, prop.type, prop.arguments):
            issues.append(f"{prefix} {issue}")

    return issues


@cache
def parser():
    return LogicParser()


def validate_rule(ix: int, rule: TBoxRule, class_names: set[str], prop_names: set[str]):
    issues = []
    prefix = f"rule at {ix}:"

    if not rule.implication_property_name:
        issues.append(f"{prefix} implication property name not defined")
        return issues
    if rule.implication_property_name in prop_names:
        issues.append(
            f"{prefix} implication property name conflicts with TBox properties"
        )
        return issues
    for issue in validate_property(
        class_names, rule.implication_property_type, rule.implication_property_arguments
    ):
        issues.append(f"{prefix} {issue}")
    if not rule.fol_expression:
        issues.append(f"{prefix} empty FOL expression")
        return issues
    try:
        expr = parser().parse(rule.fol_expression)
        preds = preds_from_fol(expr)
        if not preds:
            issues.append(f"{prefix} predicates not found")
        for pred in preds:
            if (
                pred.replace("~", "") not in prop_names
                or pred.replace("¬", "") not in prop_names
            ):
                possible_negation = pred.startswith("~") or pred.startswith("¬")
                issues.append(
                    " ".join(
                        [
                            f"{prefix} property '{pred}' is referenced by FOL expression but not found in the TBox",
                            (
                                "(is this a valid negation syntax? prefer 'negative' properties in the TBox instead e.g., isHuman vs notIsHuman)"
                                if possible_negation
                                else ""
                            ),
                        ]
                    )
                )
    except Exception as err:
        issues.append(f"{prefix} error when parsing expression - {err}")

    return issues


def validate_rules(
    rules: list[TBoxRule], properties: list[TBoxProperty], class_names: set[str]
) -> list[str]:
    issues = []
    prop_names = {prop.name for prop in properties}

    for ix, rule in enumerate(rules):
        issues.extend(validate_rule(ix, rule, class_names, prop_names))

    return issues


def validate_tbox(tbox: TBox) -> list[str]:
    class_names, issues = validate_classes(tbox.classes)
    issues.extend(validate_properties(tbox.properties, class_names))
    issues.extend(validate_rules(tbox.rules, tbox.properties, class_names))
    return issues
