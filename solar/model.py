from typing import Literal

from pydantic import BaseModel, Field


class TBoxClass(BaseModel):
    name: str = Field(
        ..., description="The name of the legal concept or entity. PascalCase."
    )
    description: str = Field(
        ..., description="A concise description of what this class represents."
    )


class TBoxProperty(BaseModel):
    type: Literal["unary", "object", "datatype"] = Field(
        ..., description="The type of the property."
    )
    name: str = Field(
        ..., description="The name of the property. camelCase or PascalCase"
    )
    arguments: list[str] = Field(
        ...,
        description=(
            "For 'unary' type: a list containing the Class name it applies to, e.g. ['Person']. "
            "For 'object' type: a list of two Class names the property binds together, e.g., ['TaxPayer', 'Dependent']. "
            "For 'datatype' type: a list with the Class name and the literal type, e.g., ['Income', decimal]."
        ),
    )
    description: str = Field(
        ...,
        description="A concise description of what this property represents or links.",
    )


class TBoxRule(BaseModel):
    fol_expression: str = Field(
        ...,
        description=(
            "First-Order-Logic (FOL) expression representing the premises (conditions) of the rule. "
            "This part does NOT include the resulting implication (e.g., no '->' or ':-'). "
            "It should be a formula or conjunction of formulas to be evaluated by an SMT solver (NLTK SMT Solver - nltk.sem.logic). "
            "Example: 'IsTaxpayer(Person) & HasIncome(Person, Income)'. "
            "The predicates used (e.g., IsTaxpayer, HasIncome) should be plausible property names that might be defined elsewhere or are common. "
            "The integration step will later reconcile these with explicitly extracted properties."
        ),
    )
    implication_property_type: Literal["unary", "object", "datatype"] = Field(
        ..., description="The type of the property"
    )
    implication_property_name: str = Field(
        ...,
        description=(
            "The name of the NEW property that is implied if the 'fol_expression' (premises) holds true. E.g., 'MustFileReturn'. "
            "This implied property is introduced by the rule. Use PascalCase."
        ),
    )
    implication_property_arguments: list[str] = Field(
        ...,
        description=(
            "A list of plausible Class names that are the arguments for the 'implication_property_name'. "
            "E.g., if 'implication_property_name' is 'MustFileReturn' and it applies to a Taxpayer, this would be ['Taxpayer']."
            "These class names should be general concepts identifiable from the text; they will be reconciled later."
        ),
    )
    description: str = Field(
        ...,
        description="A concise English description of what the rule establishes or defines, including the implication.",
    )


class TBox(BaseModel):
    classes: list[TBoxClass] = Field(..., description="Coherent list of TBox classes.")
    properties: list[TBoxProperty] = Field(
        ...,
        description=(
            "Coherent list of explicitly defined TBox properties. Properties introduced by rule implications "
            "are defined by the rules themselves and are NOT listed here."
        ),
    )
    rules: list[TBoxRule] = Field(
        ...,
        description="Coherent list of TBox rules, where each rule defines an implied property.",
    )


class ExtractionEnvelope[T](BaseModel):
    object: T = Field(..., description="Object being extracted")
    confidence: float = Field(
        ..., description="Confidence score for the extraction (0.0 to 1.0)."
    )
    explanation: str = Field(
        ...,
        description="Brief explanation of why this object was extracted and how the fields were determined.",
    )


class ABoxIndividual(BaseModel):
    name: str = Field(..., description="Unique identifier for this individual")
    class_name: str = Field(
        ..., description="The TBox class this individual belongs to"
    )


class ABoxAssertion(BaseModel):
    property_name: str = Field(
        ..., description="Name of the TBox property this assertion instantiates"
    )
    arguments: list[str] = Field(
        ...,
        description="Arguments for the property (individual names or literal values)",
    )


class ABox(BaseModel):
    individuals: list[ExtractionEnvelope[ABoxIndividual]] = Field(default_factory=list)
    assertions: list[ExtractionEnvelope[ABoxAssertion]] = Field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "individuals": {
                ind.object.name: {"type": ind.object.class_name}
                for ind in self.individuals
            },
            "assertions": [
                {
                    "predicate": assertion.object.property_name,
                    "args": assertion.object.arguments,
                }
                for assertion in self.assertions
            ],
        }
