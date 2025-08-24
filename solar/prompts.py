from typing import Final

from langchain.prompts import PromptTemplate


CONCEPT_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["statute"],
    template="""
# Instruction

You are an expert legal AI assistant specializing in ontology engineering from legal texts.
Your task is to analyze the provided segment of a legal statute and extract:
1. **Candidate Classes (Legal Concepts/Entities)**: These are the fundamental nouns or noun phrases that represent key concepts in the legal domain described by the text.
2. **Candidate Properties**: These define attributes of classes or relationships between classes.

Note! When multiple classes may be attributed to the same individual it causes ambiguity in the systems downstream.
Thus, prefer to use the properties instead. e.g., in case a Person and a Taxpayer would aspire to be two classes,
think that a Taxpayer must also be a person; instead define a class Person and a property isTaxpayer.

For each extracted item, provide a confidence score (0.0 to 1.0) and a brief explanation.

# Important Context for Knowledge Application

The ontology you create (TBox) will be used in two critical ways:
1. To guide the extraction of case-specific facts (ABox) from user queries
2. To structure Python code that will calculate exact numeric values (such as tax amounts)

Therefore, your extracted concepts should:
- Be atomic/granular rather than composite (prefer "income" and "deduction" over "netIncome")
- Avoid concepts that would require in-memory calculations
- Focus on raw data points that Python code can later process
- Enable precise calculations of numerical values as required by the examples below

# Example Use Cases

Your ontology should support answering queries like:

Example 1:
Description: Alice and Bob got married on Feb 3rd, 1992. Alice paid each of her 87 employees $5207 for work done in 2015. Alice paid each of her 70 employees $6341 for work done in 2016. Alice and Bob file jointly in 2015 and had no income. They take the standard deduction.
Query: How much tax does Alice have to pay in 2015?
Expected answer: 27181

Example 2:
Description: Alice was paid $1200 in 2019 for services performed in jail. Alice was committed to jail from January 24, 2015 to May 5th, 2019. From May 5th 2019 to Dec 31st 2019, Alice was paid $5320 in remuneration. Alice takes the standard deduction.
Query: How much tax does Alice have to pay in 2019?
Expected answer: 0

Example 3:
Description: Alice and Bob got married on Feb 3rd, 1998, and have a son Charlie, born April 1st, 1999. Bob died on Jan 1st, 2017. In 2019, Charlie lives at the house that Alice maintains as her principal place of abode. Alice's gross income for the year 2019 is $236422. Alice takes the standard deduction.
Query: How much tax does Alice have to pay in 2019?
Expected answer: 62000

Example 4:
Description: Alice and Bob got married on November 23rd, 1994. Their son Charlie was born on July 5th, 2000. Bob died on March 15th, 2015. In 2017, Alice and Charlie lived in a house maintained by Alice. Alice's gross income for the year 2017 is $95129. Alice takes the standard deduction.
Query: How much tax does Alice have to pay in 2017?
Expected answer: 19039

# Extraction Guidelines

When extracting classes and properties:

1. For **Classes**: Focus on fundamental legal entities mentioned in the statute (e.g., Taxpayer, Income, Employer).
   - Prefer atomic concepts that represent basic units of information
   - Consider what entities would appear in an ABox for the example cases above

2. For **Properties**:
   - Extract relationships between entities (e.g., HasSpouse, EmploysIndividual)
   - Extract attributes that are intrinsic to entities (e.g., FilingStatus, WageAmount)
   - For numeric values mentioned in the statute (wages amounts, income thresholds), extract datatype properties that capture the raw values WITHOUT encoding any judgments about thresholds
   - Include properties that will be needed by Python code to perform calculations (e.g., NumberOfEmployees, WagePerEmployee, DateOfMarriage)

3. Do NOT create categorical properties that encode numeric thresholds or calculations
   - AVOID: properties like "MeetsWageThreshold", "TotalIncome", or "TaxAmount" that would require calculations
   - PREFER: raw properties like "WageAmount", "IncomeAmount" that code can use for calculations

4. Remember that Python code will later implement all calculations based on your ontology, so focus on capturing the raw data points that such code would need to calculate tax amounts for cases like the examples above.

5. The research has shown, that offering LLMs complementary predicates, forces LLMs to choose and gives better results, thus for every property, make sure to offer the negation, e.g., isBlind has a complementary isNotBlind.

CRITICAL: Entities in legal contexts often serve multiple roles simultaneously (e.g., a person can be both a "Taxpayer" AND an "Employer"). 
Your ontology should:
- Create appropriate relationship properties to connect these roles (e.g., "hasEmployerRole")
- OR define class hierarchies that allow inheritance of multiple characteristics
- AVOID creating disconnected entities for the same underlying individual

EXAMPLE:
POOR: Define "Alice" as "Taxpayer" and "AliceEmployer" as "Employer" with no connection
GOOD: Define "Alice" as "Taxpayer" with property "isAlsoEmployer(Alice, True)"

# Input Statute

{statute}

# Output Format

Return a single, valid JSON object that strictly adheres to the following structure. Do NOT include any markdown formatting (e.g., ```json) around the JSON object itself.

```json
{{
  "classes": [
    {{
      "object": {{
        "name": "ClassNameInPascalCase",
        "description": "Concise description of the class."
      }},
      "confidence": 0.9,
      "explanation": "Reason for extracting this class and determining its fields."
    }}
  ],
  "properties": [
    {{
      "object": {{
        "type": "unary_or_object_or_datatype",
        "name": "propertyNameInCamelOrPascalCase",
        "arguments": ["Arg1", "Arg2_if_applicable"],
        "description": "Concise description of the property."
      }},
      "confidence": 0.85,
      "explanation": "Reason for extracting this property, its type, arguments, and other fields."
    }}
  ]
}}
```

Detailed Field Explanations for the "object" within "properties":

* **type**: Must be one of "unary", "object", or "datatype".
    - "unary": Represents a characteristic of a single class (e.g., IsTaxpayer). arguments will be
      a list with one string: the class name it applies to (e.g., ["Taxpayer"]).
    - "object": Represents a relationship between two classes (e.g., hasSpouse). arguments will be
      a list of two strings: ["DomainClassName", "RangeClassName"] (e.g., ["Person", "Person"]).
    - "datatype": Represents an attribute of a class with a literal value (e.g., hasIncomeAmount).
      arguments will be a list of two strings: ["DomainClassName", "LiteralType"] (e.g.,
      ["Income", "decimal"], ["Person", "integer"], ["LegalDocument", "date"]). Literal types can include
      string, integer, decimal, boolean and date.

* **name**: The name of the property. Use camelCase or PascalCase (e.g., isTaxpayer, HasIncomeAmount).

* **arguments**: A list of strings as described above, depending on the property type. These should
  refer to Class names defined in classes or common literal types.

* **description**: A concise explanation of the property.

Ensure all class names and property names are descriptive and consistently cased.
Focus on extracting core, reusable concepts and properties directly supported by the provided text.
If no relevant classes or properties are found in a segment, an empty list for that category is acceptable.
Do not include results of low confidence.
The overall JSON must be valid.
""",
)


RULE_FORMULATION_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=["statute"],
    template="""
# Instruction

You are an expert legal AI assistant specializing in extracting logical rules for ontology engineering from legal texts.
Your task is to analyze the provided segments of a legal statute to extract logical rules that represent the core logical relationships between concepts, WITHOUT encoding any numeric thresholds or calculations.

Each rule defines a new property (the implication) based on a set of conditions (the FOL expression/premises).

# Important Direction
- Focus ONLY on extracting logical relationships between concepts
- DO NOT create rules that encode numeric thresholds or calculations
- Extract rules about what entities ARE, not whether they MEET CRITERIA
- Numeric calculations and threshold checks will be handled separately by Python code
- Do not produce single predicate rules

# Input Statute
```
{statute}
```

# Task

Extract logical rules that define core conceptual relationships. For each rule:
-   `fol_expression`: Write the First-Order Logic (FOL) expression representing the **premises (conditions) only**.
    -   The expressions will be interpreted by NLTK SMT solver (python package nltk.sem.logic)
    -   Focus on relationships like "IsEmployer(X) & HasEmployee(X, Y)" rather than threshold conditions
    -   DO NOT include numeric comparisons in rules (>, <, >=, etc.)
    -   DO NOT create predicates that encode thresholds (e.g., avoid "MeetsWageThreshold")
-   `implication_property_name`: Specify the name (PascalCase) of the **NEW property** that is true if the `fol_expression` is true. This property is introduced by this rule.
-   `implication_property_type`: Specify the type of the **NEW property** - one of 'unary' / 'object' / 'datatype'.
-   `implication_property_arguments`: Provide a list of plausible Class names (e.g., `["Taxpayer"]`, `["Person", "Benefit"]`) that are the arguments for the `implication_property_name`. These class names should be general concepts identifiable from the text.
-   `description`: Write a concise English description of the rule, explaining what conditions lead to what implication.
-   Provide a `confidence` score (0.0 to 1.0) and an `explanation` for the entire rule extraction, including why you chose certain predicate or class names.
- You must extract rules that properly handle entities with multiple roles. For example:
    - if "Taxpayer(X) & isAlsoEmployer(X, True) & EmployerPaysTax(X, T) → MustPayTax(X, T)"
    - if "Person(X) & Employer(X) & HasEmployeeCount(X, N) & N > 0 → SubjectToExciseTax(X)"

# Examples of What to Extract and What to Avoid

## GOOD EXAMPLES (focus on conceptual relationships):
- "IsMarriedIndividual(X) & FilesJointReturn(X)" → "QualifiesForJointFilingStatus(X)"
- "Individual(X) & HasDependent(X, Y)" → "MayQualifyAsHeadOfHousehold(X)"

## BAD EXAMPLES (numeric thresholds that should be handled by code):
- "HasWages(X, W) & WageAmount(W, A) & A >= 1500" → "IsEmployer(X)"
- "Individual(X) & HasTaxableIncome(X, I) & AmountIs(I, A) & A <= 36900" → "InFirstTaxBracket(X)"
- "IsFoo(X) -> IsBar(X)"

Instead, simply extract rules about what entities ARE and their relationships. The Python code will handle all numeric comparisons and threshold checks.

# Output Format

Return a single, valid JSON object that strictly adheres to the following structure. Do NOT include any markdown formatting (e.g., ```json) around the JSON object itself.

```json
{{
  "rules": [
    {{
      "object": {{
        "fol_expression": "IsTaxpayer(X) & HasIncome(X, I)",
        "implication_property_name": "MustFileReturn",
        "implication_property_type": "unary",
        "implication_property_arguments": ["Person"],
        "description": "If Person is a taxpayer and has income must file a tax return."
      }},
      "confidence": 0.9,
      "explanation": "Rule derived from text stating high-income taxpayers must file. 'IsTaxpayer', 'HasIncome' and 'Taxpayer' (for argument) are inferred as plausible concepts. 'MustFileReturn' is a new property implied by these conditions."
    }}
    // ... more rules
  ]
}}
```

If no rules can be confidently extracted, return an empty list for "rules".
The overall JSON must be valid.
""",
)


RULE_INTEGRATION_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=[
        "candidate_classes",
        "candidate_properties",
        "candidate_rules",
        "human_feedback",
        "last_tbox",
        "tbox_validation_issues",
    ],
    template_format="jinja2",
    template="""
# Instruction

You are an expert ontology engineer AI. Your task is to integrate candidate classes, properties, and rules extracted from a legal statute into a single, coherent, and consistent Terminological Box (TBox).
The candidate elements were extracted by separate, parallel processes and may contain redundancies, inconsistencies, or require refinement.

# Inputs

## 1. Candidate Classes:
Each class includes the extracted object, confidence score, and an explanation.
```json
{{ candidate_classes }}
```

## 2. Candidate Properties (Explicitly Extracted):
These are properties identified directly from the text. Each includes the extracted object, confidence score, and an explanation.
```json
{{ candidate_properties }}
```

## 3. Candidate Rules:
Each rule includes the extracted object (premises, implied property name & args, description), confidence, and explanation.
The `implication_property_name` in a rule introduces a new property that is defined *by that rule*.
```json
{{ candidate_rules }}
```

# Task: Construct the Final TBox

Create a final TBox with three main lists: `classes`, `properties` (explicit only), and `rules`.

You are allowed to modify input classes, properties and rules if it's necessary. The inputs are only
suggestions. In particular, in case of a subsequent iteration, it's more important to incorporate feedback
than to preserve the structure of the input.

## Steps & Considerations:

### A. Finalize Classes:
1.  **Review `candidate_classes`**: Consider their descriptions, confidence, and explanations.
2.  **Consolidate & Deduplicate**: Merge classes that represent the same concept (e.g., "TaxpayerEntity" and "Taxpayer" might become "Taxpayer"). Standardize names to PascalCase.
3.  **Filter**: You may discard very low-confidence or irrelevant classes.
4.  **Output**: A final list of `TBoxClass` objects.

### B. Finalize Explicit Properties:
1.  **Review `candidate_properties`**: These are the properties extracted directly, not those implied by rules. Consider their type, arguments, descriptions, confidence, and explanations.
2.  **Consolidate & Deduplicate**: Merge `candidate_properties` that represent the same concept. Standardize names (camelCase or PascalCase).
3.  **Validate Arguments**: Ensure all class names used in property `arguments` for these explicit properties exist in your finalized list of classes (from Step A). Adjust if necessary.
4.  **Filter**: Discard low-confidence or redundant explicit properties.
5.  **Output**: A final list of `TBoxProperty` objects. This list should *ONLY* contain these explicitly defined and refined properties. Properties introduced by rule implications (e.g., `implication_property_name` from a rule) are defined by the rules themselves and are *NOT* to be added to this explicit `properties` list.

### C. Finalize Rules:
1.  **Review `candidate_rules`**: Consider their FOL expressions, implied properties, descriptions, confidence, and explanations.
2.  **Validate Predicates in `fol_expression`**:
    -   The predicates used in a rule's `fol_expression` (e.g., `IsTaxpayer(X)`, `HasIncome(X,I)`) should correspond to property names in your finalized list of explicit properties (from Step B).
    -   If a predicate in a rule's premises does not match an existing finalized explicit property, try to map it to one. If it's essential and clearly implied by the text as a premise-predicate, you might need to add it to the explicit properties list in Step B (and document this decision in the summary). Avoid inventing premise-predicates that aren't supported.
3.  **Validate `implication_property_name`, `implication_property_type` and `implication_property_arguments`**:
    -   For each rule, its `implication_property_name` defines a *new* predicate. Ensure this name is well-formed (e.g., PascalCase) and does *not* conflict with names in the finalized list of explicit properties (from Step B). It is defined *by* the rule.
    -   Ensure the class names used in the rule's `implication_property_arguments` are consistent with your finalized list of classes (from Step A).
    -   Ensure `implication_property_type` is one of 'unary' / 'object' / 'datatype',
    -   Ensure that when type is unary, there's exactly one argument referring to a valid class name,
    -   Ensure that when type is object, there are exactly two arguments, both referring to valid class names,
    -   Ensure that then type is datatype, there are exactly two arguments, first referring to a class, second is a datatype literal (one of 'string' / 'integer' / 'decimal' / 'boolean')
4.  **Filter**: Discard low-confidence, inconsistent, or redundant rules. A rule is inconsistent if its premise predicates cannot be mapped to finalized explicit properties or its implication arguments don't map to finalized classes.
5.  **Output**: A final list of `TBoxRule` objects. Each rule implicitly defines the property named in its `implication_property_name`.

{% if last_tbox %}
# Previous Iteration

CRITICAL! It is more important to incorporate human feedback than preserve the input structure of classes and properties.

Current ask is a subsequent iteration on the same problem. Last attempt:
{{ last_tbox }}

{% if human_feedback %}
## Human feedback for that iteration
{{ human_feedback }}
{% endif %}

{% if tbox_validation_issues %}
## TBox validation issues found
{{ tbox_validation_issues }}
{% endif %}
{% endif %}

# Output Format

Return a single, valid JSON object. Do NOT include any markdown formatting (e.g., ```json) around the JSON object itself.

```json
{
  "tbox": {
    "classes": [
        {
        "name": "FinalClassName",
        "description": "Final description."
        }
        // ... more classes
    ],
    "properties": [ // Explicitly defined properties ONLY
        {
        "type": "unary",
        "name": "finalExplicitPropertyName",
        "arguments": ["FinalClassName"],
        "description": "Final description from candidate property."
        }
        // ... more explicit properties
    ],
    "rules": [ // Rules define their own implied properties
        {
        "fol_expression": "FinalExplicitPredicateName(X) And AnotherFinalExplicitPredicate(X, Y)",
        "implication_property_name": "NewlyDefinedByRulePropertyName",
        "implication_property_type": "unary",
        "implication_property_arguments": ["FinalClassName"],
        "description": "Final rule description, this rule defines 'NewlyDefinedByRulePropertyName'."
        }
        // ... more rules
    ],
  }
}
```
The output must be a valid JSON object.
""",
)

CODE_GENERATION_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=[
        "classes",
        "properties",
        "statute",
        "last_error",
        "last_interpreter",
    ],
    template_format="jinja2",
    template="""
# Instruction

You are an expert AI assistant specializing in translating legal statutes into executable
Python code. Your task is to generate Python code that implements the tax calculation logic
from a legal statute using a defined ontology vocabulary.

# Ontology Vocabulary

## Classes
These represent the core legal concepts/entities in the statute:
```json
{{ classes }}
```

## Properties

These properties represent relationships and attributes of the classes:
```json
{{ properties }}
```

# Statute Text

{{ statute }}

# Task: Generate Executable Python Code

Create a Python module with a specific function that:
1. Accepts a user query and ABox assertions as input (in a precisely defined format)
2. Interprets the query to identify the specific individual(s) and time period of interest
3. Applies the logic defined in the statute to the relevant individuals/entities
4. Calculates precise numerical values as required
5. Returns the final calculation results with explanations

## Required Function Signature

Your code MUST implement exactly this function signature:

```python
def calculate(abox: dict, taxpayer_id: str, year: int) -> float:
    \"\"\"
    Calculates the numeric result based on the specified taxpayer ID, year and circumstances
    described by the abox.

    Args:
        abox (dict): Dictionary with the following structure:
            {
                "individuals": {
                    "id1": {"type": "ClassType1"},
                    "id2": {"type": "ClassType2"},
                    ...
                },
                "assertions": [
                    {"predicate": "PropertyName1", "args": ["id1", "id2"]},
                    {"predicate": "PropertyName2", "args": ["id1", "value"]},
                    ...
                ]
            }
        taxpayer_id (str): ID of the taxpayer to calculate for
        year (int): the calculation period (whole year)

    Returns:
        float: The calculated numeric result
    \"\"\"

    <ABoxRequirements>
        ...
    </ABoxRequirements>
    ...
```

## Chain-of-Code Technique

You MUST use the chain-of-code approach where each logical step is:
1. Explained clearly in comments BEFORE the code implementation
2. Implemented with descriptive variable names
3. Organized in a logical sequence that follows the statute's structure

Example of chain-of-code style:
```python
# Step 1: Extract the taxpayer ID and the year from the query
# This step unpacks the query tuple to identify the specific individual and calculation period
taxpayer_id, year = query

# Step 2: Verify that the taxpayer exists in the ABox individuals
# This ensures we're calculating for a valid individual
individuals = abox["individuals"]
if taxpayer_id not in individuals:
    raise ValueError(f"Taxpayer ID '{taxpayer_id}' not found in the ABox")

# Step 3: Extract assertions relevant to the specified taxpayer
# This filters the ABox to focus only on the relevant individual
assertions = abox["assertions"]
relevant_assertions = []
for assertion in assertions:
    if assertion["args"] and assertion["args"][0] == taxpayer_id:
        relevant_assertions.append(assertion)

# Step 4: Determine whether the taxpayer has multiple roles
# This identifies if the taxpayer is also an employer or has other roles
all_taxpayer_roles = [taxpayer_type]  # Start with explicit type
related_entities = []

# Look for employer entities that may be the same person/business as the taxpayer
# This is critical for cases where separate entities represent different roles
for entity_id, entity_info in individuals.items():
    # Check for name similarity, relationship properties, or other clues
    if entity_info["type"] == "Employer" and (
        taxpayer_id in entity_id or  # E.g., taxpayer "Alice" and employer "Alice_2015"
        any(a["predicate"] == "ownsBusinessEntity" and a["args"][0] == taxpayer_id and a["args"][1] == entity_id for a in assertions) or
        any(a["predicate"] == "representsSamePerson" and a["args"][0] == taxpayer_id and a["args"][1] == entity_id for a in assertions)
    ):
        all_taxpayer_roles.append("Employer")
        related_entities.append(entity_id)
```

## Requirements for the Generated Code:

1. Query Processing:
- Extract the taxpayer ID and given year from the query tuple
- Verify that the taxpayer exists in the ABox individuals collection
- Use this information to filter ABox assertions to focus only on the relevant taxpayer
- Ensure calculations account for the specified time period
- Code must handle entities with multiple roles. For example:
    - If taxpayer_id refers to an individual who is also an employer, you must calculate BOTH personal income tax AND employer excise tax.
- Remember to traverse relationships to find all relevant properties for the entity

2. ABox Processing:
- Use the EXACT ABox format specified in the function signature
- Extract all relevant values needed for calculation
- Validate that required properties are present

3. Calculation Logic:
- Implement numerical calculations as specified in the statute
- Use the exact numerical values, thresholds, and rates from the statute
- Calculate all intermediate values with appropriate precision
- Return the final result as a float value

4. Documentation:
- Use chain-of-code comments that explain WHAT and WHY for each step
- Reference specific sections of the statute in comments
- Use variable names that align with the ontology terms

5. (CRITICAL!) ABox Documentation
- In the Python docstring of the main method, within the <ABoxRequirement></ABoxRequirement>
  tags, thoroughly document the EXACT expected structure of the ABox for successful calculation
- List ALL required properties that must exist in the ABox, and what they represent
- Explain the meaning of each critical property and how it affects the calculation
- Document EACH property's expected format, data type, and purpose
- Explain how missing properties should be handled (errors, defaults, etc.)
- For each type of calculation (e.g., income tax vs. excise tax), document which properties are required
- For joint vs. individual filings, clearly document how property values should be structured
- If year-specific rules apply, document how the year parameter affects property interpretation
- Provide complete examples showing property combinations for different filing scenarios
- Document property relationships and dependencies: which properties must co-exist

+ ADD THESE CRITICAL MULTI-ROLE INSTRUCTIONS:
+ - Document how entities with multiple roles (e.g., a person who is both a taxpayer and an employer) 
+   must be represented and connected in the ABox
+ - Explicitly document required relationships between person entities, taxpayer entities, and employer entities
+ - Include examples showing how wage information must be represented for both direct wage amounts and 
+   per-employee wages
+ - Emphasize that any mention of wages or employees in text MUST result in complete wage information 
+   in the ABox with appropriate linkage to relevant taxpayers
+ - Include specific guidance on how the calculation code should identify all relevant entities 
+   for a given taxpayer_id, including employer entities that may represent the same person

6. Error Handling:
- Validate inputs and handle missing or invalid values
- Raise appropriate exceptions for invalid inputs with clear error messages
- Handle cases where the query doesn't match any individuals in the ABox

{% if last_error %}

# Previous Attempts

This is an attempt to fix the previous code generation. The code was evaluated against
training dataset and it failed, here's the test case and error:

{{ last_error }}

The exact code we evaluated on:

```python
{{ last_interpreter }}
```

Make sure the next iteration improves with regards to robustness and accuracy.

{% endif %}

# Output Format

Return ONLY the Python code without any explanations or markdown formatting. The code must:
1. Include the exact function signature def calculate(abox: dict, taxpayer_id: str, year: int) -> float
2. Be complete and executable without additional dependencies
3. Follow the chain-of-code technique with clear comment explanations
4. Implement all numerical calculations from the statute
5. Return a single float value as the final result

Do not include any explanation before or after the code - just provide the complete, executable Python code.
Do not wrap the output with any markdown formatting. Plain python only.
""",
)


FACT_EXTRACTION_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=[
        "classes",
        "properties",
        "description",
        "tbox_interpreter_docstring",
    ],
    template_format="jinja2",
    template="""
# Task: Legal Fact Extraction

You are an AI legal assistant tasked with extracting facts from a user situation description about a legal case
and mapping them to a formal ontology schema. Your goal is to produce a complete and accurate ABox (assertion box)
that contains ALL required information needed to answer the user's question.

# TBox Schema

## Classes
{{ classes }}

## Properties
{{ properties }}

# User's Situation Description
{{ description }}

# User's Question
The extracted ABox, will be used to answer the following question:

    {{ question }}

Make sure that all properties required by the target function to answer this question are present in the resulting ABox.

# CRITICAL INSTRUCTIONS - READ CAREFULLY

The calculation function has specific requirements for the ABox structure. These requirements are specified below.
You MUST ensure ALL required properties are extracted, even if they need to be reasonably inferred.

ABox Structure Requirements from the calculation function:
{{ tbox_interpreter_docstring }}

## Step-by-Step Extraction Process

1. First, identify all individuals mentioned in the situation description and assign them appropriate class types.
   * Look for all taxpayers, dependents, employers, etc.
   * Assign meaningful and consistent IDs to each individual

2. Parse the description to find EVERY property mentioned in the text:
   * All explicit property values stated in the text
   * Any properties that can be reasonably inferred
   * Use the class and property definitions to guide your extraction

3. CRITICAL CHECK: Review the ABoxRequirements section above and ensure you extract EVERY required property
   * Make sure all properties needed for calculation are present
   * Pay special attention to financial values, filing status, and year-specific attributes
   * Missing required properties will cause calculation failures

4. For each property, provide:
   * A confidence score (0.0 to 1.0)
   * A specific explanation of why you created this assertion
   * The source in the text that supports this assertion

5. For any numeric values mentioned (incomes, wages, tax thresholds, etc.):
   * Extract ALL numeric values as datatype property assertions
   * Connect each value to the appropriate individual through relevant properties
   * Ensure these are correctly linked to the appropriate entities

# ALWAYS EXTRACT THESE CRITICAL PROPERTIES

Regardless of the specific scenario, ALWAYS extract the following critical properties when relevant:
- Tax filing status (e.g., filing jointly, separately, as head of household, etc.)
- Income and financial information (e.g., hasAdjustedGrossIncomeAmount, hasTaxableIncomeAmount)
- Personal attributes (e.g., age, blindness, dependent status)
- Business attributes (e.g., employer status, wage payments)
- Relationships between individuals (e.g., spouse, dependent)
- Standard deduction information if mentioned

CRITICAL! The input may contain some temporal concepts, not possible to be represented by by the ABox.
Always lean towards the 'year' asked for in the User's Question. Double check if the output is not
contradictory, e.g., [isFoo(x), isNotFoo(x)], but when taken only the year from question, [isFoo(x)] is the only predicate.

# Output Format

Return a single, valid JSON object. Do NOT include any markdown formatting (e.g., ```json) around the JSON object itself.

```json
{
    "abox": {
        "individuals": [
            {
                "object": {
                    "name": "individualname",
                    "class_name": "ClassNameFromTBox"
                },
                "confidence": 0.8,
                "explanation": "brief explanation why this individual was extracted"
        ],
        "assertions": [
            {
                "object": {
                    "property_name": "NameOfTBoxProperty",
                    "arguments": ["ClassNames", "OrLiterals"]
                },
                "confidence": 0.95,
                "explanation": "brief explanation why this property was extracted"
            }
        ]
    }
}
```
""",
)

ANSWER_GENERATION_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=[
        "query",
        "individuals",
    ],
    template="""
Based on the following user query and ABox data, determine the appropriate query tuple to pass to the calculate function.

# User Query:
{query}

# Available Individuals in ABox
{individuals}

# Instruction
1. You MUST ONLY use taxpayer_id values that are EXACTLY MATCHING the individual IDs shown above.
2. Do not invent, modify, or combine IDs. Use the exact strings as they appear in the individuals list.
3. Look carefully at the individual types to identify which ones represent taxpayers.
4. The taxpayer_id must refer to an individual that exists in the ABox.

Your task is to analyze this information and determine which query should be passed to the calculate function.
The query should include:
- taxpayer_id: MUST be an exact match from the list of individual IDs above
- year: the year for the calculation (integer)
""",
)
