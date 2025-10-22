def calculate(abox: dict, taxpayer_id: str, year: int) -> float:
    """
    Calculates the numeric result based on the specified taxpayer ID, year and circumstances
    described by the abox.

    <ABoxRequirements>
    The abox dictionary must have the following structure:
      {
        "individuals": {
          "id1": {"type": "ClassType1"},
          "id2": {"type": "ClassType2"},
          ...
        },
        "assertions": [
          {"predicate": "PropertyName1", "args": ["id1"]},
          {"predicate": "PropertyName2", "args": ["id1", "value"]},
          ...
        ]
      }

    - "individuals" is a dictionary where keys are entity IDs (e.g., "T1", "E1") and values
      are dictionaries with at least a "type" field, for example: {"type": "Taxpayer"}.

    - "assertions" is a list of property assertions. Each assertion has:
       {
         "predicate": "PropertyName",
         "args": ["subjectID", "objectIDOrValue"]
       }

    Required properties and usage in this calculation:
      1) "isEmployer" (Taxpayer): If present, the taxpayer is subject to FUTA tax under §3301.
         Then "hasTotalWagesPaid" must be found to determine total wages. FUTA rate is 6%.
      2) One of the following sets to determine filing status for income tax under §1:
         - "isSurvivingSpouse" (Taxpayer)
         - "filesJointReturn" (Taxpayer) AND "isMarriedIndividual" (Taxpayer)
         - "IsTaxedAsMarriedFilingSeparately" (Taxpayer)
         - "isHeadOfHousehold" (Taxpayer)
         - "IsTaxedAsUnmarriedIndividual" (Taxpayer) [fallback if none apply => single]
      3) "hasTaxableIncomeAmount" (Taxpayer, decimal): Only if present, this is used directly for income tax computation.
      4) "hasAdjustedGrossIncomeAmount" (Taxpayer, decimal): If needed to compute TI.
      5) If a property needed for the calculation is missing, an exception is raised.

    Example minimal ABox:
    {
      "individuals": {
        "T1": {"type": "Taxpayer"}
      },
      "assertions": [
        {"predicate": "isEmployer", "args": ["T1"]},
        {"predicate": "hasTotalWagesPaid", "args": ["T1", "40000"]},
        {"predicate": "filesJointReturn", "args": ["T1"]},
        {"predicate": "isMarriedIndividual", "args": ["T1"]},
        {"predicate": "hasAdjustedGrossIncomeAmount", "args": ["T1", "70000"]},
        {"predicate": "isAged", "args": ["T1"]},
        {"predicate": "claimsDependent", "args": ["T1", "D1"]}
      ]
    }
    </ABoxRequirements>
    """

    # Step 1: Validate that the taxpayer_id is in the ABox individuals.
    individuals = abox.get("individuals", {})
    if taxpayer_id not in individuals:
        raise ValueError(f"Taxpayer ID '{taxpayer_id}' not found in the ABox.")

    # Step 2: Gather all assertions in separate structures for straightforward lookup.
    # We identify unary predicates, datatype-based predicates, and object-based predicates.
    assertions = abox.get("assertions", [])
    unary_predicates_for_taxpayer = set()
    datatype_values_for_taxpayer = {}
    all_assertions_for_taxpayer = []

    for assertion in assertions:
        predicate = assertion.get("predicate")
        args = assertion.get("args", [])
        if not predicate or not isinstance(args, list):
            continue

        # Case 1: unary property => only 1 argument
        if len(args) == 1:
            if args[0] == taxpayer_id:
                # This is a unary property about our taxpayer
                unary_predicates_for_taxpayer.add(predicate)

        # Case 2: datatype property => 2 arguments, second is a numeric or literal
        elif len(args) == 2:
            subject, obj = args[0], args[1]
            if subject == taxpayer_id:
                # Attempt numeric conversion
                try:
                    numeric_val = float(obj)
                    datatype_values_for_taxpayer[predicate] = numeric_val
                except ValueError:
                    # Not numeric => store as object
                    all_assertions_for_taxpayer.append((predicate, subject, obj))
            else:
                # Not about our taxpayer => store for reference anyway
                all_assertions_for_taxpayer.append((predicate, subject, obj))
        else:
            # More complex => store separately
            all_assertions_for_taxpayer.append((predicate, args))

    is_nonresident_alien = "isNonresidentAlien" in unary_predicates_for_taxpayer

    spouse_id = None
    if "filesJointReturn" in unary_predicates_for_taxpayer:
        for assertion in assertions:
            if (
                assertion.get("predicate") == "hasSpouse"
                and assertion.get("args")[0] == taxpayer_id
            ):
                spouse_id = assertion.get("args")[1]
                break

    # Combine AGI for joint returns
    if spouse_id and "filesJointReturn" in unary_predicates_for_taxpayer:
        spouse_agi = 0.0
        for assertion in assertions:
            if (
                assertion.get("predicate") == "hasAdjustedGrossIncomeAmount"
                and assertion.get("args")[0] == spouse_id
            ):
                try:
                    spouse_agi = float(assertion.get("args")[1])
                except (ValueError, TypeError):
                    pass
                break

        # If both taxpayer and spouse have AGI, combine them
        if "hasAdjustedGrossIncomeAmount" in datatype_values_for_taxpayer:
            taxpayer_agi = datatype_values_for_taxpayer["hasAdjustedGrossIncomeAmount"]
            combined_agi = taxpayer_agi + spouse_agi
            datatype_values_for_taxpayer["hasAdjustedGrossIncomeAmount"] = combined_agi

    # Step 3: Compute FUTA tax if taxpayer is subject to excise tax under §3301.
    # Check for "isEmployer". Then read "hasTotalWagesPaid".
    futa_tax = 0.0
    if "isEmployer" in unary_predicates_for_taxpayer:
        if "hasTotalWagesPaid" not in datatype_values_for_taxpayer:
            raise ValueError(
                "Taxpayer is subject to excise tax but 'hasTotalWagesPaid' is missing."
            )
        total_wages = datatype_values_for_taxpayer["hasTotalWagesPaid"]
        futa_tax = 0.06 * total_wages

    # Step 4: Determine filing status for income tax from the relevant properties.
    def get_filing_status():
        if is_nonresident_alien:
            return "single"
        if "isSurvivingSpouse" in unary_predicates_for_taxpayer:
            return "mfj"
        if (
            "filesJointReturn" in unary_predicates_for_taxpayer
            and "isMarriedIndividual" in unary_predicates_for_taxpayer
        ):
            return "mfj"
        if "IsTaxedAsMarriedFilingSeparately" in unary_predicates_for_taxpayer:
            return "mfs"
        if "isHeadOfHousehold" in unary_predicates_for_taxpayer:
            return "hoh"
        if "IsTaxedAsUnmarriedIndividual" in unary_predicates_for_taxpayer:
            return "single"
        return "single"

    filing_status = get_filing_status()
    if is_nonresident_alien:
        filing_status = "single"
        spouse_id = None

    # Step 5: Obtain or compute taxable income per §§63, 151, 152, etc.
    # If we have 'hasTaxableIncomeAmount', use it. Otherwise, compute:
    #   TI = AGI - standard_deduction - personal_exemptions
    def count_claimed_dependents():
        count = 0
        for a in assertions:
            if a.get("predicate") == "claimsDependent":
                cargs = a.get("args", [])
                if len(cargs) == 2 and cargs[0] == taxpayer_id:
                    count += 1
        return count

    def standard_deduction_calc(
        status: str, year_num: int, aged: bool, blind: bool
    ) -> float:
        if is_nonresident_alien:
            return 0.0
        # Base standard deduction from §§63(c)(2) & (7).
        if 2018 <= year_num < 2026:
            if status == "mfj":
                base = 24000.0
            elif status == "hoh":
                base = 18000.0
            else:
                # single or mfs
                base = 12000.0
        else:
            # year < 2018 or year >= 2026
            if status == "mfj":
                base = 6000.0
            elif status == "hoh":
                base = 4400.0
            else:
                # single or mfs
                base = 3000.0

        # Additional amounts for aged/blind from §63(f).
        # $600 if married or surviving spouse, otherwise $750
        additional = 0.0
        if aged:
            if status == "mfj":
                additional += 600.0
            else:
                additional += 750.0
        if blind:
            if status == "mfj":
                additional += 600.0
            else:
                additional += 750.0

        return base + additional

    if "hasTaxableIncomeAmount" in datatype_values_for_taxpayer:
        taxable_income = datatype_values_for_taxpayer["hasTaxableIncomeAmount"]
    else:
        # Need AGI
        if "hasAdjustedGrossIncomeAmount" not in datatype_values_for_taxpayer:
            # Check if we have wages information
            if "hasTotalWagesPaid" in datatype_values_for_taxpayer:
                # Use wages as AGI if no specific AGI is given
                datatype_values_for_taxpayer["hasAdjustedGrossIncomeAmount"] = (
                    datatype_values_for_taxpayer["hasTotalWagesPaid"]
                )
            else:
                raise ValueError(
                    "Cannot compute taxable income because 'hasAdjustedGrossIncomeAmount' is missing."
                )
        agi = datatype_values_for_taxpayer["hasAdjustedGrossIncomeAmount"]
        is_aged = "isAged" in unary_predicates_for_taxpayer
        is_blind = "isBlind" in unary_predicates_for_taxpayer

        std_ded = standard_deduction_calc(filing_status, year, is_aged, is_blind)

        # Personal exemption simplified from §151(d)(5): 0 for 2018-2025, else $2000 each
        # ignoring spouse logic, ignoring phaseouts, ignoring whether the taxpayer is a
        # dependent of someone else.
        if year < 2018 or year >= 2026:
            # $2,000 was the personal exemption amount for 2015
            pe_amount = 2000.0  # Adjust if needed for other years
        else:
            pe_amount = 0.0  # 2018-2025 has zero personal exemptions

        dep_count = count_claimed_dependents()
        # 1 personal exemption for taxpayer + one for each dependent if year < 2018 or year >= 2026
        total_pe = (1 + dep_count) * pe_amount

        computed_ti = agi - std_ded - total_pe
        taxable_income = computed_ti if computed_ti > 0 else 0.0

    # Step 6: Calculate income tax from the statute's brackets in §1 for each filing status.
    def calc_income_tax(ti: float, status: str) -> float:
        if status == "mfj":
            # §1(a) thresholds
            brackets = [
                (36900.0, 0.15, 0.0, 0.0),
                (89150.0, 0.28, 5535.0, 36900.0),
                (140000.0, 0.31, 20165.0, 89150.0),
                (250000.0, 0.36, 35928.50, 140000.0),
                (1e15, 0.396, 75528.50, 250000.0),
            ]
        elif status == "hoh":
            # §1(b)
            brackets = [
                (29600.0, 0.15, 0.0, 0.0),
                (76400.0, 0.28, 4440.0, 29600.0),
                (127500.0, 0.31, 17544.0, 76400.0),
                (250000.0, 0.36, 33385.0, 127500.0),
                (1e15, 0.396, 77485.0, 250000.0),
            ]
        elif status == "mfs":
            # §1(d)
            brackets = [
                (18450.0, 0.15, 0.0, 0.0),
                (44575.0, 0.28, 2767.50, 18450.0),
                (70000.0, 0.31, 10082.50, 44575.0),
                (125000.0, 0.36, 17964.25, 70000.0),
                (1e15, 0.396, 37764.25, 125000.0),
            ]
        else:
            # single => §1(c)
            brackets = [
                (22100.0, 0.15, 0.0, 0.0),
                (53500.0, 0.28, 3315.0, 22100.0),
                (115000.0, 0.31, 12107.0, 53500.0),
                (250000.0, 0.36, 31172.0, 115000.0),
                (1e15, 0.396, 79772.0, 250000.0),
            ]

        total_tax = 0.0
        for boundary, rate, base_tax, base_income in brackets:
            if ti <= boundary:
                return base_tax + rate * (ti - base_income)
        return total_tax

    income_tax = 0.0
    if taxable_income > 0:
        income_tax = calc_income_tax(taxable_income, filing_status)
        if income_tax < 0:
            income_tax = 0.0

    # Step 7: Combine FUTA tax and income tax to get the final result and return it.
    return float(futa_tax + income_tax)
