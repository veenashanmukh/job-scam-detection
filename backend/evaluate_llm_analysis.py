
"""Evaluate job-scam analysis structure and warning-sign coverage."""

import json
from pathlib import Path

from services.llm_analysis import analyze


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = (
    PROJECT_ROOT / "backend" / "data" / "llm_evaluation_results.json"
)

TEST_CASES = [
    {
        "name": "Clear warning signs",
        "posting": (
            "Remote online job. Earn high daily income by completing simple "
            "tasks. Deposit money before withdrawing your earnings."
        ),
        "expected_signs": [
            "upfront_fee",
            "unrealistic_earnings",
            "task_scam",
        ],
    },
    {
        "name": "Vague job offer",
        "posting": (
            "We are hiring remote assistants. Flexible hours and competitive "
            "pay. Contact us for more information."
        ),
        "expected_signs": [],
    },
    {
        "name": "Ordinary job description",
        "posting": (
            "We are hiring a software developer. Responsibilities include "
            "writing Python code, reviewing pull requests, and fixing bugs. "
            "Applicants should have two years of Python experience."
        ),
        "expected_signs": [],
    },
]


# Basic phrase matching for preliminary manual review.
# This is not a definitive semantic or factual-support checker.
SIGN_CUES = {
    "upfront_fee": (
        "deposit money",
        "pay upfront",
        "upfront payment",
        "pay a fee",
        "pay before starting",
        "pay to get the job",
        "pay to withdraw",
        "withdrawing your earnings",
        "unlock your earnings",
    ),
    "unrealistic_earnings": (
        "high daily income",
        "high earnings",
        "unrealistic earnings",
        "easy money",
        "earn a lot",
        "guaranteed income",
    ),
    "task_scam": (
        "online tasks",
        "simple tasks",
        "complete tasks",
        "task platform",
        "rating or liking",
    ),
    "money_transfer": (
        "transfer money",
        "receive money",
        "send money on behalf",
        "move money",
        "forward payments",
    ),
    "fake_check": (
        "fake check",
        "deposit this check",
        "deposit a cheque",
    ),
    "reshipping": (
        "reship packages",
        "reshipping",
        "receive packages and ship",
        "forward packages",
    ),
}


def check_result(result):
    """Check required response fields and their data types."""
    errors = []

    if not isinstance(result, dict):
        return ["Result is not a dictionary."]

    required_top_level = {
        "analysis",
        "supporting_evidence",
        "model",
    }

    missing = required_top_level - result.keys()
    if missing:
        return [f"Missing top-level keys: {sorted(missing)}"]

    if not isinstance(result["analysis"], dict):
        return ["The analysis field is not a dictionary."]

    if not isinstance(result["supporting_evidence"], list):
        errors.append("supporting_evidence must be a list.")

    analysis = result["analysis"]

    required_analysis = {
        "assessment",
        "explanation",
        "warning_signs",
        "recommended_actions",
        "limitations",
    }

    missing = required_analysis - analysis.keys()
    if missing:
        errors.append(f"Missing analysis fields: {sorted(missing)}")

    for field in ("assessment", "explanation"):
        if not isinstance(analysis.get(field), str):
            errors.append(f"{field} must be a string.")

    for field in (
        "warning_signs",
        "recommended_actions",
        "limitations",
    ):
        value = analysis.get(field)

        if not isinstance(value, list):
            errors.append(f"{field} must be a list.")
        elif not all(isinstance(item, str) for item in value):
            errors.append(f"Every item in {field} must be a string.")

    return errors


def review_warning_signs(posting, warning_signs):
    """Flag warning signs without an obvious matching phrase."""
    posting_lower = posting.casefold()
    flags = []

    for sign in warning_signs:
        if not isinstance(sign, str):
            flags.append({
                "warning_sign": repr(sign),
                "reason": "Warning sign is not a string.",
            })
            continue

        cues = SIGN_CUES.get(sign)

        if cues is None:
            flags.append({
                "warning_sign": sign,
                "reason": (
                    "Unknown warning-sign label. Add it to SIGN_CUES "
                    "or review it manually."
                ),
            })
        elif not any(cue in posting_lower for cue in cues):
            flags.append({
                "warning_sign": sign,
                "reason": (
                    "No matching phrase found. Manually check whether "
                    "the posting supports this warning sign."
                ),
            })

    return flags


def compare_expected_signs(expected_signs, actual_signs):
    """Compare expected labels with model labels.

    Exact label matching is used. A missing label is an evaluation flag,
    not proof of an incorrect answer.
    """
    expected = set(expected_signs)
    actual = set(actual_signs)

    return {
        "missing_expected_signs": sorted(expected - actual),
        "unexpected_signs": sorted(actual - expected),
    }


def main():
    results = []

    for case in TEST_CASES:
        print(f"\nTesting: {case['name']}")

        try:
            result = analyze(case["posting"])
            structure_errors = check_result(result)

            analysis = result.get("analysis", {})
            actual_signs = analysis.get("warning_signs", [])

            if not isinstance(actual_signs, list):
                actual_signs = []

            unsupported_flags = review_warning_signs(
                case["posting"],
                actual_signs,
            )

            coverage = compare_expected_signs(
                case["expected_signs"],
                actual_signs,
            )

            record = {
                "test": case["name"],
                "posting": case["posting"],
                "expected_signs": case["expected_signs"],
                "actual_signs": actual_signs,
                "passed_structure_check": not structure_errors,
                "structure_errors": structure_errors,
                "unsupported_sign_review_flags": unsupported_flags,
                "missing_expected_signs": coverage[
                    "missing_expected_signs"
                ],
                "unexpected_signs": coverage["unexpected_signs"],
                "result": result,
            }

            results.append(record)

            print(
                "Structure:",
                "PASS" if not structure_errors else "FAIL",
            )
            for error in structure_errors:
                print("  Structure error:", error)

            print("Expected signs:", case["expected_signs"])
            print("AI warning signs:", actual_signs)

            if coverage["missing_expected_signs"]:
                print(
                    "Missing expected signs:",
                    coverage["missing_expected_signs"],
                )
            else:
                print("Missing expected signs: None")

            if coverage["unexpected_signs"]:
                print(
                    "Unexpected signs:",
                    coverage["unexpected_signs"],
                )
            else:
                print("Unexpected signs: None")

            if unsupported_flags:
                print("Unsupported-sign review: CHECK")
                for flag in unsupported_flags:
                    print(
                        f"  - {flag['warning_sign']}: {flag['reason']}"
                    )
            else:
                print("Unsupported-sign review: No phrase-match flags")

            print(
                "Evidence sources:",
                [
                    source.get("id", "unknown")
                    for source in result.get("supporting_evidence", [])
                ],
            )

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print("Test failed:", error)

            results.append({
                "test": case["name"],
                "passed_structure_check": False,
                "structure_errors": [error],
                "unsupported_sign_review_flags": [],
                "missing_expected_signs": [],
                "unexpected_signs": [],
            })

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    structure_passed = sum(
        item.get("passed_structure_check", False)
        for item in results
    )
    missing_count = sum(
        len(item.get("missing_expected_signs", []))
        for item in results
    )
    unsupported_count = sum(
        len(item.get("unsupported_sign_review_flags", []))
        for item in results
    )

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Structure checks: {structure_passed}/{len(results)}")
    print(f"Missing expected labels: {missing_count}")
    print(f"Labels flagged for support review: {unsupported_count}")
    print(f"Results saved to: {RESULTS_PATH}")
    print(
        "\nNote: label comparisons and phrase matching are preliminary "
        "checks. Review the evidence manually."
    )


if __name__ == "__main__":
    main()
