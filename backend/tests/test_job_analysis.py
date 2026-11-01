import pytest

from backend.services.job_analysis import build_job_report, classifier_risk_score, risk_category

THRESHOLD = 0.291163


def test_score_stays_in_range_and_agrees_with_threshold():
    previous = -1
    for i in range(0, 1001):
        p = i / 1000
        score = classifier_risk_score(p, THRESHOLD)
        assert 0 <= score <= 100
        assert score >= previous  # never decreases as probability rises
        previous = score
        if p < THRESHOLD:
            assert score <= 29 and risk_category(score) == "Lower risk"
        else:
            assert score >= 30 and risk_category(score) != "Lower risk"


def test_score_end_points():
    assert classifier_risk_score(0.0, THRESHOLD) == 0
    assert classifier_risk_score(1.0, THRESHOLD) == 100
    assert classifier_risk_score(THRESHOLD, THRESHOLD) == 30


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.5])
def test_invalid_threshold_is_rejected(bad):
    with pytest.raises(ValueError):
        classifier_risk_score(0.5, bad)


def test_category_ranges():
    assert risk_category(29) == "Lower risk"
    assert risk_category(30) == "Moderate risk"
    assert risk_category(59) == "Moderate risk"
    assert risk_category(60) == "Higher risk"


def _prediction(probability, flagged):
    return {"model": "DistilBERT", "prediction": int(flagged), "label": "x",
            "fraud_probability": probability, "legitimate_probability": 1 - probability, "threshold": THRESHOLD}


def test_report_has_every_required_field():
    report = build_job_report(_prediction(0.93, True))
    for field in ("risk_score", "risk_category", "classifier_prediction", "warning_signs",
                  "supporting_evidence", "explanation", "uncertainty_notes"):
        assert field in report
    assert report["classifier_prediction"] == "Potentially fraudulent"
    assert report["risk_category"] != "Lower risk"
    assert report["classifier_details"]["threshold"] == THRESHOLD
    assert isinstance(report["warning_signs"], list) and isinstance(report["supporting_evidence"], list)
    assert report["uncertainty_notes"]  # never empty: the score is provisional


def test_unflagged_report_is_lower_risk_and_uses_careful_wording():
    report = build_job_report(_prediction(0.02, False))
    assert report["classifier_prediction"] == "Likely legitimate"
    assert report["risk_category"] == "Lower risk"
    assert "guarantee" in report["explanation"]
    assert "definitely" not in report["explanation"].lower()


def test_no_application_fee_is_not_flagged():
    from backend.services.job_analysis import detect_warning_signs

    text = (
        "Applicants can submit their resume through the careers portal. "
        "No application fee is required."
    )

    findings = detect_warning_signs(text)

    assert not any(
        finding["label"] == "Upfront payment"
        for finding in findings
    )


def test_explicit_application_fee_is_flagged():
    from backend.services.job_analysis import detect_warning_signs

    findings = detect_warning_signs(
        "Applicants must pay an application fee before starting employment."
    )

    assert any(
        finding["label"] == "Upfront payment"
        for finding in findings
    )


# --- three-outcome verdict -------------------------------------------------

from backend.services.job_analysis import job_verdict  # noqa: E402

ORIGINAL_KEYS = {"risk_score", "risk_category", "classifier_risk_score", "warning_adjustment",
                 "classifier_prediction", "classifier_details", "warning_signs", "supporting_evidence",
                 "explanation", "uncertainty_notes"}


@pytest.mark.parametrize("probability, profile, verdict", [
    (0.0, "Acme Ltd builds tools.", "Likely Legitimate"),
    (0.291162, "Acme Ltd builds tools.", "Likely Legitimate"),
    (0.291162, None, "Likely Legitimate"),
    (0.291162, "", "Likely Legitimate"),
    (THRESHOLD, "Acme Ltd builds tools.", "High Fraud Risk"),
    (THRESHOLD, "", "Needs Verification"),
    (THRESHOLD, None, "Needs Verification"),
    (1.0, "Acme Ltd builds tools.", "High Fraud Risk"),
])
def test_verdict_boundaries(probability, profile, verdict):
    assert job_verdict(probability, THRESHOLD, profile)[0] == verdict


def test_missing_and_empty_profile_give_different_reasons():
    missing = job_verdict(0.9, THRESHOLD, None)
    empty = job_verdict(0.9, THRESHOLD, "")
    blank = job_verdict(0.9, THRESHOLD, "  \n\t ")
    assert missing == ("Needs Verification", "company profile not provided")
    assert empty[0] == blank[0] == "Needs Verification"
    assert empty[1] == blank[1] != missing[1]


def test_report_without_profile_argument_needs_verification_when_flagged():
    report = build_job_report(_prediction(0.93, True), "some text")
    assert report["verdict"] == "Needs Verification"
    assert report["verdict_reason"] == "company profile not provided"


def test_every_verdict_has_a_reason():
    for probability, profile in ((0.01, None), (0.9, None), (0.9, ""), (0.9, "Acme")):
        verdict, reason = job_verdict(probability, THRESHOLD, profile)
        assert verdict in {"Likely Legitimate", "Needs Verification", "High Fraud Risk"}
        assert isinstance(reason, str) and reason


@pytest.mark.parametrize("probability, text, expected", [
    # Values produced by build_job_report before the verdict was added.
    (0.93, "Applicants must pay an application fee before starting employment. Act now.",
     {"risk_score": 100, "risk_category": "Higher risk", "classifier_risk_score": 93,
      "warning_adjustment": 7, "classifier_prediction": "Potentially fraudulent"}),
    (0.4, "Software engineer role.",
     {"risk_score": 41, "risk_category": "Moderate risk", "classifier_risk_score": 41,
      "warning_adjustment": 0, "classifier_prediction": "Potentially fraudulent"}),
    (0.02, "Software engineer role.",
     {"risk_score": 2, "risk_category": "Lower risk", "classifier_risk_score": 2,
      "warning_adjustment": 0, "classifier_prediction": "Likely legitimate"}),
])
def test_existing_outputs_unchanged(probability, text, expected):
    prediction = _prediction(probability, probability >= THRESHOLD)
    reports = [build_job_report(prediction, text, profile) for profile in (None, "", "Acme Ltd")]
    for report in reports:
        assert ORIGINAL_KEYS | {"verdict", "verdict_reason", "display_risk_score"} == set(report)
        assert {k: report[k] for k in expected} == expected
        assert report["classifier_details"]["fraud_probability"] == round(probability, 6)
        assert report["classifier_details"]["threshold"] == THRESHOLD
    # The company profile changes only the verdict, never the original fields.
    original = [{k: r[k] for k in ORIGINAL_KEYS - {"classifier_details"}} for r in reports]
    assert original[0] == original[1] == original[2]


def test_no_stale_heuristic_adjustment_note():
    report = build_job_report(_prediction(0.93, True), "Pay a registration fee today. Act now.")
    assert not any("point adjustment" in n or "heuristic score" in n.lower() for n in report["uncertainty_notes"])
    assert report["uncertainty_notes"]  # the provisional-score and model-limit notes remain


def test_logodds_is_passed_through_unrounded():
    prediction = _prediction(0.93, True)
    prediction["fraud_logodds"] = 2.5866893123456789
    report = build_job_report(prediction)
    assert report["classifier_details"]["fraud_logodds"] == 2.5866893123456789
    assert build_job_report(_prediction(0.93, True))["classifier_details"]["fraud_logodds"] is None


# --- display_risk_score ----------------------------------------------------

import math  # noqa: E402

from backend.services.job_analysis import display_risk_score  # noqa: E402

LOGIT_T = math.log(THRESHOLD / (1 - THRESHOLD))


def test_display_score_is_50_at_threshold_and_matches_formula():
    assert display_risk_score(LOGIT_T, THRESHOLD) == 50
    for lo in (-6.0, -2.0, -0.5, 0.0, 0.7, 2.0, 5.0):
        expected = int(min(99, max(1, 100 / (1 + math.exp(-(lo - LOGIT_T) / 3)))))
        assert display_risk_score(lo, THRESHOLD) == expected


def test_display_score_is_int_clamped_and_monotonic():
    previous = 0
    for i in range(-400, 401):
        lo = i / 10
        score = display_risk_score(lo, THRESHOLD)
        assert isinstance(score, int) and 1 <= score <= 99
        assert score >= previous
        previous = score
        # Truncation keeps the 50 line on the decision threshold.
        assert (score >= 50) == (lo >= LOGIT_T)
    assert display_risk_score(-1e6, THRESHOLD) == 1
    assert display_risk_score(1e6, THRESHOLD) == 99


def test_display_score_is_none_without_logodds():
    assert display_risk_score(None, THRESHOLD) is None
    assert build_job_report(_prediction(0.93, True))["display_risk_score"] is None


def test_display_score_does_not_change_existing_fields_or_verdict():
    for probability in (0.02, 0.4, 0.93):
        base = _prediction(probability, probability >= THRESHOLD)
        with_lo = dict(base, fraud_logodds=math.log(probability / (1 - probability)))
        for profile in (None, "", "Acme"):
            a = build_job_report(base, "Pay a fee today.", profile)
            b = build_job_report(with_lo, "Pay a fee today.", profile)
            assert a.pop("display_risk_score") is None
            assert isinstance(b.pop("display_risk_score"), int)
            a["classifier_details"].pop("fraud_logodds")
            b["classifier_details"].pop("fraud_logodds")
            assert a == b
