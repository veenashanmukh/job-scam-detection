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
