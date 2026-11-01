"""Category-level job-posting rules and the escalation-only verdict rule. Fresh wording, not taken from the synthetic files."""
import pytest

from backend.services import job_analysis as ja
from backend.services.job_analysis import (
    ESCALATION_MIN_CATEGORIES, VERDICT_HIGH_RISK, VERDICT_LEGITIMATE, VERDICT_NEEDS_VERIFICATION,
    build_job_report, detect_warning_signs, escalate_verdict, job_verdict,
)

CHEQUE = "Cheque or overpayment scam"
PAYMENT = "Upfront payment"
GIFT = "Gift card or crypto payment"
NO_INTERVIEW = "No interview"
PERSONAL = "Requests personal information"
FORWARD = "Parcel or money forwarding"

POSITIVE = {
    CHEQUE: [
        "We will mail you a cheque for $2,500; deposit it and wire the remaining amount to our equipment vendor.",
        "A bank draft arrives by courier. Cash it, keep your first week's wage and transfer the rest to the supplier.",
        "You will receive a money order. Pay it into your account, then forward the difference to our shipping agent.",
        "The check is larger than your pay because of supply costs; remit the excess to the vendor.",
        "If we overpay you by mistake, wire the extra back to our account.",
        "Your first cheque will include funds for supplies. Transfer the balance to our approved supplier within a day.",
        "Deposit the cheque we send, then pass on the remaining funds to a third-party agent.",
    ],
    PAYMENT: [
        "You must pay a security deposit of $200 before your laptop ships.",
        "Candidates are asked to send a documentation fee of 50 to hold the place.",
        "Kindly transfer the activation charge to the account below.",
        "The courier cost has to be paid by the applicant in advance.",
        "Remit a medical fee of $120 by transfer.",
        "Insurance fee of $75 is required from every new starter.",
        "Pay $300 registration to confirm your slot.",
        "Applicants wire a visa processing charge before travel documents are issued.",
    ],
    GIFT: [
        "Purchase gift cards worth $200 from a local shop and send us the codes.",
        "Buy two iTunes vouchers to activate your account.",
        "Top up your task wallet with USDT to unlock the next batch of work.",
        "Deposit 100 in bitcoin first and you can start working.",
        "A starter balance of 200 in Tether is needed for each task set.",
        "To unlock your tasks you must fund the account with crypto.",
        "Send a prepaid card number to confirm your identity.",
    ],
    NO_INTERVIEW: [
        "No experience or interview required, apply today.",
        "You can start without any interview.",
        "The interview is not required for this role.",
        "We skip the interview and hire everyone who applies.",
        "Honestly, you don't need an interview here.",
        "No interviews are conducted; offers go out within the hour.",
        "Selected candidates are hired on the spot, no interview, no CV.",
        "You are hired instantly after registering.",
    ],
    PERSONAL: [
        "Please send a photo of your passport and a selfie to start.",
        "We need your card number, expiry date and CVV to set up payroll.",
        "Share the OTP you receive by SMS so we can verify you.",
        "Provide your online banking password for the salary setup.",
        "Upload a scan of your driving licence before we send the offer.",
        "Reply with the verification code sent to your phone.",
        "Tell us your social security number today.",
    ],
    FORWARD: [
        "Receive parcels at your home address and forward them to our clients abroad.",
        "Your job is reshipping packages to overseas customers.",
        "You will repackage goods and post them on.",
        "Forward the payments you receive to our agent and keep a 10% commission.",
        "Accept transfers on behalf of our company and pass the money on to our suppliers.",
        "Forwarding parcels to another address is your daily task.",
        "Pass on funds from customers, and you keep your share.",
    ],
}

NEGATIVE = {
    CHEQUE: [
        "Salary is paid by bank transfer on the last working day.",
        "Please check your email for the interview invitation and transfer your files to the shared drive.",
        "We never ask you to deposit a cheque and send money on.",
        "Reimbursements for approved travel are paid by cheque after you submit receipts.",
        "Our accountant will transfer the balance of your signing bonus after probation.",
        "We do not issue cheques in advance, and no one should wire funds to a stranger.",
        "Payroll cheques are collected from the front desk every Friday; staff may transfer their pay to savings.",
    ],
    PAYMENT: [
        "We pay for your training and provide all equipment at no cost.",
        "The company will cover your visa and medical expenses.",
        "You will never be asked to pay a registration fee.",
        "Our onboarding kit is free of charge.",
        "Employees transfer money to their savings accounts through payroll deductions.",
        "Pay is weekly, by direct deposit into your bank account.",
        "Transfer students and returning interns are welcome to apply.",
        "Applicants do not pay any fee at any stage.",
        "Training costs are covered by the employer.",
    ],
    GIFT: [
        "We pay salaries by bank transfer; no gift cards or crypto are ever used.",
        "Experience with blockchain and Bitcoin infrastructure is a plus.",
        "Employees receive a gift card on their work anniversary.",
        "Our staff discount lets you buy products in store.",
        "You will never be asked to buy gift cards.",
        "We send every new starter a welcome gift card.",
        "Candidates can redeem referral bonuses as vouchers after six months.",
    ],
    NO_INTERVIEW: [
        "Interviews are held on video with the hiring manager.",
        "There is no experience requirement, but a strong portfolio helps.",
        "The final interview is with the team lead; no interview fee ever applies.",
        "We will schedule an interview within a week of your application.",
        "Candidates who skip the assessment stage cannot be considered.",
        "No interview rooms are available in the old building, so we use video.",
        "Please do not skip your interview slot.",
    ],
    PERSONAL: [
        "We will never ask for your OTP or card details.",
        "Bring your passport on your first day for the right-to-work check.",
        "After you accept the offer, please send a copy of your passport for onboarding.",
        "Do not share your password with anyone, including recruiters.",
        "Send your CV and a short cover letter to the address below.",
        "Our background check provider will ask for your ID after the offer is signed.",
        "Upload a portfolio of your recent work.",
    ],
    FORWARD: [
        "Warehouse staff pack and ship customer orders from our depot.",
        "We forward your CV to hiring managers within two days.",
        "Customers receive parcels at their home address within three days.",
        "Accounts staff pass on supplier invoices to the finance manager.",
        "We never ask employees to forward payments or keep commission.",
        "The courier role involves forwarding parcels between our own depots.",
        "You will receive your salary payment on the 28th and keep your payslip.",
    ],
}


def _labels(text):
    return [sign["label"] for sign in detect_warning_signs(text)]


@pytest.mark.parametrize("category", POSITIVE)
def test_each_category_has_enough_phrasings(category):
    assert len(POSITIVE[category]) >= 6 and len(NEGATIVE[category]) >= 6


@pytest.mark.parametrize("category, text", [(c, t) for c, texts in POSITIVE.items() for t in texts])
def test_category_fires_on_positive_phrasing(category, text):
    assert category in _labels(text), text


@pytest.mark.parametrize("category, text", [(c, t) for c, texts in NEGATIVE.items() for t in texts])
def test_category_does_not_fire_on_negative_phrasing(category, text):
    assert category not in _labels(text), text


def test_matched_text_is_part_of_the_posting_and_each_label_appears_once():
    text = ("You are hired instantly. Send your bank login today. Pay a $90 training fee. "
            "Buy gift cards. Deposit the cheque and wire the balance to the supplier. Reship parcels.")
    signs = detect_warning_signs(text)
    labels = [s["label"] for s in signs]
    assert len(labels) == len(set(labels)) and {CHEQUE, PAYMENT, GIFT, NO_INTERVIEW, PERSONAL, FORWARD} <= set(labels)
    assert all(s["matched_text"] in text for s in signs)


def test_base_rules_and_labels_are_unchanged():
    labels = [label for label, _ in ja.WARNING_RULES]
    assert labels == ["Upfront payment", "Pressure tactic", "Guaranteed employment", "No interview",
                      "Unusually high salary claim", "Requests personal information"]
    assert [s["matched_text"] for s in detect_warning_signs("Pay a registration fee now.")][:1] == ["registration fee"]


# --- escalation ----------------------------------------------------------------------------------

def _signs(*labels):
    return [{"label": label, "matched_text": "x"} for label in labels]


def test_two_distinct_categories_escalate_a_likely_legitimate_verdict():
    verdict, reason = escalate_verdict(VERDICT_LEGITIMATE, "below threshold", _signs(CHEQUE, PERSONAL))
    assert verdict == VERDICT_NEEDS_VERIFICATION
    assert reason == f"scam-pattern indicators present ({CHEQUE}, {PERSONAL})"
    assert ESCALATION_MIN_CATEGORIES == 2


def test_one_non_specific_category_or_a_repeated_one_does_not_escalate():
    for signs in (_signs(PAYMENT), _signs(PERSONAL), _signs(NO_INTERVIEW), _signs("Pressure tactic"),
                  _signs(PAYMENT, PAYMENT), [], [{"label": ""}, {"label": "  "}]):
        assert escalate_verdict(VERDICT_LEGITIMATE, "r", signs) == (VERDICT_LEGITIMATE, "r")


@pytest.mark.parametrize("category", [CHEQUE, GIFT, FORWARD])
def test_each_specific_category_alone_escalates_and_the_reason_names_it(category):
    assert escalate_verdict(VERDICT_LEGITIMATE, "below threshold", _signs(category)) == (
        VERDICT_NEEDS_VERIFICATION, f"specific scam-pattern indicator present ({category})")
    assert category in ja.SINGLE_ESCALATION_CATEGORIES


def test_only_the_three_specific_categories_escalate_alone():
    assert set(ja.SINGLE_ESCALATION_CATEGORIES) == {CHEQUE, GIFT, FORWARD}
    others = {label for label, _ in ja.WARNING_RULES} | {r.label for r in ja.CATEGORY_RULES}
    for label in others - set(ja.SINGLE_ESCALATION_CATEGORIES):
        assert escalate_verdict(VERDICT_LEGITIMATE, "r", _signs(label)) == (VERDICT_LEGITIMATE, "r"), label


def test_specific_category_plus_another_uses_the_multi_category_reason():
    verdict, reason = escalate_verdict(VERDICT_LEGITIMATE, "r", _signs(CHEQUE, NO_INTERVIEW))
    assert verdict == VERDICT_NEEDS_VERIFICATION and reason == f"scam-pattern indicators present ({CHEQUE}, {NO_INTERVIEW})"


@pytest.mark.parametrize("verdict", [VERDICT_NEEDS_VERIFICATION, VERDICT_HIGH_RISK])
@pytest.mark.parametrize("category", [CHEQUE, GIFT, FORWARD])
def test_a_specific_category_never_changes_a_higher_verdict(verdict, category):
    assert escalate_verdict(verdict, "original reason", _signs(category)) == (verdict, "original reason")


CHEQUE_ONLY_TEXT = "We will mail you a cheque for $2,000. Deposit it and wire the balance to our supplier."


def test_a_lone_specific_category_escalates_the_report_without_changing_any_score(monkeypatch):
    report = build_job_report(_prediction(0.05), CHEQUE_ONLY_TEXT, "")
    assert [s["label"] for s in report["warning_signs"]] == [CHEQUE]
    assert report["verdict"] == VERDICT_NEEDS_VERIFICATION
    assert report["verdict_reason"] == f"specific scam-pattern indicator present ({CHEQUE})"
    monkeypatch.setattr(ja, "escalate_verdict", lambda v, r, s: (v, r))
    plain = build_job_report(_prediction(0.05), CHEQUE_ONLY_TEXT, "")
    assert plain["verdict"] == VERDICT_LEGITIMATE
    for key in ("risk_score", "display_risk_score", "risk_category", "classifier_risk_score", "warning_adjustment",
                "classifier_prediction", "classifier_details"):
        assert report[key] == plain[key], key


@pytest.mark.parametrize("verdict", [VERDICT_NEEDS_VERIFICATION, VERDICT_HIGH_RISK])
def test_escalation_never_lowers_or_rewrites_a_higher_verdict(verdict):
    assert escalate_verdict(verdict, "original reason", _signs(CHEQUE, PERSONAL, GIFT)) == (verdict, "original reason")


def _prediction(probability, threshold=0.291163):
    return {"fraud_probability": probability, "threshold": threshold, "prediction": int(probability >= threshold),
            "fraud_logodds": 0.0, "model": "DistilBERT"}


SCAMMY = "You are hired instantly. We need your online banking password and a selfie today."
BENIGN_ONE_CATEGORY = "Interviews are on video. No experience or interview required for the optional workshop."


def test_report_escalates_but_scores_are_exactly_what_they_would_be_without_the_rule(monkeypatch):
    report = build_job_report(_prediction(0.05), SCAMMY, "")
    assert report["verdict"] == VERDICT_NEEDS_VERIFICATION
    assert report["verdict_reason"].startswith("scam-pattern indicators present (")
    monkeypatch.setattr(ja, "escalate_verdict", lambda v, r, s: (v, r))
    plain = build_job_report(_prediction(0.05), SCAMMY, "")
    assert plain["verdict"] == VERDICT_LEGITIMATE
    for key in ("risk_score", "display_risk_score", "risk_category", "classifier_risk_score", "warning_adjustment",
                "classifier_prediction"):
        assert report[key] == plain[key], key


def test_benign_text_with_one_category_is_not_escalated():
    report = build_job_report(_prediction(0.05), BENIGN_ONE_CATEGORY, "Some firm")
    assert [s["label"] for s in report["warning_signs"]] == [NO_INTERVIEW]
    assert report["verdict"] == VERDICT_LEGITIMATE


def test_a_flagged_classifier_verdict_is_untouched_by_escalation():
    flagged = build_job_report(_prediction(0.9), SCAMMY, "A Firm")
    assert flagged["verdict"] == job_verdict(0.9, 0.291163, "A Firm")[0] == VERDICT_HIGH_RISK
    assert flagged["verdict_reason"] == job_verdict(0.9, 0.291163, "A Firm")[1]


def test_job_verdict_itself_is_unchanged_by_the_escalation_rule():
    assert job_verdict(0.05, 0.291163, "") == (VERDICT_LEGITIMATE, "fraud probability is below the decision threshold")
