"""Display-only job result screen. The API fields and scores must not change; the frontend only changes how they
are shown. Structure checks read the frontend source; wording and logic are tested in the Node tests."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from backend.services.job_analysis import build_job_report

BACKEND = Path(__file__).resolve().parents[1]
FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
APP = (FRONTEND / "src" / "App.jsx").read_text(encoding="utf-8")
JOB_REPORT = (FRONTEND / "src" / "JobReport.jsx").read_text(encoding="utf-8")
VIEW = (FRONTEND / "src" / "services" / "reportView.js").read_text(encoding="utf-8")

FORBIDDEN = ("provisional", "probability", "classifier", "threshold")
SENTENCES = [
    "This posting looks low risk. We found no warning signs. Still check the company before sharing documents or money.",
    "This posting looks low risk. We found some minor warning signs. Still check the company before sharing documents or money.",
    "Be careful. Check this employer before you share anything or pay anything.",
    "This looks like a scam. Do not pay or share personal details.",
]

# The main screen is everything in the JobReport component; HowDecided (collapsed) is defined above it.
MAIN = JOB_REPORT[JOB_REPORT.index("export default function JobReport"):]
HOW_DECIDED = JOB_REPORT[JOB_REPORT.index("function HowDecided"):JOB_REPORT.index("export default function JobReport")]
BUBBLE = APP[APP.index("className={`bubble ${m.role}"):APP.index("m === lastWithReport")]


def test_the_plain_sentences_are_in_the_frontend_module():
    for sentence in SENTENCES:
        assert sentence in VIEW


def test_job_reports_use_the_new_screen_and_only_chat_reports_keep_the_old_panel():
    route = 'if (report.type !== "chat") return <JobReport report={report} />;'
    assert route in APP
    assert APP.count("<EvidenceSection") == 1 and APP.index("<EvidenceSection") > APP.index(route)  # chat panel only
    assert "hasDisplayScore" not in APP and "hasVerdict" not in APP


def test_the_main_screen_follows_the_requested_order():
    markers = ["{screen.headline && (", "{screen.sentence && (", "{screen.scoreText && (", "<h3>What we checked</h3>",
               "<h3>What to do next</h3>", "<h3>Why</h3>", "<h3>Similar known postings</h3>",
               "<h3>Linked companies and contacts</h3>", '<p className="footer-line">', "<HowDecided report={report} />"]
    positions = [MAIN.index(marker) for marker in markers]
    assert positions == sorted(positions)


def test_why_similar_and_linked_blocks_only_render_with_content_and_there_is_no_filler():
    for guard in ("{screen.why && (", "{screen.similar.length > 0 && (", "{screen.linked.length > 0 && ("):
        assert guard in MAIN
    assert "{screen.next.length > 0 && (" in MAIN
    for filler in ("No specific evidence found", "None detected", "No close matches", "No linked companies",
                   "AI explanation unavailable", "Evidence</h3>"):
        assert filler not in MAIN


def test_the_main_screen_shows_one_score_in_the_requested_form():
    assert MAIN.count("screen.scoreText") == 2 and "screen.scoreNote" in MAIN  # guard + render
    assert "{screen.scoreText}" in MAIN and "{screen.scoreNote}" in MAIN
    assert 'return {score}' not in MAIN
    for source in (APP, JOB_REPORT):
        assert "classifier_risk_score" not in source  # the second figure is not read by the screen components
    assert "/100" not in VIEW.replace(" / 100", "") and 'label: "Raw classifier probability"' in VIEW  # raw probability only
    assert 'scoreText: typeof score === "number" ? `${score} / 100` : ""' in VIEW
    assert 'export const SCORE_NOTE = "Higher means more risk";' in VIEW


def test_how_this_was_decided_is_collapsed_by_default_and_holds_everything_technical():
    assert JOB_REPORT.count("How this was decided") == 1
    open_tag = HOW_DECIDED[HOW_DECIDED.index("<details"):].split(">")[0]
    assert open_tag == '<details className="how-decided"' and "open" not in open_tag
    assert "<summary>How this was decided</summary>" in HOW_DECIDED
    for part in ("tech.rows.map", "tech.rules.length > 0", "<GuidanceCards", "tech.notes.length > 0", "report.source_text"):
        assert part in HOW_DECIDED
    for part in ("tech.rows", "tech.rules", "GuidanceCards", "tech.notes", "verdict_reason", "report.source_text"):
        assert part not in MAIN.replace("<HowDecided report={report} />", "")


def test_forbidden_words_are_not_in_the_main_screen_code_or_the_chat_bubble():
    for name, source in (("JobReport main", MAIN), ("chat bubble", BUBBLE)):
        lowered = source.lower()
        for word in FORBIDDEN:
            assert word not in lowered, f"{word!r} in {name}"
    assert "Risk score {clampDisplayScore(m.report.display_risk_score)} / 100." in BUBBLE and "Higher means more risk." in BUBBLE
    assert "escalationNote" not in APP  # the note mentions the classifier; it now lives in the collapsed section
    assert "{(m.report && plainSentence(m.report)) || splitExplanationNote(m.text).body}" in BUBBLE


def test_the_chat_bubble_uses_the_same_sentence_and_no_second_score():
    assert "plainSentence(m.report)" in BUBBLE and "classifier" not in BUBBLE.lower()
    assert len(re.findall(r"Risk score \{clampDisplayScore\(m\.report\.display_risk_score\)\}", BUBBLE)) == 1


def test_the_company_description_tick_uses_the_backend_field_not_a_client_side_guess():
    assert "company_profile_provided" not in APP and "profileText" not in APP     # the browser no longer decides this
    route = (BACKEND / "routes" / "job_routes.py").read_text(encoding="utf-8")
    assert 'report["company_profile_present"] = bool(company_profile and str(company_profile).strip())' in route
    assert "report.company_profile_present === true" in VIEW and "company_profile_provided" not in VIEW


def _prediction(probability, threshold=0.291163):
    return {"fraud_probability": probability, "threshold": threshold, "prediction": int(probability >= threshold),
            "fraud_logodds": -2.0 if probability < 0.3 else 1.5, "model": "DistilBERT"}


EXPECTED_KEYS = {"classifier_details", "classifier_prediction", "classifier_risk_score", "display_risk_score",
                 "explanation", "risk_category", "risk_score", "supporting_evidence", "uncertainty_notes", "verdict",
                 "verdict_reason", "warning_adjustment", "warning_signs"}


@pytest.mark.parametrize("probability, text, profile, expected", [
    (0.05, "Interviews are held on video with the team.", "",
     {"risk_score": 5, "risk_category": "Lower risk", "classifier_risk_score": 5, "warning_adjustment": 0,
      "classifier_prediction": "Likely legitimate", "verdict": "Likely Legitimate", "display_risk_score": 40,
      "verdict_reason": "fraud probability is below the decision threshold"}),
    (0.9, "Pay a registration fee today.", "A Firm",
     {"risk_score": 100, "risk_category": "Higher risk", "classifier_risk_score": 90, "warning_adjustment": 10,
      "classifier_prediction": "Potentially fraudulent", "verdict": "High Fraud Risk", "display_risk_score": 68,
      "verdict_reason": "classifier flagged the posting even though a company profile is present"}),
    (0.05, "We will mail you a cheque for $2,000. Deposit it and wire the balance to our supplier.", "",
     {"risk_score": 15, "risk_category": "Lower risk", "classifier_risk_score": 5, "warning_adjustment": 10,
      "classifier_prediction": "Likely legitimate", "verdict": "Needs Verification", "display_risk_score": 40,
      "verdict_reason": "specific scam-pattern indicator present (Cheque or overpayment scam)"}),
])
def test_api_fields_and_scores_are_exactly_what_they_were(probability, text, profile, expected):
    report = build_job_report(_prediction(probability), text, profile)
    assert set(report) == EXPECTED_KEYS  # no field added, removed or renamed
    assert {key: report[key] for key in expected} == expected
    assert report["classifier_details"]["fraud_probability"] == probability and report["classifier_details"]["threshold"] == 0.291163


def test_the_classifier_score_is_still_in_the_api_output():
    report = build_job_report(_prediction(0.05), "Interviews are held on video.", "")
    assert report["classifier_risk_score"] == 5 and report["classifier_prediction"] == "Likely legitimate"


def test_no_backend_file_mentions_the_display_modules():
    for path in BACKEND.rglob("*.py"):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "reportView" not in text and "JobReport" not in text, path


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_tests_for_the_report_view_main_screen_and_rendered_html_pass():
    result = subprocess.run(["node", "--test", "src/services/reportView.test.mjs", "src/services/mainScreen.test.mjs",
                             "src/services/jobReportRender.test.mjs", "src/services/cautionScreen.test.mjs"],
                            cwd=FRONTEND, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


def test_the_job_screen_never_ties_linked_or_similar_blocks_to_the_ai_status():
    assert "llm_status" not in JOB_REPORT and "aiAvailable" not in JOB_REPORT
    assert "llm_status" not in VIEW[VIEW.index("export function buildMainScreen"):VIEW.index("export function mainScreenText")]


def test_the_old_score_wording_is_gone_from_all_frontend_code():
    for path in (FRONTEND / "src").rglob("*"):
        if path.suffix in {".js", ".jsx", ".css", ".html"}:
            assert "Lower is safer" not in path.read_text(encoding="utf-8"), path
    for path in (FRONTEND / "src" / "services").glob("*.test.mjs"):
        text = path.read_text(encoding="utf-8")
        # only the tests that assert the old wording is absent may mention it
        assert "Lower is safer" not in text or path.name == "cautionScreen.test.mjs", path.name


def test_tick_lines_are_only_built_for_likely_legitimate_and_the_caution_line_is_defined():
    build = VIEW[VIEW.index("export function buildMainScreen"):VIEW.index("export function mainScreenText")]
    assert 'const showTicks = report.verdict === "Likely Legitimate";' in build
    assert "if (showTicks && Array.isArray(report.warning_signs))" in build
    assert "if (showTicks && report.company_profile_present === true)" in build
    assert "if (showTicks && similar.length > 0 && similar.every((s) => s.genuine))" in build
    assert 'if (cautious && firedLabels.length === 0)' in build
    assert 'export const RESEMBLES_SCAMS_LINE = "The wording of this posting resembles postings that were reported as scams.";' in VIEW
    assert "if (label === MONEY_TICK_LABEL && paymentFired) continue;" in build


def test_why_is_shown_only_for_an_accepted_explanation_that_cites_evidence():
    why = VIEW[VIEW.index("export function whyView"):]
    why = why[:why.index("\n}\n")]
    assert 'report.llm_status !== "ok"' in why and "citedEvidenceIds(report).length === 0) return null" in why
    assert r"/^[SGRK]\d+$/" in VIEW
    assert "{screen.why && (" in MAIN
    # the pipeline that accepts or rejects explanations is not touched by this display rule
    explainer = (BACKEND / "services" / "llm_explainer.py").read_text(encoding="utf-8")
    assert "citedEvidenceIds" not in explainer


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_tests_for_the_why_rule_pass():
    result = subprocess.run(["node", "--test", "src/services/whyCitations.test.mjs"], cwd=FRONTEND,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
