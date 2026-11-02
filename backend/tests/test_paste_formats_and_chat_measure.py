"""Paste formats end to end (frontend tidy-up -> chat route) and the chat-rules measurement script.
No rule, score, threshold or verdict logic is changed by any of this."""
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

flask = pytest.importorskip("flask")
from flask import Flask  # noqa: E402

from backend.routes.chat_routes import chat_bp  # noqa: E402
from backend.services.conversation_parser import parse_chat_text  # noqa: E402

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import eval_chat_rules_on_val as cm  # noqa: E402

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

C1 = ("Recruiter: Hello, we found your profile and have a data entry role for you.\nYou: Thanks, which company is this?\n"
      "Recruiter: Pay the registration fee of Rs 3000 within 2 hours.\nRecruiter: Send your bank details and add me on Telegram.")
C2 = ("[24/10/26, 9:02 AM] Recruiter: Hello, we found your profile.\n[24/10/26, 9:03 AM] You: Thanks, can you share the details?\n"
      "[24/10/26, 9:05 AM] Recruiter: Pay the registration fee of Rs 3000 within 2 hours.\n"
      "[24/10/26, 9:06 AM] Recruiter: Send your bank details and add me on Telegram.")
C3 = ("24/10/26, 9:00 - Priya: Why do I need to pay? Is my data safe?\n24/10/26, 9:01 - HR Desk: Pay the registration fee of Rs 3000 within 2 hours.\n"
      "24/10/26, 9:02 - Priya: I want to speak to someone first.\n24/10/26, 9:03 - HR Desk: Send your bank details and add me on Telegram.")


def _formats(text):
    lines = text.split("\n")
    return {"lines": text, "paragraph": " ".join(lines), "messy": "\n\n\t" + "\n\n\t".join(lines).replace(": ", ":\t") + "\n",
            "windows": "\r\n".join(lines) + "\r\n"}


def _tidy(texts):
    """Run the real frontend tidy-up (analyzePaste) in Node on each text."""
    url = (FRONTEND / "src" / "services" / "conversation.js").resolve().as_uri()
    code = (f'import {{ analyzePaste }} from "{url}"; const c = []; for await (const x of process.stdin) c.push(x); '
            "const t = JSON.parse(Buffer.concat(c).toString()); console.log(JSON.stringify(t.map(analyzePaste)));")
    done = subprocess.run(["node", "--input-type=module", "-e", code], input=json.dumps(texts), capture_output=True,
                          text=True, timeout=60, encoding="utf-8")
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.fixture()
def client():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(chat_bp)
    return app.test_client()


def _post(client, text):
    return client.post("/api/analyze-chat", data={"file": (io.BytesIO(text.encode()), "pasted_conversation.txt"),
                                                  "untimestamped": "1"}, content_type="multipart/form-data")


@needs_node
@pytest.mark.parametrize("name, chat", [("C1", C1), ("C2", C2), ("C3", C3)])
def test_the_same_conversation_scores_identically_however_it_was_pasted(client, name, chat):
    variants = _formats(chat)
    tidied = _tidy(list(variants.values()))
    reports = {}
    for (label, _), result in zip(variants.items(), tidied):
        assert result["conversation"] is True, f"{name} as {label}"
        response = _post(client, result["normalized"])
        assert response.status_code == 200, f"{name} as {label}: {response.get_json()}"
        reports[label] = response.get_json()
    reference = reports["lines"]
    assert len(reference["participants"]) == 2 and reference["warning_signs"]
    for label, report in reports.items():
        for key in ("risk_score", "risk_category", "warning_signs", "participants", "you", "classifier_prediction"):
            assert report[key] == reference[key], f"{name} as {label}: {key}"
        assert [t["sign"] for t in report["timeline"]] == [t["sign"] for t in reference["timeline"]], f"{name} as {label}"
        assert [t["cumulative"] for t in report["timeline"]] == [t["cumulative"] for t in reference["timeline"]]


@needs_node
def test_timestamped_pastes_keep_their_times_through_the_tidy_up(client):
    paragraph = _formats(C2)["paragraph"]
    normalized = _tidy([paragraph])[0]["normalized"]
    assert [m["time"] for m in parse_chat_text(normalized)] == ["24/10/26 9:02 AM", "24/10/26 9:03 AM", "24/10/26 9:05 AM", "24/10/26 9:06 AM"]
    body = _post(client, normalized).get_json()
    assert body["timeline"][0]["time"] == "24/10/26 9:02 AM"


@needs_node
def test_telegram_headers_and_dotted_dates_become_lines_the_strict_parser_reads(client):
    telegram = ("Recruiter, [24.10.2024 9:02]\nHello, we have a role.\n\nYou, [24.10.2024 9:03]\nWhich company?\n\n"
                "Recruiter, [24.10.2024 9:04]\nPay the registration fee today.\nSend your bank details.")
    result = _tidy([telegram])[0]
    assert result["conversation"] is True and result["mode"] == "lines"
    messages = parse_chat_text(result["normalized"])
    assert [(m["time"], m["speaker"]) for m in messages] == [("24/10/2024 9:02", "Recruiter"), ("24/10/2024 9:03", "You"),
                                                              ("24/10/2024 9:04", "Recruiter")]
    assert messages[2]["text"] == "Pay the registration fee today.\nSend your bank details."
    assert _post(client, result["normalized"]).status_code == 200


@needs_node
def test_non_conversations_are_not_tidied_into_chats():
    texts = ["Title: Packer Company: Acme Description: Pack boxes. Requirements: None. Benefits: Holiday.",
             "Contact: Sam. We are hiring packers in Leeds, start Monday.",
             "Remote Customer Support Associate: We need 10 people. After you accept we mail you a cheque."]
    assert [r["conversation"] for r in _tidy(texts)] == [False, False, False]


def test_app_sends_the_tidied_text_to_the_chat_analyzer():
    app = (FRONTEND / "src" / "App.jsx").read_text(encoding="utf-8")
    module = (FRONTEND / "src" / "services" / "conversation.js").read_text(encoding="utf-8")
    assert "conversationFile(text)" in app  # the file is built from the tidied text inside conversationFile
    assert "const { normalized } = analyzePaste(text);" in module and 'new File([normalized || String(text)]' in module


@needs_node
def test_node_tests_for_the_paste_formats_pass():
    result = subprocess.run(["node", "--test", "src/services/pasteFormats.test.mjs", "src/services/conversation.test.mjs"],
                            cwd=FRONTEND, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


# --- chat-rules measurement script ----------------------------------------------------------------------------------

def _val():
    rows = [
        {"job_id": 1, "group_id": 1, "fraudulent": 0, "title": "Analyst", "company_profile": "Firm", "description": "Review reports with the team.",
         "requirements": "Degree", "benefits": "Holiday"},
        {"job_id": 2, "group_id": 2, "fraudulent": 0, "title": "Clerk", "company_profile": "", "description": "Pay the registration fee of Rs 3000 within 2 hours.",
         "requirements": "None", "benefits": "Cash"},                                         # legitimate but the chat rules flag it
        {"job_id": 3, "group_id": 3, "fraudulent": 1, "title": "Agent", "company_profile": "", "description": "Pay the fee of Rs 5000 within 2 hours.",
         "requirements": "None", "benefits": "Cash"},                                         # fraud the classifier missed
        {"job_id": 4, "group_id": 4, "fraudulent": 1, "title": "Packer", "company_profile": "", "description": "Pack boxes at home.",
         "requirements": "None", "benefits": "Cash"},                                         # fraud, no chat rule fires
    ]
    return pd.DataFrame(rows)


def test_chat_rules_measurement_counts_levels_for_each_population():
    result = cm.compute(_val(), np.array([0.05, 0.05, 0.05, 0.9]))
    t = result["table"]
    assert t["legitimate, all"] == {"Higher risk": 0, "Moderate risk": 1, "Lower risk": 1, "n": 2}
    assert t["legitimate, currently Likely Legitimate"]["Moderate risk"] == 1
    assert t["fraud, all"]["n"] == 2 and t["fraud, all"]["Moderate risk"] + t["fraud, all"]["Higher risk"] == 1
    assert t["fraud, currently Likely Legitimate (missed)"]["n"] == 1  # job 4 is flagged by the classifier, job 3 is not
    assert result["causes_of_false_alarms"]["Payment request"] == 1


def test_chat_rules_measurement_reports_the_bar_and_the_banner():
    lines = cm.report_lines(cm.compute(_val(), np.array([0.05, 0.05, 0.05, 0.9])))
    text = "\n".join(lines)
    assert lines[0] == cm.BANNER and lines[-1] == cm.BANNER and "VALIDATION ONLY" in text
    assert "Higher only" in text and "Moderate or higher" in text and "bar <= 10" in text
    assert "within the bar" in text  # one false alarm is within 10
    assert cm.MAX_FALSE_ALARMS == 10


def test_chat_rules_measurement_rejects_misaligned_probabilities_and_changes_nothing():
    with pytest.raises(ValueError):
        cm.compute(_val(), np.zeros(2))
    source = (Path(__file__).resolve().parents[2] / "scripts" / "eval_chat_rules_on_val.py").read_text(encoding="utf-8")
    assert "test.csv" not in source.replace("Never", "") and "client_from_env" not in source
    assert "app.register_blueprint" not in source  # not wired into the app
