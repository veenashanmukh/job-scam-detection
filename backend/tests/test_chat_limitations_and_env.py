"""Rule-label spelling, the chat keyword-limitation note, and scripts loading .env without printing it."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import eval_grounding_val as gv  # noqa: E402
import eval_paraphrase_val as ep  # noqa: E402
import script_env  # noqa: E402
from backend.services.conversation_analysis import KEYWORD_LIMITATION_NOTE, RULES, analyze_messages  # noqa: E402
from backend.services.conversation_parser import parse_chat_text  # noqa: E402
from backend.services.job_analysis import WARNING_RULES  # noqa: E402

KNOWN_SINGLE_WORDS: set[str] = set()  # no single-word sign labels exist today


def test_every_sign_label_has_a_space_or_is_a_known_single_word():
    labels = [r[0] for r in RULES] + [label for label, _ in WARNING_RULES]
    for label in labels:
        assert " " in label or label in KNOWN_SINGLE_WORDS, label
    assert len(set(labels)) == len(labels) - 1  # "Pressure tactic" is the one label both tables share


def test_chat_report_keeps_existing_limitations_and_adds_the_keyword_note():
    messages = parse_chat_text("24/10/26, 9:00 - HR: Thanks for applying.\n24/10/26, 9:01 - You: When is my interview?\n")
    notes = analyze_messages(messages, self_name="You")["uncertainty_notes"]
    assert notes.count(KEYWORD_LIMITATION_NOTE) == 1
    assert KEYWORD_LIMITATION_NOTE == (
        "This score uses keyword rules only. It can miss scams that do not use explicit payment, "
        "personal-data or pressure wording, so a low score is not a sign that a chat is safe.")
    assert any("rules" in n and "DistilBERT" in n for n in notes) and any("employer verification" in n for n in notes)
    assert any(n.startswith("Messages from You") for n in notes)


def test_the_ui_lists_every_uncertainty_note():
    app = (Path(__file__).resolve().parents[2] / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")
    assert "uncertaintyNotes.map" in app  # chat notes, including the new one, are rendered from the report


def _fake_env(tmp_path):
    env = tmp_path / ".env"
    env.write_text("GROQ_API_KEY=fake-key-do-not-print\nLLM_MODEL=fake-model\n", encoding="utf-8")
    return env


def test_loader_sets_missing_variables_only_and_prints_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL", "already-set")
    assert script_env.load_project_env(_fake_env(tmp_path)) is True
    import os
    assert os.environ["GROQ_API_KEY"] == "fake-key-do-not-print"
    assert os.environ["LLM_MODEL"] == "already-set"  # existing environment wins
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    monkeypatch.delenv("GROQ_API_KEY")


def test_missing_env_file_is_not_an_error(tmp_path):
    assert script_env.load_project_env(tmp_path / "absent.env") is False


@pytest.mark.parametrize("module", [gv, ep])
def test_llm_scripts_load_env_and_never_print_the_key(module, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(script_env, "PROJECT_ENV", _fake_env(tmp_path))
    monkeypatch.setattr(sys, "argv", ["script"])
    assert (module.main() if module is gv else module.main([])) == 2  # refuses: no --run-live
    import os
    assert os.environ.get("GROQ_API_KEY") == "fake-key-do-not-print" and os.environ.get("LLM_MODEL") == "fake-model"
    captured = capsys.readouterr()
    assert "fake-key-do-not-print" not in captured.out + captured.err
    monkeypatch.delenv("GROQ_API_KEY")
    monkeypatch.delenv("LLM_MODEL")
