import assert from "node:assert/strict";
import test from "node:test";

import {
  CHAT_SCORE_NOTE, NO_PATTERNS_LINE, NO_PATTERNS_PREDICTION, REUSED_NOTE, chatPrediction, chatScore,
  clampDisplayScore, splitExplanationNote,
} from "./chatView.js";

test("a chat with raw score 0 shows 1 and raw score 100 shows 99", () => {
  assert.equal(chatScore({ risk_score: 0, display_risk_score: 1 }), 1);
  assert.equal(chatScore({ risk_score: 100, display_risk_score: 99 }), 99);
  // older responses without the new field: the same clamp is applied here
  assert.equal(chatScore({ risk_score: 0 }), 1);
  assert.equal(chatScore({ risk_score: 100 }), 99);
  assert.equal(chatScore({ risk_score: 42 }), 42);
  assert.equal(chatScore({}), null);
  assert.equal(chatScore(null), null);
});

test("the server's display score wins over the raw score when both are present", () => {
  assert.equal(chatScore({ risk_score: 55, display_risk_score: 55 }), 55);
  assert.equal(chatScore({ risk_score: 100, display_risk_score: 99 }), 99);
});

test("the clamp never shows 0 or 100 and keeps the order of scores", () => {
  const raw = Array.from({ length: 101 }, (_, i) => i);
  const shown = raw.map(clampDisplayScore);
  assert.ok(shown.every((s) => s >= 1 && s <= 99));
  for (let i = 1; i < shown.length; i++) assert.ok(shown[i] >= shown[i - 1], `order at ${i}`);
  for (let r = 1; r <= 99; r++) assert.equal(clampDisplayScore(r), r); // distinct scores in range stay distinct
  assert.equal(clampDisplayScore(-5), 1);
  assert.equal(clampDisplayScore(250), 99);
  assert.equal(clampDisplayScore(37.9), 37);
});

test("the nothing-found prediction becomes the plain sentence; other predictions keep their text", () => {
  assert.equal(NO_PATTERNS_PREDICTION, "No strong rule-based warning pattern detected");
  assert.equal(NO_PATTERNS_LINE, "No warning patterns detected in the recruiter's messages.");
  assert.deepEqual(chatPrediction({ classifier_prediction: NO_PATTERNS_PREDICTION }),
    { text: "No warning patterns detected in the recruiter's messages.", labelled: false });
  assert.deepEqual(chatPrediction({ classifier_prediction: "Warning signs require review" }),
    { text: "Warning signs require review", labelled: true });
  assert.deepEqual(chatPrediction({}), { text: "", labelled: false });
});

test("the old wording is not produced for any prediction", () => {
  for (const prediction of [NO_PATTERNS_PREDICTION, "Warning signs require review", "Potentially fraudulent conversation"]) {
    const { text } = chatPrediction({ classifier_prediction: prediction });
    assert.equal(/No strong rule-based/.test(text), false);
  }
  assert.equal(CHAT_SCORE_NOTE, "Higher means more risk");
});

test("the reused-explanation note is split off so it can go into the collapsed section", () => {
  const body = "The recruiter asks for a fee [T1].";
  const { body: shown, note } = splitExplanationNote(`${body}\n\n${REUSED_NOTE}`);
  assert.equal(shown, body);
  assert.equal(note, "(Explanation based on the first analysis; only the rule score was recomputed for the chosen name.)");
  assert.deepEqual(splitExplanationNote(body), { body, note: "" });
  assert.deepEqual(splitExplanationNote(undefined), { body: "", note: "" });
  assert.equal(splitExplanationNote(`${body}\n\n${REUSED_NOTE}`).body.includes("first analysis"), false);
});
