import assert from "node:assert/strict";
import test from "node:test";

import { NO_FLAGGED_MESSAGES, showChatGuidance } from "./chatView.js";

const chat = (risk_category, warning_signs) => ({ type: "chat", risk_category, warning_signs, risk_score: 10 });

test("the empty-timeline line", () => {
  assert.equal(NO_FLAGGED_MESSAGES, "No warning signs in any message");
});

test("guidance is hidden only for a calm chat: no warning sign and Lower risk", () => {
  assert.equal(showChatGuidance(chat("Lower risk", [])), false);
  assert.equal(showChatGuidance(chat("Lower risk", undefined)), false);
});

test("guidance is kept for Moderate and Higher, and whenever a warning sign fired", () => {
  assert.equal(showChatGuidance(chat("Moderate risk", [])), true);
  assert.equal(showChatGuidance(chat("Higher risk", [])), true);
  assert.equal(showChatGuidance(chat("Lower risk", ["Moves to another app"])), true);
  assert.equal(showChatGuidance(chat("Higher risk", ["Payment request"])), true);
  assert.equal(showChatGuidance(null), false);
});

test("deciding never changes the report", () => {
  const report = chat("Lower risk", []);
  const before = JSON.stringify(report);
  showChatGuidance(report);
  assert.equal(JSON.stringify(report), before);
});
