import assert from "node:assert/strict";
import test from "node:test";

import { highlightSegments, signPhrases } from "./highlight.js";

const marked = (segments) => segments.filter((s) => s.label).map((s) => s.text);
const joined = (segments) => segments.map((s) => s.text).join("");

test("a word that is not part of a listed sign is never marked", () => {
  const text = "Earn Rs 5000 per week. Contact us on Telegram. Urgent deposit needed, no registration fee.";
  const signs = [{ label: "Unusually high salary claim", matched_text: "Rs 5000 per week" }];
  const segments = highlightSegments(text, signs);
  assert.deepEqual(marked(segments), ["Rs 5000 per week"]);
  assert.equal(joined(segments), text);
});

test("every marked span is the matched text of a listed sign", () => {
  const text = "Pay a registration fee today. Send your bank details. Guaranteed placement. We use Telegram.";
  const signs = [
    { label: "Upfront payment", matched_text: "registration fee" },
    { label: "Requests personal information", matched_text: "Send your bank details" },
  ];
  const allowed = new Set(signPhrases(signs).map((p) => p.phrase.toLowerCase()));
  const segments = highlightSegments(text, signs);
  for (const span of marked(segments)) assert.ok(allowed.has(span.toLowerCase()), span);
  assert.equal(marked(segments).length, 2);
  assert.equal(joined(segments), text);
});

test("no signs, chat-style label strings or empty phrases mark nothing", () => {
  const text = "Move to Telegram for a guaranteed deposit.";
  for (const signs of [undefined, [], ["Pressure tactic"], [{ label: "x", matched_text: "  " }], [null]]) {
    assert.deepEqual(highlightSegments(text, signs), [{ text, label: null }]);
  }
});

test("matching ignores case and tolerates different whitespace, and carries the sign label", () => {
  const text = "You must PAY   a registration\nfee now";
  const segments = highlightSegments(text, [{ label: "Upfront payment", matched_text: "pay a registration fee" }]);
  assert.deepEqual(marked(segments), ["PAY   a registration\nfee"]);
  assert.equal(segments.find((s) => s.label).label, "Upfront payment");
  assert.equal(joined(segments), text);
});

test("regex characters in a matched phrase are treated literally", () => {
  const text = "Earn $1,000+ (weekly) fast";
  const segments = highlightSegments(text, [{ label: "Salary", matched_text: "$1,000+ (weekly)" }]);
  assert.deepEqual(marked(segments), ["$1,000+ (weekly)"]);
});

test("the longest phrase wins when phrases overlap", () => {
  const text = "A refundable registration fee is required";
  const signs = [
    { label: "Short", matched_text: "registration fee" },
    { label: "Long", matched_text: "refundable registration fee" },
  ];
  assert.deepEqual(marked(highlightSegments(text, signs)), ["refundable registration fee"]);
});
