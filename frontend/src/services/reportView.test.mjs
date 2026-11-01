import assert from "node:assert/strict";
import test from "node:test";

import { NO_EVIDENCE_LINE, decisionDetails, evidenceView, plainSentence } from "./reportView.js";

const report = (verdict, extra = {}) => ({
  verdict,
  verdict_reason: "fraud probability is below the decision threshold",
  display_risk_score: 17,
  risk_score: 20,
  classifier_risk_score: 0,
  classifier_prediction: "Likely legitimate",
  risk_category: "Lower risk",
  warning_signs: [],
  ...extra,
});

test("the three plain sentences, word for word", () => {
  assert.equal(
    plainSentence(report("Likely Legitimate")),
    "This posting looks low risk. We found no warning signs. Still check the company before sharing documents or money."
  );
  assert.equal(
    plainSentence(report("Needs Verification")),
    "Be careful. Check this employer before you share anything or pay anything."
  );
  assert.equal(
    plainSentence(report("High Fraud Risk")),
    "This looks like a scam. Do not pay or share personal details."
  );
});

test("Likely Legitimate with warning signs says minor warning signs instead", () => {
  const withSigns = report("Likely Legitimate", { warning_signs: [{ label: "No interview", matched_text: "no interview needed" }] });
  assert.equal(
    plainSentence(withSigns),
    "This posting looks low risk. We found some minor warning signs. Still check the company before sharing documents or money."
  );
  assert.equal(plainSentence(report("Likely Legitimate", { warning_signs: undefined })).includes("no warning signs"), true);
});

test("no sentence for chats, insufficient information, unknown verdicts or empty input", () => {
  assert.equal(plainSentence(report("Needs Verification", { type: "chat" })), "");
  assert.equal(plainSentence(report("Insufficient Information")), "");
  for (const bad of [null, undefined, {}, report("Something else")]) assert.equal(plainSentence(bad), "");
});

test("the technical lines show the raw classifier probability, never a 0-100 classifier score", () => {
  const rows = decisionDetails(report("Likely Legitimate", { classifier_details: { model: "DistilBERT", fraud_probability: 0.999614, threshold: 0.291163 } }));
  assert.deepEqual(rows, [
    { label: "Reason", text: "fraud probability is below the decision threshold" },
    { label: "Raw classifier probability", text: "0.999614 (saturated)" },
  ]);
  assert.equal(JSON.stringify(rows).includes("/100"), false);
  assert.deepEqual(decisionDetails(report("Likely Legitimate")).map((r) => r.label), ["Reason", "Classifier"]); // no details sent
  assert.deepEqual(decisionDetails(report("Needs Verification", { type: "chat" })), []);
  assert.deepEqual(decisionDetails(null), []);
});

test("nothing is changed on the report: scores, category and API fields stay as sent", () => {
  const r = report("Needs Verification", { warning_signs: [{ label: "x", matched_text: "y" }], evidence: { rule_signs: [] }, llm_status: "unavailable" });
  const before = JSON.stringify(r);
  plainSentence(r);
  decisionDetails(r);
  evidenceView(r);
  assert.equal(JSON.stringify(r), before);
  assert.equal(r.classifier_risk_score, 0);
  assert.equal(r.display_risk_score, 17);
});

const withEvidence = (llm_status, evidence, explanation = "Verdict: X. Reason: Y. Risk score 17 out of 100.") => ({
  llm_status, explanation, evidence,
});
const SIGN = { id: "R1", label: "No interview", matched_text: "no interview needed" };
const SIMILAR = { id: "S1", label: 1, similarity: 0.5, snippet: "similar" };

test("with an AI explanation the evidence section shows everything as before", () => {
  const view = evidenceView(withEvidence("ok", { rule_signs: [SIGN], similar_postings: [SIMILAR] }, "AI text [R1]"));
  assert.deepEqual(view, { aiAvailable: true, showExplanation: true, showSimilar: true, showGraph: true, showRules: true, emptyLine: "" });
});

test("without an AI explanation: only matched signs and similar postings, and never the template explanation", () => {
  for (const status of ["unavailable", "error", "quota", "inconsistent"]) {
    const both = evidenceView(withEvidence(status, { rule_signs: [SIGN], similar_postings: [SIMILAR] }));
    assert.deepEqual(both, { aiAvailable: false, showExplanation: false, showSimilar: true, showGraph: false, showRules: true, emptyLine: "" });
  }
  const onlySigns = evidenceView(withEvidence("unavailable", { rule_signs: [SIGN], similar_postings: [] }));
  assert.equal(onlySigns.showSimilar, false);
  assert.equal(onlySigns.showRules, true);
  assert.equal(onlySigns.emptyLine, "");
  const onlySimilar = evidenceView(withEvidence("unavailable", { rule_signs: [], similar_postings: [SIMILAR] }));
  assert.equal(onlySimilar.showRules, false);
  assert.equal(onlySimilar.showSimilar, true);
});

test("linked contacts show whenever they have content, even without an AI explanation", () => {
  const linked = { graph: { items: [{ id: "G1", message: "This email address appears in 3 known postings (2 labelled fraud)." }] } };
  for (const status of ["unavailable", "error", "quota", "inconsistent"]) {
    const view = evidenceView(withEvidence(status, { rule_signs: [], similar_postings: [], ...linked }));
    assert.equal(view.showGraph, true, status);
    assert.equal(view.emptyLine, "", status);
    assert.equal(view.showExplanation, false, status);
  }
  const emptyGraph = evidenceView(withEvidence("unavailable", { rule_signs: [], similar_postings: [], graph: { items: [], note: "No linked companies or contacts found." } }));
  assert.equal(emptyGraph.showGraph, false);
  assert.equal(emptyGraph.emptyLine, "No specific evidence found.");
});

test("without an AI explanation and nothing to show, the only line is 'No specific evidence found.'", () => {
  const view = evidenceView(withEvidence("unavailable", { rule_signs: [], similar_postings: [], graph: { items: [], note: "No linked companies or contacts found." } }));
  assert.equal(NO_EVIDENCE_LINE, "No specific evidence found.");
  assert.deepEqual(view, { aiAvailable: false, showExplanation: false, showSimilar: false, showGraph: false, showRules: false, emptyLine: "No specific evidence found." });
});
