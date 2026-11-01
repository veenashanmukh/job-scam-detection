import assert from "node:assert/strict";
import test from "node:test";

import { buildMainScreen, citedEvidenceIds, whyView } from "./reportView.js";

const EVIDENCE = {
  similar_postings: [{ id: "S1", label: 1, similarity: 0.6, snippet: "a fee posting" }],
  graph: { items: [{ id: "G1", message: "This website appears in 3 known postings (3 labelled fraud)." }] },
  rule_signs: [{ id: "R1", label: "Upfront payment", matched_text: "registration fee" }],
  guidance: [{ id: "K1", title: "Job scams", url: "https://consumer.ftc.gov/jobs" }],
};
const report = (explanation, cited, extra = {}) => ({
  verdict: "Needs Verification", display_risk_score: 70, warning_signs: [], llm_status: "ok",
  explanation, cited_evidence_ids: cited, evidence: EVIDENCE, ...extra,
});

test("no citations: no Why section, including a generic sentence", () => {
  for (const text of ["Based on the evidence provided, this posting may be risky.", "Be careful with this posting."]) {
    assert.equal(whyView(report(text, [])), null, text);
    assert.equal(buildMainScreen(report(text, [])).why, null, text);
  }
  assert.equal(whyView(report("Something.", undefined)), null);
});

test("with a citation to an evidence item: the Why section is shown", () => {
  for (const id of ["S1", "G1", "R1", "K1"]) {
    const why = whyView(report(`The posting asks for a registration fee [${id}].`, [id]));
    assert.ok(why, id);
    assert.equal(why.text, "The posting asks for a registration fee.");
  }
  assert.ok(buildMainScreen(report("It asks for a fee [R1].", ["R1"])).why);
});

test("cited ids that are not evidence items present in the report do not count", () => {
  assert.equal(whyView(report("Fee [R9].", ["R9"])), null);                    // not in the evidence
  assert.equal(whyView(report("Message [T1].", ["T1"])), null);                // T (chat message) is not S, G, R or K
  assert.equal(whyView(report("Fee [R1].", ["R1"], { evidence: {} })), null);  // no evidence at all
  assert.deepEqual(citedEvidenceIds(report("x", ["R9", "R1", "T1", "K1"])), ["R1", "K1"]);
});

test("only an explanation the checker accepted can show Why", () => {
  for (const status of ["unavailable", "error", "quota", "inconsistent", undefined]) {
    assert.equal(whyView(report("It asks for a fee [R1].", ["R1"], { llm_status: status })), null, String(status));
  }
});

test("sources are still the cited guidance links only", () => {
  assert.deepEqual(whyView(report("Read the advice [K1] about the fee [R1].", ["K1", "R1"])).sources,
    [{ title: "Job scams", url: "https://consumer.ftc.gov/jobs" }]);
  assert.deepEqual(whyView(report("It asks for a fee [R1].", ["R1"])).sources, []);
});

test("reading the report changes nothing", () => {
  const r = report("It asks for a fee [R1].", ["R1"]);
  const before = JSON.stringify(r);
  whyView(r);
  citedEvidenceIds(r);
  assert.equal(JSON.stringify(r), before);
});
