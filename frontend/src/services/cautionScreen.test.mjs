import assert from "node:assert/strict";
import test from "node:test";

import {
  PAYMENT_RELATED_LABELS, RESEMBLES_SCAMS_LINE, SCORE_NOTE, buildMainScreen, mainScreenText,
} from "./reportView.js";

const sign = (label, matched_text) => ({ label, matched_text });
const job = (verdict, extra = {}) => ({
  verdict,
  verdict_reason: "classifier flagged the posting and the company profile is empty",
  display_risk_score: 52,
  risk_score: 60,
  classifier_risk_score: 55,
  classifier_prediction: "Potentially fraudulent",
  warning_signs: [],
  ...extra,
});
const GENUINE = { evidence: { similar_postings: [{ id: "S1", label: 0, similarity: 0.5, snippet: "x" }] } };
const kinds = (screen) => screen.checked.map((c) => c.kind);
const MONEY = "We did not find requests for money or fees.";

test("Needs Verification and High Fraud Risk show no tick lines at all, whatever fired", () => {
  for (const verdict of ["Needs Verification", "High Fraud Risk"]) {
    for (const warning_signs of [[], [sign("No interview", "no interview needed")], [sign("Upfront payment", "fee"), sign("Cheque or overpayment scam", "cheque")]]) {
      const screen = buildMainScreen(job(verdict, { warning_signs, company_profile_present: true, ...GENUINE }));
      assert.equal(kinds(screen).includes("ok"), false, `${verdict} with ${warning_signs.length} fired`);
      assert.equal(screen.checked.some((c) => /We did not find|company description|known genuine/.test(c.text)), false);
    }
  }
});

test("caution verdicts list only the categories that fired, with their phrases", () => {
  const screen = buildMainScreen(job("High Fraud Risk", {
    warning_signs: [sign("Upfront payment", "registration fee"), sign("Pressure tactic", "urgent")],
  }));
  assert.deepEqual(screen.checked.map((c) => [c.kind, c.label, c.phrase]), [
    ["warn", "Upfront payment", "registration fee"],
    ["warn", "Pressure tactic", "urgent"],
  ]);
});

test("a caution verdict with no fired category shows the one classifier-raised line", () => {
  for (const verdict of ["Needs Verification", "High Fraud Risk"]) {
    const screen = buildMainScreen(job(verdict));
    assert.equal(screen.checked.length, 1);
    assert.equal(screen.checked[0].text, "The wording of this posting resembles postings that were reported as scams.");
    assert.equal(screen.checked[0].text, RESEMBLES_SCAMS_LINE);
    assert.equal(screen.checked[0].kind, "warn");
  }
});

test("that line never appears for Likely Legitimate or when a category fired", () => {
  assert.equal(buildMainScreen(job("Likely Legitimate")).checked.some((c) => c.text === RESEMBLES_SCAMS_LINE), false);
  for (const verdict of ["Needs Verification", "High Fraud Risk"]) {
    const fired = buildMainScreen(job(verdict, { warning_signs: [sign("No interview", "no interview needed")] }));
    assert.equal(fired.checked.some((c) => c.text === RESEMBLES_SCAMS_LINE), false);
  }
});

test("Likely Legitimate still gets ticks for the categories that did not fire", () => {
  const screen = buildMainScreen(job("Likely Legitimate", { company_profile_present: true, ...GENUINE }));
  assert.equal(screen.checked.filter((c) => /^We did not find /.test(c.text)).length, 7);
  assert.equal(screen.checked.some((c) => c.text === "The posting includes a company description."), true);
  assert.equal(screen.checked.some((c) => /known genuine posting/.test(c.text)), true);
});

test("the money tick is suppressed when any payment-related category fired", () => {
  assert.deepEqual(PAYMENT_RELATED_LABELS, [
    "Upfront payment", "Cheque or overpayment scam", "Gift card or crypto payment", "Parcel or money forwarding",
  ]);
  for (const label of PAYMENT_RELATED_LABELS) {
    const screen = buildMainScreen(job("Likely Legitimate", { warning_signs: [sign(label, "x")] }));
    assert.equal(screen.checked.some((c) => c.text === MONEY), false, label);
    assert.equal(screen.checked.some((c) => c.kind === "warn" && c.label === label), true, label);
  }
  // the other ticks stay: fired payment category removes only the money tick (and its own tick)
  const cheque = buildMainScreen(job("Likely Legitimate", { warning_signs: [sign("Cheque or overpayment scam", "cheque")] }));
  assert.equal(cheque.checked.filter((c) => c.kind === "ok").length, 5); // 7 categories - cheque - money
});

test("the money tick stays when only a non-payment category fired, and when nothing fired", () => {
  for (const warning_signs of [[], [sign("No interview", "no interview needed")], [sign("Unusually high salary claim", "x")], [sign("Requests personal information", "x")]]) {
    const screen = buildMainScreen(job("Likely Legitimate", { warning_signs }));
    assert.equal(screen.checked.some((c) => c.text === MONEY), true, JSON.stringify(warning_signs));
  }
});

test("the old score wording is gone and the new wording is used", () => {
  assert.equal(SCORE_NOTE, "Higher means more risk");
  for (const verdict of ["Likely Legitimate", "Needs Verification", "High Fraud Risk"]) {
    const screen = buildMainScreen(job(verdict));
    assert.equal(screen.scoreNote, "Higher means more risk");
    assert.equal(mainScreenText(screen).some((t) => /Lower is safer/i.test(t)), false);
  }
});

test("none of this changes the report", () => {
  const report = job("Needs Verification", { warning_signs: [sign("Upfront payment", "fee")], company_profile_present: true, ...GENUINE });
  const before = JSON.stringify(report);
  buildMainScreen(report);
  assert.equal(JSON.stringify(report), before);
  assert.equal(report.display_risk_score, 52);
  assert.equal(report.risk_score, 60);
});
