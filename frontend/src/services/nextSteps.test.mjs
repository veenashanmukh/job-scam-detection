import assert from "node:assert/strict";
import test from "node:test";

import {
  ADVICE_ORDER, CATEGORY_ADVICE, CHAT_LABEL_TO_CATEGORY, HELP_LINE, MAX_ADVICE_LINES, NEXT_STEPS, SEARCH_LINE, buildMainScreen,
  nextStepsFor,
} from "./reportView.js";

const sign = (label) => ({ label, matched_text: "x" });
const job = (verdict, labels = [], extra = {}) => ({ verdict, display_risk_score: 40, risk_score: 45, warning_signs: labels.map(sign), ...extra });
const chat = (risk_category, labels = []) => ({ type: "chat", risk_category, risk_score: 50, display_risk_score: 50, warning_signs: labels });

test("the fixed map, word for word", () => {
  assert.deepEqual(CATEGORY_ADVICE, {
    "Cheque or overpayment scam": "Do not deposit the cheque or send any of the money onward. Tell your bank.",
    "Gift card or crypto payment": "Do not buy or send gift cards or crypto to anyone about a job.",
    "Parcel or money forwarding": "Do not receive or forward packages or money for an employer. Scammers use this to move stolen goods.",
    "Upfront payment": "Do not pay any fee to get a job, even if it is described as refundable.",
    "Requests personal information": "Do not send ID numbers, bank details or photos of documents before you have verified the employer.",
    "No interview": "Genuine employers interview you. Be wary of offers with no interview.",
    "Unusually high salary claim": "Be wary of very high pay for easy work. Honest employers do not promise big money for little effort.",
    "Guaranteed employment": "Be wary of a guaranteed job. No honest employer can promise you a job before they have met you.",
    "Pressure tactic": "Do not let anyone rush you. Scammers use deadlines to stop you checking.",
  });
  assert.equal(SEARCH_LINE, "Search the company name with the word 'scam', and check that emails come from the company's official website domain.");
  assert.equal(HELP_LINE, "If you have already paid or shared details: contact your bank at once. In India, report at cybercrime.gov.in or call 1930.");
});

test("severity order: money moving first, then payments and personal data, then the softer signs", () => {
  assert.deepEqual(ADVICE_ORDER, ["Cheque or overpayment scam", "Gift card or crypto payment", "Parcel or money forwarding", "Upfront payment",
    "Requests personal information", "No interview", "Unusually high salary claim", "Guaranteed employment", "Pressure tactic"]);
});

test("each fired category adds its own line, in severity order whatever order the signs came in", () => {
  const lines = nextStepsFor(job("Needs Verification", ["Pressure tactic", "No interview", "Upfront payment"]));
  assert.deepEqual(lines.slice(NEXT_STEPS["Needs Verification"].length, -1), [
    CATEGORY_ADVICE["Upfront payment"], CATEGORY_ADVICE["No interview"], CATEGORY_ADVICE["Pressure tactic"],
  ]);
  const single = nextStepsFor(job("Likely Legitimate", ["Unusually high salary claim"]));
  assert.ok(single.includes(CATEGORY_ADVICE["Unusually high salary claim"]));
});

test("at most four category lines, keeping the most severe", () => {
  const all = Object.keys(CATEGORY_ADVICE).reverse();
  const lines = nextStepsFor(job("High Fraud Risk", all));
  const advice = lines.slice(NEXT_STEPS["High Fraud Risk"].length, -1);
  assert.equal(MAX_ADVICE_LINES, 4);
  assert.deepEqual(advice, ADVICE_ORDER.slice(0, 4).map((c) => CATEGORY_ADVICE[c]));
  assert.equal(lines.length, NEXT_STEPS["High Fraud Risk"].length + 4 + 1);
  const six = nextStepsFor(job("Needs Verification", ["Pressure tactic", "Guaranteed employment", "No interview", "Requests personal information", "Upfront payment", "Gift card or crypto payment"]));
  assert.equal(six.slice(NEXT_STEPS["Needs Verification"].length, -1).length, 4);
  assert.equal(six.includes(CATEGORY_ADVICE["Pressure tactic"]), false); // the softest signs are the ones dropped
});

test("the verdict lines are kept and each verdict gets the right closing line", () => {
  const legit = nextStepsFor(job("Likely Legitimate"));
  assert.deepEqual(legit, [...NEXT_STEPS["Likely Legitimate"], SEARCH_LINE]);
  assert.deepEqual(legit.slice(0, 2), ["Check the company's official website.", "Never pay to get a job."]);
  for (const verdict of ["Needs Verification", "High Fraud Risk"]) {
    const lines = nextStepsFor(job(verdict));
    assert.deepEqual(lines, [...NEXT_STEPS[verdict], HELP_LINE]);
    assert.equal(lines.at(-1), HELP_LINE);
    assert.equal(lines.includes(SEARCH_LINE), false);
  }
  assert.equal(nextStepsFor(job("Likely Legitimate", ["Upfront payment"])).includes(HELP_LINE), false);
  assert.deepEqual(NEXT_STEPS["High Fraud Risk"], ["Do not reply.", "Do not pay or share personal details."]);
  assert.equal(NEXT_STEPS["Needs Verification"][0], "Do not pay or share ID documents until you have confirmed the employer through their official website or phone.");
});

test("no invented categories: an unknown or inherited label adds no line", () => {
  const base = nextStepsFor(job("Needs Verification"));
  for (const label of ["Moves to another app", "Something new", "constructor", "__proto__", "toString", ""]) {
    assert.deepEqual(nextStepsFor(job("Needs Verification", [label])), base, label);
  }
  assert.deepEqual(nextStepsFor(job("Needs Verification", ["Upfront payment", "Upfront payment"])).length, base.length + 1); // repeated label once
});

test("chat reports get the same section from their risk level and the chat labels that have a counterpart", () => {
  assert.deepEqual(CHAT_LABEL_TO_CATEGORY, {
    "Payment request": "Upfront payment", "Sensitive information request": "Requests personal information",
    "Pressure tactic": "Pressure tactic", "Unrealistic job promise": "Unusually high salary claim",
  });
  const high = nextStepsFor(chat("Higher risk", ["Payment request", "Sensitive information request", "Pressure tactic"]));
  assert.deepEqual(high.slice(0, 2), NEXT_STEPS["High Fraud Risk"]);
  assert.deepEqual(high.slice(2, -1), [CATEGORY_ADVICE["Upfront payment"], CATEGORY_ADVICE["Requests personal information"], CATEGORY_ADVICE["Pressure tactic"]]);
  assert.equal(high.at(-1), HELP_LINE);
  assert.deepEqual(nextStepsFor(chat("Lower risk")), [...NEXT_STEPS["Likely Legitimate"], SEARCH_LINE]);
  assert.deepEqual(nextStepsFor(chat("Moderate risk", ["Moves to another app"])), [...NEXT_STEPS["Needs Verification"], HELP_LINE]);
  assert.ok(nextStepsFor(chat("Moderate risk", ["Unrealistic job promise"])).includes(CATEGORY_ADVICE["Unusually high salary claim"]));
});

test("no section without a verdict or a chat risk level", () => {
  for (const report of [null, undefined, {}, { verdict: "Insufficient Information" }, { type: "chat" }, chat("Unknown")]) {
    assert.deepEqual(nextStepsFor(report), []);
  }
});

test("the main screen uses it, and building it leaves the report and its scores untouched", () => {
  const report = job("High Fraud Risk", ["Cheque or overpayment scam", "Upfront payment"], { risk_category: "Higher risk" });
  const before = JSON.stringify(report);
  assert.deepEqual(buildMainScreen(report).next, nextStepsFor(report));
  assert.equal(buildMainScreen(report).next.includes(CATEGORY_ADVICE["Cheque or overpayment scam"]), true);
  assert.equal(JSON.stringify(report), before);
  assert.equal(report.display_risk_score, 40);
  assert.equal(report.risk_score, 45);
});
