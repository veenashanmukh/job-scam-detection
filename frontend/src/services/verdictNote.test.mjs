import assert from "node:assert/strict";
import test from "node:test";

import { escalationCategories, escalationNote } from "./verdictNote.js";

const escalated = (reason, extra = {}) => ({
  verdict: "Needs Verification",
  verdict_reason: reason,
  risk_category: "Lower risk",
  display_risk_score: 17,
  risk_score: 20,
  ...extra,
});

test("two categories: the line names both and says the score is the classifier only", () => {
  const report = escalated("scam-pattern indicators present (Cheque or overpayment scam, No interview)");
  assert.deepEqual(escalationCategories(report), ["Cheque or overpayment scam", "No interview"]);
  assert.equal(
    escalationNote(report),
    "Score reflects the text classifier only. The verdict was raised by warning signs: Cheque or overpayment scam, No interview."
  );
});

test("a lone specific category is named too", () => {
  const report = escalated("specific scam-pattern indicator present (Parcel or money forwarding)");
  assert.equal(
    escalationNote(report),
    "Score reflects the text classifier only. The verdict was raised by warning signs: Parcel or money forwarding."
  );
});

test("a verdict raised by the classifier has no such line", () => {
  for (const reason of [
    "classifier flagged the posting and the company profile is empty",
    "company profile not provided",
    "fraud probability is below the decision threshold",
  ]) {
    assert.equal(escalationNote(escalated(reason)), "");
  }
});

test("only Needs Verification can carry the line; other verdicts and chats never do", () => {
  const reason = "scam-pattern indicators present (Cheque or overpayment scam, No interview)";
  assert.equal(escalationNote(escalated(reason, { verdict: "High Fraud Risk" })), "");
  assert.equal(escalationNote(escalated(reason, { verdict: "Likely Legitimate" })), "");
  assert.equal(escalationNote(escalated(reason, { type: "chat" })), "");
  for (const bad of [null, undefined, {}, { verdict: "Needs Verification" }, { verdict: "Needs Verification", verdict_reason: 5 }]) {
    assert.equal(escalationNote(bad), "");
  }
});

test("presenting the note never changes the report, so risk_category and the scores stay as the API sent them", () => {
  const report = escalated("scam-pattern indicators present (Cheque or overpayment scam, No interview)");
  const before = JSON.stringify(report);
  escalationNote(report);
  escalationCategories(report);
  assert.equal(JSON.stringify(report), before);
  assert.equal(report.risk_category, "Lower risk");
  assert.equal(report.display_risk_score, 17);
});
