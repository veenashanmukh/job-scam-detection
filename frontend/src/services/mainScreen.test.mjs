import assert from "node:assert/strict";
import test from "node:test";

import {
  CHECKED_CATEGORIES, FOOTER_LINE, NEXT_STEPS, buildMainScreen, mainScreenText, stripCitations, technicalDetails,
  whyView,
} from "./reportView.js";

const FORBIDDEN = ["provisional", "probability", "classifier", "threshold", "fraud probability"];
const sign = (label, matched_text) => ({ label, matched_text });
const job = (verdict, extra = {}) => ({
  verdict,
  verdict_reason: "fraud probability is below the decision threshold",
  display_risk_score: 11,
  risk_score: 15,
  classifier_risk_score: 5,
  classifier_prediction: "Likely legitimate",
  classifier_details: { model: "DistilBERT", fraud_probability: 0.05, threshold: 0.291163 },
  uncertainty_notes: ["The classifier can miss some scams."],
  risk_category: "Lower risk",
  warning_signs: [],
  ...extra,
});
const PLAIN_NAMES = [
  "requests for money or fees",
  "requests for bank or ID details",
  "claims of no interview",
  "unrealistic pay",
  "cheque or overpayment requests",
  "gift card or crypto payments",
  "parcel forwarding",
];

test("with nothing fired every real category gets a tick line, with the plain descriptions word for word", () => {
  const screen = buildMainScreen(job("Likely Legitimate"));
  const ticks = screen.checked.filter((c) => c.kind === "ok").map((c) => c.text);
  assert.deepEqual(ticks, PLAIN_NAMES.map((name) => `We did not find ${name}.`));
  assert.equal(screen.checked.filter((c) => c.kind === "warn").length, 0);
  assert.deepEqual(CHECKED_CATEGORIES.map((c) => c.plain), PLAIN_NAMES);
});

test("tick lines appear only for categories that did not fire; fired ones are warning lines with the matched phrase", () => {
  // Ticks exist only for "Likely Legitimate", so this uses a posting with one minor sign and that verdict.
  const screen = buildMainScreen(job("Likely Legitimate", {
    warning_signs: [sign("No interview", "no interview needed"), sign("Unusually high salary claim", "earn 9000 per month")],
  }));
  const warns = screen.checked.filter((c) => c.kind === "warn");
  assert.deepEqual(warns.map((w) => [w.label, w.text, w.phrase]), [
    ["No interview", "We found a warning sign: claims of no interview", "no interview needed"],
    ["Unusually high salary claim", "We found a warning sign: unrealistic pay", "earn 9000 per month"],
  ]);
  const tickLabels = screen.checked.filter((c) => c.kind === "ok").map((c) => c.label);
  assert.equal(tickLabels.includes("No interview"), false);
  assert.equal(tickLabels.includes("Unusually high salary claim"), false);
  assert.equal(tickLabels.length, 5);
  for (const w of warns) assert.equal(screen.checked.some((c) => c.kind === "ok" && c.label === w.label), false);
});

test("a category without a plain name warns under its own label and never gets a tick; no messaging-app line exists", () => {
  const fired = buildMainScreen(job("Needs Verification", { warning_signs: [sign("Pressure tactic", "urgent")] }));
  assert.deepEqual(fired.checked.filter((c) => c.kind === "warn").map((c) => c.text), ["We found a warning sign: pressure tactic"]);
  const quiet = buildMainScreen(job("Likely Legitimate"));
  assert.equal(quiet.checked.some((c) => /pressure|guarantee|messaging/i.test(c.text)), false);
});

test("tick lines never say verified, confirmed or safe", () => {
  const screen = buildMainScreen(job("Likely Legitimate", { company_profile_present: true }));
  for (const c of screen.checked) assert.equal(/verified|confirmed|safe/i.test(c.text), false, c.text);
  assert.ok(screen.checked.filter((c) => c.kind === "ok" && /^We did not find /.test(c.text)).length >= 7);
});

test("the company description line appears only when a company profile was provided", () => {
  const line = "The posting includes a company description.";
  assert.equal(buildMainScreen(job("Likely Legitimate", { company_profile_present: true })).checked.some((c) => c.text === line), true);
  for (const flag of [false, undefined, "yes"]) {
    assert.equal(buildMainScreen(job("Likely Legitimate", { company_profile_present: flag })).checked.some((c) => c.text === line), false);
  }
});

const similar = (...labels) => ({
  evidence: { similar_postings: labels.map((label, i) => ({ id: `S${i + 1}`, label, similarity: 0.5, snippet: "x" })) },
});

test("the similar genuine postings line is absent unless similar known postings exist and all are genuine", () => {
  const has = (r) => buildMainScreen(r).checked.some((c) => /known genuine posting/.test(c.text));
  assert.equal(has(job("Likely Legitimate")), false); // no evidence at all
  assert.equal(has(job("Likely Legitimate", { evidence: { similar_postings: [] } })), false);
  assert.equal(has(job("Likely Legitimate", similar(1, 1))), false); // all fraud
  assert.equal(has(job("Likely Legitimate", similar(0, 1))), false); // mixed
  assert.equal(buildMainScreen(job("Likely Legitimate", similar(0, 0, 0))).checked.find((c) => /genuine/.test(c.text)).text,
    "It looks similar to 3 known genuine postings.");
  assert.equal(buildMainScreen(job("Likely Legitimate", similar(0))).checked.find((c) => /genuine/.test(c.text)).text,
    "It looks similar to 1 known genuine posting.");
});

test("similar and linked blocks are empty arrays when there is no content, and listed when there is", () => {
  const none = buildMainScreen(job("Likely Legitimate", {
    evidence: { similar_postings: [], graph: { items: [], note: "No linked companies or contacts found." } },
  }));
  assert.deepEqual([none.similar, none.linked], [[], []]);
  const some = buildMainScreen(job("Needs Verification", {
    evidence: {
      similar_postings: [{ id: "S1", label: 1, similarity: 0.6, snippet: "a fee posting" }],
      graph: { items: [{ id: "G1", message: "This email address appears in 3 known postings (2 labelled fraud)." }] },
    },
  }));
  assert.equal(some.similar.length, 1);
  assert.equal(some.linked[0].text, "This email address appears in 3 known postings (2 labelled fraud).");
});

test("score line, next steps and footer", () => {
  const screen = buildMainScreen(job("Likely Legitimate"));
  assert.equal(screen.scoreText, "11 / 100");
  assert.equal(screen.scoreNote, "Higher means more risk");
  assert.equal(screen.footer, "This is a warning-sign check, not a guarantee.");
  assert.equal(FOOTER_LINE, screen.footer);
  assert.deepEqual(NEXT_STEPS["Likely Legitimate"], ["Check the company's official website.", "Never pay to get a job."]);
  assert.equal(NEXT_STEPS["Needs Verification"][0],
    "Do not pay or share ID documents until you have confirmed the employer through their official website or phone.");
  assert.deepEqual(NEXT_STEPS["High Fraud Risk"].slice(0, 2), ["Do not reply.", "Do not pay or share personal details."]);
  for (const verdict of Object.keys(NEXT_STEPS)) assert.ok(NEXT_STEPS[verdict].length >= 2 && NEXT_STEPS[verdict].length <= 3);
  assert.deepEqual(buildMainScreen(job("High Fraud Risk")).next.slice(0, 2), NEXT_STEPS["High Fraud Risk"]); // verdict lines first
});

test("none of the forbidden words appear anywhere on the main screen, for any verdict or content", () => {
  const raised = job("Needs Verification", {
    verdict_reason: "scam-pattern indicators present (Cheque or overpayment scam, No interview)",
    warning_signs: [sign("Cheque or overpayment scam", "cheque for $2,800"), sign("No interview", "no interview required")],
    company_profile_present: true,
    llm_status: "ok",
    explanation: "The posting asks you to deposit a cheque and send money on [R1]. Check the employer first [K1].",
    cited_evidence_ids: ["R1", "K1"],
    evidence: {
      similar_postings: [{ id: "S1", label: 0, similarity: 0.4, snippet: "A genuine posting." }],
      graph: { items: [{ id: "G1", message: "This email address appears in 2 known postings (0 labelled fraud)." }] },
      guidance: [{ id: "K1", title: "Job scams", url: "https://example.org/jobs" }],
    },
  });
  for (const report of [job("Likely Legitimate"), job("High Fraud Risk"), raised, job("Likely Legitimate", similar(0, 0))]) {
    const text = mainScreenText(buildMainScreen(report)).join(" \n ").toLowerCase();
    for (const word of FORBIDDEN) assert.equal(text.includes(word), false, `${word} on the main screen`);
  }
});

test("the technical words moved into the collapsed section instead of disappearing", () => {
  const raised = job("Needs Verification", { verdict_reason: "scam-pattern indicators present (Cheque or overpayment scam, No interview)" });
  const tech = technicalDetails(raised);
  const rows = tech.rows.map((r) => `${r.label}: ${r.text}`).join(" | ");
  assert.match(rows, /Reason: scam-pattern indicators present/);
  assert.match(rows, /Raw classifier probability: 0\.050000$|Raw classifier probability: 0\.050000 \|/);
  assert.match(rows, /Verdict note: Score reflects the text classifier only/);
  assert.match(rows, /Model details: DistilBERT, decision threshold 0\.291163/);
  assert.deepEqual(tech.notes, ["The classifier can miss some scams."]);
  assert.deepEqual(technicalDetails(job("Likely Legitimate", { warning_signs: [sign("No interview", "no interview needed")] })).rules,
    [{ id: "R1", label: "No interview", matched_text: "no interview needed" }]);
});

test("why: only with an AI explanation, in plain words, with cited http(s) guidance as links", () => {
  const base = {
    llm_status: "ok",
    explanation: "You are asked to pay a fee [R1]. See official advice [K1][K2].",
    cited_evidence_ids: ["R1", "K1", "K2"],
    evidence: {
      guidance: [
        { id: "K1", title: "FTC job scams", url: "https://consumer.ftc.gov/jobs" },
        { id: "K2", title: "Bad link", url: "javascript:alert(1)" },
        { id: "K3", title: "Not cited", url: "https://example.org/x" },
      ],
    },
  };
  const why = whyView(job("Needs Verification", base));
  assert.equal(why.text, "You are asked to pay a fee. See official advice.");
  assert.deepEqual(why.sources, [{ title: "FTC job scams", url: "https://consumer.ftc.gov/jobs" }]);
  for (const status of ["unavailable", "error", "quota", "inconsistent", undefined]) {
    assert.equal(whyView(job("Needs Verification", { ...base, llm_status: status })), null);
    assert.equal(buildMainScreen(job("Needs Verification", { ...base, llm_status: status })).why, null);
  }
  assert.equal(whyView(job("Needs Verification", { ...base, explanation: "" })), null);
  assert.equal(stripCitations("A [S1] and [R1, R2]."), "A and.");
});

test("building the main screen leaves the report untouched, so no score or API field changes", () => {
  const report = job("Needs Verification", {
    warning_signs: [sign("Upfront payment", "fee")],
    company_profile_present: true,
    evidence: { similar_postings: [{ id: "S1", label: 0, similarity: 0.5, snippet: "x" }] },
  });
  const before = JSON.stringify(report);
  buildMainScreen(report);
  mainScreenText(buildMainScreen(report));
  technicalDetails(report);
  whyView(report);
  assert.equal(JSON.stringify(report), before);
  assert.equal(buildMainScreen(null), null);
  assert.equal(buildMainScreen({ type: "chat", verdict: "Needs Verification" }), null);
});
