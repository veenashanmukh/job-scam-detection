import assert from "node:assert/strict";
import test from "node:test";

import { SERVER_UNREACHABLE, analyzeChat, analyzeJob, analyzeJobFields, requestErrorMessage } from "./api.js";
import { chatScore } from "./chatView.js";
import { buildMainScreen, decisionDetails, rawProbability, technicalDetails } from "./reportView.js";

// --- A: raw classifier probability, never a 0 or 100 score ------------------------------------------------------------

test("the raw probability is shown with six decimals and marked saturated near 0 or 1", () => {
  assert.equal(rawProbability(0.999614), "0.999614 (saturated)");
  assert.equal(rawProbability(0.004), "0.004000 (saturated)");
  assert.equal(rawProbability(0.5), "0.500000");
  assert.equal(rawProbability(0.99), "0.990000");     // not above 0.99
  assert.equal(rawProbability(0.01), "0.010000");     // not below 0.01
});

const job = (extra = {}) => ({
  verdict: "Needs Verification", verdict_reason: "company profile not provided", classifier_risk_score: 100,
  classifier_prediction: "Potentially fraudulent", display_risk_score: 94, risk_score: 100, warning_signs: [],
  classifier_details: { model: "DistilBERT", fraud_probability: 0.999614, threshold: 0.291163 }, ...extra,
});

test("How this was decided shows the raw probability and no classifier N/100 line", () => {
  const rows = technicalDetails(job()).rows.map((r) => `${r.label}: ${r.text}`);
  assert.ok(rows.includes("Raw classifier probability: 0.999614 (saturated)"));
  assert.ok(rows.includes("Model details: DistilBERT, decision threshold 0.291163"));
  assert.equal(rows.some((r) => /Score details|\/100|\b100\b/.test(r)), false);
  assert.deepEqual(decisionDetails(job()).map((r) => r.label), ["Reason", "Raw classifier probability"]);
});

test("no score of exactly 0 or 100 is ever shown, for job or chat reports", () => {
  for (const [raw, shown] of [[0, "1 / 100"], [100, "99 / 100"], [55, "55 / 100"]]) {
    assert.equal(buildMainScreen({ verdict: "Likely Legitimate", risk_score: raw, warning_signs: [] }).scoreText, shown);
    assert.equal(buildMainScreen({ verdict: "Likely Legitimate", display_risk_score: raw, warning_signs: [] }).scoreText, shown);
  }
  assert.equal(chatScore({ display_risk_score: 0 }), 1);
  assert.equal(chatScore({ display_risk_score: 100 }), 99);
  assert.equal(chatScore({ risk_score: 100 }), 99);
});

// --- D: a failed request without a server message ---------------------------------------------------------------------

async function withFetch(fake, run) {
  const original = globalThis.fetch;
  globalThis.fetch = fake;
  try {
    return await run();
  } finally {
    globalThis.fetch = original;
  }
}

const calls = [
  ["analyzeJob", () => analyzeJob("A posting long enough to be analysed by the server.")],
  ["analyzeJobFields", () => analyzeJobFields({ title: "Clerk", description: "Enter customer data into spreadsheets." })],
  ["analyzeChat", () => analyzeChat(new File(["24/10/26, 9:00 - HR: Hi\n"], "chat.txt", { type: "text/plain" }))],
];

test("a bare 500 with no message shows the plain 'could not reach' message", async () => {
  assert.equal(SERVER_UNREACHABLE, "Could not reach the analysis server. Please try again in a moment.");
  for (const [name, call] of calls) {
    await withFetch(async () => new Response("", { status: 500 }), () =>
      assert.rejects(call, { message: SERVER_UNREACHABLE }, name));
    await withFetch(async () => new Response("<html>Internal Server Error</html>", { status: 500 }), () =>
      assert.rejects(call, { message: SERVER_UNREACHABLE }, name));
  }
});

test("a network failure shows the same message", async () => {
  for (const [name, call] of calls) {
    await withFetch(async () => { throw new TypeError("Failed to fetch"); }, () =>
      assert.rejects(call, { message: SERVER_UNREACHABLE }, name));
  }
});

test("a message from the server is still shown as sent, and a success still returns the report", async () => {
  for (const [name, call] of calls) {
    await withFetch(async () => Response.json({ error: "Please provide the job posting text." }, { status: 400 }), () =>
      assert.rejects(call, { message: "Please provide the job posting text." }, name));
    const body = await withFetch(async () => Response.json({ verdict: "Likely Legitimate" }), call);
    assert.equal(body.verdict, "Likely Legitimate", name);
  }
  assert.equal(requestErrorMessage({ message: "Too large" }), "Too large");
  assert.equal(requestErrorMessage(null), SERVER_UNREACHABLE);
  assert.equal(requestErrorMessage({}), SERVER_UNREACHABLE);
});

test("the old 'failed (500)' wording is gone from the requests", async () => {
  const source = await import("node:fs").then((fs) => fs.readFileSync(new URL("./api.js", import.meta.url), "utf8"));
  assert.equal(/failed \(\$\{res\.status\}\)/.test(source), false);
});
