// Renders the real JobReport component to HTML (bundled with esbuild, which ships with vite) and checks what the
// user would actually see. No browser, no network.
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { build } from "esbuild";

const frontend = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");

const bundle = await build({
  stdin: {
    contents: `
      import { renderToStaticMarkup } from "react-dom/server";
      import JobReport from "./src/JobReport.jsx";
      export const render = (report) => renderToStaticMarkup(<JobReport report={report} />);
    `,
    resolveDir: frontend,
    loader: "jsx",
  },
  bundle: true,
  write: false,
  format: "cjs",
  platform: "node",
  jsx: "automatic",
  logLevel: "silent",
});
const bundled = { exports: {} };
new Function("module", "exports", "require", bundle.outputFiles[0].text)(
  bundled, bundled.exports, createRequire(import.meta.url)
);
const { render } = bundled.exports;

const LINKED = {
  items: [{ id: "G1", kind_name: "email address", message: "This email address appears in 3 known postings (2 labelled fraud)." }],
};
const SIMILAR = [{ id: "S1", label: 1, similarity: 0.61, snippet: "A known posting that asked for a fee." }];
const report = (extra = {}) => ({
  verdict: "Needs Verification",
  verdict_reason: "classifier flagged the posting and the company profile is empty",
  display_risk_score: 52,
  risk_score: 60,
  classifier_risk_score: 55,
  classifier_prediction: "Potentially fraudulent",
  warning_signs: [],
  uncertainty_notes: [],
  llm_status: "unavailable",
  explanation: "Verdict: Needs Verification. Reason: restated headline.",
  evidence: { similar_postings: [], rule_signs: [], guidance: [], graph: { items: [], note: "No linked companies or contacts found." } },
  ...extra,
});
const withEvidence = (evidence, extra = {}) => report({ evidence: { similar_postings: [], rule_signs: [], guidance: [], graph: { items: [] }, ...evidence }, ...extra });

test("linked companies present and AI explanation unavailable: the block is visible", () => {
  const html = render(withEvidence({ graph: LINKED }, { llm_status: "unavailable" }));
  assert.match(html, /<h3>Linked companies and contacts<\/h3>/);
  assert.match(html, /This email address appears in 3 known postings \(2 labelled fraud\)\./);
  assert.doesNotMatch(html, /<h3>Why<\/h3>/); // no AI explanation, so no Why section
  assert.doesNotMatch(html, /<h3>Evidence<\/h3>/);
});

test("similar known postings present and AI explanation unavailable: the block is visible", () => {
  for (const status of ["unavailable", "error", "quota", "inconsistent"]) {
    const html = render(withEvidence({ similar_postings: SIMILAR }, { llm_status: status }));
    assert.match(html, /<h3>Similar known postings<\/h3>/, status);
    assert.match(html, /A known posting that asked for a fee\./, status);
  }
});

test("both blocks are visible with an AI explanation too, next to Why", () => {
  const html = render(withEvidence({ graph: LINKED, similar_postings: SIMILAR, rule_signs: [{ id: "R1", label: "Upfront payment", matched_text: "fee" }] },
    { llm_status: "ok", explanation: "You are asked to pay a fee [R1].", cited_evidence_ids: ["R1"] }));
  assert.match(html, /<h3>Why<\/h3>/);
  assert.match(html, /<h3>Linked companies and contacts<\/h3>/);
  assert.match(html, /<h3>Similar known postings<\/h3>/);
});

test("empty blocks are hidden in every AI state, including a graph that only has a note", () => {
  for (const status of ["ok", "unavailable"]) {
    const html = render(report({ llm_status: status, explanation: "", cited_evidence_ids: [] }));
    assert.doesNotMatch(html, /Linked companies and contacts/, status);
    assert.doesNotMatch(html, /Similar known postings/, status);
    assert.doesNotMatch(html, /No linked companies or contacts found/, status);
    assert.doesNotMatch(html, /Evidence<\/h3>/, status);
    assert.doesNotMatch(html, /No specific evidence found/, status);
  }
});

test("the rendered main screen keeps the collapsed technical section closed", () => {
  const html = render(withEvidence({ graph: LINKED }));
  assert.match(html, /<details class="how-decided"><summary>How this was decided<\/summary>/);
  assert.doesNotMatch(html, /<details class="how-decided" open/);
  const main = html.split('<details class="how-decided">')[0].toLowerCase();
  for (const word of ["provisional", "probability", "classifier", "threshold"]) {
    assert.equal(main.includes(word), false, `${word} on the rendered main screen`);
  }
});

test("the matched rules table sits in a scroll box so the Id column never wraps", () => {
  const html = render(report({ warning_signs: [{ label: "Upfront payment", matched_text: "registration fee" }],
    evidence: { similar_postings: [], guidance: [], graph: { items: [] },
                rule_signs: [{ id: "R1", label: "Upfront payment", matched_text: "registration fee" }] } }));
  assert.match(html, /<div class="table-scroll"><table class="rules-table">/);
  assert.match(html, /<td>R1<\/td>/);
});
