// Presentation only. Nothing here changes a score, a category, a verdict or any API field; functions only read
// the report and return text or flags for the UI.
import { clampDisplayScore } from "./chatView.js";
import { escalationNote } from "./verdictNote.js";

export const PLAIN_SENTENCES = {
  lowNoSigns:
    "This posting looks low risk. We found no warning signs. Still check the company before sharing documents or money.",
  lowMinorSigns:
    "This posting looks low risk. We found some minor warning signs. Still check the company before sharing documents or money.",
  careful: "Be careful. Check this employer before you share anything or pay anything.",
  scam: "This looks like a scam. Do not pay or share personal details.",
};

// One plain sentence per verdict, shown at the top of a job report and in its chat bubble. "" when there is none
// (chat reports, "Insufficient Information", reports without a verdict).
export function plainSentence(report) {
  if (!report || report.type === "chat") return "";
  switch (report.verdict) {
    case "Likely Legitimate":
      return Array.isArray(report.warning_signs) && report.warning_signs.length > 0
        ? PLAIN_SENTENCES.lowMinorSigns
        : PLAIN_SENTENCES.lowNoSigns;
    case "Needs Verification":
      return PLAIN_SENTENCES.careful;
    case "High Fraud Risk":
      return PLAIN_SENTENCES.scam;
    default:
      return "";
  }
}

// "0.999614", with "(saturated)" when the model is pinned near 0 or 1, where small differences mean little.
export function rawProbability(probability) {
  const p = Number(probability);
  return `${p.toFixed(6)}${p > 0.99 || p < 0.01 ? " (saturated)" : ""}`;
}

// The technical lines kept in the collapsed "How this was decided" section (job reports only).
export function decisionDetails(report) {
  if (!report || report.type === "chat") return [];
  const rows = [];
  if (report.verdict_reason) rows.push({ label: "Reason", text: String(report.verdict_reason) });
  const probability = report.classifier_details ? Number(report.classifier_details.fraud_probability) : NaN;
  if (report.classifier_details && Number.isFinite(probability)) {
    rows.push({ label: "Raw classifier probability", text: rawProbability(probability) });
  } else if (report.classifier_prediction) {
    rows.push({ label: "Classifier", text: String(report.classifier_prediction) });
  }
  return rows;
}

export const NO_EVIDENCE_LINE = "No specific evidence found.";

// What the Evidence section of a CHAT report shows. With an AI explanation it shows everything. Without one it
// shows only the matched warning signs and any similar known postings, or the single line "No specific evidence
// found."; the template explanation (which only restates the headline) and the rest are hidden.
export function evidenceView(report) {
  const evidence = (report && report.evidence) || {};
  const similar = evidence.similar_postings || [];
  const rules = evidence.rule_signs || [];
  const aiAvailable = !report || !report.llm_status || report.llm_status === "ok";
  if (aiAvailable) {
    return {
      aiAvailable: true,
      showExplanation: Boolean(report && report.explanation),
      showSimilar: true,
      showGraph: true,
      showRules: true,
      emptyLine: "",
    };
  }
  // Similar postings and linked contacts show whenever they have content, whether or not an AI explanation exists.
  const linked = (evidence.graph && evidence.graph.items) || [];
  const hasAny = similar.length > 0 || rules.length > 0 || linked.length > 0;
  return {
    aiAvailable: false,
    showExplanation: false,
    showSimilar: similar.length > 0,
    showGraph: linked.length > 0,
    showRules: rules.length > 0,
    emptyLine: hasAny ? "" : NO_EVIDENCE_LINE,
  };
}

// ---------------------------------------------------------------------------------------------------------------
// Main result screen for a job posting, built only from data in the report.

// Plain names for the rule categories the job rules really have. There is no job rule for "contact only through a
// messaging app" (that category exists for chats only), so no line is shown for it: a tick would claim a check
// that was never made. Pressure tactic and Guaranteed employment have no plain name here either; they appear as a
// warning line, under their own label, when they fire.
export const CHECKED_CATEGORIES = [
  { label: "Upfront payment", plain: "requests for money or fees" },
  { label: "Requests personal information", plain: "requests for bank or ID details" },
  { label: "No interview", plain: "claims of no interview" },
  { label: "Unusually high salary claim", plain: "unrealistic pay" },
  { label: "Cheque or overpayment scam", plain: "cheque or overpayment requests" },
  { label: "Gift card or crypto payment", plain: "gift card or crypto payments" },
  { label: "Parcel or money forwarding", plain: "parcel forwarding" },
];

export const NEXT_STEPS = {
  "Likely Legitimate": ["Check the company's official website.", "Never pay to get a job."],
  "Needs Verification": [
    "Do not pay or share ID documents until you have confirmed the employer through their official website or phone.",
    "Never pay to get a job.",
  ],
  "High Fraud Risk": ["Do not reply.", "Do not pay or share personal details."],
};

// One line per fired warning category, from this fixed map. Nothing is invented: a category that is not here adds no line.
export const CATEGORY_ADVICE = {
  "Cheque or overpayment scam": "Do not deposit the cheque or send any of the money onward. Tell your bank.",
  "Gift card or crypto payment": "Do not buy or send gift cards or crypto to anyone about a job.",
  "Parcel or money forwarding":
    "Do not receive or forward packages or money for an employer. Scammers use this to move stolen goods.",
  "Upfront payment": "Do not pay any fee to get a job, even if it is described as refundable.",
  "Requests personal information":
    "Do not send ID numbers, bank details or photos of documents before you have verified the employer.",
  "No interview": "Genuine employers interview you. Be wary of offers with no interview.",
  "Unusually high salary claim":
    "Be wary of very high pay for easy work. Honest employers do not promise big money for little effort.",
  "Guaranteed employment": "Be wary of a guaranteed job. No honest employer can promise you a job before they have met you.",
  "Pressure tactic": "Do not let anyone rush you. Scammers use deadlines to stop you checking.",
};
// Severity order: money moving first, then payments and personal data, then the softer signs.
export const ADVICE_ORDER = Object.keys(CATEGORY_ADVICE);
export const MAX_ADVICE_LINES = 4;

// The chat rules have their own labels; these are the ones with a counterpart above. "Moves to another app" has no
// fixed line, so it adds none.
export const CHAT_LABEL_TO_CATEGORY = {
  "Payment request": "Upfront payment",
  "Sensitive information request": "Requests personal information",
  "Pressure tactic": "Pressure tactic",
  "Unrealistic job promise": "Unusually high salary claim",
};
const CHAT_VERDICT = { "Higher risk": "High Fraud Risk", "Moderate risk": "Needs Verification", "Lower risk": "Likely Legitimate" };

export const SEARCH_LINE =
  "Search the company name with the word 'scam', and check that emails come from the company's official website domain.";
export const HELP_LINE =
  "If you have already paid or shared details: contact your bank at once. In India, report at cybercrime.gov.in or call 1930.";

function firedAdviceCategories(report) {
  const chat = report.type === "chat";
  const found = new Set();
  for (const sign of Array.isArray(report.warning_signs) ? report.warning_signs : []) {
    const label = typeof sign === "string" ? sign : sign && sign.label;
    const category = chat ? CHAT_LABEL_TO_CATEGORY[label] : label;
    if (category && Object.hasOwn(CATEGORY_ADVICE, category)) found.add(category);
  }
  return found;
}

// "What to do next" for a job or chat report: the verdict lines, then up to four category lines in severity order,
// then a closing line (a search tip for Likely Legitimate, where to get help for the two caution verdicts).
export function nextStepsFor(report) {
  if (!report) return [];
  const verdict = report.type === "chat" ? CHAT_VERDICT[report.risk_category] : report.verdict;
  const base = NEXT_STEPS[verdict];
  if (!base) return [];
  const fired = firedAdviceCategories(report);
  const advice = ADVICE_ORDER.filter((category) => fired.has(category)).slice(0, MAX_ADVICE_LINES).map((c) => CATEGORY_ADVICE[c]);
  return [...base, ...advice, verdict === "Likely Legitimate" ? SEARCH_LINE : HELP_LINE];
}

export const FOOTER_LINE = "This is a warning-sign check, not a guarantee.";
export const SCORE_NOTE = "Higher means more risk";

// Categories about money moving; if one fired, the "requests for money or fees" tick is not shown.
export const PAYMENT_RELATED_LABELS = [
  "Upfront payment",
  "Cheque or overpayment scam",
  "Gift card or crypto payment",
  "Parcel or money forwarding",
];
const MONEY_TICK_LABEL = "Upfront payment";
export const RESEMBLES_SCAMS_LINE = "The wording of this posting resembles postings that were reported as scams.";

const LEVEL = { "Likely Legitimate": "low", "Needs Verification": "mid", "High Fraud Risk": "high" };
const ICON = { "Likely Legitimate": "✓", "Needs Verification": "!", "High Fraud Risk": "!" };

const isObject = (value) => value !== null && typeof value === "object";

// "[S1]" and "[R1, R2]" markers are for the technical view; plain words do not carry them.
export function stripCitations(text) {
  return String(text || "")
    .replace(/\s*\[\s*[A-Z]\d{1,3}(?:\s*,\s*[A-Z]\d{1,3})*\s*\]/g, "")
    .replace(/\s+([.,;:!?])/g, "$1")
    .replace(/\s{2,}/g, " ")
    .trim();
}

// "Why": the AI explanation in plain words with its cited guidance sources as links. null when there is no AI
// explanation, so no section and no filler text is shown.
// The ids an accepted explanation cited that point to a similar posting (S), entity link (G), rule sign (R) or guidance
// item (K) present in the report's evidence. The pipeline already rejects unknown ids; this only reads its result.
export function citedEvidenceIds(report) {
  const evidence = (report && report.evidence) || {};
  const present = new Set([
    ...(evidence.similar_postings || []), ...((evidence.graph && evidence.graph.items) || []),
    ...(evidence.rule_signs || []), ...(evidence.guidance || []),
  ].map((item) => item && item.id));
  const cited = Array.isArray(report && report.cited_evidence_ids) ? report.cited_evidence_ids : [];
  return cited.filter((id) => /^[SGRK]\d+$/.test(String(id)) && present.has(id));
}

export function whyView(report) {
  if (!report || report.llm_status !== "ok" || !report.explanation) return null;
  // A generic sentence that cites nothing ("Based on the evidence provided...") is not shown.
  if (citedEvidenceIds(report).length === 0) return null;
  const text = stripCitations(report.explanation);
  if (!text) return null;
  const cited = new Set(citedEvidenceIds(report));
  const guidance = (report.evidence && report.evidence.guidance) || [];
  const sources = guidance
    .filter((k) => cited.has(k.id) && /^https?:\/\//i.test(String(k.url || "")))
    .map((k) => ({ title: k.title || k.publisher || "Source", url: k.url }));
  return { text, sources };
}

const describe = (label) => {
  const known = CHECKED_CATEGORIES.find((c) => c.label === label);
  return known ? known.plain : String(label).toLowerCase();
};

// Everything the main screen shows, as plain data. Never claims a posting is verified or safe; tick lines say
// "We did not find ..." and exist only for categories that did not fire.
export function buildMainScreen(report) {
  if (!report || report.type === "chat") return null;
  const signs = Array.isArray(report.warning_signs) ? report.warning_signs.filter(isObject) : [];
  const firedLabels = [...new Set(signs.map((s) => String(s.label)))];
  const phraseFor = (label) => {
    const sign = signs.find((s) => String(s.label) === label);
    return sign && typeof sign.matched_text === "string" ? sign.matched_text : "";
  };

  const checked = firedLabels.map((label) => ({
    kind: "warn",
    label,
    text: `We found a warning sign: ${describe(label)}`,
    phrase: phraseFor(label),
  }));
  // Tick lines are reassurance, so they exist only for "Likely Legitimate". The two caution verdicts show warning
  // lines and nothing that could read as reassurance.
  const showTicks = report.verdict === "Likely Legitimate";
  const cautious = report.verdict === "Needs Verification" || report.verdict === "High Fraud Risk";
  const paymentFired = firedLabels.some((label) => PAYMENT_RELATED_LABELS.includes(label));
  if (showTicks && Array.isArray(report.warning_signs)) {
    // The rules ran, so every category that did not fire gets a tick line, except that "no requests for money or
    // fees" would sit badly next to a payment-related warning.
    for (const { label, plain } of CHECKED_CATEGORIES) {
      if (firedLabels.includes(label)) continue;
      if (label === MONEY_TICK_LABEL && paymentFired) continue;
      checked.push({ kind: "ok", label, text: `We did not find ${plain}.` });
    }
  }
  if (cautious && firedLabels.length === 0) {
    // The classifier raised the verdict and no rule fired.
    checked.push({ kind: "warn", label: "similar wording", text: RESEMBLES_SCAMS_LINE, phrase: "" });
  }
  if (showTicks && report.company_profile_present === true) {
    checked.push({ kind: "ok", label: "company profile", text: "The posting includes a company description." });
  }

  const evidence = report.evidence || {};
  const similar = (evidence.similar_postings || []).map((s) => ({
    id: s.id, genuine: s.label === 0, similarity: Number(s.similarity), snippet: s.snippet || "",
  }));
  if (showTicks && similar.length > 0 && similar.every((s) => s.genuine)) {
    checked.push({
      kind: "ok",
      label: "similar genuine postings",
      text: `It looks similar to ${similar.length} known genuine posting${similar.length === 1 ? "" : "s"}.`,
    });
  }
  const linked = ((evidence.graph && evidence.graph.items) || []).map((g) => ({
    id: g.id,
    text: g.message || `This ${g.kind_name} appears in ${g.linked_count} known postings (${g.fraud_count} labelled fraud).`,
  }));

  // Never show exactly 0 or 100: the display score is already clamped; an older response without it is clamped here.
  const score = typeof report.display_risk_score === "number" ? clampDisplayScore(report.display_risk_score)
    : typeof report.risk_score === "number" ? clampDisplayScore(report.risk_score) : null;
  return {
    verdict: report.verdict || "",
    level: LEVEL[report.verdict] || "neutral",
    icon: ICON[report.verdict] || "?",
    headline: report.verdict || "",
    sentence: plainSentence(report),
    score,
    scoreText: typeof score === "number" ? `${score} / 100` : "",
    scoreNote: typeof score === "number" ? SCORE_NOTE : "",
    checked,
    next: nextStepsFor(report),
    why: whyView(report),
    similar,
    linked,
    footer: FOOTER_LINE,
  };
}

// Every visible string of the main screen, for checks on its wording.
export function mainScreenText(screen) {
  if (!screen) return [];
  return [
    screen.headline, screen.sentence, screen.scoreText, screen.scoreNote,
    ...screen.checked.flatMap((c) => [c.text, c.phrase || ""]),
    ...screen.next,
    ...(screen.why ? [screen.why.text, ...screen.why.sources.map((s) => s.title)] : []),
    ...screen.similar.map((s) => s.snippet),
    ...screen.linked.map((g) => g.text),
    screen.footer,
  ].filter(Boolean);
}

// The collapsed "How this was decided" section: everything technical.
export function technicalDetails(report) {
  const rows = decisionDetails(report);
  const note = escalationNote(report);
  if (note) rows.push({ label: "Verdict note", text: note });
  const d = report && report.classifier_details;
  if (isObject(d)) {
    rows.push({ label: "Model details", text: `${d.model || "model"}, decision threshold ${d.threshold}` });
  }
  const evidenceRules = (report && report.evidence && report.evidence.rule_signs) || [];
  const rules = evidenceRules.length > 0
    ? evidenceRules.map((r) => ({ id: r.id, label: r.label, matched_text: r.matched_text || "" }))
    : ((report && Array.isArray(report.warning_signs) ? report.warning_signs : []).filter(isObject)
        .map((s, i) => ({ id: `R${i + 1}`, label: s.label, matched_text: s.matched_text || "" })));
  const notes = report && Array.isArray(report.uncertainty_notes) ? report.uncertainty_notes.map(String) : [];
  return { rows, rules, notes };
}
