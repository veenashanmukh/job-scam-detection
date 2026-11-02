// Presentation only, for chat reports. Nothing here changes a score, a rule or an API field; functions read the report
// and return the text or number to show.

// The note the server appends to a reused explanation after "You in this chat" is changed.
export const REUSED_NOTE =
  "(Explanation based on the first analysis; only the rule score was recomputed for the chosen name.)";

export const NO_PATTERNS_PREDICTION = "No strong rule-based warning pattern detected";
export const NO_PATTERNS_LINE = "No warning patterns detected in the recruiter's messages.";
export const CHAT_SCORE_NOTE = "Higher means more risk";

// The shown score is never exactly 0 or 100 (same clamp as the server's display_risk_score).
export function clampDisplayScore(value) {
  return Math.trunc(Math.min(99, Math.max(1, Number(value))));
}

// The number to show for a chat report: the server's display_risk_score, or the raw score clamped for older responses.
export function chatScore(report) {
  if (report && typeof report.display_risk_score === "number") return clampDisplayScore(report.display_risk_score);
  if (report && typeof report.risk_score === "number") return clampDisplayScore(report.risk_score);
  return null;
}

// The line under the score. The "nothing found" prediction becomes a plain sentence; others keep their text.
export function chatPrediction(report) {
  const prediction = report && report.classifier_prediction ? String(report.classifier_prediction) : "";
  if (!prediction) return { text: "", labelled: false };
  if (prediction === NO_PATTERNS_PREDICTION) return { text: NO_PATTERNS_LINE, labelled: false };
  return { text: prediction, labelled: true };
}

// Splits the reused-explanation note off the explanation so it can sit in the collapsed section.
export function splitExplanationNote(text) {
  const full = String(text || "");
  const index = full.lastIndexOf(REUSED_NOTE);
  if (index === -1) return { body: full, note: "" };
  return { body: (full.slice(0, index) + full.slice(index + REUSED_NOTE.length)).trim(), note: REUSED_NOTE };
}

// Shown under the risk timeline when no message carries a warning sign (instead of an empty list).
export const NO_FLAGGED_MESSAGES = "No warning signs in any message";

// Scam-prevention guidance is hidden for a calm chat (no warning sign and Lower risk); kept for Moderate and Higher.
export function showChatGuidance(report) {
  if (!report) return false;
  const signs = Array.isArray(report.warning_signs) ? report.warning_signs : [];
  return !(signs.length === 0 && report.risk_category === "Lower risk");
}
