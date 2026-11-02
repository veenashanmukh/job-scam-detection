// Presentation only. Which warning-sign categories raised a verdict, read from the verdict reason the backend
// wrote ("scam-pattern indicators present (A, B)" or "specific scam-pattern indicator present (A)").
// Nothing here changes a score, a category or a verdict, and risk_category is never touched.

const ESCALATED_REASON = /^(?:specific )?scam-pattern indicators? present \((.+)\)$/;

export function escalationCategories(report) {
  if (!report || report.type === "chat" || report.verdict !== "Needs Verification") return [];
  const match = ESCALATED_REASON.exec(String(report.verdict_reason || "").trim());
  return match ? match[1].split(",").map((name) => name.trim()).filter(Boolean) : [];
}

// One line shown next to the score when warning signs, not the classifier, raised the verdict.
export function escalationNote(report) {
  const categories = escalationCategories(report);
  return categories.length
    ? `Score reflects the text classifier only. The verdict was raised by warning signs: ${categories.join(", ")}.`
    : "";
}
