// Display-only text helpers.

// "upfront_payment" -> "Upfront payment": underscores and hyphens become spaces, then sentence case.
export function plainTag(tag) {
  const words = String(tag || "").replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim().toLowerCase();
  return words ? words[0].toUpperCase() + words.slice(1) : "";
}

// Plain, de-duplicated phrases for a list of tags; empty and non-string entries are dropped.
export function plainTags(tags) {
  if (!Array.isArray(tags)) return [];
  return [...new Set(tags.filter((t) => typeof t === "string").map(plainTag).filter(Boolean))];
}
