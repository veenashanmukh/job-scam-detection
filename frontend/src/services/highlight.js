// Which parts of the submitted text to mark. Only phrases the backend returned as the matched text of a
// listed warning sign are ever marked; there is no separate word list, so a highlight always belongs to a sign.

const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const normalise = (s) => s.toLowerCase().replace(/\s+/g, " ").trim();

// Signs from a job report are { label, matched_text }; chat reports list plain labels, which have no phrase.
export function signPhrases(signs) {
  return (Array.isArray(signs) ? signs : [])
    .filter((s) => s && typeof s === "object" && typeof s.matched_text === "string" && s.matched_text.trim())
    .map((s) => ({ label: String(s.label || "Warning sign"), phrase: s.matched_text.trim() }));
}

// Segments [{ text, label }] covering the whole text; label is null for unmarked text.
export function highlightSegments(text, signs) {
  const source = typeof text === "string" ? text : "";
  const phrases = signPhrases(signs);
  if (!source || !phrases.length) return [{ text: source, label: null }];

  // Longest phrases first so a long match is not cut short by a shorter one; whitespace-flexible because the
  // backend matched on cleaned text (tags removed, spaces collapsed) while this is the text as typed.
  const ordered = [...phrases].sort((a, b) => b.phrase.length - a.phrase.length);
  const pattern = new RegExp(
    ordered.map((p) => escapeRegex(p.phrase).replace(/\s+/g, "\\s+")).join("|"),
    "gi"
  );
  const labelFor = (matched) => {
    const key = normalise(matched);
    return (ordered.find((p) => normalise(p.phrase) === key) || ordered[0]).label;
  };

  const segments = [];
  let last = 0;
  for (const match of source.matchAll(pattern)) {
    if (!match[0]) continue;
    if (match.index > last) segments.push({ text: source.slice(last, match.index), label: null });
    segments.push({ text: match[0], label: labelFor(match[0]) });
    last = match.index + match[0].length;
  }
  if (last < source.length) segments.push({ text: source.slice(last), label: null });
  return segments.length ? segments : [{ text: source, label: null }];
}
