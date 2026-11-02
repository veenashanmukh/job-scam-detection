// Routing only. Decides whether pasted text is a conversation (to be checked as a chat) or a job posting, and tidies
// a pasted conversation into the "one message per line" form the chat parser reads. It reads the text and returns
// booleans and text; no score, rule or verdict is involved.

export const ROUTED_NOTE = "This looks like a conversation, so it was checked as a chat.";
export const OVERRIDE_LABEL = "Check as job posting instead";

// Labels that are usually a speaker in a pasted chat.
const KNOWN_SPEAKERS = new Set([
  "recruiter", "hr", "you", "me", "candidate", "applicant", "manager", "hiring manager", "agent", "interviewer",
  "employer", "support", "admin", "assistant", "client", "customer", "sender", "user", "bot",
]);

// Labels that start a field of a job posting (including the app's own "Title:", "Description:" ... format) or a
// note, never a speaker.
const FIELD_LABELS = new Set([
  "title", "job title", "company", "company profile", "description", "job description", "requirements", "benefits",
  "responsibilities", "qualifications", "skills", "location", "salary", "pay", "department", "industry", "function",
  "employment type", "job type", "experience", "education", "schedule", "hours", "apply", "how to apply", "contact",
  "email", "phone", "website", "about", "about us", "note", "notes", "important", "subject", "from", "to", "cc",
  "date", "re", "summary", "overview", "duties", "perks", "compensation", "position", "role", "vacancy", "deadline",
  "warning", "tip", "tips", "update", "reminder", "example", "disclaimer", "what we offer", "who we are",
]);

const DATE = String.raw`\d{1,4}[/.-]\d{1,2}[/.-]\d{2,4}`;
const CLOCK = String.raw`\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp][Mm])?`;
// A timestamp some chat exports put before the speaker: "[24/10/26, 9:02 AM] " or "24/10/26, 9:00 - ".
const TIMESTAMP = new RegExp(String.raw`^\[?${DATE},?\s+${CLOCK}\]?\s*(?:[-–]\s*)?`);
const SPEAKER_LINE = /^([^:\n]{1,30}):\s*(\S.*)$/;
const NAME_WORD = /^[A-Z][a-z'’.-]+$/;
const ACRONYM = /^[A-Z]{2,4}$/;

// "Recruiter", "HR", "You", or a name of up to three capitalised words ("Priya", "John Smith").
export function isSpeakerLabel(label) {
  const trimmed = String(label || "").trim();
  if (!/^[A-Za-z]/.test(trimmed)) return false;
  const lower = trimmed.toLowerCase();
  if (FIELD_LABELS.has(lower)) return false;
  if (KNOWN_SPEAKERS.has(lower)) return true;
  const words = trimmed.split(/\s+/);
  if (words.length > 3) return false;
  // A name is capitalised words; a short acronym may sit inside a longer label ("HR Desk"), but a lone all-caps
  // word that is not a known speaker ("NOTE", "FAQ") is not a speaker.
  const hasName = words.some((word) => NAME_WORD.test(word));
  return hasName && words.every((word) => NAME_WORD.test(word) || (words.length > 1 && ACRONYM.test(word)));
}

// Line endings, tabs and runs of spaces; blank lines are kept (the parsers ignore them).
function tidy(text) {
  return String(text || "").replace(/\r\n?/g, "\n").replace(/[\t ]/g, " ").replace(/ {2,}/g, " ");
}

// Lines that start with a speaker label (after an optional timestamp): [{ label, text }].
export function speakerLines(text) {
  const found = [];
  for (const raw of tidy(text).split("\n")) {
    const match = SPEAKER_LINE.exec(raw.trim().replace(TIMESTAMP, ""));
    if (match && isSpeakerLabel(match[1])) found.push({ label: match[1].trim(), text: match[2] });
  }
  return found;
}

// At least 3 speaker lines with at least 2 different labels, and at least one speaker talks more than once (a real
// back-and-forth). A posting whose lines are each a different "Field: value" does not qualify.
function enoughSpeakers(labels) {
  if (labels.length < 3) return false;
  const counts = new Map();
  for (const label of labels) counts.set(label.toLowerCase(), (counts.get(label.toLowerCase()) || 0) + 1);
  return counts.size >= 2 && Math.max(...counts.values()) >= 2;
}

const linesLookLikeChat = (text) => enoughSpeakers(speakerLines(text).map((l) => l.label));

// Telegram Desktop copies a message as a header line "Name, [24.10.2024 9:02]" followed by its text lines. Rewrite it
// as "24/10/2024, 9:02 - Name: text" so the chat parser reads it with its time.
const TELEGRAM_HEADER = new RegExp(String.raw`^\s*([^,\n]{1,40}?),\s*\[(${DATE})[ ,]+(${CLOCK})\]\s*$`);

function convertTelegramHeaders(text) {
  const out = [];
  let pendingPrefix = null;
  for (const line of text.split("\n")) {
    const header = TELEGRAM_HEADER.exec(line);
    if (header) {
      pendingPrefix = `${header[2]}, ${header[3]} - ${header[1].trim()}:`;
    } else if (pendingPrefix && line.trim()) {
      out.push(`${pendingPrefix} ${line.trim()}`);
      pendingPrefix = null;
    } else if (!pendingPrefix || line.trim()) {
      out.push(line);
    }
  }
  return out.join("\n");
}

// "[24.10.24 9:02]" and "24.10.2024, 10:15 -" use dots; the chat parser reads slashes.
const dottedDates = (text) => text.replace(new RegExp(String.raw`(\d{1,4})\.(\d{1,2})\.(\d{2,4})(?=,?\s+${CLOCK})`, "g"), "$1/$2/$3");

// A whole chat pasted as one paragraph still has a timestamp in front of each message: break before each one.
const INLINE_TIMESTAMP = new RegExp(String.raw`(\S)\s+(?=\[?${DATE},?\s+${CLOCK})`, "g");
const splitBeforeTimestamps = (text) => text.replace(INLINE_TIMESTAMP, "$1\n");

// Inline speaker labels in running text ("Recruiter: Hi. You: Hello. Recruiter: ..."): the word or two before each
// ": ". Supported: a known speaker (one or two words), an acronym plus a name ("HR Desk"), or one capitalised name.
function findInlineLabels(flat) {
  const labels = [];
  for (const colon of flat.matchAll(/:(?=\s)/g)) {
    const before = flat.slice(Math.max(0, colon.index - 40), colon.index);
    const two = /(^|[\s.!?"'’()[\]])([A-Za-z][A-Za-z'’-]*) ([A-Za-z][A-Za-z'’-]*)$/.exec(before);
    const one = /(^|[\s.!?"'’()[\]])([A-Za-z][A-Za-z'’-]*)$/.exec(before);
    let label = "";
    if (two) {
      const candidate = `${two[2]} ${two[3]}`;
      const knownTwo = KNOWN_SPEAKERS.has(candidate.toLowerCase());
      const acronymName = ACRONYM.test(two[2]) && NAME_WORD.test(two[3]);
      if (isSpeakerLabel(candidate) && (knownTwo || acronymName)) label = candidate;
    }
    if (!label && one && isSpeakerLabel(one[2])) label = one[2];
    if (!label) continue;
    const start = colon.index - label.length;
    // A bulleted or numbered "Label:" is a list item, not a speaker.
    const lead = flat.slice(0, start).trimEnd();
    if (/[-*•·–—]$/.test(lead) || /\d[.)]$/.test(lead)) continue;
    labels.push({ label, start, end: colon.index + 1 });
  }
  return labels;
}

function inlineLooksLikeChat(labels) {
  if (!enoughSpeakers(labels.map((l) => l.label))) return false;
  if (labels.some((l) => KNOWN_SPEAKERS.has(l.label.toLowerCase()))) return true;
  // Names only: ask for a clear back-and-forth between at least two people.
  const counts = new Map();
  for (const { label } of labels) counts.set(label.toLowerCase(), (counts.get(label.toLowerCase()) || 0) + 1);
  return labels.length >= 4 && [...counts.values()].filter((n) => n >= 2).length >= 2;
}

// Decide and tidy in one pass. normalized is one message per line when the text is a conversation.
// mode: "lines" (labels at the start of lines), "timestamps" (one paragraph, a timestamp before each message),
// "inline" (one paragraph, speaker labels only) or "" (not a conversation).
export function analyzePaste(text) {
  const base = dottedDates(convertTelegramHeaders(tidy(text)));
  if (linesLookLikeChat(base)) {
    const lines = base.split("\n").map((line) => line.trim()).filter(Boolean);
    return { conversation: true, normalized: lines.join("\n"), mode: "lines" };
  }

  const flat = base.replace(/\s*\n\s*/g, " ").trim();
  const byTimestamps = splitBeforeTimestamps(flat);
  if (byTimestamps !== flat && linesLookLikeChat(byTimestamps)) {
    return { conversation: true, normalized: byTimestamps, mode: "timestamps" };
  }

  const labels = findInlineLabels(flat);
  if (inlineLooksLikeChat(labels)) {
    const lines = labels.map((l, i) => {
      const end = i + 1 < labels.length ? labels[i + 1].start : flat.length;
      return `${l.label}: ${flat.slice(l.end, end).trim()}`;
    });
    return { conversation: true, normalized: lines.join("\n"), mode: "inline" };
  }
  return { conversation: false, normalized: "", mode: "" };
}

export function looksLikeConversation(text) {
  return analyzePaste(text).conversation;
}

// The pasted text as a .txt file for the chat analyzer, tidied into one message per line.
export function conversationFile(text) {
  const { normalized } = analyzePaste(text);
  return new File([normalized || String(text)], "pasted_conversation.txt", { type: "text/plain" });
}

// Where pasted text goes: "chat" for a conversation, "job" otherwise. forceJob is the user's override.
export function routePastedText(text, { forceJob = false } = {}) {
  return !forceJob && looksLikeConversation(text) ? "chat" : "job";
}
