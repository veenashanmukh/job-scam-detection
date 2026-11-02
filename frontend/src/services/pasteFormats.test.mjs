import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { analyzePaste, looksLikeConversation, routePastedText, speakerLines } from "./conversation.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..");

// C1: plain labels; C2: WhatsApp brackets; C3: Android dash with names.
const C1 = [
  "Recruiter: Hello, we found your profile and have a data entry role for you.",
  "You: Thanks, which company is this?",
  "Recruiter: A global firm. You must pay a registration fee today to confirm your seat.",
  "You: Why do I need to pay?",
];
const C2 = [
  "[24/10/26, 9:02 AM] Recruiter: Hello, we found your profile.",
  "[24/10/26, 9:03 AM] You: Thanks, can you share the details?",
  "[24/10/26, 9:05 AM] Recruiter: Send your bank details and add me on Telegram.",
];
const C3 = [
  "24/10/26, 9:00 - Priya: Why do I need to pay? Is my data safe?",
  "24/10/26, 9:01 - HR Desk: Pay the registration fee within 2 hours.",
  "24/10/26, 9:02 - Priya: I want to speak to someone first.",
  "24/10/26, 9:03 - HR Desk: Send your documents on Telegram today.",
];

const asLines = (lines) => lines.join("\n");
const asParagraph = (lines) => lines.join(" ");
const asMessy = (lines) => `\n\n${lines.join("\n\n\t")}\n\n`.replace(/: /g, ":\t");
const asWindows = (lines) => lines.join("\r\n") + "\r\n";
const FORMATS = { "separate lines": asLines, "one paragraph": asParagraph, "blank lines and tabs": asMessy, "Windows line endings": asWindows };

test("C1, C2 and C3 are detected as conversations in every paste format", () => {
  for (const [name, chat] of Object.entries({ C1, C2, C3 })) {
    for (const [format, make] of Object.entries(FORMATS)) {
      const result = analyzePaste(make(chat));
      assert.equal(result.conversation, true, `${name} as ${format}`);
      assert.equal(routePastedText(make(chat)), "chat", `${name} as ${format}`);
    }
  }
});

test("the tidy-up gives one message per line with the right speakers, whatever the paste format", () => {
  const speakers = (text) => text.split("\n").filter((l) => l.trim()).map((l) => l.replace(/^\[?\d[^\]]*?\]?\s*-?\s*/, "").split(":")[0].trim());
  for (const [format, make] of Object.entries(FORMATS)) {
    const c1 = analyzePaste(make(C1)).normalized.split("\n").filter((l) => l.trim());
    assert.equal(c1.length, 4, `C1 ${format}`);
    assert.deepEqual(speakers(c1.join("\n")), ["Recruiter", "You", "Recruiter", "You"], `C1 ${format}`);
    assert.match(c1[2], /^Recruiter: A global firm\. You must pay a registration fee today to confirm your seat\.$/, format);
    const c2 = analyzePaste(make(C2)).normalized.split("\n").filter((l) => l.trim());
    assert.equal(c2.length, 3, `C2 ${format}`);
    const c3 = analyzePaste(make(C3)).normalized.split("\n").filter((l) => l.trim());
    assert.equal(c3.length, 4, `C3 ${format}`);
    assert.match(c3[1], /HR Desk: Pay the registration fee within 2 hours\.$/, format);
  }
});

test("which path found it: lines, a timestamp before each message, or inline labels", () => {
  assert.equal(analyzePaste(asLines(C1)).mode, "lines");
  assert.equal(analyzePaste(asParagraph(C1)).mode, "inline");
  assert.equal(analyzePaste(asParagraph(C2)).mode, "timestamps");
  assert.equal(analyzePaste(asParagraph(C3)).mode, "timestamps");
});

test("WhatsApp (bracket and dash) and Telegram-style exports are accepted", () => {
  const whatsappBracket = ["[12/03/24, 10:15] Sam: Hi, are you available?", "[12/03/24, 10:16] You: Yes.", "[12/03/24, 10:17] Sam: Send your CV here.", "[12/03/24, 10:18] You: Ok."].join("\n");
  const whatsappDash = ["12/03/24, 10:15 - Sam: Hi, are you available?", "12/03/24, 10:16 - You: Yes.", "12/03/24, 10:17 - Sam: Send your CV here."].join("\n");
  const telegram = ["Recruiter, [24.10.2024 9:02]", "Hello, we have a role for you.", "", "You, [24.10.2024 9:03]", "Which company?", "", "Recruiter, [24.10.2024 9:04]", "A global firm.", "Pay the fee today."].join("\n");
  const telegramDotted = ["[24.10.24 9:02] Recruiter: Hello there.", "[24.10.24 9:03] You: Hi.", "[24.10.24 9:04] Recruiter: Pay the fee."].join("\n");
  for (const text of [whatsappBracket, whatsappDash, telegram, telegramDotted]) assert.equal(looksLikeConversation(text), true, text.slice(0, 30));
  // Telegram headers become "date, time - Name: text" with continuation lines kept, so the chat parser keeps the times
  assert.deepEqual(analyzePaste(telegram).normalized.split("\n").filter((l) => l.trim()), [
    "24/10/2024, 9:02 - Recruiter: Hello, we have a role for you.",
    "24/10/2024, 9:03 - You: Which company?",
    "24/10/2024, 9:04 - Recruiter: A global firm.",
    "Pay the fee today.",
  ]);
  assert.match(analyzePaste(telegramDotted).normalized, /^\[24\/10\/24 9:02\] Recruiter: Hello there\./);
  assert.equal(speakerLines(analyzePaste(telegram).normalized).length, 3);
});

test("names-only chats need a clear back-and-forth; three different names once each do not count", () => {
  assert.equal(looksLikeConversation("Priya: Is this real? Sam: Yes, it is. Priya: Who is the employer? Sam: A firm in Leeds."), true);
  assert.equal(looksLikeConversation("Shift: days. Team: sales. Office: Leeds."), false);
  assert.equal(looksLikeConversation("Option: A is cheaper. Option: B is faster. Result: both work."), false);
});

// --- things that must NOT be detected ------------------------------------------------------------------------------

const labelled = (p) =>
  [["Title", p.title], ["Company profile", p.company_profile], ["Description", p.description], ["Requirements", p.requirements], ["Benefits", p.benefits]]
    .filter(([, v]) => v).map(([l, v]) => `${l}: ${v}`);

function postingTexts(p) {
  const lines = labelled(p);
  const plain = [p.title, p.company_profile, p.description, p.requirements, p.benefits].filter(Boolean);
  return {
    "labelled lines": lines.join("\n"),
    "labelled, one paragraph": lines.join(" "),
    "plain lines": plain.join("\n"),
    "plain paragraph": plain.join(" "),
    "tabs and blank lines": `\n\n${lines.join("\n\n\t")}\n`,
    "Windows line endings": lines.join("\r\n"),
  };
}

test("the 12 synthetic dev job postings are never detected, in any paste format", () => {
  const postings = JSON.parse(readFileSync(path.join(root, "data", "synthetic_jobs", "postings.json"), "utf8"));
  assert.equal(postings.length, 12);
  for (const posting of postings) {
    for (const [format, text] of Object.entries(postingTexts(posting))) {
      assert.equal(looksLikeConversation(text), false, `${posting.id} as ${format}`);
    }
  }
});

test("the fake-cheque paragraph, the parcel posting, the legitimate example and the demo examples are never detected", () => {
  const demo = JSON.parse(readFileSync(path.join(root, "backend", "data", "demo_examples.json"), "utf8"));
  const known = [
    "Remote Customer Support Associate (Immediate Start): We are expanding our remote support team and need 10 people to start this week. Earn $35 per hour from home with no experience or interview required. After you accept, we will mail you a cheque for $2,800. Deposit it, keep $800 as your first week's pay, and transfer the remaining $2,000 to our equipment vendor so your laptop and software can ship. Reply on WhatsApp today to secure your place. Requirements: smartphone, bank account, and a quiet place to work. Benefits: weekly pay, free laptop, flexible hours, and a signing bonus.",
    "Title: Parcel Inspector, Work From Home. Description: Receive packages at your home address, check them, and forward them to our overseas customers using the prepaid labels we email you. 25 per parcel, paid weekly.",
    ...demo.flatMap((p) => Object.values(postingTexts(p))),
  ];
  assert.ok(known.length >= 20);
  for (const text of known) assert.equal(looksLikeConversation(text), false, text.slice(0, 50));
});

test("postings with labelled fields or a single Contact line are not conversations", () => {
  const fields = "Title: Packer Company: Acme Description: Pack boxes. Requirements: None. Benefits: Holiday. Salary: 20 an hour. Location: Leeds.";
  assert.equal(looksLikeConversation(fields), false);
  assert.equal(looksLikeConversation(fields.replace(/ (?=[A-Z][a-z]+:)/g, "\n")), false);
  assert.equal(looksLikeConversation("We are hiring packers in Leeds. Contact: jobs at example dot org. Start Monday."), false);
  assert.equal(looksLikeConversation("Description: Earn from home.\nContact: Sam on 555 0100."), false);
  assert.equal(looksLikeConversation("1. Salary: 20 an hour 2. Location: Leeds 3. Hours: 9 to 5 4. Holiday: 25 days"), false);
});

test("bulleted lists of Name: value stay out, also when pasted as one paragraph", () => {
  assert.equal(looksLikeConversation("• Recruiter: ask for Sam • You: bring a pen • Recruiter: arrive early"), false);
  assert.equal(looksLikeConversation("- Recruiter: ask for Sam\n- You: bring a pen\n- Recruiter: arrive early"), false);
});
