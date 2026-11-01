import assert from "node:assert/strict";
import test from "node:test";

import {
  OVERRIDE_LABEL, ROUTED_NOTE, isSpeakerLabel, looksLikeConversation, routePastedText, speakerLines,
} from "./conversation.js";

// C1: plain labels, no timestamps
const C1 = [
  "Recruiter: Hello, we found your profile and have a data entry role for you.",
  "You: Thanks, which company is this?",
  "Recruiter: A global firm. You must pay a registration fee today to confirm your seat.",
  "You: Why do I need to pay?",
].join("\n");

// C2: WhatsApp style, square-bracket timestamps
const C2 = [
  "[24/10/26, 9:02 AM] Recruiter: Hello, we found your profile.",
  "[24/10/26, 9:03 AM] You: Thanks, can you share the details?",
  "[24/10/26, 9:05 AM] Recruiter: Send your bank details and add me on Telegram.",
].join("\n");

// C3: Android export style, dash after the timestamp, names as speakers
const C3 = [
  "24/10/26, 9:00 - Priya: Why do I need to pay? Is my data safe?",
  "24/10/26, 9:01 - HR Desk: Pay the registration fee within 2 hours.",
  "24/10/26, 9:02 - Priya: I want to speak to someone first.",
  "24/10/26, 9:03 - HR Desk: Send your documents on Telegram today.",
].join("\n");

test("detection is true for C1, C2 and C3 style conversations", () => {
  for (const text of [C1, C2, C3]) assert.equal(looksLikeConversation(text), true, text.slice(0, 40));
});

test("other speaker labels work: HR, Me, Candidate, a full name, and multi-line messages", () => {
  assert.equal(looksLikeConversation("HR: Welcome aboard.\nMe: Thank you.\nHR: Please send your ID.\nCandidate: Why?"), true);
  assert.equal(looksLikeConversation("John Smith: Hello there.\nMe: Hi.\nJohn Smith: Are you available?\nOn Monday please.\nMe: Yes."), true);
  assert.deepEqual(speakerLines(C2).map((l) => l.label), ["Recruiter", "You", "Recruiter"]);
});

const POSTING = [
  "Title: Data Entry Clerk",
  "Company profile: Acme Ltd builds software.",
  "Description: Type forms from home. Pay is weekly.",
  "Requirements: A computer.",
  "Benefits: Flexible hours.",
].join("\n");

test("ordinary postings are not conversations, including the app's own labelled format", () => {
  assert.equal(looksLikeConversation(POSTING), false);
  assert.equal(looksLikeConversation("Description: Earn from home. Send your bank details to start."), false);
  assert.equal(looksLikeConversation("We are hiring.\nRequirements: a phone and a bank account.\nApply today."), false);
  assert.equal(looksLikeConversation("Job title: Packer\nDescription: Pack boxes.\nRequirements: None.\nSalary: 20 an hour."), false);
});

test("bulleted lists and other 'Name: value' lines are not conversations", () => {
  assert.equal(looksLikeConversation("Benefits:\n- Salary: 20 an hour\n- Location: Leeds\n- Hours: 9 to 5\n- Holiday: 25 days"), false);
  assert.equal(looksLikeConversation("• Recruiter: ask for Sam\n• You: bring a pen\n• Recruiter: arrive early"), false);
  assert.equal(looksLikeConversation("Shift: days\nTeam: sales\nOffice: Leeds"), false); // three different labels, none repeated
  assert.equal(looksLikeConversation("Q: Is it remote?\nA: Yes.\nQ: Is it paid?\nA: Weekly."), false);
  assert.equal(looksLikeConversation("See https://example.org/jobs\nhttp://example.org/apply\nhttps://example.org"), false);
});

test("it needs at least 3 speaker lines and 2 different labels", () => {
  assert.equal(looksLikeConversation("Recruiter: Hi.\nYou: Hello."), false); // only 2 lines
  assert.equal(looksLikeConversation("Recruiter: Hi.\nRecruiter: Are you there?\nRecruiter: Please reply."), false); // one label
  assert.equal(looksLikeConversation(""), false);
  assert.equal(looksLikeConversation(undefined), false);
});

test("speaker labels: known words and capitalised names count, field labels never do", () => {
  for (const label of ["Recruiter", "HR", "You", "Me", "Candidate", "Priya", "John Smith", "Hiring Manager", "HR Desk"]) {
    assert.equal(isSpeakerLabel(label), true, label);
  }
  for (const label of ["Title", "Description", "Requirements", "Benefits", "Company profile", "Note", "NOTE", "FAQ", "HR IT", "Q", "https", "- Salary", "", "a very long label that is not a speaker name"]) {
    assert.equal(isSpeakerLabel(label), false, label);
  }
});

test("routing: a conversation goes to the chat analyzer and the override sends it to the job analyzer", () => {
  assert.equal(routePastedText(C1), "chat");
  assert.equal(routePastedText(C3), "chat");
  assert.equal(routePastedText(POSTING), "job");
  assert.equal(routePastedText(C1, { forceJob: true }), "job"); // "Check as job posting instead"
  assert.equal(routePastedText(C1, { forceJob: false }), "chat");
});

test("the visible wording", () => {
  assert.equal(ROUTED_NOTE, "This looks like a conversation, so it was checked as a chat.");
  assert.equal(OVERRIDE_LABEL, "Check as job posting instead");
});
