import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { plainTag, plainTags } from "./textDisplay.js";

test("tags become plain phrases in sentence case", () => {
  assert.equal(plainTag("upfront_payment"), "Upfront payment");
  assert.equal(plainTag("REQUESTS_PERSONAL_INFO"), "Requests personal info");
  assert.equal(plainTag("move-to-messaging__app"), "Move to messaging app");
  assert.equal(plainTag("  no_interview  "), "No interview");
  assert.equal(plainTag(""), "");
  assert.equal(plainTag(undefined), "");
});

test("tag lists drop empties, non-strings and duplicates, so no underscore is ever shown", () => {
  assert.deepEqual(plainTags(["upfront_payment", "Upfront payment", "", null, 7, "no_interview"]), ["Upfront payment", "No interview"]);
  assert.deepEqual(plainTags(undefined), []);
  assert.deepEqual(plainTags("upfront_payment"), []);
  for (const phrase of plainTags(["a_b_c", "pressure_tactic", "gift-card_or_crypto"])) assert.equal(phrase.includes("_"), false);
});

test("the guidance cards use the plain phrases and hide the line when there are none", () => {
  const cards = readFileSync(new URL("../components/GuidanceCards.jsx", import.meta.url), "utf8");
  assert.ok(cards.includes('import { plainTags } from "../services/textDisplay.js";'));
  assert.ok(cards.includes("Related warning signs: {plainTags(e.warning_sign_tags).join(\", \")}"));
  assert.ok(cards.includes("{plainTags(e.warning_sign_tags).length > 0 && ("));
  assert.equal(cards.includes("e.warning_sign_tags.join"), false);
});
