import test from "node:test";
import assert from "node:assert/strict";
import { splitTextIntoChunks, joinPcm } from "../src/text-chunking.js";
import { estimateDuration } from "../src/duration.js";

test("long Badini text is split into chunks under the target-frame limit", () => {
  const text = Array.from({ length: 80 }, (_, i) => i % 2 ? "سڵاو" : "چۆنی").join(" ") + "؟";
  const chunks = splitTextIntoChunks(text, "سڵاو", 48, 120);
  assert.ok(chunks.length > 1);
  assert.ok(chunks.every(chunk => estimateDuration(chunk, "سڵاو", 48) <= 120));
  assert.equal(chunks.join(" ").replace(/\s+/g, " ").trim().replace(/\s+([؟.!?])/g, "$1"), text);
});

test("a single oversized token is split safely instead of being truncated", () => {
  const word = "کوردی".repeat(100);
  const chunks = splitTextIntoChunks(word, "سڵاو", 48, 120);
  assert.ok(chunks.length > 1);
  assert.ok(chunks.every(chunk => estimateDuration(chunk, "سڵاو", 48) <= 120));
  assert.equal(chunks.join(""), word);
});

test("PCM chunks are joined with the requested silence gap", () => {
  const joined = joinPcm([new Float32Array([1, 2]), new Float32Array([3])], 10, 0.2);
  assert.deepEqual(Array.from(joined), [1, 2, 0, 0, 3]);
});
