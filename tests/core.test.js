import test from "node:test";
import assert from "node:assert/strict";
import { getTimeSteps, computeSchedule, logSoftmax1D, topkFlatIndices } from "../src/math-utils.js";
import { estimateDuration } from "../src/duration.js";
import { splitTextIntoChunks, joinPcm } from "../src/text-chunking.js";

test("time schedule starts at zero and ends at one", () => {
  const steps = getTimeSteps(0, 1, 32, 0.1);
  assert.equal(steps.length, 33);
  assert.equal(steps[0], 0);
  assert.ok(Math.abs(steps.at(-1) - 1) < 1e-12);
});

test("mask schedule allocates every codebook position exactly once", () => {
  const times = getTimeSteps(0, 1, 32, 0.1);
  const schedule = computeSchedule(37, 8, 32, times);
  assert.equal(schedule.reduce((sum, n) => sum + n, 0), 37 * 8);
  assert.ok(schedule.every(n => n >= 0));
});

test("log-softmax is normalized", () => {
  const values = logSoftmax1D(new Float32Array([1, 2, 3]), 0, 3);
  const sum = values.reduce((acc, x) => acc + Math.exp(x), 0);
  assert.ok(Math.abs(sum - 1) < 1e-6);
});

test("top-k returns highest score indices", () => {
  assert.deepEqual(topkFlatIndices(new Float32Array([0.1, 2, -1, 1]), 2), [1, 3]);
});

test("duration estimator preserves reference length for identical text", () => {
  assert.equal(estimateDuration("دەقێ کوردی", "دەقێ کوردی", 42), 47);
  assert.equal(estimateDuration("Same text", "Same text", 50), 50);
});

test("long text is split without exceeding the per-chunk frame budget", () => {
  const text = "This is a long sentence that should be split into several small chunks, because we need keep every part within a modest token budget.";
  const chunks = splitTextIntoChunks(text, "hello there", 80, 120);
  assert.ok(chunks.length > 1);
  assert.equal(chunks.join(" "), text);
  assert.ok(chunks.every(chunk => estimateDuration(chunk, "hello there", 80) <= 120));
});

test("PCM chunks are joined with the requested silence gap", () => {
  const joined = joinPcm([new Float32Array([1, 2]), new Float32Array([3])], 10, 0.2);
  assert.deepEqual([...joined], [1, 2, 0, 0, 3]);
});
