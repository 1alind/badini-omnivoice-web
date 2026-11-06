import { estimateDuration } from "./duration.js";

export function splitTextIntoChunks(text, referenceText, referenceFrames, maxFrames) {
  const sentences = text.match(/[^.!?؟۔…]+[.!?؟۔…]?/gu) || [text];
  const chunks = [];
  let current = "";
  const appendWords = sentence => {
    for (const word of sentence.trim().split(/\s+/u).filter(Boolean)) {
      if (estimateDuration(word, referenceText, referenceFrames) > maxFrames) {
        throw new Error("One unbroken word is too long for a single generation. Add spaces or shorten it.");
      }
      const candidate = current ? current + " " + word : word;
      if (current && estimateDuration(candidate, referenceText, referenceFrames) > maxFrames) {
        chunks.push(current);
        current = word;
      } else current = candidate;
    }
  };
  for (const sentence of sentences) {
    const value = sentence.trim();
    if (!value) continue;
    const candidate = current ? current + " " + value : value;
    if (estimateDuration(candidate, referenceText, referenceFrames) <= maxFrames) {
      current = candidate;
    } else {
      if (current) { chunks.push(current); current = ""; }
      appendWords(value);
    }
  }
  if (current) chunks.push(current);
  return chunks.length ? chunks : [text.trim()];
}

export function joinPcm(chunks, sampleRate, pauseSeconds = 0.1) {
  const pause = Math.round(sampleRate * pauseSeconds);
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0) + pause * Math.max(0, chunks.length - 1);
  const output = new Float32Array(total);
  let offset = 0;
  chunks.forEach((chunk, index) => {
    output.set(chunk, offset);
    offset += chunk.length;
    if (index < chunks.length - 1) offset += pause;
  });
  return output;
}

