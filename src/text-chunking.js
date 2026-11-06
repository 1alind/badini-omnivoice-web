import { estimateDuration } from "./duration.js";

function splitLongWord(word, referenceText, referenceFrames, maxFrames) {
  const parts = [];
  let current = "";
  for (const character of [...word]) {
    const candidate = current + character;
    if (current && estimateDuration(candidate, referenceText, referenceFrames) > maxFrames) {
      parts.push(current);
      current = character;
    } else current = candidate;
  }
  if (current) parts.push(current);
  return parts;
}

export function splitTextIntoChunks(text, referenceText, referenceFrames, maxFrames) {
  const sentences = text.match(/[^.!?؟۔…]+[.!?؟۔…]?/gu) || [text];
  const chunks = [];
  let current = "";
  const appendWords = sentence => {
    for (const word of sentence.trim().split(/\s+/u).filter(Boolean)) {
      const pieces = estimateDuration(word, referenceText, referenceFrames) > maxFrames
        ? splitLongWord(word, referenceText, referenceFrames, maxFrames)
        : [word];
      for (const piece of pieces) {
        const candidate = current ? current + " " + piece : piece;
        if (current && estimateDuration(candidate, referenceText, referenceFrames) > maxFrames) {
          chunks.push(current);
          current = piece;
        } else current = candidate;
      }
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

export function joinPcm(chunks, sampleRate, pauseSeconds = 0.18) {
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

