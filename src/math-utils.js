export function getTimeSteps(start, end, count, shift) {
  return Array.from({ length: count + 1 }, (_, i) => {
    const t = start + (end - start) * i / count;
    return shift * t / (1 + (shift - 1) * t);
  });
}
export function computeSchedule(length, codebooks, steps, times) {
  const total = length * codebooks;
  let remaining = total;
  return Array.from({ length: steps }, (_, i) => {
    const n = i === steps - 1 ? remaining : Math.min(Math.ceil(total * (times[i + 1] - times[i])), remaining);
    remaining -= n;
    return n;
  });
}
export function logSoftmax1D(data, offset, length) {
  let max = -Infinity;
  for (let i = 0; i < length; i++) max = Math.max(max, data[offset + i]);
  let sum = 0;
  for (let i = 0; i < length; i++) sum += Math.exp(data[offset + i] - max);
  const normalizer = max + Math.log(sum);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) out[i] = data[offset + i] - normalizer;
  return out;
}
export function topkFlatIndices(scores, k) {
  return Array.from(scores, (score, index) => ({ score, index }))
    .sort((a, b) => b.score - a.score).slice(0, k).map(x => x.index);
}
export function gumbelSample(logits, temperature) {
  const out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    const u = Math.random();
    out[i] = logits[i] / temperature - Math.log(-Math.log(u + 1e-10) + 1e-10);
  }
  return out;
}
export function argmax1D(data, offset, length) {
  let best = -Infinity, index = 0;
  for (let i = 0; i < length; i++) if (data[offset + i] > best) { best = data[offset + i]; index = i; }
  return index;
}
