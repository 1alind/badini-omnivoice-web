const WEIGHT = { arabic: 1.45, latin: 1, digit: 3, punctuation: .5, space: .2, mark: 0, other: 1 };
function weightOf(char) {
  const cp = char.codePointAt(0);
  if (/\p{M}/u.test(char)) return WEIGHT.mark;
  if (/\p{P}|\p{S}/u.test(char)) return WEIGHT.punctuation;
  if (/\p{Z}/u.test(char) || char === " ") return WEIGHT.space;
  if (/\p{N}/u.test(char)) return WEIGHT.digit;
  if ((cp >= 0x0600 && cp <= 0x08FF) || (cp >= 0xFB50 && cp <= 0xFEFF)) return WEIGHT.arabic;
  if ((cp >= 0x0041 && cp <= 0x024F)) return WEIGHT.latin;
  return WEIGHT.other;
}
export function estimateDuration(text, refText, refFrames) {
  const reference = [...(refText || "")].reduce((s, c) => s + weightOf(c), 0);
  const target = [...(text || "")].reduce((s, c) => s + weightOf(c), 0);
  if (!reference || !refFrames) return Math.max(40, Math.ceil(target * 1.1));
  const estimate = target * refFrames / reference;
  return Math.max(24, Math.ceil(estimate < 50 ? 50 * Math.pow(estimate / 50, 1 / 3) : estimate));
}
