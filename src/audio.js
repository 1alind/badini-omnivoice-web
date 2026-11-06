const SAMPLE_RATE = 24000;
export async function decodeAudioFile(file) {
  const decodeContext = new OfflineAudioContext(1, 1, SAMPLE_RATE);
  const decoded = await decodeContext.decodeAudioData(await file.arrayBuffer());
  const frames = Math.max(1, Math.round(decoded.length * SAMPLE_RATE / decoded.sampleRate));
  const context = new OfflineAudioContext(1, frames, SAMPLE_RATE);
  const source = context.createBufferSource();
  source.buffer = decoded;
  source.connect(context.destination);
  source.start(0);
  const result = await context.startRendering();
  return new Float32Array(result.getChannelData(0));
}
export function encodeWAV(samples, sampleRate = SAMPLE_RATE) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset, value) => { for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i)); };
  write(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true);
  write(8, "WAVE"); write(12, "fmt "); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  write(36, "data"); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, sample < 0 ? sample * 32768 : sample * 32767, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}
