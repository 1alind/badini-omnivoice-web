/*
 * Adapted from RabbitDaisuke/typed-voice, Apache License 2.0.
 * Source: https://github.com/RabbitDaisuke/typed-voice
 * See LICENSE-APACHE-2.0 and NOTICE.
 */
import { Tokenizer } from "@huggingface/tokenizers";
import * as ort from "onnxruntime-web/webgpu";
import { buildOmniVoiceAttentionMask, createPythonRandom, generateOmniVoiceCodes, prepareOmniVoiceInputs } from "./omnivoice-generation.js";

function halfToFloat(value) {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >>> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

function tensorToFloat32(tensor) {
  if (tensor.data instanceof Float32Array) return new Float32Array(tensor.data);
  if (tensor.type === "float16" && tensor.data instanceof Uint16Array) {
    return Float32Array.from(tensor.data, halfToFloat);
  }
  return Float32Array.from(tensor.data, Number);
}

function normalizePcm(input, refRms = null) {
  let pcm = new Float32Array(input);
  const sampleRate = 24000;
  const windowSize = 240;
  const threshold = Math.pow(10, -50 / 20);
  const makeActivity = samples => {
    const count = Math.ceil(samples.length / windowSize);
    const active = new Uint8Array(count);
    for (let w = 0; w < count; w += 1) {
      const start = w * windowSize;
      const end = Math.min(samples.length, start + windowSize);
      let energy = 0;
      for (let i = start; i < end; i += 1) energy += samples[i] * samples[i];
      active[w] = Math.sqrt(energy / Math.max(1, end - start)) > threshold ? 1 : 0;
    }
    return active;
  };
  let activity = makeActivity(pcm);
  let first = -1, last = -1;
  for (let i = 0; i < activity.length; i += 1) if (activity[i]) { if (first < 0) first = i; last = i; }
  if (first >= 0) {
    const lead = first * windowSize;
    const trail = pcm.length - (last + 1) * windowSize;
    const left = lead > 2400 ? lead - 2400 : 0;
    const right = trail > 2400 ? Math.min(pcm.length, (last + 1) * windowSize + 2400) : pcm.length;
    pcm = pcm.slice(left, right);
  }
  activity = makeActivity(pcm);
  const pieces = [];
  let cursor = 0;
  let w = 0;
  while (w < activity.length) {
    if (activity[w]) { w += 1; continue; }
    const silentStart = w;
    while (w < activity.length && !activity[w]) w += 1;
    const silentEnd = w;
    if (silentEnd - silentStart > 50) {
      const cutStart = Math.min(pcm.length, (silentStart + 25) * windowSize);
      const cutEnd = Math.min(pcm.length, Math.max(cutStart, (silentEnd - 25) * windowSize));
      if (cutStart > cursor) pieces.push(pcm.subarray(cursor, cutStart));
      cursor = cutEnd;
    }
  }
  if (cursor < pcm.length) pieces.push(pcm.subarray(cursor));
  if (pieces.length > 1) {
    const total = pieces.reduce((sum, piece) => sum + piece.length, 0);
    const joined = new Float32Array(total);
    let offset = 0;
    for (const piece of pieces) { joined.set(piece, offset); offset += piece.length; }
    pcm = joined;
  }
  if (refRms !== null && refRms < 0.1) {
    const scale = refRms / 0.1;
    for (let i = 0; i < pcm.length; i += 1) pcm[i] *= scale;
  } else if (refRms === null) {
    let peak = 0;
    for (const sample of pcm) peak = Math.max(peak, Math.abs(sample));
    if (peak > 1e-6) {
      const scale = 0.5 / peak;
      for (let i = 0; i < pcm.length; i += 1) pcm[i] *= scale;
    }
  }
  const fade = Math.min(Math.floor(sampleRate * 0.1), Math.floor(pcm.length / 2));
  for (let i = 0; i < fade; i += 1) {
    const gain = fade <= 1 ? 0 : i / (fade - 1);
    pcm[i] *= gain;
    pcm[pcm.length - 1 - i] *= 1 - gain;
  }
  const pad = Math.floor(sampleRate * 0.1);
  const output = new Float32Array(pcm.length + pad * 2);
  output.set(pcm, pad);
  return output;
}

function bigintRange(length, batch) {
  const result = new BigInt64Array(batch * length);
  for (let row = 0; row < batch; row += 1) {
    for (let index = 0; index < length; index += 1) result[row * length + index] = BigInt(index);
  }
  return result;
}

function tokenHash(tokens) {
  let hash = 0x811c9dc5;
  for (const token of tokens) {
    const value = Number(token);
    hash ^= value & 0xff;
    hash = Math.imul(hash, 0x01000193);
    hash ^= (value >>> 8) & 0xff;
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

export class OmniVoiceEngine {
  constructor() {
    this.manifest = null; this.tokenizer = null; this.sessions = null;
    this.backend = null; this.runtime = null; this.ort = ort; this.files = null; this.urls = new Map();
  }

  async load(files, onStatus = () => {}) {
    const manifestBlob = files.get("manifest.json");
    if (!manifestBlob) throw new Error("manifest.json is missing from the model bundle");
    const manifest = JSON.parse(await manifestBlob.text());
    if (!manifest.runtime?.sessions) throw new Error("Model manifest has no split OmniVoice runtime sessions");
    this.manifest = manifest; this.runtime = manifest.runtime; this.files = files;
    ort.env.wasm.wasmPaths = new URL("./vendor/ort/", document.baseURI).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1)) : 1;
    this.threadCount = ort.env.wasm.numThreads;
    onStatus(self.crossOriginIsolated
      ? `Cross-origin isolated · WASM threads: ${ort.env.wasm.numThreads}`
      : "Single-thread WASM. Reload after the app's first visit if you want to enable multi-threading.");
    this.tokenizer = new Tokenizer(JSON.parse(await this.#readText("tokenizer.json")), JSON.parse(await this.#readText("tokenizer_config.json")));
    const plans = buildSessionPlans(manifest, Boolean(navigator.gpu && this.runtime.preferWebGpu !== false));
    let lastError;
    for (const plan of plans) {
      try {
        onStatus(`Loading split OmniVoice models using ${plan.label.toUpperCase()}…`);
        this.sessions = await this.#createSessions(plan, onStatus);
        this.backend = plan.label;
        await this.#warmup();
        onStatus("Model loaded on this device.");
        return { backend: plan.label, sampleRate: this.runtime.sampleRate };
      } catch (error) {
        lastError = error;
        await this.#releaseSessions();
        if (plan.defaultBackend === "webgpu") onStatus("WebGPU path failed. Retrying with local WASM CPU…");
      }
    }
    throw new Error(`OmniVoice initialization failed: ${lastError?.message || "no usable backend"}`);
  }
  async #readText(name) {
    const blob = this.files.get(name);
    if (!blob) throw new Error(`Missing required model file: ${name}`);
    return blob.text();
  }
  #fileUrl(name) {
    const blob = this.files.get(name);
    if (!blob) throw new Error(`Missing required model file: ${name}`);
    if (!this.urls.has(name)) this.urls.set(name, URL.createObjectURL(blob));
    return this.urls.get(name);
  }
  async synthesize(text, options = {}) {
    if (!this.sessions || !this.tokenizer) throw new Error("Load the model files first.");
    const config = this.runtime.generation;
    const inputs = await prepareOmniVoiceInputs(text, this.tokenizer, config, options);
    const numStep = options.numStep ?? config.numStep ?? 32;
    const generated = await generateOmniVoiceCodes({
      inputs, config, runBackboneStep: step => this.#runBackboneStep(step), numStep,
      guidanceScale: options.guidanceScale ?? config.guidanceScale ?? 2,
      tShift: options.tShift ?? config.tShift ?? 0.1,
      layerPenalty: options.layerPenalty ?? config.layerPenalty ?? 5,
      positionTemperature: options.positionTemperature ?? config.positionTemperature ?? 5,
      classTemperature: options.classTemperature ?? config.classTemperature ?? 0,
      attentionMode: this.runtime.llmAttention?.mode ?? "omnivoice-noncausal",
      isCancelled: options.isCancelled, onStep: progress => options.onStep?.(progress),
      random: options.seed == null ? Math.random : createPythonRandom(options.seed),
    });
    const pcm = await this.#decode(generated.tokens, generated.codebooks, generated.targetLength);
    return { pcm: normalizePcm(pcm, options.refRms ?? null), sampleRate: this.runtime.sampleRate || 24000, backend: this.backend, targetLength: generated.targetLength };
  }
  async #createSessions(plan, onStatus) {
    const sessions = {};
    try {
      for (const [name, definition] of Object.entries(this.runtime.sessions)) {
        const provider = plan.backends[name] || plan.defaultBackend;
        const sessionProvider = definition.executionProvider || provider;
        const options = { executionProviders: [sessionProvider], graphOptimizationLevel: "all" };
        if (definition.externalData?.length) {
          options.externalData = [];
          for (const entry of definition.externalData) options.externalData.push({ path: entry.path, data: this.#fileUrl(entry.localPath) });
        }
        sessions[name] = await ort.InferenceSession.create(this.#fileUrl(definition.model), options);
        onStatus(`Loaded ${name} (${sessionProvider.toUpperCase()})`);
      }
      return sessions;
    } catch (error) {
      await Promise.allSettled(Object.values(sessions).map(session => session.release?.()));
      throw error;
    }
  }
  async #releaseSessions() {
    if (this.sessions) await Promise.allSettled(Object.values(this.sessions).map(session => session.release?.()));
    this.sessions = null;
  }
  async #warmup() {
    const codebooks = this.runtime.generation.num_audio_codebook;
    const sequenceLength = 4;
    const inputIds = new BigInt64Array(2 * codebooks * sequenceLength).fill(BigInt(this.runtime.generation.audio_mask_id));
    const audioMask = new Uint8Array(2 * sequenceLength).fill(1);
    const attention = buildOmniVoiceAttentionMask({
      sequenceLength,
      targetLength: sequenceLength,
      mode: this.runtime.llmAttention?.mode ?? "legacy-causal-2d",
    });
    const logits = await this.#runBackboneStep({
      inputIds,
      audioMask,
      attentionMask: attention.data,
      attentionMaskType: attention.type,
      attentionMaskShape: attention.shape,
      batch: 2,
      codebooks,
      sequenceLength,
    });
    if (logits.length !== 2 * codebooks * sequenceLength * this.runtime.generation.audio_vocab_size) {
      throw new Error(`Unexpected warmup logits length: ${logits.length}`);
    }
    const decoderInputName = this.runtime.decoderInputName || "codes";
    const decoderOutputName = this.runtime.decoderOutputName || "waveform_24k";
    const decoderInput = new this.ort.Tensor("int64", new BigInt64Array(codebooks * 8), [codebooks, 1, 8]);
    let decoderResult;
    try {
      decoderResult = await this.sessions.higgsDecoder.run({ [decoderInputName]: decoderInput });
      const decoded = decoderResult[decoderOutputName];
      if (!decoded || !decoded.data?.length) throw new Error("Higgs decoder warmup returned no waveform.");
      decoded.dispose?.();
    } finally {
      decoderInput.dispose?.();
    }
  }

  async #runBackboneStep({ inputIds, audioMask, attentionMask, attentionMaskType = "bool", attentionMaskShape, batch, codebooks, sequenceLength }) {
    const embIds = new ort.Tensor("int64", inputIds, [batch, codebooks, sequenceLength]);
    const embMask = new ort.Tensor("bool", audioMask, [batch, sequenceLength]);
    let embeddingsResult;
    let llmResult;
    let headsResult;
    let embedsTensor;
    let attentionTensor;
    let hiddenTensor;
    try {
      embeddingsResult = await this.sessions.audioEmbeddings.run({ input_ids: embIds, audio_mask: embMask });
      embedsTensor = new ort.Tensor("float32", tensorToFloat32(embeddingsResult.inputs_embeds), [batch, sequenceLength, this.runtime.hiddenSize]);
      attentionTensor = new ort.Tensor(attentionMaskType, attentionMask, attentionMaskShape ?? [batch, 1, sequenceLength, sequenceLength]);
      llmResult = await this.sessions.llm.run({ inputs_embeds: embedsTensor, attention_mask: attentionTensor });
      hiddenTensor = new ort.Tensor("float32", tensorToFloat32(llmResult.hidden_states), [batch, sequenceLength, this.runtime.hiddenSize]);
      headsResult = await this.sessions.audioHeads.run({ hidden_states: hiddenTensor });
      return tensorToFloat32(headsResult.logits);
    } finally {
      embIds.dispose?.(); embMask.dispose?.(); embedsTensor?.dispose?.(); attentionTensor?.dispose?.(); hiddenTensor?.dispose?.();
      embeddingsResult?.inputs_embeds?.dispose?.(); llmResult?.hidden_states?.dispose?.(); headsResult?.logits?.dispose?.();
    }
  }
  async #decode(tokens, codebooks, targetLength) {
    const decoderInputName = this.runtime.decoderInputName || "codes";
    const decoderOutputName = this.runtime.decoderOutputName || "waveform_24k";
    const input = new this.ort.Tensor("int64", tokens, [codebooks, 1, targetLength]);
    let result;
    let waveform;
    try {
      result = await this.sessions.higgsDecoder.run({ [decoderInputName]: input });
      waveform = result[decoderOutputName];
      if (!waveform) throw new Error(`Higgs decoder output not found: ${decoderOutputName}`);
      return tensorToFloat32(waveform);
    } finally {
      input.dispose?.();
      waveform?.dispose?.();
    }
  }

  async dispose() {
    await this.#releaseSessions();
    for (const url of this.urls.values()) URL.revokeObjectURL(url);
    this.urls.clear();
    this.backend = null; this.tokenizer = null; this.files = null;
  }
}

function buildSessionPlans(manifest, webGpuAvailable) {
  if (!webGpuAvailable) {
    return [{ label: "wasm", defaultBackend: "wasm", backends: {} }];
  }

  const profile = manifest.runtime?.qualityProfile
    || manifest.runtimeSource?.qualityProfile
    || manifest.conversion?.targetProfile;
  const isMobileInt8 = profile === "mobile-int8-weight-only" || profile === "mobile-int8-dynamic-wasm";
  const isMobileInt4 = profile === "mobile-int4-weight-only";
  if (isMobileInt8 || isMobileInt4) {
    return [
      {
        label: "webgpu+llm-wasm+higgs-wasm",
        defaultBackend: "webgpu",
        backends: { llm: "wasm", higgsDecoder: "wasm" },
      },
      { label: "wasm", defaultBackend: "wasm", backends: {} },
    ];
  }

  return [
    {
      label: "webgpu+higgs-wasm",
      defaultBackend: "webgpu",
      backends: { higgsDecoder: "wasm" },
    },
    { label: "wasm", defaultBackend: "wasm", backends: {} },
  ];
}

