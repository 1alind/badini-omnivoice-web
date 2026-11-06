import { Tokenizer } from "@huggingface/tokenizers";
import * as ort from "onnxruntime-web/webgpu";
import { buildOmniVoiceAttentionMask, createPythonRandom, generateOmniVoiceCodes, prepareOmniVoiceInputs } from "./omnivoice-generation-v2.js";

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
  let start = 0;
  let end = input.length;
  const threshold = 0.0005;
  while (start < end && Math.abs(input[start]) < threshold) start += 1;
  while (end > start && Math.abs(input[end - 1]) < threshold) end -= 1;
  const pcm = input.slice(start, end);
  if (refRms !== null && refRms > 0 && refRms < 0.1) {
    const scale = refRms / 0.1;
    for (let index = 0; index < pcm.length; index += 1) pcm[index] *= scale;
  }
  let peak = 0;
  for (const sample of pcm) peak = Math.max(peak, Math.abs(sample));
  if (peak > 0) {
    const scale = Math.min(0.95 / peak, 3);
    for (let index = 0; index < pcm.length; index += 1) pcm[index] *= scale;
  }
  const fade = Math.min(2400, Math.floor(pcm.length / 2));
  for (let index = 0; index < fade; index += 1) {
    const gain = fade <= 1 ? 0 : index / (fade - 1);
    pcm[index] *= gain; pcm[pcm.length - 1 - index] *= 1 - gain;
  }
  const padded = new Float32Array(pcm.length + 4800);
  padded.set(pcm, 2400);
  return padded;
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
    this.tokenizer = new Tokenizer(JSON.parse(await this.#readText("tokenizer.json")), JSON.parse(await this.#readText("tokenizer_config.json")));
    const providers = navigator.gpu && this.runtime.preferWebGpu !== false ? ["webgpu", "wasm"] : ["wasm"];
    let lastError;
    for (const provider of providers) {
      try {
        onStatus(`Loading split OmniVoice models using ${provider.toUpperCase()}…`);
        this.sessions = await this.#createSessions(provider, onStatus);
        this.backend = provider;
        await this.#warmup();
        onStatus("Model loaded on this device.");
        return { backend: provider, sampleRate: this.runtime.sampleRate };
      } catch (error) {
        lastError = error;
        await this.#releaseSessions();
        if (provider === "webgpu") onStatus("WebGPU failed. Retrying with local WASM CPU…");
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
  async #createSessions(provider, onStatus) {
    const sessions = {};
    try {
      for (const [name, definition] of Object.entries(this.runtime.sessions)) {
        const options = { executionProviders: [provider], graphOptimizationLevel: "all" };
        if (definition.externalData?.length) {
          options.externalData = [];
          for (const entry of definition.externalData) options.externalData.push({ path: entry.path, data: this.#fileUrl(entry.localPath) });
        }
        sessions[name] = await ort.InferenceSession.create(this.#fileUrl(definition.model), options);
        onStatus(`Loaded ${name} (${provider.toUpperCase()})`);
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
    const result = await this.sessions.higgsDecoder.run({ [decoderInputName]: input });
    input.dispose?.();
    const waveform = result[decoderOutputName];
    if (!waveform) throw new Error(`Higgs decoder output not found: ${decoderOutputName}`);
    const pcm = tensorToFloat32(waveform);
    waveform.dispose?.();
    return pcm;
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

  const isMobileInt8 = manifest.runtimeSource?.qualityProfile === "mobile-int8-weight-only"
    || manifest.conversion?.targetProfile === "mobile-int8-weight-only";
  const isMobileInt4 = manifest.runtimeSource?.qualityProfile === "mobile-int4-weight-only"
    || manifest.conversion?.targetProfile === "mobile-int4-weight-only";
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

async function loadTokenizer(assetRoot, tokenizerDirectory) {
  const base = new URL(`${tokenizerDirectory.replace(/\/$/, "")}/`, assetRoot);
  const [tokenizerResponse, configResponse] = await Promise.all([
    fetch(new URL("tokenizer.json", base)),
    fetch(new URL("tokenizer_config.json", base)),
  ]);
  if (!tokenizerResponse.ok) {
    throw new Error(`Tokenizer model is unavailable offline: ${tokenizerResponse.status}`);
  }
  if (!configResponse.ok) {
    throw new Error(`Tokenizer config is unavailable offline: ${configResponse.status}`);
  }
  const [tokenizerJson, tokenizerConfig] = await Promise.all([
    tokenizerResponse.json(),
    configResponse.json(),
  ]);
  return new Tokenizer(tokenizerJson, tokenizerConfig);
}
