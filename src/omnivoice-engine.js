import * as ort from "onnxruntime-web/webgpu";
import { Tokenizer } from "@huggingface/tokenizers";
import { estimateDuration } from "./duration.js";
import { getTimeSteps, computeSchedule, logSoftmax1D, topkFlatIndices, gumbelSample, argmax1D } from "./math-utils.js";

const CODEBOOKS = 8;
const AUDIO_VOCAB = 1025;
const MASK_ID = 1024;
const SAMPLE_RATE = 24000;

export class OmniVoiceEngine {
  constructor() {
    this.lm = null;
    this.encoder = null;
    this.decoder = null;
    this.tokenizer = null;
    this.backend = "wasm";
    this.runtime = ort;
  }

  async load(files, onProgress = () => {}) {
    const get = async name => {
      const blob = files.get(name);
      if (!blob) throw new Error(`Missing required model file: ${name}`);
      return new Uint8Array(await blob.arrayBuffer());
    };

    // Keep every runtime asset on this same static origin. No CDN is used.
    ort.env.wasm.wasmPaths = new URL("./vendor/ort/", document.baseURI).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1)) : 1;

    this.backend = navigator.gpu ? "webgpu" : "wasm";
    const options = { executionProviders: [this.backend], graphOptimizationLevel: "all" };
    const manifest = files.has("manifest.json") ? JSON.parse(await files.get("manifest.json").text()) : {};
    const sidecarsFor = async graph => {
      const names = manifest.external_data?.[graph] || [];
      const assets = [];
      for (const name of names) {
        onProgress(`Reading weights: ${name}`);
        assets.push({ path: name, data: await get(name) });
      }
      return assets;
    };
    const loadAll = async provider => {
      const common = { executionProviders: [provider], graphOptimizationLevel: "all" };
      this.lm = await ort.InferenceSession.create(await get("omnivoice_lm.onnx"), {
        ...common, externalData: await sidecarsFor("omnivoice_lm.onnx")
      });
      onProgress("Loading Higgs audio encoder…");
      this.encoder = await ort.InferenceSession.create(await get("audio_encoder.onnx"), {
        ...common, externalData: await sidecarsFor("audio_encoder.onnx")
      });
      onProgress("Loading Higgs audio decoder…");
      this.decoder = await ort.InferenceSession.create(await get("audio_decoder.onnx"), {
        ...common, externalData: await sidecarsFor("audio_decoder.onnx")
      });
    };
    try {
      onProgress(`Loading OmniVoice models (${this.backend.toUpperCase()})…`);
      await loadAll(this.backend);
    } catch (error) {
      this.lm?.release(); this.encoder?.release(); this.decoder?.release();
      this.lm = this.encoder = this.decoder = null;
      if (this.backend !== "webgpu") throw error;
      onProgress("WebGPU failed. Retrying all models with local WASM CPU runtime…");
      this.backend = "wasm";
      await loadAll("wasm");
    }
    onProgress("Loading local tokenizer…");
    const tokenizerJson = JSON.parse(await files.get("tokenizer.json").text());
    const tokenizerConfig = files.has("tokenizer_config.json") ? JSON.parse(await files.get("tokenizer_config.json").text()) : {};
    this.tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
    onProgress("Model loaded on this device.");
    return { backend: this.backend, sampleRate: SAMPLE_RATE };
  }

  _tokens(text) {
    return this.tokenizer.encode(text, null, false).ids;
  }

  async encodeReference(waveform) {
    const input = new ort.Tensor("float32", waveform, [1, 1, waveform.length]);
    const result = await this.encoder.run({ waveform: input });
    input.dispose();
    const codes = result.audio_codes;
    const output = { data: new Int32Array(codes.data), dims: Array.from(codes.dims) };
    codes.dispose?.();
    return output;
  }

  async decode(codes, dims) {
    const input = new ort.Tensor("int32", codes, dims);
    const result = await this.decoder.run({ audio_codes: input });
    input.dispose();
    const values = result.audio_values;
    const output = new Float32Array(values.data);
    values.dispose?.();
    return output;
  }

  _buildInput(text, targetFrames, refText, refCodes, language = "kmr") {
    const denoise = refCodes ? "<|denoise|>" : "";
    const style = this._tokens(`${denoise}<|lang_start|>${language}<|lang_end|><|instruct_start|>None<|instruct_end|>`);
    let fullText = refText ? `${refText.trim()} ${text.trim()}` : text.trim();
    fullText = fullText.replace(/[ \t]*\r?\n[\s]*/g, ".");
    const textTokens = this._tokens(`<|text_start|>${fullText}<|text_end|>`);
    const refFrames = refCodes ? refCodes.dims[2] : 0;
    const seq = style.length + textTokens.length + refFrames + targetFrames;
    const ids = new Int32Array(CODEBOOKS * seq).fill(MASK_ID);
    const audioMask = new Float32Array(seq);
    for (let c = 0; c < CODEBOOKS; c++) {
      style.forEach((id, i) => { ids[c * seq + i] = id; });
      textTokens.forEach((id, i) => { ids[c * seq + style.length + i] = id; });
      if (refCodes) for (let t = 0; t < refFrames; t++) ids[c * seq + style.length + textTokens.length + t] = refCodes.data[c * refFrames + t];
    }
    const audioStart = style.length + textTokens.length;
    audioMask.fill(1, audioStart);
    return { ids, audioMask, seq, targetStart: audioStart + refFrames };
  }

  async _forward(ids, audioMask, attentionMask, seq) {
    const idTensor = new ort.Tensor("int32", ids, [2, CODEBOOKS, seq]);
    const audioTensor = new ort.Tensor("float32", audioMask, [2, seq]);
    const attentionTensor = new ort.Tensor("float32", attentionMask, [2, 1, seq, seq]);
    const result = await this.lm.run({ input_ids: idTensor, audio_mask: audioTensor, attention_mask: attentionTensor });
    idTensor.dispose(); audioTensor.dispose(); attentionTensor.dispose();
    const logits = new Float32Array(result.logits.data);
    result.logits.dispose?.();
    return logits;
  }

  async generate(text, { refAudio, refText, language = "kmr", steps = 32, guidance = 2, positionTemperature = 5, onStep = () => {} } = {}) {
    if (!this.lm || !this.tokenizer) throw new Error("Load the model files first.");
    if (!text.trim()) throw new Error("Enter some Badini text.");
    const refCodes = refAudio ? await this.encodeReference(refAudio) : null;
    const refFrames = refCodes?.dims?.[2] || 25;
    const targetFrames = Math.max(1, estimateDuration(text, refText || "سلاڤ", refFrames));
    const { ids, audioMask, seq, targetStart } = this._buildInput(text, targetFrames, refText, refCodes, language);

    const batchIds = new Int32Array(2 * CODEBOOKS * seq).fill(MASK_ID);
    const batchMask = new Float32Array(2 * seq);
    const attention = new Float32Array(2 * seq * seq);
    batchIds.set(ids, 0);
    batchMask.set(audioMask, 0);
    for (let c = 0; c < CODEBOOKS; c++) {
      for (let t = 0; t < targetFrames; t++) batchIds[CODEBOOKS * seq + c * seq + t] = ids[c * seq + targetStart + t];
    }
    batchMask.fill(1, seq, seq + targetFrames);
    for (let q = 0; q < seq; q++) for (let k = 0; k < seq; k++) attention[q * seq + k] = 1;
    for (let q = 0; q < targetFrames; q++) for (let k = 0; k < targetFrames; k++) attention[seq * seq + q * seq + k] = 1;
    for (let i = targetFrames; i < seq; i++) attention[seq * seq + i * seq + i] = 1;

    const tokens = new Int32Array(CODEBOOKS * targetFrames).fill(MASK_ID);
    const schedule = computeSchedule(targetFrames, CODEBOOKS, steps, getTimeSteps(0, 1, steps, 0.1));
    const conditionalStride = CODEBOOKS * seq * AUDIO_VOCAB;

    for (let step = 0; step < steps; step++) {
      const logits = await this._forward(batchIds, batchMask, attention, seq);
      const predicted = new Int32Array(CODEBOOKS * targetFrames);
      const scores = new Float32Array(CODEBOOKS * targetFrames).fill(-Infinity);
      for (let c = 0; c < CODEBOOKS; c++) {
        for (let t = 0; t < targetFrames; t++) {
          const condOffset = c * seq * AUDIO_VOCAB + (targetStart + t) * AUDIO_VOCAB;
          const uncondOffset = conditionalStride + c * seq * AUDIO_VOCAB + t * AUDIO_VOCAB;
          const cond = logSoftmax1D(logits, condOffset, AUDIO_VOCAB);
          const uncond = logSoftmax1D(logits, uncondOffset, AUDIO_VOCAB);
          const guided = new Float32Array(AUDIO_VOCAB);
          for (let v = 0; v < AUDIO_VOCAB; v++) guided[v] = cond[v] + guidance * (cond[v] - uncond[v]);
          const finalLogits = logSoftmax1D(guided, 0, AUDIO_VOCAB);
          finalLogits[MASK_ID] = -Infinity;
          const token = argmax1D(finalLogits, 0, AUDIO_VOCAB);
          const index = c * targetFrames + t;
          predicted[index] = token;
          scores[index] = finalLogits[token] - c * 5;
        }
      }
      if (positionTemperature > 0) {
        const randomized = gumbelSample(scores, positionTemperature);
        scores.set(randomized);
      }
      for (let i = 0; i < scores.length; i++) if (tokens[i] !== MASK_ID) scores[i] = -Infinity;
      const chosen = topkFlatIndices(scores, schedule[step]);
      for (const index of chosen) {
        const c = Math.floor(index / targetFrames), t = index % targetFrames;
        const token = predicted[index];
        tokens[index] = token;
        batchIds[c * seq + targetStart + t] = token;
        batchIds[CODEBOOKS * seq + c * seq + t] = token;
      }
      onStep(step + 1, steps);
    }
    return this.decode(tokens, [1, CODEBOOKS, targetFrames]);
  }

  dispose() {
    this.lm?.release(); this.encoder?.release(); this.decoder?.release();
    this.lm = this.encoder = this.decoder = this.tokenizer = null;
  }
}
