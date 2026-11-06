# Badini Voice

An offline-first browser app for Badini Kurdish text-to-speech using the Badini fine-tuned checkpoint from Hugging Face.

## Model source

The browser app is intended to run the Badini fine-tuned weights, not the original unmodified `k2-fsa/OmniVoice` weights. The current exporter and Colab notebook target:

`Dezheen89/omnivoice`

The checkpoint repository is `Dezheen89/omnivoice`. The exporter reads its exact `model.safetensors`, config and tokenizer. The OmniVoice Python package supplies only the architecture implementation for conversion. The `k2-fsa/OmniVoice` repository is used only for the Higgs Audio V2 codec needed to decode audio tokens. Its original TTS backbone weights are never substituted for the Badini checkpoint.

## What runs in the browser

The app uses four split ONNX graphs: `audio_embeddings_encoder.onnx`, `llm_decoder.onnx`, `audio_heads_decoder.onnx`, and `higgs_decoder.onnx`.

The one-time bundle builder exports the fine-tuned backbone into four split ONNX graphs, pre-encodes the Hayfa, Abdullah, and Bafreen reference audio samples into local `.i32` prompt-token files, then dynamically quantizes the LLM's constant MatMul weights to INT8. The quantized graph uses standard ONNX `DynamicQuantizeLinear` and `MatMulInteger` operators. Audio embeddings, audio heads and the Higgs decoder remain FP32. Hayfa's natural sample comes from the Badini fine-tune; additional voice/emotion samples come from `Dezheen89/cosyvoice`.

The browser runtime uses ONNX Runtime Web. The mobile INT8 LLM and Higgs decoder are pinned to WASM; other graphs may use WebGPU where supported. iOS Safari currently uses the WASM path. Generation speed, peak memory use, and speech quality still need a real iPad test with the exported bundle.

## No inference server

- No Python backend, inference API, cloud GPU, analytics, telemetry, or inference-server connection.
- Vercel hosts the static website files only.
- The app and runtime assets are downloaded from Vercel and cached by the service worker.
- The model bundle is imported into browser storage. IndexedDB is used when available; otherwise files remain in memory for the current tab.
- After the app and model bundle are cached, speech generation is designed to work offline.

The Colab notebook is used only for one-time model conversion, not inference.

## Deploy on Vercel

Connect this GitHub repository to Vercel. Use these settings:

- Framework preset: **Vite**
- Build command: `npm run build`
- Output directory: `dist`
- Install command: `npm install`

No GitHub Pages setup or GitHub Actions deployment workflow is required. Vercel deploys when you push commits to the connected branch.

## Build the model bundle once

The model checkpoint is large. Use a Google Colab **High-RAM** runtime if available. The exporter uses CPU PyTorch and may take a while.

1. Open the [Colab export notebook](https://colab.research.google.com/github/1alind/badini-omnivoice-web/blob/main/notebooks/export_badini_omnivoice.ipynb).
2. Run the notebook cells. It downloads the Badini fine-tuned checkpoint, exports the four ONNX graphs, pre-encodes the reference prompts, converts the LLM to INT8, and validates the bundle against the FP32 reference.
3. Download `badini-omnivoice-model.zip` when the notebook finishes.
4. Extract the ZIP in the iPad Files app.
5. Open your Vercel deployment, select all extracted files, and import them.

The notebook prints the actual ZIP size before downloading. Do not commit model weights or generated prompt-token files to this GitHub repository.

## Run locally

Requirements: Node.js 22 or newer.

```bash
npm install
npm test
npm run build
npm run preview
```

The build copies ONNX Runtime Web WASM assets locally and generates a service worker precache list. `vercel.json` sets COOP/COEP/CORP headers from the first request so the browser can enable cross-origin isolation and WASM threading where supported. No runtime CDN is used.

## Browser and device notes

The target is Safari on the M2 iPad Pro. WebGPU support in iOS Safari is not assumed; the app has a WASM fallback. The full model bundle is large and peak memory use may be a limiting factor on iPad. A successful JavaScript build does **not** prove that the exported model runs or generates correct speech. End-to-end verification requires exporting the bundle and testing it on the iPad.

## Licensing

- Original project files: MIT, see `LICENSE`.
- Adapted OmniVoice browser runtime components: Apache-2.0, see `LICENSE-APACHE-2.0` and `NOTICE`.
- Adapted exporter and mobile INT8 quantizer: see the attribution and license files in the repository.
- Bundled ONNX Runtime Web and Hugging Face Tokenizers: see `THIRD_PARTY_NOTICES.md`.
- Model code, model weights and the Higgs Audio V2 tokenizer may have different license terms. Review the included upstream model cards and tokenizer license before redistributing a generated bundle.

## Current status

The static app, split-graph browser runtime, exporter, mobile INT8 quantizer, validator, speaker prompt encoder, chunking and local runtime assets are in the repository. The project is **not end-to-end verified yet**: the model bundle must still be generated and tested on the iPad.
