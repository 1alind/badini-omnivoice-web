# Badini Voice

An offline-first browser app for Badini Kurdish text-to-speech using the fine-tuned [Dezheen89/omnivoice-badini](https://huggingface.co/Dezheen89/omnivoice-badini) checkpoint.

## What runs in the browser

The app uses four split ONNX graphs:

- `audio_embeddings_encoder.onnx`
- `llm_decoder.onnx`
- `audio_heads_decoder.onnx`
- `higgs_decoder.onnx`

The one-time bundle builder exports the fine-tuned backbone in FP32, splits large external weight files into browser-importable parts, and pre-encodes the Hayfa, Abdullah, and Bafreen reference audio samples into local `.i32` prompt-token files. Safari does not need to upload or encode reference audio during generation.

The browser runtime uses ONNX Runtime Web. It attempts WebGPU for the backbone graphs when available, runs the Higgs decoder through WASM, and falls back to all-WASM if the mixed path cannot initialize. Long text is split into shorter generation chunks to keep the per-run token budget bounded on iPad.

## No inference server

- No Python backend, inference API, cloud GPU, analytics, telemetry, or CDN calls during speech generation.
- The static app and runtime assets are downloaded from the host once and precached by the service worker.
- The model bundle is imported into browser storage. IndexedDB is used when available; otherwise files remain in memory for the current tab.
- After the app and model bundle are cached, speech generation is designed to work offline.

A hosted static page still needs an initial connection to download the app. The Colab notebook below is used only for one-time model conversion, not for inference.

## Build the model bundle once

The model checkpoint is large. Use a Google Colab **High-RAM** runtime if available. The exporter uses CPU PyTorch and may take a while.

1. Open the [Colab export notebook](https://colab.research.google.com/github/1alind/badini-omnivoice-web/blob/main/notebooks/export_badini_omnivoice.ipynb).
2. Run the notebook cells. It downloads the exact fine-tuned checkpoint, exports and validates the four ONNX graphs, and pre-encodes the reference prompts.
3. Download `badini-omnivoice-model.zip` when the notebook finishes.
4. Extract the ZIP in the iPad Files app.
5. Open the [Badini Voice app](https://1alind.github.io/badini-omnivoice-web/) after GitHub Pages is enabled, select all extracted files, and import them.

The ZIP can be several gigabytes. Do not commit model weights or generated prompt-token files to this GitHub repository.

## Run locally

Requirements: Node.js 22 or newer.

```bash
npm install
npm test
npm run build
npm run preview
```

The build copies the pinned ONNX Runtime Web WASM assets locally and generates a service worker precache list. No runtime CDN is used.

## Enable GitHub Pages

In the repository, open **Settings → Pages**, choose **GitHub Actions** as the build and deployment source, and save. The workflow builds the static app and deploys it to:

`https://1alind.github.io/badini-omnivoice-web/`

## Browser and device notes

The target is Safari on the M2 iPad Pro. WebGPU support in iOS Safari and ONNX Runtime's operator coverage must be confirmed on the actual device. The app falls back to WASM CPU if WebGPU initialization fails. Generation speed and peak memory use still need a real iPad test with the exported bundle.

## Licensing

- Original project files: MIT, see `LICENSE`.
- Adapted legacy OmniVoice Web components: Apache-2.0, see `LICENSE-APACHE-2.0`.
- Adapted FP32 exporter: MIT, see `LICENSE-TSUKUYOMICHAN-MIT`.
- Bundled ONNX Runtime Web and Hugging Face Tokenizers: see `THIRD_PARTY_NOTICES.md`.
- Model weights and Higgs Audio V2 tokenizer assets have separate upstream license terms. Review them before redistributing a model bundle.

## Current status

The static app, split-graph browser runtime, bundle exporter, validator, chunking, and CI checks are in the repository. The project is **not end-to-end verified yet**: the model bundle still needs to be generated in Colab and tested on the iPad.
