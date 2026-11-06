# Badini Voice · Browser-local OmniVoice

A static, offline-first web app for generating Badini Kurdish speech using the fine-tuned OmniVoice model.

## What it does

- Imports ONNX model files from the user's device.
- Stores them in browser IndexedDB.
- Runs the OmniVoice inference loop in JavaScript with ONNX Runtime Web.
- Uses Hayfa, Abdullah, and Bafreen reference recordings (including available emotion samples), or a user-provided reference recording.
- Exports generated speech as 24 kHz mono WAV.
- Caches the static app/runtime locally so it can be reopened offline after the first visit.

**No inference server, inference API, analytics, or CDN is used by the app.** The static page must be downloaded once from its host. A cloud GPU is used only for the one-time model conversion in Colab; it is not involved in speech generation. After the app and model bundle have been imported, generation is local.

## 1. Export the model bundle

Open the [Colab export notebook](https://colab.research.google.com/github/1alind/badini-omnivoice-web/blob/main/notebooks/export_badini_omnivoice.ipynb) and run its cells.

Default checkpoint: `Dezheen89/omnivoice-badini` (Badini Kurdish, `kmr`). The notebook exports the fine-tuned checkpoint, not the base OmniVoice weights. It also downloads small reference WAV/TXT pairs from `Dezheen89/cosyvoice`. Those clips are used only as zero-shot voice prompts; no CosyVoice model weights are included. It creates a ZIP containing:
- `omnivoice_lm.onnx` and optional external weight data
- `audio_encoder.onnx` and `audio_decoder.onnx`
- tokenizer JSON files
- model manifest
- speaker/emotion reference WAVs and exact transcripts

The notebook runs `tools/validate_bundle.py` before zipping, checking ONNX graph names, tokenizer files, and prompt pairs.

Download the ZIP, extract it in the iPad Files app, and keep the extracted folder. The ZIP can be large. Do not upload model weights to this GitHub repository.

## 2. Build / host the static app

The GitHub Actions workflow builds the static site on each push. To enable hosting:
1. Open this repository's **Settings → Pages**.
2. Set the build and deployment source to **GitHub Actions**.
3. Open the Pages URL after the workflow succeeds.

Alternatively, run `npm install` then `npm run build` and host the `dist/` folder on any static host. The ONNX Runtime WASM files are copied into the build locally. No CDN runtime dependency is used.

## 3. First-time setup on iPad

1. Open the Pages URL in Safari.
2. Add it to the Home Screen if you want an app-like launch.
3. In Badini Voice, select all extracted model files together.
4. Tap **Load model on this iPad**.
5. Wait for the model to load, then enter Badini text and generate.

The app tries to store imported model files in IndexedDB. If Safari's storage quota rejects the large bundle, the files remain usable in the current tab only. Keep Safari open during loading and generation; after a tab-only import, you must select the files again after closing or reloading.

## Browser limitations

- ONNX Runtime Web's current compatibility table lists WASM CPU for Safari on iOS, but still marks its WebGPU execution provider unsupported there. Safari 26 itself has begun exposing the WebGPU API on iOS. The app probes for it and attempts WebGPU, then falls back to WASM if ONNX Runtime or a required operator is unsupported.
- This model is about 0.6B parameters. FP16 export is the default to reduce file size, but memory use and generation speed still need real-device validation on the target iPad.
- The current engine is an initial JavaScript port of OmniVoice's masked-token generation loop. It must be compared against the Python reference output before treating the generated voice as fully validated.
- The first export notebook uses a cloud notebook only for conversion. It is not used for speech generation.

## Project status

The UI, local file import, IndexedDB storage, static runtime bundling, ONNX export script, Colab notebook, and browser inference engine are now in the repository. Remaining validation:
1. Run the Colab export against the exact checkpoint.
2. Confirm the exported graph names and external data layout.
3. Build successfully in GitHub Actions.
4. Load the bundle in Safari on the M2 iPad.
5. Compare generated Badini WAV output with the Python model's output and fix any numerical or text-preprocessing differences.

## Licenses

The original project files are MIT licensed. The adapted OmniVoice browser inference components include Apache-2.0-licensed upstream code; see `NOTICE` and `LICENSE-APACHE-2.0`. The model weights and Higgs Audio V2 tokenizer are separate assets with separate license terms. Check those licenses before redistributing model bundles.
