# Model bundle

Do not commit model weights or generated prompt-token files to this repository.

The browser app expects an ONNX export of the **fine-tuned** checkpoint `Dezheen89/omnivoice-badini` (Badini Kurdish, `kmr`). The Colab build exports FP32 graphs and splits large external weight data into separate files:

- `audio_embeddings_encoder.onnx` and its external data
- `llm_decoder.onnx` and its external data
- `audio_heads_decoder.onnx` and its external data
- `higgs_decoder.onnx` and its external data
- `tokenizer.json`
- `tokenizer_config.json`
- `manifest.json`
- Pre-encoded voice/emotion token files (`.i32`) and exact transcripts listed in `manifest.json`

Hayfa's natural reference sample is taken from the fine-tuned model repository itself. The additional voice/emotion reference samples are downloaded from `Dezheen89/cosyvoice`. All samples are encoded with the Higgs Audio V2 codec from `k2-fsa/OmniVoice`; the CosyVoice model weights are not included or used.

Generate the bundle with `notebooks/export_badini_omnivoice.ipynb`. The notebook runs `tools/validate_bundle.py` before download. Select all extracted files together in the browser app.

The frontend runs the four-graph OmniVoice pipeline in ONNX Runtime Web. The app tries WebGPU for the backbone graphs when available and uses WASM for the Higgs decoder, with an all-WASM fallback. Reference prompts are pre-encoded during the one-time export, so Safari does not need to encode raw prompt audio.
