# Model bundle

Do not commit model weights or prompt WAV files to this repository.

The browser app expects an ONNX export of the **fine-tuned** checkpoint `Dezheen89/omnivoice-badini` (Badini Kurdish, `kmr`). The exporter writes:

- `audio_embeddings_encoder.onnx` and its external data
- `llm_decoder.onnx` and its external data
- `audio_heads_decoder.onnx` and its external data
- `higgs_decoder.onnx` and its external data
- `tokenizer.json`
- `tokenizer_config.json`
- `manifest.json`
- Pre-encoded voice/emotion token files (`.i32`) and exact transcripts listed in `manifest.json`

The reference prompt files are downloaded from `Dezheen89/cosyvoice` and are used only as voice-cloning prompts. The CosyVoice model weights are not included or used.

Generate the bundle with `notebooks/export_badini_omnivoice.ipynb`. Before importing it on the iPad, run `tools/validate_bundle.py` (the notebook does this automatically). Select all extracted files together in the browser app.

The frontend runs the four-graph OmniVoice inference pipeline in ONNX Runtime Web. The prompt WAVs are encoded once during model export; browser inference consumes only the resulting audio-token IDs. Actual audio quality, memory use, and generation speed still require validation on the target iPad.
