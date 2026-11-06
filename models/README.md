# Model bundle

Do not commit model weights or prompt WAV files to this repository.

The browser app expects an ONNX export of the **fine-tuned** checkpoint `Dezheen89/omnivoice-badini` (Badini Kurdish, `kmr`). The exporter writes:

- `omnivoice_lm.onnx` and optional `omnivoice_lm.onnx.data`
- `audio_encoder.onnx` and optional external data
- `audio_decoder.onnx` and optional external data
- `tokenizer.json`
- `tokenizer_config.json`
- `manifest.json`
- The voice/emotion reference WAV/TXT pairs listed in `manifest.json`

The reference prompt files are downloaded from `Dezheen89/cosyvoice` and are used only as voice-cloning prompts. The CosyVoice model weights are not included or used.

Generate the bundle with `notebooks/export_badini_omnivoice.ipynb`. Before importing it on the iPad, run `tools/validate_bundle.py` (the notebook does this automatically). Select all extracted files together in the browser app.

The frontend includes an initial JavaScript inference port and ONNX Runtime Web integration. Actual audio quality, memory use, and generation speed still require validation on the target iPad.
