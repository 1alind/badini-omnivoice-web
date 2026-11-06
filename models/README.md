# Model bundle

Do not commit model weights or generated prompt-token files to this repository.

The Colab build first exports the fine-tuned checkpoint `Dezheen89/omnivoice` (Badini Kurdish, `kmr`) into four FP32 ONNX graphs, then dynamically quantizes **only the LLM's constant MatMul weights** to INT8 using standard ONNX operators:

- `audio_embeddings_encoder.onnx` and external data: FP32
- `llm_decoder.onnx` and external data: INT8 weights with `DynamicQuantizeLinear` + `MatMulInteger`
- `audio_heads_decoder.onnx` and external data: FP32
- `higgs_decoder.onnx` and external data: FP32
- `tokenizer.json`
- `tokenizer_config.json`
- `manifest.json`
- Pre-encoded voice/emotion token files (`.i32`) and exact transcripts listed in `manifest.json`

The quantizer compares the INT8 LLM against the FP32 reference before replacing it. Graph inputs and outputs remain FP32, while activations are dynamically quantized inside standard ONNX operators. The manifest marks the `mobile-int8-dynamic-wasm` profile so the browser runs the LLM and Higgs decoder through WASM, while audio embeddings and audio heads may use WebGPU. The app falls back to all-WASM if the mixed path fails.

Hayfa's natural reference sample is taken from the fine-tuned model repository itself. Additional voice/emotion samples come from `Dezheen89/cosyvoice`. All samples are encoded with the Higgs Audio V2 codec from `k2-fsa/OmniVoice`; CosyVoice model weights are not included or used.

Generate the bundle with `notebooks/export_badini_omnivoice.ipynb`. The notebook runs FP32 export, prompt encoding, mobile INT8 conversion, and bundle validation before download. Select all extracted files together in the browser app.
