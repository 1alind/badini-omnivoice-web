# Model files

Do not commit model weights to this repository.

The browser build will need a compatible export of the **fine-tuned** `Dezheen89/omnivoice` checkpoint, not the unrelated base checkpoint. Expected backbone files:

- `audio_embeddings_encoder.onnx`
- `llm_decoder.onnx`
- `audio_heads_decoder.onnx`

It also needs the matching tokenizer assets and the Higgs Audio V2 decoder assets. Exact filenames depend on the exporter and must be recorded in a manifest.

The current frontend can select and cache files locally, but does not yet execute the model. Do not treat file selection as a working TTS engine.
