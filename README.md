# Badini Voice · Browser-local OmniVoice

Static frontend scaffold for local Badini Kurdish text-to-speech using the fine-tuned `Dezheen89/omnivoice` checkpoint.

## Runtime rule
- No inference server, Python backend, cloud GPU, hosted inference API, analytics, telemetry, or CDN imports in the final offline build.
- Model and runtime assets must be local and cached in IndexedDB.
- A hosted static site still needs a network to open initially. For literal zero network requests during use, package the site, runtime, and model files locally.

## Current status
This is a frontend scaffold, not a completed TTS engine. It includes the responsive interface, voice/reference-style selectors, local model-file selection, and IndexedDB caching. Speech generation remains disabled until the exact fine-tuned checkpoint is exported to browser-compatible ONNX graphs, the Higgs Audio V2 decoder is exported, and the OmniVoice inference loop is ported and validated in JavaScript.

The fine-tuned checkpoint is `Dezheen89/omnivoice`. Do not substitute base OmniVoice ONNX weights, because that would discard the Badini fine-tuning. Review the model and tokenizer licenses before redistributing model assets.
