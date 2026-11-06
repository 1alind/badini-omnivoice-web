# Vendored browser runtime

To keep runtime network requests at **zero**, vendor a pinned, locally built copy of ONNX Runtime Web here and reference it with a relative path.

Do not import ONNX Runtime from a CDN in the final offline build. Confirm the package license and include its license notices.

The runtime is intentionally not bundled in this starter because its WebGPU/WebAssembly build and operator support must be validated against the exported OmniVoice graphs first.
