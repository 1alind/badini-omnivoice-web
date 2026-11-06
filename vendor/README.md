# Vendored browser runtime

The build script copies the pinned ONNX Runtime Web WASM and JavaScript runtime assets from `node_modules/onnxruntime-web/dist` into `public/vendor/ort/`.

Run `npm install` and `npm run build` to generate these local assets in `dist/vendor/ort/`. The app points ONNX Runtime at this same-origin directory. Do not replace it with a CDN URL.

The generated runtime files are build artifacts and are intentionally ignored by Git. The dependency version is pinned in `package.json`.
