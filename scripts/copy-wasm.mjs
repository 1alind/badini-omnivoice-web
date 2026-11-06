import { cp, mkdir, readdir } from "node:fs/promises";
import { resolve } from "node:path";

const source = resolve("node_modules/onnxruntime-web/dist");
const target = resolve("public/vendor/ort");
await mkdir(target, { recursive: true });
const files = await readdir(source);
const wasm = files.filter(name => name.endsWith(".wasm") || name.endsWith(".mjs"));
for (const name of wasm) await cp(resolve(source, name), resolve(target, name));
console.log(`Copied ${wasm.length} ONNX Runtime assets to public/vendor/ort`);
