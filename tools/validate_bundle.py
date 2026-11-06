#!/usr/bin/env python3
"""Validate the exported browser model bundle before moving it to the iPad."""
import json
import sys
from pathlib import Path

import onnx

EXPECTED = {
    "omnivoice_lm.onnx": ({"input_ids", "audio_mask", "attention_mask"}, {"logits"}),
    "audio_encoder.onnx": ({"waveform"}, {"audio_codes"}),
    "audio_decoder.onnx": ({"audio_codes"}, {"audio_values"}),
}

def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "./badini-model-bundle")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("ERROR: manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for filename, (expected_inputs, expected_outputs) in EXPECTED.items():
        path = root / filename
        if not path.is_file():
            raise SystemExit(f"ERROR: {filename} is missing")
        graph = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(graph)
        inputs = {item.name for item in graph.graph.input}
        outputs = {item.name for item in graph.graph.output}
        if not expected_inputs.issubset(inputs):
            raise SystemExit(f"ERROR: {filename} inputs are {sorted(inputs)}")
        if not expected_outputs.issubset(outputs):
            raise SystemExit(f"ERROR: {filename} outputs are {sorted(outputs)}")
        print(f"OK: {filename} · inputs={sorted(inputs)} · outputs={sorted(outputs)}")

    for graph_name, sidecars in manifest.get("external_data", {}).items():
        for sidecar in sidecars:
            if not (root / sidecar).is_file():
                raise SystemExit(f"ERROR: missing ONNX sidecar {sidecar}")

    for filename in ("tokenizer.json", "tokenizer_config.json"):
        path = root / filename
        if not path.is_file():
            raise SystemExit(f"ERROR: {filename} is missing")
        json.loads(path.read_text(encoding="utf-8"))

    prompts = manifest.get("reference_prompts", [])
    valid_prompts = 0
    for prompt in prompts:
        audio = root / prompt["audio"]
        text = root / prompt["text"]
        if audio.is_file() and text.is_file():
            valid_prompts += 1
    if not valid_prompts:
        print("WARNING: no complete voice prompt pairs found. A custom WAV and transcript will be needed.")
    else:
        print(f"OK: {valid_prompts} complete reference prompt pairs")

    total = sum(p.stat().st_size for p in root.iterdir() if p.is_file())
    print(f"\nBundle validation passed. Files: {sum(1 for p in root.iterdir() if p.is_file())}; size: {total / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
