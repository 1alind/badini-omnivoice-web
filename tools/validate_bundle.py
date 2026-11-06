#!/usr/bin/env python3
"""Validate graph contracts, sidecar weights and pre-encoded Badini voice prompts."""
import json
import sys
from pathlib import Path
import onnx
from onnx import TensorProto

EXPECTED = {
    "audioEmbeddings": ("audio_embeddings_encoder.onnx", {"input_ids", "audio_mask"}, {"inputs_embeds"}),
    "llm": ("llm_decoder.onnx", {"inputs_embeds", "attention_mask"}, {"hidden_states"}),
    "audioHeads": ("audio_heads_decoder.onnx", {"hidden_states"}, {"logits"}),
    "higgsDecoder": ("higgs_decoder.onnx", {"codes"}, {"waveform_24k"}),
}

def tensor_type(value):
    return value.type.tensor_type

def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "./badini-model-bundle")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("ERROR: manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = manifest.get("runtime", {})
    sessions = runtime.get("sessions", {})
    if set(sessions) != set(EXPECTED):
        raise SystemExit("ERROR: manifest must describe audioEmbeddings, llm, audioHeads and higgsDecoder")

    for key, (filename, expected_inputs, expected_outputs) in EXPECTED.items():
        path = root / filename
        if not path.is_file():
            raise SystemExit(f"ERROR: {filename} is missing")
        model = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(model, full_check=False)
        inputs = {item.name: item for item in model.graph.input}
        outputs = {item.name for item in model.graph.output}
        if set(inputs) != expected_inputs:
            raise SystemExit(f"ERROR: {filename} input contract is {sorted(inputs)}")
        if outputs != expected_outputs:
            raise SystemExit(f"ERROR: {filename} output contract is {sorted(outputs)}")

        if key == "audioEmbeddings":
            if tensor_type(inputs["input_ids"]).elem_type != TensorProto.INT64:
                raise SystemExit("ERROR: audio embedding input_ids must be int64")
            if tensor_type(inputs["audio_mask"]).elem_type != TensorProto.BOOL:
                raise SystemExit("ERROR: audio embedding audio_mask must be bool")
        elif key == "llm":
            attention = tensor_type(inputs["attention_mask"])
            if attention.elem_type != TensorProto.BOOL or len(attention.shape.dim) != 4:
                raise SystemExit("ERROR: LLM must retain rank-4 bool non-causal attention")
            embeds = tensor_type(inputs["inputs_embeds"])
            if embeds.elem_type != TensorProto.FLOAT or len(embeds.shape.dim) != 3:
                raise SystemExit("ERROR: LLM inputs_embeds must be rank-3 float32")
        elif key == "audioHeads":
            if tensor_type(inputs["hidden_states"]).elem_type != TensorProto.FLOAT:
                raise SystemExit("ERROR: audio heads hidden_states must be float32")
        elif key == "higgsDecoder":
            if tensor_type(inputs["codes"]).elem_type != TensorProto.INT64:
                raise SystemExit("ERROR: Higgs decoder codes must be int64")

        definition = sessions[key]
        if definition.get("model") != filename:
            raise SystemExit(f"ERROR: manifest graph filename mismatch for {key}")
        for sidecar in definition.get("externalData", []):
            if not (root / sidecar["localPath"]).is_file():
                raise SystemExit(f"ERROR: missing ONNX external data {sidecar['localPath']}")
        print(f"OK: {filename} · inputs={sorted(inputs)} · outputs={sorted(outputs)}")

    for filename in ("tokenizer.json", "tokenizer_config.json"):
        path = root / filename
        if not path.is_file():
            raise SystemExit(f"ERROR: {filename} is missing")
        json.loads(path.read_text(encoding="utf-8"))

    for filename in ("BADINI_MODEL_CARD.md", "OMNIVOICE_MODEL_CARD.md", "HIGGS_AUDIO_LICENSE.txt"):
        if not (root / filename).is_file():
            raise SystemExit(f"ERROR: upstream license/provenance file {filename} is missing")

    prompts = manifest.get("reference_prompts", [])
    valid = 0
    for prompt in prompts:
        token_path = root / prompt["audio_tokens"]
        text_path = root / prompt["text"]
        frames = int(prompt["frames"])
        codebooks = int(prompt["codebooks"])
        if codebooks != 8 or frames <= 0:
            raise SystemExit(f"ERROR: invalid prompt dimensions for {prompt['voice']}/{prompt['emotion']}")
        if not token_path.is_file() or token_path.stat().st_size != codebooks * frames * 4:
            raise SystemExit(f"ERROR: invalid prompt token file {token_path.name}")
        if not text_path.is_file() or not text_path.read_text(encoding="utf-8").strip():
            raise SystemExit(f"ERROR: missing prompt transcript {text_path.name}")
        valid += 1
    if valid == 0:
        raise SystemExit("ERROR: no usable Badini speaker reference prompts")
    available_voices = {prompt["voice"] for prompt in prompts}
    if not {"hayfa", "abdullah", "bafreen"}.issubset(available_voices):
        raise SystemExit("ERROR: reference prompts must include Hayfa, Abdullah and Bafreen")
    if not any(prompt["voice"] == "hayfa" and prompt["emotion"] == "natural" for prompt in prompts):
        raise SystemExit("ERROR: the default Hayfa natural prompt is missing")
    if runtime.get("llmAttention", {}).get("mode") != "omnivoice-noncausal":
        raise SystemExit("ERROR: manifest does not declare OmniVoice non-causal attention")
    quantization = manifest.get("quantization")
    if quantization and quantization.get("bits") == 8:
        if runtime.get("qualityProfile") != "mobile-int8-dynamic-wasm":
            raise SystemExit("ERROR: manifest must declare the mobile-int8-dynamic-wasm runtime profile")
        llm = onnx.load(str(root / "llm_decoder.onnx"), load_external_data=False)
        matmul_integer = sum(
            1 for node in llm.graph.node
            if node.domain in ("", "ai.onnx") and node.op_type == "MatMulInteger"
        )
        dynamic_quantize = sum(
            1 for node in llm.graph.node
            if node.domain in ("", "ai.onnx") and node.op_type == "DynamicQuantizeLinear"
        )
        nbits = sum(
            1 for node in llm.graph.node
            if node.domain == "com.microsoft" and node.op_type == "MatMulNBits"
        )
        if nbits:
            raise SystemExit("ERROR: WASM-compatible bundle must not contain MatMulNBits")
        if matmul_integer == 0 or dynamic_quantize == 0:
            raise SystemExit(
                "ERROR: INT8 LLM must use standard ONNX DynamicQuantizeLinear + MatMulInteger"
            )
        if sessions["llm"].get("executionProvider") != "wasm":
            raise SystemExit("ERROR: mobile INT8 LLM must use the WASM execution provider")
        if sessions["higgsDecoder"].get("executionProvider") != "wasm":
            raise SystemExit("ERROR: Higgs decoder must use WASM for the validated audio path")
        if quantization.get("format") != "ONNX DynamicQuantizeLinear + MatMulInteger":
            raise SystemExit("ERROR: manifest does not declare the WASM-compatible INT8 format")
        print(
            "OK: mobile WASM INT8 LLM · "
            f"MatMulInteger nodes={matmul_integer}; DynamicQuantizeLinear nodes={dynamic_quantize}"
        )
    total = sum(path.stat().st_size for path in root.iterdir() if path.is_file())
    print(f"\nBundle validation passed. Voice prompts: {valid}; files: {sum(1 for p in root.iterdir() if p.is_file())}; size: {total / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
