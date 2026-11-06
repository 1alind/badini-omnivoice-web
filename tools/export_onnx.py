#!/usr/bin/env python3
"""Export the Badini OmniVoice checkpoint for local browser inference.

Run this once in Google Colab or a Linux Python environment. This is a
build-time conversion step only; speech generation does not use this script.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


EXPORT_EXTERNAL_THRESHOLD = 1024

def export_module(module, args, path: Path, input_names, output_names, dynamic_axes):
    import onnx
    torch.onnx.export(
        module, args, str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    data_name = path.name + ".data"
    graph = onnx.load(str(path), load_external_data=True)
    onnx.save_model(
        graph, str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        convert_attribute=False,
        size_threshold=EXPORT_EXTERNAL_THRESHOLD,
    )
    print(f"Exported {path.name} (+ {data_name} when needed)")


class LMWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, audio_mask, attention_mask):
        return self.model(
            input_ids=input_ids.long(),
            audio_mask=audio_mask.bool(),
            attention_mask=attention_mask.bool(),
        ).logits


class AudioEncoderWrapper(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, waveform):
        return self.tokenizer.encode(waveform).audio_codes.int()


class AudioDecoderWrapper(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, audio_codes):
        return self.tokenizer.decode(audio_codes.long()).audio_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Dezheen89/omnivoice-badini")
    parser.add_argument("--reference_repo", default="Dezheen89/cosyvoice")
    parser.add_argument("--output_dir", default="./badini-model-bundle")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    args = parser.parse_args()
    global EXPORT_EXTERNAL_THRESHOLD
    EXPORT_EXTERNAL_THRESHOLD = (2**31 - 1) if args.dtype == "fp16" else 1024

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found. In Colab, choose Runtime → Change runtime type → T4 GPU.")
    device = "cuda:0" if args.device == "cuda" else "cpu"
    dtype = torch.float16 if args.dtype == "fp16" and device.startswith("cuda") else torch.float32

    from omnivoice.models.omnivoice import OmniVoice
    from huggingface_hub import hf_hub_download

    print(f"Loading fine-tuned checkpoint: {args.model_path} on {device} ({dtype})")
    model = OmniVoice.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="eager",
        device_map=device,
    )
    model.eval()

    with torch.inference_mode():
        batch, codebooks, seq = 2, model.config.num_audio_codebook, 64
        dummy_ids = torch.randint(0, 1024, (batch, codebooks, seq), dtype=torch.int32, device=device)
        dummy_audio = torch.zeros(batch, seq, dtype=torch.float32, device=device)
        dummy_attention = torch.ones(batch, 1, seq, seq, dtype=torch.float32, device=device)
        export_module(
            LMWrapper(model).eval().to(device),
            (dummy_ids, dummy_audio, dummy_attention),
            out / "omnivoice_lm.onnx",
            ["input_ids", "audio_mask", "attention_mask"],
            ["logits"],
            {
                "input_ids": {0: "batch", 2: "seq"},
                "audio_mask": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 2: "seq", 3: "seq"},
                "logits": {0: "batch", 2: "seq"},
            },
        )

        tokenizer = model.audio_tokenizer
        hop = getattr(tokenizer.config, "hop_length", 960)
        dummy_wave = torch.randn(1, 1, hop * 10, dtype=torch.float32, device=device)
        export_module(
            AudioEncoderWrapper(tokenizer).eval().to(device),
            (dummy_wave,),
            out / "audio_encoder.onnx",
            ["waveform"], ["audio_codes"],
            {"waveform": {0: "batch", 2: "samples"}, "audio_codes": {0: "batch", 2: "frames"}},
        )

        dummy_codes = torch.randint(0, 1024, (1, codebooks, 10), dtype=torch.int32, device=device)
        export_module(
            AudioDecoderWrapper(tokenizer).eval().to(device),
            (dummy_codes,),
            out / "audio_decoder.onnx",
            ["audio_codes"], ["audio_values"],
            {"audio_codes": {0: "batch", 2: "frames"}, "audio_values": {0: "batch", 2: "samples"}},
        )

    model.text_tokenizer.save_pretrained(out)
    prompt_specs = [
        ("hayfa", "natural", "prompt-hayfa"),
        ("hayfa", "angry", "prompt-angry"),
        ("hayfa", "sad", "prompt-sad"),
        ("hayfa", "happy", "prompt-happy"),
        ("hayfa", "fear", "prompt-fear"),
        ("hayfa", "surprise", "prompt-surprise"),
        ("abdullah", "angry", "prompt-abdullah-angry"),
        ("abdullah", "sad", "prompt-abdullah-sad"),
        ("abdullah", "happy", "prompt-abdullah-happy"),
        ("abdullah", "fear", "prompt-abdullah-fear"),
        ("abdullah", "surprise", "prompt-abdullah-surprise"),
        ("bafreen", "angry", "prompt-bafreen-angry"),
        ("bafreen", "sad", "prompt-bafreen-sad"),
        ("bafreen", "happy", "prompt-bafreen-happy"),
        ("bafreen", "fear", "prompt-bafreen-fear"),
        ("bafreen", "surprise", "prompt-bafreen-surprise"),
    ]
    reference_prompts = []
    for voice, emotion, stem in prompt_specs:
        try:
            wav_path = hf_hub_download(args.reference_repo, stem + ".wav")
            txt_path = hf_hub_download(args.reference_repo, stem + ".txt")
            shutil.copy2(wav_path, out / (stem + ".wav"))
            shutil.copy2(txt_path, out / (stem + ".txt"))
            reference_prompts.append({
                "voice": voice, "emotion": emotion,
                "audio": stem + ".wav", "text": stem + ".txt"
            })
        except Exception as exc:
            print(f"WARNING: prompt {stem} was not copied: {exc}")

    if not reference_prompts:
        print("WARNING: No reference prompts were copied. Import your own reference WAV and exact transcript.")

    external_data = {}
    for graph_name in ("omnivoice_lm.onnx", "audio_encoder.onnx", "audio_decoder.onnx"):
        sidecar = graph_name + ".data"
        external_data[graph_name] = [sidecar] if (out / sidecar).exists() else []
    manifest = {
        "model_id": args.model_path,
        "language": "kmr",
        "sample_rate": 24000,
        "num_audio_codebooks": codebooks,
        "external_data": external_data,
        "reference_prompts": reference_prompts,
        "precision": args.dtype if device.startswith("cuda") else "fp32",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "BUNDLE_README.txt").write_text(
        "Badini OmniVoice browser model bundle.\n"
        "Select all files in this folder in the Badini Voice app.\n"
        "This bundle contains model weights; do not commit it to GitHub.\n"
        "Output audio is 24 kHz mono.\n",
        encoding="utf-8",
    )
    print(f"\nDONE. Bundle files are in: {out.resolve()}")
    print("Zip this folder and move the ZIP to your iPad. Extract it in Files before importing all files.")


if __name__ == "__main__":
    main()
