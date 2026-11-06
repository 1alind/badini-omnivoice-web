#!/usr/bin/env python3
"""Build a browser-ready, reference-conditioned Badini OmniVoice bundle."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
import torch, torchaudio, onnx
from huggingface_hub import hf_hub_download, model_info, snapshot_download
from transformers import HiggsAudioV2TokenizerModel
from export_fp32 import save_backbone, save_higgs

PROMPTS = [
 ("hayfa","natural","prompt-hayfa"),("hayfa","angry","prompt-angry"),
 ("hayfa","sad","prompt-sad"),("hayfa","happy","prompt-happy"),
 ("hayfa","fear","prompt-fear"),("hayfa","surprise","prompt-surprise"),
 ("abdullah","angry","prompt-abdullah-angry"),("abdullah","sad","prompt-abdullah-sad"),
 ("abdullah","happy","prompt-abdullah-happy"),("abdullah","fear","prompt-abdullah-fear"),
 ("abdullah","surprise","prompt-abdullah-surprise"),("bafreen","angry","prompt-bafreen-angry"),
 ("bafreen","sad","prompt-bafreen-sad"),("bafreen","happy","prompt-bafreen-happy"),
 ("bafreen","fear","prompt-bafreen-fear"),("bafreen","surprise","prompt-bafreen-surprise"),
]

def external_files(graph):
    model = onnx.load(str(graph), load_external_data=False)
    locations = set()
    for tensor in model.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            locations.update(item.value for item in tensor.external_data if item.key == "location")
    return [{"path": name, "localPath": name} for name in sorted(locations)]

def encode_prompts(codec, model_repo, model_revision, reference_repo, reference_revision, out):
    prompts = []
    sr_target = int(getattr(codec.config, "sampling_rate", 24000) or 24000)
    hop = int(codec.config.hop_length)
    for voice, emotion, stem in PROMPTS:
        try:
            prompt_repo = model_repo if (voice == "hayfa" and emotion == "natural") else reference_repo
            prompt_revision = model_revision if prompt_repo == model_repo else reference_revision
            wav = hf_hub_download(prompt_repo, stem + ".wav", revision=prompt_revision)
            txt = hf_hub_download(prompt_repo, stem + ".txt", revision=prompt_revision)
            audio, sr = torchaudio.load(wav)
            if audio.shape[0] > 1: audio = audio.mean(dim=0, keepdim=True)
            if sr != sr_target: audio = torchaudio.functional.resample(audio, sr, sr_target)
            rms = torch.sqrt(torch.mean(audio.square())).item()
            if 0 < rms < 0.1: audio = audio * (0.1 / rms)
            # Match OmniVoice prompt preprocessing: retain speech plus 100 ms lead / 200 ms tail.
            window = max(1, int(sr_target * 0.01))
            active = []
            threshold = 10 ** (-50 / 20)
            for start in range(0, audio.shape[-1], window):
                segment = audio[..., start:min(audio.shape[-1], start + window)]
                if torch.sqrt(torch.mean(segment.square())).item() > threshold:
                    active.append(start)
            if active:
                first = active[0]
                last = min(audio.shape[-1], active[-1] + window)
                left = max(0, first - int(sr_target * 0.1))
                right = min(audio.shape[-1], last + int(sr_target * 0.2))
                audio = audio[..., left:right]
            remainder = audio.shape[-1] % hop
            if remainder: audio = audio[..., :-remainder]
            if audio.numel() == 0: raise RuntimeError("empty after codec-hop alignment")
            with torch.inference_mode(): codes = codec.encode(audio.unsqueeze(0)).audio_codes.squeeze(0)
            arr = codes.detach().cpu().numpy().astype("<i4", copy=False)
            token_file = stem + ".i32"
            arr.tofile(out / token_file)
            text_file = stem + ".txt"
            shutil.copy2(txt, out / text_file)
            prompts.append({"voice":voice,"emotion":emotion,"audio_tokens":token_file,"frames":int(arr.shape[-1]),"codebooks":int(arr.shape[0]),"rms":float(rms),"text":text_file,"source_repo":prompt_repo})
            print(f"Encoded {voice}/{emotion}: {arr.shape[-1]} frames")
        except Exception as exc:
            print(f"WARNING {stem}: {exc}")
    if not prompts: raise RuntimeError("No voice prompt token files were created")
    return prompts, sr_target, hop

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-id",default="Dezheen89/omnivoice")
    p.add_argument("--higgs-repo",default="k2-fsa/OmniVoice")
    p.add_argument("--reference-repo",default="Dezheen89/cosyvoice")
    p.add_argument("--out",default="./badini-model-bundle")
    p.add_argument("--work",default="./badini-export-work")
    a=p.parse_args()
    out=Path(a.out).resolve(); work=Path(a.work).resolve()
    out.mkdir(parents=True,exist_ok=True); work.mkdir(parents=True,exist_ok=True)
    voice_revision = model_info(a.model_id).sha
    higgs_revision = model_info(a.higgs_repo).sha
    reference_revision = model_info(a.reference_repo).sha
    source=Path(snapshot_download(repo_id=a.model_id,revision=voice_revision,allow_patterns=["model.safetensors","config.json","tokenizer.json","tokenizer_config.json","chat_template.jinja","README.md"],local_dir=work/"fine-tuned-source"))
    higgs_root=Path(snapshot_download(repo_id=a.higgs_repo,revision=higgs_revision,allow_patterns=["README.md","LICENSE","audio_tokenizer/**"],local_dir=work/"higgs-source"))
    higgs=higgs_root/"audio_tokenizer"
    if (source/"README.md").is_file():
        shutil.copy2(source/"README.md",out/"BADINI_MODEL_CARD.md")
    if (higgs_root/"README.md").is_file():
        shutil.copy2(higgs_root/"README.md",out/"OMNIVOICE_MODEL_CARD.md")
    print("Exporting the exact fine-tuned Badini checkpoint…")
    save_backbone(source,work,out)
    print("Exporting Higgs decoder…")
    save_higgs(higgs,work,out)
    if (higgs / "LICENSE").is_file():
        shutil.copy2(higgs / "LICENSE", out / "HIGGS_AUDIO_LICENSE.txt")
    codec=HiggsAudioV2TokenizerModel.from_pretrained(str(higgs),dtype=torch.float32)
    codec.eval()
    prompts,sample_rate,hop=encode_prompts(codec,a.model_id,voice_revision,a.reference_repo,reference_revision,out)
    del codec
    for name in ("tokenizer.json","tokenizer_config.json"):
        if not (out/name).is_file(): raise RuntimeError(f"Missing {name}")
    sessions={
      "audioEmbeddings":{"model":"audio_embeddings_encoder.onnx","externalData":external_files(out/"audio_embeddings_encoder.onnx")},
      "llm":{"model":"llm_decoder.onnx","externalData":external_files(out/"llm_decoder.onnx")},
      "audioHeads":{"model":"audio_heads_decoder.onnx","externalData":external_files(out/"audio_heads_decoder.onnx")},
      "higgsDecoder":{"model":"higgs_decoder.onnx","externalData":external_files(out/"higgs_decoder.onnx"),"executionProvider":"wasm"},
    }
    manifest={
      "model_id":a.model_id,"language":"kmr","sample_rate":sample_rate,"audio_hop_length":hop,"source_revisions":{"voice":voice_revision,"higgs":higgs_revision,"references":reference_revision},
      "num_audio_codebooks":8,"precision":"fp32","reference_prompts":prompts,
      "runtime":{
        "sampleRate":sample_rate,"preferWebGpu":True,"hiddenSize":1024,"numKvHeads":8,"headDim":128,
        "llmAttention":{"mode":"omnivoice-noncausal","maskDtype":"bool","maskRank":4,"useCache":False},
        "decoderInputName":"codes","decoderOutputName":"waveform_24k",
        "generation":{"num_audio_codebook":8,"audio_mask_id":1024,"audio_vocab_size":1025,"numStep":32,"maxTargetTokens":320,"guidanceScale":2.0,"tShift":0.1,"layerPenalty":5.0,"positionTemperature":5.0,"classTemperature":0.0,"denoise":True},
        "sessions":sessions
      }
    }
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"BUNDLE_README.txt").write_text("Badini OmniVoice local browser bundle.\nSelect all files in this folder in Badini Voice.\nSpeech generation runs on this device; no inference server or API is used.\nOutput is mono 24 kHz.\nDo not redistribute unless every upstream model and codec license permits it. Review BADINI_MODEL_CARD.md, OMNIVOICE_MODEL_CARD.md and HIGGS_AUDIO_LICENSE.txt.\n",encoding="utf-8")
    print(f"Bundle ready: {out} · {len(prompts)} voice prompts. Run tools/validate_bundle.py next.")

if __name__=="__main__": main()
