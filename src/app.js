import { OmniVoiceEngine } from "./omnivoice-engine.js";
import { decodeAudioFile, encodeWAV } from "./audio.js";

const $ = id => document.getElementById(id);
const DB_NAME = "badini-voice-local";
const STORE = "model-files";
const REQUIRED = ["omnivoice_lm.onnx", "audio_encoder.onnx", "audio_decoder.onnx", "tokenizer.json", "tokenizer_config.json"];

let engine = null;
let outputUrl = null;
let loadedManifest = null;
let loadedFiles = new Map();
const volatileFiles = new Map();

function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE, { keyPath: "name" });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
async function putFiles(files) {
  const db = await openDB();
  await new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    for (const file of files) tx.objectStore(STORE).put({ name: file.name, blob: file, size: file.size, type: file.type });
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
  db.close();
}
async function readFiles() {
  let rows = [];
  try {
    const db = await openDB();
    rows = await new Promise((resolve, reject) => {
      const request = db.transaction(STORE).objectStore(STORE).getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    db.close();
  } catch {}
  const merged = new Map(rows.map(row => [row.name, row]));
  for (const [name, blob] of volatileFiles) {
    merged.set(name, { name, blob, size: blob.size, type: blob.type });
  }
  return [...merged.values()];
}
async function removeFiles() {
  const db = await openDB();
  await new Promise((resolve, reject) => {
    const request = db.transaction(STORE, "readwrite").objectStore(STORE).clear();
    request.onsuccess = resolve;
    request.onerror = () => reject(request.error);
  });
  db.close();
}
function sizeLabel(bytes) {
  return bytes >= 1e9 ? (bytes / 1e9).toFixed(2) + " GB" : (bytes / 1e6).toFixed(1) + " MB";
}
function renderFiles(rows) {
  const list = $("fileList");
  list.replaceChildren();
  const names = new Set(rows.map(row => row.name));
  const missing = REQUIRED.filter(name => !names.has(name));
  if (!rows.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No model files imported.";
    list.append(li);
  } else {
    for (const row of rows) {
      const li = document.createElement("li");
      li.textContent = row.name + " · " + sizeLabel(row.size);
      list.append(li);
    }
  }
  const total = rows.reduce((sum, row) => sum + row.size, 0);
  $("modelStatus").textContent = rows.length ? (missing.length ? "INCOMPLETE" : "READY TO LOAD") : "NOT LOADED";
  const sessionOnly = rows.filter(row => volatileFiles.has(row.name)).length;
  $("modelStatusText").textContent = rows.length
    ? `${rows.length} files · ${sizeLabel(total)}. ${sessionOnly ? sessionOnly + " file(s) are session-only; keep this tab open." : "Stored in browser IndexedDB."} ${missing.length ? "Missing: " + missing.join(", ") : "Required model files found."}`
    : "Model files stay on this device in browser storage. They are never uploaded.";
  $("loadBtn").disabled = missing.length > 0;
}

function selectedPrompt() {
  return (loadedManifest?.reference_prompts || []).find(prompt =>
    prompt.voice === $("voiceSelect").value && prompt.emotion === $("emotionSelect").value
  );
}
function refreshEmotionOptions() {
  const prompts = loadedManifest?.reference_prompts || [];
  const available = new Set(prompts.filter(p => p.voice === $("voiceSelect").value).map(p => p.emotion));
  const select = $("emotionSelect");
  for (const option of select.options) option.disabled = !available.has(option.value);
  const active = [...select.options].find(option => !option.disabled);
  if (select.selectedOptions[0]?.disabled && active) select.value = active.value;
}
async function applySelectedPrompt() {
  const prompt = selectedPrompt();
  $("refAudio").value = "";
  if (!prompt) {
    $("promptStatus").textContent = "CUSTOM PROMPT";
    $("refText").value = "";
    return;
  }
  $("promptStatus").textContent = `${prompt.voice.toUpperCase()} · ${prompt.emotion.toUpperCase()}`;
  const transcript = loadedFiles.get(prompt.text);
  $("refText").value = transcript ? (await transcript.text()).trim() : "";
}


$("textInput").addEventListener("input", () => {
  $("charCount").textContent = `${$("textInput").value.length} / 1200`;
  $("generateBtn").disabled = !engine || !$("textInput").value.trim();
});

$("modelFiles").addEventListener("change", async event => {
  const selected = [...event.target.files];
  if (!selected.length) return;
  engine?.dispose();
  engine = null;
  loadedManifest = null;
  loadedFiles = new Map();
  $("engineStatus").textContent = "ENGINE OFF";
  $("generateBtn").disabled = true;
  for (const file of selected) volatileFiles.set(file.name, file);
  $("modelStatusText").textContent = "Importing model files…";
  try {
    await putFiles(selected);
    for (const file of selected) volatileFiles.delete(file.name);
    try { await navigator.storage?.persist?.(); } catch {}
    $("generationStatus").textContent = "Files imported and saved locally. Load the model to begin.";
  } catch (error) {
    $("generationStatus").textContent = "Browser storage could not keep the large files. They remain available for this tab only. Keep this page open.";
  }
  renderFiles(await readFiles());
  event.target.value = "";
});

$("loadBtn").addEventListener("click", async () => {
  $("loadBtn").disabled = true;
  $("modelStatus").textContent = "LOADING";
  $("engineStatus").textContent = "LOADING";
  $("generationStatus").textContent = "Loading model files into browser memory. Keep this tab open…";
  try {
    const rows = await readFiles();
    const files = new Map(rows.map(row => [row.name, row.blob]));
    const missing = REQUIRED.filter(name => !files.has(name));
    if (missing.length) throw new Error("Missing files: " + missing.join(", "));
    const manifest = files.has("manifest.json") ? JSON.parse(await files.get("manifest.json").text()) : {};
    const sidecars = Object.values(manifest.external_data || {}).flat();
    for (const external of sidecars) {
      if (!files.has(external)) throw new Error("Missing external weight file: " + external);
    }
    engine?.dispose();
    engine = new OmniVoiceEngine();
    await engine.load(files, message => {
      $("generationStatus").textContent = message;
      $("modelStatusText").textContent = message;
    });
    loadedManifest = manifest;
    loadedFiles = files;
    refreshEmotionOptions();
    await applySelectedPrompt();
    $("modelStatus").textContent = "LOADED";
    $("engineStatus").textContent = engine.backend.toUpperCase();
    $("generateBtn").disabled = !$("textInput").value.trim();
    $("generationStatus").textContent = `Ready. Inference backend: ${engine.backend.toUpperCase()}. The model runs locally on this device.`;

  } catch (error) {
    engine?.dispose();
    engine = null;
    $("modelStatus").textContent = "LOAD FAILED";
    $("engineStatus").textContent = "ENGINE OFF";
    $("generationStatus").textContent = "Could not load the model: " + error.message;
  } finally {
    const latestRows = await readFiles();
    renderFiles(latestRows);
    if (engine) $("modelStatus").textContent = "LOADED";
    $("loadBtn").disabled = false;
    $("generateBtn").disabled = !engine || !$("textInput").value.trim();
  }
});

$("clearBtn").addEventListener("click", async () => {
  engine?.dispose();
  engine = null;
  loadedManifest = null;
  loadedFiles = new Map();
  await removeFiles();
  volatileFiles.clear();
  renderFiles([]);
  $("engineStatus").textContent = "ENGINE OFF";
  $("generateBtn").disabled = true;
  $("generationStatus").textContent = "Local model cache cleared.";
  if (outputUrl) URL.revokeObjectURL(outputUrl);
  $("audioPlayer").hidden = true;
  $("downloadLink").hidden = true;
});


$("voiceSelect").addEventListener("change", async () => {
  refreshEmotionOptions();
  await applySelectedPrompt();
});
$("emotionSelect").addEventListener("change", applySelectedPrompt);
$("refAudio").addEventListener("change", () => {
  if ($("refAudio").files[0]) {
    $("promptStatus").textContent = "CUSTOM AUDIO";
    $("refText").value = "";
  } else {
    applySelectedPrompt();
  }
});

$("generateBtn").addEventListener("click", async () => {
  if (!engine) return;
  const text = $("textInput").value.trim();
  if (!text) return;
  $("generateBtn").disabled = true;
  $("progressWrap").hidden = false;
  $("progressBar").value = 0;
  $("progressBar").max = Number($("stepsSelect").value);
  $("audioPlayer").hidden = true;
  $("downloadLink").hidden = true;
  $("audioStatus").textContent = "GENERATING";
  $("generationStatus").textContent = "Preparing the reference voice…";

  try {
    const customReference = $("refAudio").files[0];
    const prompt = selectedPrompt();
    let referenceFile = customReference;
    let referenceText = $("refText").value.trim();
    if (!customReference) {
      if (!prompt) throw new Error("That speaker/emotion reference is not included. Choose another preset or upload your own WAV.");
      referenceFile = loadedFiles.get(prompt.audio);
      const transcriptFile = loadedFiles.get(prompt.text);
      if (!referenceText && transcriptFile) referenceText = (await transcriptFile.text()).trim();
    }
    if (!referenceFile) throw new Error("Reference WAV is missing. Re-import the selected prompt files or choose a custom recording.");
    if (!referenceText) throw new Error("The exact transcript for the selected reference WAV is required.");

    const refAudio = await decodeAudioFile(referenceFile);
    const started = performance.now();
    const samples = await engine.generate(text, {
      refAudio,
      refText: referenceText || null,
      language: $("languageSelect").value,
      steps: Number($("stepsSelect").value),
      onStep(step, total) {
        $("progressBar").value = step;
        $("progressText").textContent = `Generating audio · step ${step} / ${total}`;
      }
    });
    const wav = encodeWAV(samples, 24000);
    if (outputUrl) URL.revokeObjectURL(outputUrl);
    outputUrl = URL.createObjectURL(wav);
    $("audioPlayer").src = outputUrl;
    $("audioPlayer").hidden = false;
    $("downloadLink").href = outputUrl;
    $("downloadLink").hidden = false;
    $("audioStatus").textContent = "READY";
    const seconds = (samples.length / 24000).toFixed(1);
    const elapsed = ((performance.now() - started) / 1000).toFixed(1);
    $("generationStatus").textContent = `Generated ${seconds}s of audio in ${elapsed}s on this device.`;
  } catch (error) {
    $("audioStatus").textContent = "ERROR";
    $("generationStatus").textContent = error.message;
  } finally {
    $("progressWrap").hidden = true;
    $("generateBtn").disabled = !engine;
  }
});

(async () => {
  try {
    renderFiles(await readFiles());
  } catch (error) {
    $("modelStatusText").textContent = "This browser could not open IndexedDB: " + error.message;
  }
  if ("serviceWorker" in navigator && location.protocol.startsWith("http")) {
    navigator.serviceWorker.register("./sw.js").catch(() => {});
  }
})();
