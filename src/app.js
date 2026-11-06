const $ = (id) => document.getElementById(id);
const DB_NAME = "badini-voice-local";
const STORE = "model-files";
const EXPECTED = ["audio_embeddings_encoder.onnx", "llm_decoder.onnx", "audio_heads_decoder.onnx"];

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: "name" });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
async function saveFiles(files) {
  const db = await openDB();
  const tx = db.transaction(STORE, "readwrite");
  for (const file of files) tx.objectStore(STORE).put({ name: file.name, blob: file, size: file.size, type: file.type });
  await new Promise((resolve, reject) => { tx.oncomplete = resolve; tx.onerror = () => reject(tx.error); });
  db.close();
}
async function getFiles() {
  const db = await openDB();
  const rows = await new Promise((resolve, reject) => {
    const req = db.transaction(STORE).objectStore(STORE).getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  db.close();
  return rows;
}
async function clearFiles() {
  const db = await openDB();
  await new Promise((resolve, reject) => {
    const req = db.transaction(STORE, "readwrite").objectStore(STORE).clear();
    req.onsuccess = resolve; req.onerror = () => reject(req.error);
  });
  db.close();
}
function humanSize(n) {
  return n >= 1e9 ? `${(n/1e9).toFixed(2)} GB` : n >= 1e6 ? `${(n/1e6).toFixed(1)} MB` : `${(n/1e3).toFixed(1)} KB`;
}
function renderFiles(rows) {
  const list = $("fileList");
  list.replaceChildren();
  if (!rows.length) {
    const li = document.createElement("li"); li.className = "empty"; li.textContent = "No model files selected."; list.append(li);
    $("modelStatus").textContent = "NOT LOADED";
    return;
  }
  for (const row of rows) {
    const li = document.createElement("li");
    li.textContent = `${row.name} · ${humanSize(row.size)}`;
    list.append(li);
  }
  const names = new Set(rows.map(x => x.name));
  const missing = EXPECTED.filter(x => !names.has(x));
  $("modelStatus").textContent = missing.length ? `${rows.length} FILE(S)` : "BACKBONE FOUND";
}
$("textInput").addEventListener("input", () => {
  $("charCount").textContent = `${$("textInput").value.length} / 1200`;
});
$("modelFiles").addEventListener("change", async (event) => {
  const files = [...event.target.files];
  if (!files.length) return;
  try {
    await saveFiles(files);
    renderFiles(await getFiles());
    $("generationStatus").textContent = "Files saved locally. The model-specific browser inference adapter is still required before speech generation can run.";
  } catch (error) {
    $("generationStatus").textContent = `Could not store files: ${error.message}`;
  }
});
$("clearBtn").addEventListener("click", async () => {
  await clearFiles();
  renderFiles([]);
  $("generationStatus").textContent = "Local model cache cleared.";
});
$("voiceSelect").addEventListener("change", () => {
  const isHayfa = $("voiceSelect").value === "hayfa";
  const natural = [...$("emotionSelect").options].find(o => o.value === "natural");
  natural.disabled = !isHayfa;
  if (!isHayfa && $("emotionSelect").value === "natural") $("emotionSelect").value = "happy";
});
(async () => {
  try { renderFiles(await getFiles()); }
  catch { $("generationStatus").textContent = "This browser does not allow IndexedDB storage in the current context. Open the site from localhost or HTTPS to enable local caching."; }
})();
