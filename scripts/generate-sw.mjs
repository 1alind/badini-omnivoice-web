import { readFile, readdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const dist = resolve("dist");
const manifestPath = resolve(dist, ".vite/manifest.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
const assets = new Set(["./", "./index.html", "./manifest.webmanifest", "./icons/icon.svg", "./sw.js"]);

for (const entry of Object.values(manifest)) {
  if (entry.file) assets.add("./" + entry.file);
  for (const file of entry.css || []) assets.add("./" + file);
  for (const file of entry.assets || []) assets.add("./" + file);
}

const ortDir = resolve(dist, "vendor/ort");
for (const file of await readdir(ortDir)) assets.add("./vendor/ort/" + file);

const precache = JSON.stringify([...assets]);
const sw = `const CACHE = "badini-voice-shell-v3";
const PRECACHE = ${precache};
self.addEventListener("install", event => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    const urls = PRECACHE.map(path => new URL(path, self.registration.scope).href);
    await cache.addAll(urls);
    await self.skipWaiting();
  })());
});
self.addEventListener("activate", event => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) if (key !== CACHE) await caches.delete(key);
    await self.clients.claim();
  })());
});
self.addEventListener("fetch", event => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    const cached = await cache.match(request);
    const response = cached || await fetch(request);
    if (!cached && response.ok && response.type === "basic") cache.put(request, response.clone());
    const headers = new Headers(response.headers);
    headers.set("Cross-Origin-Resource-Policy", "same-origin");
    if (request.mode === "navigate") {
      headers.set("Cross-Origin-Opener-Policy", "same-origin");
      headers.set("Cross-Origin-Embedder-Policy", "require-corp");
    }
    return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
  })());
});
`;
await writeFile(resolve(dist, "sw.js"), sw, "utf8");
console.log(`Generated offline service worker with ${assets.size} precached assets.`);
