// Minimal static file server for the web version (no dependencies).
// Usage: node server.js   (PORT environment variable, default 3000)
"use strict";
const http = require("http");
const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "web");
const PORT = Number(process.env.PORT) || 3000;
const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8", ".wav": "audio/wav", ".png": "image/png", ".ico": "image/x-icon",
  ".i16": "application/octet-stream",
};

http.createServer((req, res) => {
  let pathname;
  try { pathname = decodeURIComponent(new URL(req.url, "http://localhost").pathname); } catch (e) { res.writeHead(400); res.end(); return; }
  if (pathname.endsWith("/")) pathname += "index.html";
  const file = path.normalize(path.join(ROOT, pathname));
  if (file !== ROOT && !file.startsWith(ROOT + path.sep)) { res.writeHead(403); res.end("Forbidden"); return; }
  fs.stat(file, (error, stat) => {
    if (error || !stat.isFile()) { res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" }); res.end("Not found"); return; }
    const data = pathname.startsWith("/data/");
    res.writeHead(200, {
      "Content-Type": TYPES[path.extname(file).toLowerCase()] || "application/octet-stream",
      "Content-Length": stat.size,
      "Cache-Control": data ? "public, max-age=86400" : "no-cache",
      "X-Content-Type-Options": "nosniff",
    });
    if (req.method === "HEAD") { res.end(); return; }
    fs.createReadStream(file).pipe(res);
  });
}).listen(PORT, "0.0.0.0", () => console.log(`Signal Processor web: http://localhost:${PORT}`));
