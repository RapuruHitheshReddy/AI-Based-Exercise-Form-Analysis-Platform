// server.js — Local streaming + static + proxy (Express 5 safe)
import express from "express";
import cors from "cors";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { createProxyMiddleware } from "http-proxy-middleware";
import analyzeRoutes from "./routes/analyzeRoutes.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors({
  origin: "*",
  methods: ["GET", "POST"],
  allowedHeaders: ["Content-Type", "Range"],
  exposedHeaders: ["Content-Range", "Accept-Ranges"],
}));

app.use(express.json());

// ----------------------
// 📁 Directories
// ----------------------
const RESULTS_DIR = path.resolve(__dirname, "../python_service/results");
const UPLOADS_DIR = path.resolve(__dirname, "../python_service/uploads");
const PYTHON_API = "http://localhost:8000"; // always local

console.log("📁 Serving static directories:");
console.log(" → RESULTS:", RESULTS_DIR);
console.log(" → UPLOADS:", UPLOADS_DIR);

// Normalize double slashes
app.use((req, res, next) => {
  req.url = req.url.replace(/\/{2,}/g, "/");
  next();
});

// Directly serve uploaded videos
app.use("/uploads", express.static(UPLOADS_DIR));

// ----------------------
// 🔁 Proxy fallback (FastAPI)
// ----------------------
const resultsProxy = createProxyMiddleware({
  target: PYTHON_API,
  changeOrigin: true,
  logLevel: "silent",
  onError(err, req, res) {
    console.error("[PROXY ERROR]", err?.message || err);
    if (!res.headersSent)
      res.status(502).send("⚠️ FastAPI backend unreachable.");
  },
});

// ----------------------
// 🎞️ Stream / Serve results
// ----------------------
app.use("/results", (req, res, next) => {
  try {
    const relPath = req.path.replace(/^\/+/, "");
    const filePath = path.join(RESULTS_DIR, relPath);
    const resolved = path.resolve(filePath);

    if (!resolved.startsWith(RESULTS_DIR)) {
      return res.status(400).send("Invalid path access attempt.");
    }

    if (!fs.existsSync(resolved)) {
      console.warn(`[PROXY] Missing locally → forwarding ${req.originalUrl}`);
      return resultsProxy(req, res, next);
    }

    const ext = path.extname(resolved).toLowerCase();
    const mimeMap = {
      ".mp4": "video/mp4",
      ".png": "image/png",
      ".jpg": "image/jpeg",
      ".jpeg": "image/jpeg",
      ".webp": "image/webp",
      ".csv": "text/csv",
      ".json": "application/json",
      ".txt": "text/plain",
    };
    const contentType = mimeMap[ext] || "application/octet-stream";

    // ✅ Stream .mp4 with Range support
    if (ext === ".mp4") {
      const stat = fs.statSync(resolved);
      let range = req.headers.range;

      // 🧩 Force range if browser didn’t send one
      if (!range) {
        range = "bytes=0-";
      }

      const [startStr, endStr] = range.replace(/bytes=/, "").split("-");
      const start = parseInt(startStr, 10) || 0;
      const end = endStr ? parseInt(endStr, 10) : stat.size - 1;
      const chunkSize = end - start + 1;

      if (start >= stat.size) {
        res.writeHead(416, {
          "Content-Range": `bytes */${stat.size}`,
        });
        return res.end();
      }

      const fileStream = fs.createReadStream(resolved, { start, end });
      res.writeHead(206, {
        "Content-Range": `bytes ${start}-${end}/${stat.size}`,
        "Accept-Ranges": "bytes",
        "Content-Length": chunkSize,
        "Content-Type": contentType,
      });

      console.log(`[STREAM] ${path.basename(resolved)} → bytes ${start}-${end}`);
      fileStream.pipe(res);
      return;
    }

    // Non-video files
    res.type(contentType);
    fs.createReadStream(resolved).pipe(res);
  } catch (err) {
    console.error("❌ Error in /results handler:", err);
    res.status(500).send("Server error while streaming file");
  }
});


// ----------------------
// 🧠 API Routes
// ----------------------
app.use("/api", analyzeRoutes);

// ----------------------
// 💡 Health check
// ----------------------
app.get("/", (req, res) => {
  res.json({
    message: "🏋️ Node backend (local streaming + proxy) running",
    endpoints: {
      analyze: "/api/analyze",
      results: "/results/<session>/<file>",
    },
    python_service_base: PYTHON_API,
  });
});

// ----------------------
// 🚀 Launch server
// ----------------------
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀 Node server running at: http://localhost:${PORT}`);
  console.log(`🔁 Proxy fallback target → ${PYTHON_API}/results/*`);
});
