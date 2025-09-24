"""
Carb Inspector — single-file API + UI

- One command:  python carb_inspector.py
- Auto-opens browser
- Loads model.pth and class_names.txt from same folder
- reg_head outputs per-class carbs: [B, K]
- tta=auto|on|off  (auto if top-1 < 0.5)
- thresholds: CONF_THRESHOLD, TOPK_MIN, TOPK_MAX
First-time deps:
    pip install fastapi uvicorn pillow torch torchvision
"""
# =========================================================
# First-run bootstrap: install deps in-process (no relaunch)
# =========================================================
import sys, subprocess, importlib, site

REQUIRED = [
    ("fastapi",     "fastapi"),
    ("uvicorn",     "uvicorn"),
    ("PIL",         "pillow"),       # some libs (e.g., gradio) expect Pillow present
    ("torch",       "torch"),
    ("torchvision", "torchvision"),
]

def _pip_install_batch(pkgs, use_user=False):
    args = [sys.executable, "-m", "pip", "install"]
    if use_user:
        args.append("--user")
    subprocess.check_call(args + pkgs)

# 1) detect missing modules
missing_mods = []
for mod, _ in REQUIRED:
    try:
        importlib.import_module(mod)
    except ImportError:
        missing_mods.append(mod)

if missing_mods:
    missing_pkgs = [pkg for (mod, pkg) in REQUIRED if mod in missing_mods]
    print(f"[setup] Installing missing dependencies (batch): {', '.join(missing_pkgs)}")
    try:
        _pip_install_batch(missing_pkgs, use_user=False)
    except subprocess.CalledProcessError:
        # no admin / system-protected → install to user site
        _pip_install_batch(missing_pkgs, use_user=True)
        # ensure user site is importable in this running process
        try:
            user_site = site.getusersitepackages()
            if user_site and user_site not in sys.path:
                sys.path.append(user_site)
        except Exception:
            pass

    # refresh and import again (no relaunch needed)
    importlib.invalidate_caches()
    still = []
    for mod, _ in REQUIRED:
        try:
            importlib.import_module(mod)
        except ImportError:
            still.append(mod)

    if still:
        print(f"[setup] Could not import after install: {', '.join(still)}")
        print("[setup] Try reopening the terminal or run inside a venv.")
        sys.exit(1)
# =========================================================

import io, os, socket, webbrowser
from contextlib import asynccontextmanager
from threading import Timer
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import uvicorn
import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_B4_Weights
from PIL import Image

# =========================
# Constants
# =========================
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(HERE, "model.pth"))
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", os.path.join(HERE, "class_names.txt"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.10"))  # class shown if prob ≥ threshold
TOPK_MIN = int(os.getenv("TOPK_MIN", "3"))                   # always show at least this many
TOPK_MAX = int(os.getenv("TOPK_MAX", "6"))                   # cap list max length
CONF_LOW_FLAG = float(os.getenv("CONF_LOW_FLAG", "0.50"))    # "low_confidence" flag for UI
EFF_WEIGHTS = EfficientNet_B4_Weights.DEFAULT  # Model (per-class carbs) for backbone init only

class MultiTaskEfficientNet(nn.Module):
    """
    Classification: logits  [B, K]; per-class probabilities
    Regression:    carbs    [B, K]; per-class carb estimations
    the shape of both outputs is [B, K] i.e. K= num classes outputs for each of the images B
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.efficientnet_b4(weights=EFF_WEIGHTS)
        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.dropout = nn.Dropout(0.2)
        self.cls_head = nn.Linear(in_feats, num_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(in_feats, num_classes),
            nn.Softplus(beta=1.0)
        )

    def forward(self, x):
        feats = self.backbone.features(x)
        pooled = self.backbone.avgpool(feats)
        flat = torch.flatten(pooled, 1)
        flat = self.dropout(flat)
        logits = self.cls_head(flat)    # [B, K]
        carbs  = self.reg_head(flat)    # [B, K]
        return logits, carbs

# =========================
# Utils
# =========================
def _read_class_names(path: str) -> List[str]:
    if not os.path.exists(path):
        return ["unknown"]
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def _build_model(num_classes: int) -> nn.Module:
    return MultiTaskEfficientNet(num_classes)

def _load_model(model_path: str, num_classes: int) -> nn.Module:
    try:
        obj = torch.load(model_path, map_location=DEVICE)
    except Exception as e:
        raise RuntimeError(f"Could not load model file: {e}")
    model = _build_model(num_classes)
    # Accept full module, plain state_dict, or dict with 'state_dict'
    if isinstance(obj, nn.Module):
        model = obj
    else:
        state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    return model.to(DEVICE).eval()

softmax = nn.Softmax(dim=1)

# --------------------- Preprocess ----------------------
# squeeze to 380x380 (no center-crop)
IMAGENET_MEAN = (0.485, 0.456, 0.406) # from internet
IMAGENET_STD  = (0.229, 0.224, 0.225) # from internet

PREPROCESS = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def _make_tfm(resize, center_crop=None, hflip=False):
    ops = [transforms.Resize(resize)]
    if center_crop:
        ops.append(transforms.CenterCrop(center_crop))
    if hflip:
        ops.append(transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)))
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)

TTA_TRANSFORMS = [
    _make_tfm((380, 380)),
    _make_tfm((380, 380), hflip=True),
    _make_tfm(420, center_crop=380),
    _make_tfm(420, center_crop=380, hflip=True),
    _make_tfm(512, center_crop=380),
]

def _prepare_image_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

# ========= Port + auto-open helpers =========
def _get_free_port(preferred=8000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
PORT = _get_free_port(8000)

# =========================
# App
# =========================
CLASS_NAMES: List[str] = []
NUM_CLASSES: int = 0
MODEL: Optional[nn.Module] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup ----
    global CLASS_NAMES, NUM_CLASSES, MODEL
    CLASS_NAMES = _read_class_names(CLASS_NAMES_PATH)
    NUM_CLASSES = len(CLASS_NAMES)
    MODEL = _load_model(MODEL_PATH, num_classes=NUM_CLASSES)
    # auto-open browser after server starts
    url = f"http://127.0.0.1:{PORT}"
    Timer(0.4, lambda: webbrowser.open(url, new=1, autoraise=True)).start()
    yield
    # ---- shutdown ----
    return

app = FastAPI(
    title="Food Classifier API",
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index_page():
    return HTMLResponse("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Carb Inspector — Gur & Hila</title>
  <style>
    :root{
      --bg: #0b0e14;
      --card: #111827;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --primary: #4f46e5;
      --primary-press: #4338ca;
      --ring: rgba(79,70,229,.35);
      --border: #1f2937;
      --success: #16a34a;
    }
    *{ box-sizing:border-box; }
    html,body{
      margin:0;
      height:100%;
      background:var(--bg);
      color:var(--text);
      font:16px/1.5 system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji";
    }
    .container{ max-width:900px; padding:1.25rem; margin-inline:auto; }
    .site-header{ padding:2rem 0 1rem; border-bottom:1px solid var(--border); }
    .site-header h1{ margin:0 0 .5rem; font-size:2rem; }
    .tagline{ margin:.25rem 0 0; color:var(--text); }
    .instructions{ margin:.75rem 0 0 1rem; color:var(--muted); }
    .card{
      background:var(--card);
      border:1px solid var(--border);
      border-radius:1rem;
      padding:1.25rem;
      margin:1rem 0;
      box-shadow:0 10px 30px rgba(0,0,0,.25);
    }
    .card h2{ margin:.25rem 0 1rem; font-size:1.25rem; }
    .drop-area{
      display:grid; place-items:center; min-height:220px;
      border:2px dashed #334155; border-radius:1rem;
      background:linear-gradient(180deg, rgba(255,255,255,.02), transparent);
      transition:border-color .2s, box-shadow .2s, transform .05s;
      outline:none;
    }
    .drop-area:focus, .drop-area:hover{ border-color:var(--primary); box-shadow:0 0 0 4px var(--ring); }
    .drop-area.dragover{ transform:scale(0.99); border-color:var(--success); box-shadow:0 0 0 4px rgba(22,163,74,.3); }
    .drop-area__content{ text-align:center; }
    .icon{ width:36px; height:36px; opacity:.8; margin-bottom:.25rem; fill:currentColor; color:var(--muted); }
    .muted{ color:var(--muted); margin:.25rem 0; }
    .hint{ color:var(--muted); font-size:.875rem; margin:.25rem 0 0; }
    .btn{
      display:inline-block; padding:.6rem 1rem; border-radius:.8rem;
      border:1px solid transparent; background:#222; color:var(--text);
      cursor:pointer; user-select:none;
    }
    .btn:hover{ filter:brightness(1.05); }
    .btn:active{ transform:translateY(1px); }
    .btn-primary{ background:var(--primary); }
    .btn-primary:active{ background:var(--primary-press); }
    .btn-secondary{ background:transparent; border-color:#334155; }
    .preview{ display:flex; gap:1rem; align-items:center; margin-top:1rem; }
    .preview img{ max-width:260px; max-height:260px; border-radius:.75rem; border:1px solid var(--border); display:block; }
    .actions{ display:flex; gap:.75rem; margin-top:1rem; }
    .status{ color:var(--muted); margin:.25rem 0 .75rem; }
    .results-grid{ display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:.75rem; }
    .result-item{ background:#0f172a; border:1px solid var(--border); border-radius:.8rem; padding:.75rem; }
    .result-label{ color:var(--muted); font-size:.85rem; }
    .result-value{ font-size:1.25rem; margin-top:.25rem; }
    .site-footer{ border-top:1px solid var(--border); margin-top:1.5rem; padding:1rem 0 2rem; color:var(--muted); }
    @media (max-width: 520px){ .preview img{ max-width:100%; height:auto; } }
  </style>
</head>
<body>
  <header class="site-header">
    <div class="container">
      <h1>Carb Inspector</h1>
      <p class="tagline">
        Hello! We’re <strong>Gur & Hila</strong>. Upload a food photo and our model will predict the
        <em>food label</em> and estimate its <em>carbohydrates</em>. This is an early demo—thanks for testing!
      </p>
      <ol class="instructions">
        <li>Click “Choose image” or drag a photo into the box.</li>
        <li>We’ll preview the image.</li>
        <li>Then we’ll send it to the model and show: <strong>Label</strong> + <strong>Carbs (g)</strong>.</li>
      </ol>
    </div>
  </header>

  <main class="container">
    <!-- Upload card -->
    <section class="card" aria-labelledby="upload-title">
      <h2 id="upload-title">Upload your image</h2>

      <div id="drop-area" class="drop-area" tabindex="0" role="button"
           aria-label="Drop image here or press Enter to choose a file">
        <div class="drop-area__content">
          <svg class="icon" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M19 15v4H5v-4H3v4a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-4h-2zM12 3l-4 4h3v6h2V7h3l-4-4z"/>
          </svg>
          <p><strong>Drag & drop</strong> an image here</p>
          <p class="muted">or</p>
          <label class="btn">
            <input id="file-input" type="file" accept="image/*" hidden />
            Choose image
          </label>
          <p class="hint">PNG, JPG, WEBP — up to ~10MB is safe.</p>
        </div>
      </div>

      <div id="preview" class="preview" hidden>
        <img id="preview-img" alt="Selected food preview" />
        <button id="clear-btn" class="btn btn-secondary" type="button">Clear</button>
      </div>

      <div class="actions">
        <button id="predict-btn" class="btn btn-primary" type="button" disabled>
          Analyze with model
        </button>
      </div>
    </section>

    <!-- Results card -->
    <section class="card" aria-labelledby="results-title">
      <h2 id="results-title">Results</h2>
      <div id="status" class="status" aria-live="polite">No image analyzed yet.</div>

      <div id="results" class="results-grid" hidden>
        <div class="result-item">
          <div class="result-label">Predicted Label</div>
          <div id="pred-label" class="result-value">—</div>
        </div>
        <div class="result-item">
          <div class="result-label">Carbohydrates (g)</div>
          <div id="pred-carbs" class="result-value">—</div>
        </div>
        <div class="result-item">
          <div class="result-label">Confidence</div>
          <div id="pred-conf" class="result-value">—</div>
        </div>
        <div class="result-item" style="grid-column: 1 / -1;">
          <div class="result-label">Top predictions</div>
          <div id="topk" class="result-value">—</div>
        </div>
      </div>
    </section>
  </main>

  <footer class="site-footer">
    <div class="container">
      <small>© <span id="year"></span> Gur Abraham &amp; Hila Ofek, 2025 • School of Computer Science &amp; AI, Faculty of Exact Science, Tel Aviv University, 6997801 Tel Aviv-Yafo, IL • Workshop on Deep Learing • Model Demonstration</small>
    </div>
  </footer>

  <script>
  // ---------------- config ----------------
  const API_URL = "/predict";      // same-origin
  const MAX_MB = 10;
  const TTA_MODE = "auto";         // "off" | "on" | "auto"

  // ---------------- element refs ----------------
  const fileInput   = document.getElementById("file-input");
  const dropArea    = document.getElementById("drop-area");
  const previewWrap = document.getElementById("preview");
  const previewImg  = document.getElementById("preview-img");
  const clearBtn    = document.getElementById("clear-btn");
  const predictBtn  = document.getElementById("predict-btn");
  const statusEl    = document.getElementById("status");
  const resultsWrap = document.getElementById("results");
  const outLabel    = document.getElementById("pred-label");
  const outCarbs    = document.getElementById("pred-carbs");
  const outConf     = document.getElementById("pred-conf");
  const topkOut     = document.getElementById("topk");

  let selectedFile = null;
  let objectURL = null;

  function setStatus(msg){ statusEl.textContent = msg; }
  function resetResults(){
    resultsWrap.hidden = true;
    outLabel.textContent = "—";
    outCarbs.textContent = "—";
    outConf.textContent = "—";
    topkOut.textContent = "—";
  }
  function enablePredict(enable){ predictBtn.disabled = !enable; }
  function validateFile(file){
    if(!file) return "No file selected.";
    if(!file.type || !file.type.startsWith("image/")) return "Please choose an image file.";
    const mb = file.size / (1024*1024);
    if(mb > MAX_MB) return `Image is too large (${mb.toFixed(1)} MB). Max ${MAX_MB} MB.`;
    return null;
  }
  function showPreview(file){
    if(objectURL) URL.revokeObjectURL(objectURL);
    objectURL = URL.createObjectURL(file);
    previewImg.src = objectURL;
    previewWrap.hidden = false;
    enablePredict(true);
  }
  function clearSelection(){
    selectedFile = null;
    fileInput.value = "";
    previewWrap.hidden = true;
    if(objectURL){ URL.revokeObjectURL(objectURL); objectURL = null; }
    setStatus("No image analyzed yet.");
    resetResults();
    enablePredict(false);
  }

  // drag & drop + file select
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    const err = validateFile(file);
    if(err){ setStatus(err); clearSelection(); return; }
    selectedFile = file;
    setStatus("Image ready. Click “Analyze with model”.");
    resetResults();
    showPreview(file);
  });
  ["dragenter","dragover"].forEach(evt =>
    dropArea.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dropArea.classList.add("dragover"); })
  );
  ["dragleave","drop"].forEach(evt =>
    dropArea.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dropArea.classList.remove("dragover"); })
  );
  dropArea.addEventListener("drop", e => {
    const file = e.dataTransfer?.files?.[0];
    const err = validateFile(file);
    if(err){ setStatus(err); clearSelection(); return; }
    selectedFile = file;
    setStatus("Image ready. Click “Analyze with model”.");
    resetResults();
    showPreview(file);
  });
  dropArea.addEventListener("keydown", e => { if(e.key==="Enter"||e.key===" ") fileInput.click(); });
  clearBtn.addEventListener("click", clearSelection);

  // API call
  async function callApi(file){
    const fd = new FormData();
    // prefer 'image', but backend also accepts 'file'
    fd.append("image", file);
    const res = await fetch(`${API_URL}?tta=${encodeURIComponent(TTA_MODE)}`, { method:"POST", body: fd });
    if(!res.ok) throw new Error(`Server responded ${res.status}`);
    return await res.json();
  }

  predictBtn.addEventListener("click", async () => {
    if(!selectedFile) return;
    enablePredict(false);
    setStatus("Analyzing image…");
    try{
      const result = await callApi(selectedFile);
      // response fields
      outLabel.textContent = result.label ?? "n/a";
      outCarbs.textContent = (result.carbs_g != null) ? Number(result.carbs_g).toFixed(1) : "n/a";
      outConf.textContent  = (result.confidence != null) ? (Number(result.confidence)*100).toFixed(1) + "%" : "—";

      if (Array.isArray(result.top_preds) && result.top_preds.length){
        const parts = result.top_preds.map(item => {
          const prob  = (Number(item.prob)*100).toFixed(1) + "%";
          const carbs = (item.carbs_g != null ? Number(item.carbs_g).toFixed(1)
                        : (result.carbs_g != null ? Number(result.carbs_g).toFixed(1) : "n/a"));
          return `${item.label} — ${carbs}g (${prob})`;
        });
        topkOut.textContent = parts.join("  •  ");
      } else {
        topkOut.textContent = "—";
      }

      resultsWrap.hidden = false;

      if (result.used_tta) {
        if (result.note) setStatus(`Done. (${result.note}) Used TTA.`);
        else if (result.uncertain) setStatus("Done. (Low confidence — model may be unsure.) Used TTA.");
        else setStatus("Done. Used TTA.");
      } else {
        if (result.note) setStatus(`Done. (${result.note})`);
        else if (result.uncertain) setStatus("Done. (Low confidence — model may be unsure.)");
        else setStatus("Done.");
      }
    } catch (err){
      console.error(err);
      setStatus("Error: " + (err?.message || "prediction failed"));
    } finally {
      enablePredict(true);
    }
  });

  // footer year
  document.getElementById('year').textContent = new Date().getFullYear();
  // init
  clearSelection();
  </script>
</body>
</html>
""")

@app.get("/info")
def info():
    return {"ok": True, "device": DEVICE, "num_classes": NUM_CLASSES}

@app.get("/favicon.ico")
def favicon_ico():
    # Return 204 (no content) so the browser stops whining.
    return Response(status_code=204)

# ------------------------- Predict --------------------------
@app.post("/predict")
async def predict(
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    tta: str = Query("auto")
):
    up = image or file
    if up is None or not up.content_type or not up.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")

    try:
        img_bytes = await up.read()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Single forward pass with preprocess
        def forward_once(img: Image.Image):
            x = PREPROCESS(img).unsqueeze(0).to(DEVICE)     # [B=1,channels=3 (RGB),Height=380,Width=380]
            with torch.inference_mode():
                logits, carbs_vec = MODEL(x)                   # [1,K], [1,K]
                probs = softmax(logits)                        # [1,K]
                conf, idx = torch.max(probs, dim=1)            # [1]
            return logits, carbs_vec, float(conf.item()), int(idx.item())

        logits1, carbs1, conf1, idx1 = forward_once(pil)
        use_tta = (tta == "on") or (tta == "auto" and conf1 < 0.5)

        if not use_tta:
            logits, carbs_vec, confidence, pred_idx = logits1, carbs1, conf1, idx1
        else:
            imgs = [tfm(pil) for tfm in TTA_TRANSFORMS]
            x_batch = torch.stack(imgs, dim=0).to(DEVICE)      # [B,3,H,W]
            with torch.inference_mode():
                l, c = MODEL(x_batch)                          # [B,K], [B,K]
                l = l.mean(dim=0, keepdim=True)                # [1,K]
                c = c.mean(dim=0, keepdim=True)                # [1,K]
                p = softmax(l)                                 # [1,K]
                conf, idx = torch.max(p, dim=1)
            logits, carbs_vec = l, c
            confidence, pred_idx = float(conf.item()), int(idx.item())
        # Build response
        probs = softmax(logits)                                 # [1,K]
        probs_np = probs.squeeze(0).detach().cpu().numpy()      # [K]
        carbs_np = carbs_vec.squeeze(0).detach().cpu().numpy()  # [K]

        ranked = sorted(enumerate(probs_np), key=lambda t: t[1], reverse=True)
        above = [(i, p) for i, p in ranked if p >= CONF_THRESHOLD]
        if len(above) < TOPK_MIN:
            above = ranked[:TOPK_MIN]
        top = above[:TOPK_MAX]

        label = CLASS_NAMES[pred_idx] if 0 <= pred_idx < NUM_CLASSES else str(pred_idx)
        carbs_top1 = float(carbs_np[pred_idx])

        top_preds = [{
            "label": CLASS_NAMES[i] if 0 <= i < NUM_CLASSES else str(i),
            "prob": float(p),
            "carbs_g": float(carbs_np[i]),
        } for i, p in top]

        note = None
        if all(p < CONF_THRESHOLD for _, p in ranked[:TOPK_MIN]):
            note = "No credible answer found — showing top guesses."

        payload = {
            "label": label,
            "carbs_g": round(carbs_top1, 3),
            "confidence": round(confidence, 4),
            "uncertain": confidence < 0.5,
            "top_preds": [
                {"label": t["label"], "prob": round(t["prob"], 4), "carbs_g": round(t["carbs_g"], 3)}
                for t in top_preds
            ],
            "note": note,
            "used_tta": use_tta,
        }
        ui_payload = {
            "top1": {"label": label, "confidence": confidence, "carbs_g": carbs_top1},
            "low_confidence": confidence < CONF_LOW_FLAG,
            "topk": [{"label": t["label"], "confidence": t["prob"], "carbs_g": t["carbs_g"]} for t in top_preds],
        }

        return JSONResponse({**payload, **ui_payload})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

# ========= Run =========
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
