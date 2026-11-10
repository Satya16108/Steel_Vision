"""
SteelVision Desktop App
=======================
Run locally with:  python app.py
Then open:         http://127.0.0.1:7860

Requirements: pip install -r requirements.txt
Weights:      Copy your trained .pt files into the  weights/  folder
              (see README.md for exact filenames)
"""

import io, json, time, zipfile, warnings, os, sys, socket
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for desktop
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gradio as gr
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
from PIL import Image

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
WEIGHTS_DIR  = BASE_DIR / "weights"
RESULTS_DIR  = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)

CLF_WEIGHTS  = WEIGHTS_DIR / "best_clf.pt"
DET_WEIGHTS  = WEIGHTS_DIR / "best_det.pt"
SEG_WEIGHTS  = WEIGHTS_DIR / "best_seg.pt"

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 224
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Defect taxonomy
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]

CLASS_INFO = {
    "crazing": {
        "label"   : "Crazing",
        "product" : "HR Coil / Hot-rolled Sheet",
        "plants"  : "JSW Vijayanagar · SAIL Bokaro · AM/NS Hazira",
        "standard": "IS 10748:2004",
        "severity": "Major",
        "colour"  : "#E05A2B",
        "desc"    : "Network of fine surface cracks from thermal stress during rolling.",
    },
    "inclusion": {
        "label"   : "Inclusion",
        "product" : "HR/CR Coil · Plates · Wire Rod",
        "plants"  : "Tata Meramandali · JSW · SAIL · RINL",
        "standard": "IS 2062:2011",
        "severity": "Critical",
        "colour"  : "#C0392B",
        "desc"    : "Non-metallic particles (oxides/sulphides) trapped in steel matrix.",
    },
    "patches": {
        "label"   : "Patches",
        "product" : "HR Strip · Hot-rolled Sheet",
        "plants"  : "SAIL Bokaro HSM · AM/NS · JSW",
        "standard": "IS 10748:2004",
        "severity": "Moderate",
        "colour"  : "#E0A82B",
        "desc"    : "Irregular surface discolouration patches from scale variation.",
    },
    "pitted_surface": {
        "label"   : "Pitted Surface",
        "product" : "HR Coil · Wire Rod",
        "plants"  : "RINL Visakhapatnam · JSW Vijayanagar",
        "standard": "IS 10748:2004",
        "severity": "Major",
        "colour"  : "#8E44AD",
        "desc"    : "Small cavities from aggressive descaling or oxide entrapment.",
    },
    "rolled-in_scale": {
        "label"   : "Rolled-in Scale",
        "product" : "Rebar (TMT) · Wire Rod · Structurals",
        "plants"  : "RINL WRM-1/2 · JSPL · SAIL Bhilai",
        "standard": "IS 1786:2008 / IS 7887:2012",
        "severity": "Critical",
        "colour"  : "#C0392B",
        "desc"    : "Mill scale pressed into bar surface — reduces bond strength.",
    },
    "scratches": {
        "label"   : "Scratches",
        "product" : "All flat & long products",
        "plants"  : "All Indian integrated plants",
        "standard": "IS 2062:2011",
        "severity": "Minor",
        "colour"  : "#2E86C1",
        "desc"    : "Linear marks from handling, roller contact or transport.",
    },
}

SEV_DISP = {"Critical": "REJECT", "Major": "REVIEW",
            "Moderate": "REVIEW", "Minor": "PASS"}
SEV_COL  = {"Critical": "#C0392B", "Major": "#E0A82B",
            "Moderate": "#E0A82B", "Minor": "#2E86C1"}
DISP_COL = {"REJECT": "#C0392B", "REVIEW": "#E0A82B", "PASS": "#1A9E6F"}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Model definitions & loading
# ─────────────────────────────────────────────────────────────────────────────
class SteelClassifier(nn.Module):
    """EfficientNet-B0 + custom head — identical to training notebook."""
    def __init__(self, num_classes=6, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=0)
        feat = self.backbone.num_features          # 1280
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.head(self.backbone(x))


infer_tfm = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


def load_models():
    clf = det = seg = None
    msgs = []

    if CLF_WEIGHTS.exists():
        clf = SteelClassifier(6).to(DEVICE)
        clf.load_state_dict(torch.load(CLF_WEIGHTS, map_location=DEVICE))
        clf.eval()
        msgs.append(f"✅ Classifier   : {CLF_WEIGHTS.name}")
    else:
        msgs.append(f"⬜ Classifier   : NOT FOUND  ({CLF_WEIGHTS})")

    if DET_WEIGHTS.exists():
        det = YOLO(str(DET_WEIGHTS))
        msgs.append(f"✅ Detector     : {DET_WEIGHTS.name}")
    else:
        msgs.append(f"⬜ Detector     : NOT FOUND  ({DET_WEIGHTS})")

    if SEG_WEIGHTS.exists():
        seg = YOLO(str(SEG_WEIGHTS))
        msgs.append(f"✅ Segmenter    : {SEG_WEIGHTS.name}")
    else:
        msgs.append(f"⬜ Segmenter    : NOT FOUND  ({SEG_WEIGHTS})")

    return clf, det, seg, msgs


print("Loading models …")
CLF_MODEL, DET_MODEL, SEG_MODEL, load_msgs = load_models()
for m in load_msgs:
    print(" ", m)
print()

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Inference functions
# ─────────────────────────────────────────────────────────────────────────────
def run_classification(img_bgr: np.ndarray) -> dict:
    if CLF_MODEL is None:
        return {"error": "Classifier not loaded — add best_clf.pt to weights/"}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor  = infer_tfm(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(CLF_MODEL(tensor), 1).cpu().numpy()[0]
    top_i = int(probs.argmax())
    top_c = CLASS_NAMES[top_i]
    return {
        "top_class" : top_c,
        "confidence": float(probs[top_i]),
        "all_probs" : sorted(
            [(CLASS_NAMES[i], float(probs[i])) for i in range(6)],
            key=lambda x: -x[1]),
        "info": CLASS_INFO[top_c],
    }


def run_detection(img_bgr: np.ndarray, conf: float) -> dict:
    if DET_MODEL is None:
        return {"error": "Detector not loaded — add best_det.pt to weights/"}
    tmp = str(RESULTS_DIR / "_tmp_det.jpg")
    cv2.imwrite(tmp, img_bgr)
    res  = DET_MODEL(tmp, conf=conf, verbose=False)[0]
    dets = []
    for box in res.boxes:
        dets.append({
            "class"     : res.names[int(box.cls.item())],
            "confidence": round(float(box.conf.item()), 4),
            "bbox"      : [round(float(v), 1) for v in box.xyxy[0].tolist()],
        })
    return {"annotated": res.plot(line_width=2),
            "detections": dets, "count": len(dets)}


def run_segmentation(img_bgr: np.ndarray, conf: float) -> dict:
    if SEG_MODEL is None:
        return {"error": "Segmenter not loaded — add best_seg.pt to weights/"}
    tmp = str(RESULTS_DIR / "_tmp_seg.jpg")
    cv2.imwrite(tmp, img_bgr)
    res = SEG_MODEL(tmp, conf=conf, verbose=False)[0]
    h, w     = img_bgr.shape[:2]
    total_px = h * w
    instances, total_area = [], 0.0
    for idx, box in enumerate(res.boxes):
        area = 0.0
        if res.masks is not None and idx < len(res.masks):
            m    = res.masks.data[idx].cpu().numpy()
            area = float(m.sum()) / total_px * 100
            total_area += area
        instances.append({
            "class"     : res.names[int(box.cls.item())],
            "confidence": round(float(box.conf.item()), 4),
            "area_pct"  : round(area, 2),
        })
    return {"annotated" : res.plot(line_width=2),
            "instances" : instances,
            "count"     : len(instances),
            "total_area": round(total_area, 2)}


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Figure builder  (2 × 3 dark-theme panel)
# ─────────────────────────────────────────────────────────────────────────────
BG   = "#0F1117"
CARD = "#1A1F2E"
TPRI = "#EAEAEA"
TSEC = "#9BA3AF"


def build_figure(fname: str,
                 orig_bgr: np.ndarray,
                 clf: dict, det: dict, seg: dict) -> plt.Figure:
    top  = clf.get("top_class", "?")
    conf = clf.get("confidence", 0)
    info = clf.get("info", {})
    sev  = info.get("severity", "")
    disp = SEV_DISP.get(sev, "REVIEW")
    dcol = DISP_COL[disp]
    scol = SEV_COL.get(sev, "#E05A2B")

    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                             hspace=0.38, wspace=0.28,
                             left=0.03, right=0.97,
                             top=0.89, bottom=0.04)

    fig.text(0.5, 0.96, f"🏭  SteelVision  —  {Path(fname).name}",
             ha="center", fontsize=14, fontweight="bold", color=TPRI)
    fig.text(0.5, 0.925,
             f"Class: {top.upper().replace('_',' ')}  │  "
             f"Confidence: {conf*100:.1f}%  │  Disposition: {disp}",
             ha="center", fontsize=10, color=dcol, fontweight="bold")

    def sax(ax, title):
        ax.set_facecolor(CARD)
        ax.set_title(title, color=TPRI, fontsize=9.5, pad=6, fontweight="bold")
        ax.tick_params(colors=TSEC, labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2A2F3E"); sp.set_linewidth(0.5)

    # (0,0) Original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
    ax.axis("off"); ax.set_facecolor(CARD)
    ax.set_title("Input Image", color=TPRI, fontsize=9.5, pad=6, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_edgecolor(scol); sp.set_linewidth(2.5); sp.set_visible(True)

    # (0,1) Classification bars
    ax = fig.add_subplot(gs[0, 1])
    sax(ax, "Phase 1 — Classification (EfficientNet-B0)")
    probs = clf.get("all_probs", [])
    lbls  = [p[0].replace("_", " ").replace("-", " ") for p in probs]
    vals  = [p[1] * 100 for p in probs]
    cols  = [CLASS_INFO.get(p[0], {}).get("colour", "#555") for p in probs]
    bars  = ax.barh(lbls[::-1], vals[::-1],
                    color=cols[::-1], edgecolor="none", height=0.6)
    bars[-1].set_edgecolor("#FFFFFF"); bars[-1].set_linewidth(1.5)
    for b, v in zip(bars, vals[::-1]):
        ax.text(min(v + 1, 104), b.get_y() + b.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=8, color=TPRI)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Probability (%)", color=TSEC, fontsize=8)
    ax.tick_params(axis="y", colors=TPRI)
    ax.axvline(50, color="#3A3F50", linestyle="--", linewidth=0.8)

    # (0,2) Detection
    ax = fig.add_subplot(gs[0, 2])
    if det.get("error"):
        ax.text(0.5, 0.5, det["error"], ha="center", va="center",
                color=TSEC, transform=ax.transAxes, fontsize=8); ax.axis("off")
    else:
        ax.imshow(cv2.cvtColor(det["annotated"], cv2.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title(f"Phase 2 — YOLOv8s Detection  ({det['count']} box(es))",
                     color=TPRI, fontsize=9.5, pad=6, fontweight="bold")
    ax.set_facecolor(CARD)

    # (1,0) Segmentation
    ax = fig.add_subplot(gs[1, 0])
    if seg.get("error"):
        ax.text(0.5, 0.5, seg["error"], ha="center", va="center",
                color=TSEC, transform=ax.transAxes, fontsize=8); ax.axis("off")
    else:
        ax.imshow(cv2.cvtColor(seg["annotated"], cv2.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title(
            f"Phase 3 — YOLOv8s-seg  "
            f"({seg['count']} mask(s) │ area: {seg['total_area']:.1f}%)",
            color=TPRI, fontsize=9.5, pad=6, fontweight="bold")
    ax.set_facecolor(CARD)

    # (1,1) Defect intelligence card
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off"); ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2A2F3E"); sp.set_linewidth(0.5); sp.set_visible(True)
    ax.set_title("Defect Intelligence", color=TPRI, fontsize=9.5, pad=6, fontweight="bold")
    rows = [
        ("CLASS",       top.upper().replace("_", " ").replace("-", " "), info.get("colour", "#E05A2B")),
        ("DESCRIPTION", info.get("desc", ""),                            TSEC),
        ("PRODUCT",     info.get("product", ""),                         TPRI),
        ("PLANTS",      info.get("plants", ""),                          TSEC),
        ("STANDARD",    info.get("standard", ""),                        TPRI),
        ("SEVERITY",    sev,                                             SEV_COL.get(sev, TSEC)),
        ("CONFIDENCE",  f"{conf*100:.2f}%",                              TPRI),
    ]
    if not seg.get("error"):
        rows.append(("DEFECT AREA", f"{seg['total_area']:.2f}% of frame", TPRI))
    y = 0.93
    for lbl, val, col in rows:
        ax.text(0.05, y, lbl, transform=ax.transAxes,
                fontsize=7, color=TSEC, fontweight="bold")
        y -= 0.06
        ax.text(0.05, y, val, transform=ax.transAxes,
                fontsize=8.5, color=col)
        y -= 0.07
        ax.axhline(y=y + 0.035, xmin=0.03, xmax=0.97,
                   color="#2A2F3E", linewidth=0.4)

    # (1,2) Disposition gauge
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off"); ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2A2F3E"); sp.set_linewidth(0.5); sp.set_visible(True)
    ax.set_title("QC Disposition", color=TPRI, fontsize=9.5, pad=6, fontweight="bold")
    ax.text(0.5, 0.72, disp, transform=ax.transAxes,
            ha="center", fontsize=30, fontweight="bold", color=dcol)
    ax.add_patch(plt.Rectangle((0.1, 0.50), 0.8, 0.065,
                 transform=ax.transAxes, color="#2A2F3E", clip_on=False))
    ax.add_patch(plt.Rectangle((0.1, 0.50), 0.8 * conf, 0.065,
                 transform=ax.transAxes, color=dcol, clip_on=False))
    ax.text(0.5, 0.46, f"Confidence: {conf*100:.1f}%",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=TSEC)
    d_sum = (f"{det['count']} bounding box(es)" if not det.get("error") else "N/A")
    s_sum = (f"{seg['count']} mask(s) · {seg['total_area']:.1f}% area"
             if not seg.get("error") else "N/A")
    for yp, lbl, val in [(0.33, "Detection", d_sum),
                          (0.20, "Segmentation", s_sum),
                          (0.07, "Standard", info.get("standard", "—"))]:
        ax.text(0.5, yp + 0.04, lbl, transform=ax.transAxes,
                ha="center", fontsize=7.5, color=TSEC, fontweight="bold")
        ax.text(0.5, yp, val, transform=ax.transAxes,
                ha="center", fontsize=8.5, color=TPRI)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Image collection helpers
# ─────────────────────────────────────────────────────────────────────────────
def read_bgr(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        arr = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def collect_images(file_list: list[str]) -> list[tuple[str, np.ndarray]]:
    """
    file_list: paths from gr.File (tmp paths on disk).
    Returns list of (display_name, bgr_array).
    Handles: single image, multiple images, ZIP files.
    """
    out = []
    for fpath in file_list:
        p   = Path(fpath)
        ext = p.suffix.lower()
        if ext == ".zip":
            with zipfile.ZipFile(fpath) as zf:
                for zname in sorted(zf.namelist()):
                    if Path(zname).suffix.lower() in IMG_EXTS:
                        raw     = zf.read(zname)
                        arr     = np.frombuffer(raw, np.uint8)
                        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img_bgr is not None:
                            out.append((Path(zname).name, img_bgr))
        elif ext in IMG_EXTS:
            img_bgr = read_bgr(fpath)
            if img_bgr is not None:
                out.append((p.name, img_bgr))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Gradio callback functions
# ─────────────────────────────────────────────────────────────────────────────
def fig_to_pil(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def infer_single(upload, conf_thresh):
    """
    Gradio callback for the Single Image tab.
    upload: tmp file path (str) from gr.Image(type='filepath')
    """
    if upload is None:
        return (None, None, None,
                "⚠️ Please upload an image first.", "", "")

    img_bgr = read_bgr(upload)
    if img_bgr is None:
        return (None, None, None,
                "❌ Could not read image. Try a different file.", "", "")

    t0  = time.time()
    clf = run_classification(img_bgr)
    det = run_detection(img_bgr, conf_thresh)
    seg = run_segmentation(img_bgr, conf_thresh)
    elapsed = time.time() - t0

    # Save full figure
    fname    = Path(upload).name
    fig      = build_figure(fname, img_bgr, clf, det, seg)
    fig_pil  = fig_to_pil(fig)
    fig_path = RESULTS_DIR / f"{Path(fname).stem}_result.png"
    fig_pil.save(fig_path)

    # Detection annotated image
    det_img  = (Image.fromarray(
                    cv2.cvtColor(det["annotated"], cv2.COLOR_BGR2RGB))
                if not det.get("error") else None)

    # Segmentation annotated image
    seg_img  = (Image.fromarray(
                    cv2.cvtColor(seg["annotated"], cv2.COLOR_BGR2RGB))
                if not seg.get("error") else None)

    # Status line
    top      = clf.get("top_class", "?")
    conf_val = clf.get("confidence", 0)
    info     = clf.get("info", {})
    disp     = SEV_DISP.get(info.get("severity", ""), "REVIEW")
    status   = (f"✅ Done in {elapsed:.1f}s  ·  "
                f"Class: {top.replace('_',' ')}  ·  "
                f"Confidence: {conf_val*100:.1f}%  ·  "
                f"Disposition: {disp}  ·  "
                f"Saved → {fig_path.name}")

    # Defect card (markdown)
    card = _make_card_md(clf, det, seg)

    # JSON detail
    detail = json.dumps({
        "classification": {
            "top_class"  : clf.get("top_class"),
            "confidence" : round(clf.get("confidence", 0), 4),
            "all_probs"  : {k: round(v, 4)
                            for k, v in clf.get("all_probs", [])},
        },
        "detection"    : {
            "count"      : det.get("count", 0),
            "detections" : det.get("detections", []),
        },
        "segmentation" : {
            "count"      : seg.get("count", 0),
            "total_area_pct": seg.get("total_area", 0),
            "instances"  : seg.get("instances", []),
        },
    }, indent=2)

    return fig_pil, det_img, seg_img, status, card, detail


def infer_batch(file_list, conf_thresh, progress=gr.Progress()):
    """
    Gradio callback for the Batch / Folder tab.
    file_list: list of tmp file paths from gr.File(file_count='multiple')
    """
    if not file_list:
        return None, "⚠️ No files uploaded.", "", None

    images = collect_images([f.name for f in file_list])
    if not images:
        return None, "❌ No valid images found in upload.", "", None

    n       = len(images)
    log     = []
    gallery = []
    t0      = time.time()

    for i, (fname, img_bgr) in enumerate(
            progress.tqdm(images, desc="Analysing images")):

        clf = run_classification(img_bgr)
        det = run_detection(img_bgr, conf_thresh)
        seg = run_segmentation(img_bgr, conf_thresh)

        top   = clf.get("top_class", "?")
        cval  = clf.get("confidence", 0)
        sev   = clf.get("info", {}).get("severity", "")
        disp  = SEV_DISP.get(sev, "REVIEW")
        n_det = det.get("count", 0)
        n_seg = seg.get("count", 0)
        area  = seg.get("total_area", 0.0)

        log.append({
            "file"           : fname,
            "class"          : top,
            "confidence"     : round(cval, 4),
            "severity"       : sev,
            "disposition"    : disp,
            "n_detections"   : n_det,
            "n_masks"        : n_seg,
            "defect_area_pct": area,
            "product"        : clf.get("info", {}).get("product", ""),
            "standard"       : clf.get("info", {}).get("standard", ""),
        })

        # Save per-image figure
        fig = build_figure(fname, img_bgr, clf, det, seg)
        fig_pil  = fig_to_pil(fig)
        fig_path = RESULTS_DIR / f"{Path(fname).stem}_result.png"
        fig_pil.save(fig_path)

        # Gallery: use detection annotated image as thumbnail
        preview = (Image.fromarray(
                       cv2.cvtColor(det["annotated"], cv2.COLOR_BGR2RGB))
                   if not det.get("error")
                   else Image.fromarray(
                       cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
        sym   = {"REJECT": "❌", "REVIEW": "⚠️", "PASS": "✅"}[disp]
        label = f"{sym} {fname}  |  {top.replace('_',' ')}  {cval*100:.0f}%"
        gallery.append((preview, label))

    elapsed = time.time() - t0

    # Save CSV
    df       = pd.DataFrame(log)
    csv_path = RESULTS_DIR / "batch_report.csv"
    df.to_csv(csv_path, index=False)

    # Summary markdown
    n_rej = sum(1 for r in log if r["disposition"] == "REJECT")
    n_rev = sum(1 for r in log if r["disposition"] == "REVIEW")
    n_pas = sum(1 for r in log if r["disposition"] == "PASS")
    top_cls = df["class"].value_counts().index[0].replace("_", " ").title() \
              if len(df) else "—"

    summary = f"""
## Batch Results — {n} images processed in {elapsed:.1f}s

| | |
|---|---|
| ❌ REJECT | **{n_rej}** ({n_rej/n*100:.0f}%) |
| ⚠️ REVIEW | **{n_rev}** ({n_rev/n*100:.0f}%) |
| ✅ PASS   | **{n_pas}** ({n_pas/n*100:.0f}%) |
| Most common defect | **{top_cls}** |
| Avg confidence | **{df['confidence'].mean()*100:.1f}%** |
| CSV saved | `{csv_path}` |
| Figures saved | `{RESULTS_DIR}/` |
"""
    # DataFrame for display
    df_display = df[["file", "class", "confidence",
                     "disposition", "n_detections",
                     "n_masks", "defect_area_pct", "standard"]].copy()
    df_display["confidence"] = (df_display["confidence"] * 100).round(1)
    df_display.columns = ["File", "Class", "Conf %", "Disposition",
                           "Boxes", "Masks", "Area %", "Standard"]

    return gallery, summary, str(csv_path), df_display


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Helper: markdown defect card
# ─────────────────────────────────────────────────────────────────────────────
def _make_card_md(clf, det, seg) -> str:
    top  = clf.get("top_class", "?")
    conf = clf.get("confidence", 0)
    info = clf.get("info", {})
    sev  = info.get("severity", "")
    disp = SEV_DISP.get(sev, "REVIEW")
    sym  = {"REJECT": "❌", "REVIEW": "⚠️", "PASS": "✅"}[disp]
    area = seg.get("total_area", 0.0) if not seg.get("error") else "N/A"

    probs_md = "\n".join(
        f"- `{cls.replace('_',' '):<22}` {val*100:>6.2f}%"
        for cls, val in clf.get("all_probs", [])
    )
    return f"""
### {sym} {disp} — {top.replace('_',' ').title()}

| Field | Value |
|---|---|
| **Defect class** | {info.get('label', top)} |
| **Confidence** | {conf*100:.2f}% |
| **Severity** | {sev} |
| **Disposition** | {disp} |
| **Defect area** | {area if isinstance(area, str) else f'{area:.2f}%'} |
| **Product** | {info.get('product', '—')} |
| **Plants** | {info.get('plants', '—')} |
| **IS Standard** | {info.get('standard', '—')} |
| **Description** | {info.get('desc', '—')} |
| **Detections** | {det.get('count', 0)} bounding box(es) |
| **Masks** | {seg.get('count', 0)} instance(s) |

**All class probabilities:**
{probs_md}
"""


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
MODEL_STATUS_MD = "\n".join(f"- {m}" for m in load_msgs)
DEVICE_INFO     = (f"🖥️  Running on **{DEVICE.upper()}**"
                   + (f" — {torch.cuda.get_device_name(0)}"
                      if DEVICE == "cuda" else " (CPU)"))

CSS = """
body, .gradio-container { background: #F3F4F6 !important; color: #111827 !important; }
.gr-panel, .gr-box      { background: #FFFFFF !important;
                           border: 1px solid #D1D5DB !important;
                           color: #111827 !important; }
h1, h2, h3, label, .gr-markdown { color: #111827 !important; }
.gr-button-primary      { background: #E05A2B !important;
                           border: none !important; color: white !important; }
.gr-button-secondary    { background: #F8FAFC !important;
                           border: 1px solid #CBD5E1 !important;
                           color: #111827 !important; }
.gr-dropdown, .gr-input, textarea, input { background: #FFFFFF !important; color: #111827 !important; }
.det-panel .gr-label   { background: #FEE2E2 !important; color: #9B1C1C !important;
                           border-radius: 0.35rem !important; padding: 0.35rem 0.6rem !important; }
.seg-panel .gr-label   { background: #DBEAFE !important; color: #1D4ED8 !important;
                           border-radius: 0.35rem !important; padding: 0.35rem 0.6rem !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="🏭 SteelVision",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ).set(
        body_background_fill="#F3F4F6",
        block_background_fill="#FFFFFF",
        block_border_color="#D1D5DB",
        input_background_fill="#FFFFFF",
        panel_background_fill="#FFFFFF",
    ),
    css=CSS,
) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown("""
# 🏭 SteelVision — Steel Surface Defect Detection
**Classification (EfficientNet-B0)  ·  Detection (YOLOv8s)  ·  Segmentation (YOLOv8s-seg)**

Indian steel plants: RINL · JSW · SAIL · Tata Steel · JSPL · AM/NS  
Defect classes: Crazing · Inclusion · Patches · Pitted Surface · Rolled-in Scale · Scratches
""")

    with gr.Row():
        gr.Markdown(f"{DEVICE_INFO}")
        gr.Markdown(f"**Models:**\n{MODEL_STATUS_MD}")

    gr.Markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1 — Single image
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("📷 Single Image"):
            gr.Markdown("Upload one steel surface image → get all three model outputs at once.")

            with gr.Row():
                with gr.Column(scale=1):
                    img_input  = gr.Image(
                        label="Upload steel image (JPG / PNG / BMP)",
                        type="filepath",
                        height=280,
                    )
                    conf_s1    = gr.Slider(
                        minimum=0.05, maximum=0.95, value=0.20, step=0.05,
                        label="Detection / Segmentation confidence threshold",
                    )
                    run_btn_s1 = gr.Button("⚡  Analyse", variant="primary")
                    status_s1  = gr.Textbox(label="Status", interactive=False,
                                            lines=2)

                with gr.Column(scale=2):
                    full_fig   = gr.Image(label="Full Analysis Panel (saved to results/)",
                                         height=400)

            gr.Markdown("---")

            with gr.Row():
                det_out = gr.Image(label="Phase 2 — Detection (YOLOv8s)", elem_classes="det-panel")
                seg_out = gr.Image(label="Phase 3 — Segmentation (YOLOv8s-seg)", elem_classes="seg-panel")

            with gr.Row():
                with gr.Column():
                    card_out   = gr.Markdown(label="Defect Intelligence Card")
                with gr.Column():
                    json_out   = gr.Code(label="Full JSON Result",
                                         language="json", lines=20)

            run_btn_s1.click(
                fn=infer_single,
                inputs=[img_input, conf_s1],
                outputs=[full_fig, det_out, seg_out,
                         status_s1, card_out, json_out],
            )

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2 — Batch / Folder
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("📁 Batch / Folder"):
            gr.Markdown("""
Upload **multiple images** or a **ZIP file** (your image folder zipped).

| Upload type | How to do it |
|---|---|
| Multiple images | Hold **Ctrl** (Win/Linux) or **⌘** (Mac) and click files |
| Entire folder | Zip the folder first, then upload the `.zip` |
""")

            with gr.Row():
                with gr.Column(scale=1):
                    files_input = gr.File(
                        label="Upload images or ZIP",
                        file_count="multiple",
                        file_types=[".jpg", ".jpeg", ".png",
                                    ".bmp", ".webp", ".zip"],
                        height=200,
                    )
                    conf_batch  = gr.Slider(
                        minimum=0.05, maximum=0.95, value=0.20, step=0.05,
                        label="Confidence threshold",
                    )
                    run_batch   = gr.Button("⚡  Run Batch", variant="primary")
                    csv_path_out= gr.Textbox(label="CSV saved to",
                                             interactive=False)

                with gr.Column(scale=2):
                    batch_summary = gr.Markdown(label="Batch Summary")

            gr.Markdown("### Results Gallery")
            gallery_out = gr.Gallery(
                label="Annotated images (Detection output)",
                columns=4,
                height=420,
                object_fit="cover",
            )

            gr.Markdown("### Per-image Results Table")
            table_out = gr.Dataframe(
                label="Batch results",
                interactive=False,
                wrap=True,
            )

            run_batch.click(
                fn=infer_batch,
                inputs=[files_input, conf_batch],
                outputs=[gallery_out, batch_summary,
                         csv_path_out, table_out],
            )

        # ══════════════════════════════════════════════════════════════════════
        # TAB 3 — Setup / Help
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("⚙️ Setup & Help"):
            gr.Markdown(f"""
## Setup

### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. Copy your trained model weights

Copy the three `.pt` files from Kaggle into the `weights/` folder:

| Kaggle output path | Local filename |
|---|---|
| `phase1_classification/best_clf.pt` | `weights/best_clf.pt` |
| `phase2_detection/yolo_neu_det/weights/best.pt` | `weights/best_det.pt` |
| `phase3_segmentation/yolo_neu_seg/weights/best.pt` | `weights/best_seg.pt` |

**Download from Kaggle:**
1. Open your training notebook on Kaggle
2. Go to the **Output** tab (right panel)
3. Click the three `.pt` files → Download
4. Place them in `steelvision_desktop/weights/`

### 3. Run the app
```bash
python app.py
```
Open **http://127.0.0.1:7860** in your browser.

---

## Current model status
{MODEL_STATUS_MD}

## Device
{DEVICE_INFO}

---

## Output files
All results are saved to `results/` in the app folder:
- **Per-image PNG** — full 6-panel analysis figure
- **batch_report.csv** — one row per image with all metrics

---

## Defect classes & Indian plant mapping

| Class | Severity | Disposition | Product | Plants |
|---|---|---|---|---|
| Crazing | Major | REVIEW | HR Coil / Sheet | JSW · SAIL Bokaro · AM/NS |
| Inclusion | Critical | REJECT | HR/CR Coil · Plates | Tata · JSW · SAIL · RINL |
| Patches | Moderate | REVIEW | HR Strip | SAIL Bokaro · AM/NS |
| Pitted Surface | Major | REVIEW | HR Coil · Wire Rod | RINL Vizag · JSW |
| Rolled-in Scale | Critical | REJECT | Rebar · Wire Rod | RINL · JSPL · SAIL |
| Scratches | Minor | PASS | All products | All plants |
""")

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.Markdown("""
---
**SteelVision v1.0**  ·  EfficientNet-B0 + YOLOv8s + YOLOv8s-seg  ·  
NEU Surface Defect Database  ·  Indian Steel Plants  ·  BIS QCO 2024
""")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Launch
# ─────────────────────────────────────────────────────────────────────────────
def find_free_port(start_port=7860, end_port=7880):
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(f"Cannot find an empty local port in range {start_port}-{end_port}.")

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  🏭  SteelVision Desktop App")
    print("═" * 60)
    print(f"  Device   : {DEVICE.upper()}")
    for m in load_msgs:
        print(f"  {m}")
    print(f"  Results  : {RESULTS_DIR}")
    print("═" * 60 + "\n")

    port = find_free_port()
    print(f"Starting SteelVision on local port: {port}")
    demo.launch(
        server_name="127.0.0.1",   # localhost only
        server_port=port,
        inbrowser=True,            # auto-opens browser tab
        share=False,               # set True for a public Gradio link
        show_error=True,
        quiet=False,
    )
