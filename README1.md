# 🏭 SteelVision — Desktop App
### Steel Surface Defect Detection · Classification · Detection · Segmentation

Runs entirely on your local machine. No Kaggle, no internet required after setup.

---

## 📁 Folder structure

```
steelvision_desktop/
│
├── app.py                ← main application (run this)
├── requirements.txt      ← Python packages
├── README.md             ← this file
│
├── weights/              ← put your trained .pt files here
│   ├── best_clf.pt       ← EfficientNet-B0 classifier
│   ├── best_det.pt       ← YOLOv8s detector
│   └── best_seg.pt       ← YOLOv8s-seg segmenter
│
└── results/              ← auto-created, all outputs saved here
    ├── image1_result.png
    ├── image2_result.png
    └── batch_report.csv
```

---

## ⚡ Quick Start

### Step 1 — Install Python packages

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate

# Install
pip install -r requirements.txt
```

> **GPU users (NVIDIA):** Replace the `torch` line in requirements.txt with the
> CUDA version for your driver: https://pytorch.org/get-started/locally/
> Example for CUDA 12.1:
> ```
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

### Step 2 — Download model weights from Kaggle

1. Open your **training notebook** on Kaggle
2. Click the **Output** tab in the right panel
3. Download these three files:

| Kaggle path | Save as |
|---|---|
| `phase1_classification/best_clf.pt` | `weights/best_clf.pt` |
| `phase2_detection/yolo_neu_det/weights/best.pt` | `weights/best_det.pt` |
| `phase3_segmentation/yolo_neu_seg/weights/best.pt` | `weights/best_seg.pt` |

4. Place all three files inside the `weights/` folder.

---

### Step 3 — Run the app

```bash
python app.py
```

The browser opens automatically at **http://127.0.0.1:7860**

---

## 🖥️ App tabs

### 📷 Single Image
- Upload one JPG / PNG / BMP
- Adjust confidence slider
- Click **⚡ Analyse**
- Get: full 6-panel figure · detection image · segmentation image ·
  defect intelligence card · JSON result

### 📁 Batch / Folder
Three ways to upload:

| Method | How |
|---|---|
| **Single image** | Pick one file |
| **Multiple images** | Hold Ctrl (Win) / ⌘ (Mac) → click files |
| **Entire folder** | Zip it → upload the `.zip` |

Results:
- Annotated **thumbnail gallery** (YOLO boxes)
- **Batch summary** (REJECT / REVIEW / PASS counts)
- **Results table** (one row per image)
- **CSV download path** → `results/batch_report.csv`
- Per-image full-panel PNGs → `results/`

### ⚙️ Setup & Help
- Live model status (which weights loaded)
- Device info (CPU vs GPU)
- Full defect taxonomy table

---

## 📊 Output files

Everything saved to `results/` automatically:

```
results/
├── frame001_result.png    ← 6-panel analysis figure
├── frame002_result.png
├── ...
└── batch_report.csv       ← columns: file, class, confidence,
                               severity, disposition, n_detections,
                               n_masks, defect_area_pct, product, standard
```

---

## 🔧 Configuration

Edit the top of `app.py` to change defaults:

```python
WEIGHTS_DIR  = BASE_DIR / "weights"    # where .pt files live
RESULTS_DIR  = BASE_DIR / "results"    # where outputs are saved
IMG_SIZE     = 224                     # classifier input size
```

To make the app accessible on your local network (e.g. from a tablet):
```python
demo.launch(server_name="0.0.0.0", server_port=7860)
```

To create a temporary public Gradio link (shareable over internet):
```python
demo.launch(share=True)
```

---

## 🗂️ Defect classes

| Class | Severity | QC Decision | IS Standard |
|---|---|---|---|
| Crazing | Major | REVIEW | IS 10748:2004 |
| Inclusion | Critical | REJECT | IS 2062:2011 |
| Patches | Moderate | REVIEW | IS 10748:2004 |
| Pitted Surface | Major | REVIEW | IS 10748:2004 |
| Rolled-in Scale | Critical | REJECT | IS 1786:2008 |
| Scratches | Minor | PASS | IS 2062:2011 |

---

## 💻 System requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.11 |
| RAM | 4 GB | 8 GB |
| GPU | None (CPU ok) | NVIDIA 4 GB+ VRAM |
| Disk | 1 GB | 2 GB |
| OS | Windows 10 / macOS 12 / Ubuntu 20.04 | Any |

CPU inference is supported — expect ~1–3 seconds per image.  
GPU inference: ~100–300 ms per image.
