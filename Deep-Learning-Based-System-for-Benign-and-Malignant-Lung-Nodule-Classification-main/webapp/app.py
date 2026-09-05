"""
LungScan — Lung Nodule Segmentation & Malignancy Analysis
Hugging Face Spaces deployment (Gradio)
4-panel output: Original CT | Segmented Lungs | Nodule + Juxta | Malignancy Result
"""

import os, io, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnet18
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gradio as gr
from huggingface_hub import hf_hub_download

# TensorFlow for lung segmentation model (.h5)
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠  TensorFlow not available — using fallback lung mask")

# ── Config ────────────────────────────────────────────────────────
HF_REPO       = os.environ.get("HF_REPO", "")
SEG_CKPT      = "best_resunet34_v3.pth"
MAL_CKPT      = "best_malignancy_classifier.pth"
LUNG_CKPT     = "lung_segmentation_model.h5"   # ← your friend's model filename here
IMG_SIZE      = 256
HU_MIN        = -1000
HU_MAX        = 400
JUXTA_PX      = 10
CROP_SIZE     = 64
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── PyTorch model definitions ─────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c,out_c,3,padding=1,bias=False),nn.BatchNorm2d(out_c),nn.ReLU(inplace=True),
            nn.Conv2d(out_c,out_c,3,padding=1,bias=False),nn.BatchNorm2d(out_c),nn.ReLU(inplace=True),
        )
    def forward(self,x): return self.block(x)

class ResUNet34(nn.Module):
    def __init__(self):
        super().__init__()
        r = resnet34(weights=None); r.conv1.stride=(1,1)
        self.initial=nn.Sequential(r.conv1,r.bn1,r.relu)
        self.encoder1=r.layer1; self.encoder2=r.layer2; self.encoder3=r.layer3; self.encoder4=r.layer4
        self.up4=nn.ConvTranspose2d(512,256,2,stride=2); self.dec4=DoubleConv(512,256)
        self.up3=nn.ConvTranspose2d(256,128,2,stride=2); self.dec3=DoubleConv(256,128)
        self.up2=nn.ConvTranspose2d(128, 64,2,stride=2); self.dec2=DoubleConv(128, 64)
        self.up1=nn.ConvTranspose2d( 64, 64,2,stride=2); self.dec1=DoubleConv(128, 64)
        self.final=nn.Conv2d(64,1,1)
    def _c(self,up,skip):
        skip=F.interpolate(skip,size=up.shape[2:],mode="bilinear",align_corners=False)
        return torch.cat([up,skip],1)
    def forward(self,x):
        x0=self.initial(x);x1=self.encoder1(x0);x2=self.encoder2(x1);x3=self.encoder3(x2);x4=self.encoder4(x3)
        d=self.dec4(self._c(self.up4(x4),x3));d=self.dec3(self._c(self.up3(d),x2))
        d=self.dec2(self._c(self.up2(d),x1));d=self.dec1(self._c(self.up1(d),x0))
        return F.interpolate(self.final(d),size=(IMG_SIZE,IMG_SIZE),mode="bilinear",align_corners=False)

class MalignancyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        b = resnet18(weights=None)
        b.fc = nn.Sequential(
            nn.Dropout(.4),nn.Linear(b.fc.in_features,128),
            nn.ReLU(True),nn.Dropout(.4),nn.Linear(128,1))
        self.net = b
    def forward(self,x): return self.net(x).squeeze(1)

# ── Load weights ──────────────────────────────────────────────────
seg_model = ResUNet34().to(DEVICE)
mal_model = MalignancyClassifier().to(DEVICE)
lung_model = None
models_loaded = {"seg": False, "mal": False, "lung": False}

def _load(ckpt_name):
    if os.path.exists(ckpt_name):
        return ckpt_name
    if HF_REPO:
        try:
            return hf_hub_download(repo_id=HF_REPO, filename=ckpt_name)
        except Exception as e:
            print(f"HF Hub download failed for {ckpt_name}: {e}")
    return None

# Load ResUNet-34
seg_path = _load(SEG_CKPT)
if seg_path:
    ckpt = torch.load(seg_path, map_location=DEVICE)
    seg_model.load_state_dict(ckpt["model_state"]); seg_model.eval()
    models_loaded["seg"] = True
    print("✓ Segmentation model loaded")
else:
    print("⚠  Segmentation checkpoint not found")

# Load ResNet-18 malignancy classifier
mal_path = _load(MAL_CKPT)
if mal_path:
    ckpt = torch.load(mal_path, map_location=DEVICE)
    mal_model.load_state_dict(ckpt["model_state"]); mal_model.eval()
    models_loaded["mal"] = True
    print("✓ Malignancy model loaded")
else:
    print("⚠  Malignancy checkpoint not found")

# Load lung segmentation .h5 model
if TF_AVAILABLE:
    lung_path = _load(LUNG_CKPT)
    if lung_path:
        try:
            lung_model = tf.keras.models.load_model(lung_path, compile=False)
            models_loaded["lung"] = True
            print("✓ Lung segmentation model loaded")
        except Exception as e:
            print(f"⚠  Failed to load lung model: {e}")
    else:
        print("⚠  Lung segmentation checkpoint not found — using fallback")

# ── Analysis helpers ──────────────────────────────────────────────
def normalize_ct(img):
    return (np.clip(img.astype(np.float32), HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)

def clean_pred(mask, min_area=20):
    mask = mask.astype(np.uint8)
    n,labels,stats,_ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1,n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area: out[labels==i] = 1
    return out

def get_lung_mask_fallback(img):
    """Rule-based fallback if .h5 model not available."""
    H,W = img.shape
    air  = (img < 0.36).astype(np.uint8)
    n,labels = cv2.connectedComponents(air, connectivity=8)
    border = set()
    for e in [labels[0,:],labels[-1,:],labels[:,0],labels[:,-1]]:
        border.update(np.unique(e).tolist())
    border.discard(0)
    lf = np.zeros((H,W), dtype=np.uint8)
    for l in range(1,n):
        if l in border: continue
        a = int((labels==l).sum())
        if int(.005*H*W) <= a <= int(.48*H*W):
            lf = np.maximum(lf, binary_fill_holes(labels==l).astype(np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    return cv2.morphologyEx(lf, cv2.MORPH_CLOSE, k).astype(np.uint8)

def get_lung_mask(img):
    """Use .h5 model if available, else fallback."""
    if models_loaded["lung"] and lung_model is not None:
        try:
            # Most lung seg UNets expect (batch, H, W, 1) or (batch, H, W, 3)
            inp = img[..., np.newaxis]                         # (256,256,1)
            inp_shape = lung_model.input_shape                 # e.g. (None,256,256,1)
            # If model expects 3 channels, replicate
            if inp_shape[-1] == 3:
                inp = np.stack([img]*3, axis=-1)               # (256,256,3)
            # If model expects different spatial size, resize
            h_in, w_in = inp_shape[1], inp_shape[2]
            if (h_in and h_in != IMG_SIZE) or (w_in and w_in != IMG_SIZE):
                inp_resized = cv2.resize(img, (w_in, h_in))
                inp = inp_resized[..., np.newaxis]
                if inp_shape[-1] == 3:
                    inp = np.stack([inp_resized]*3, axis=-1)
            inp = inp[np.newaxis].astype(np.float32)           # (1, H, W, C)
            pred = lung_model.predict(inp, verbose=0)[0, ..., 0]  # (H, W)
            # Resize back to IMG_SIZE if needed
            if pred.shape != (IMG_SIZE, IMG_SIZE):
                pred = cv2.resize(pred, (IMG_SIZE, IMG_SIZE))
            mask = (pred > 0.5).astype(np.uint8)
            # Clean up small noise
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            return mask
        except Exception as e:
            print(f"Lung model inference failed: {e}, using fallback")
    return get_lung_mask_fallback(img)

def is_juxta(lung, nod, d=JUXTA_PX):
    if nod.sum()==0 or lung.sum()==0: return False
    dist = distance_transform_edt(lung)
    return bool(np.logical_and((lung>0)&(dist<=d), nod>0).any())

def classify_crop(img, cy, cx):
    if not models_loaded["mal"]: return None
    half = CROP_SIZE//2; H,W = img.shape
    y1 = int(np.clip(cy-half, 0, H-CROP_SIZE))
    x1 = int(np.clip(cx-half, 0, W-CROP_SIZE))
    crop = cv2.resize(img[y1:y1+CROP_SIZE, x1:x1+CROP_SIZE], (CROP_SIZE,CROP_SIZE))
    t = torch.tensor(np.stack([crop]*3)[None], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        return float(torch.sigmoid(mal_model(t)).cpu().item())

def run_analysis(arr, threshold=0.35):
    img = normalize_ct(arr) if arr.max() > 1.1 else arr.astype(np.float32)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    t   = torch.tensor(np.stack([img]*3)[None], dtype=torch.float32).to(DEVICE)
    seg_model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(seg_model(t)).cpu().numpy()[0,0]
    pred = clean_pred((prob > threshold).astype(np.uint8))
    lung = get_lung_mask(img)
    n,cl,st,_ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    nodules = []
    for cid in range(1,n):
        a = st[cid, cv2.CC_STAT_AREA]
        if a < 20: continue
        ys,xs = np.where(cl==cid)
        cy,cx = int(ys.mean()), int(xs.mean())
        mp = classify_crop(img, cy, cx)
        nodules.append({
            "centroid":      (cy, cx),
            "bbox":          (int(ys.min()),int(ys.max()),int(xs.min()),int(xs.max())),
            "area_px":       int(a),
            "diameter_mm":   round(float(np.sqrt(4*a/np.pi)*.7), 1),
            "mal_prob":      round(mp,3) if mp is not None else None,
            "mal_pred":      ("Malignant" if mp>=.30 else "Benign") if mp is not None else "Unknown",
            "juxta_pleural": is_juxta(lung, (cl==cid).astype(np.uint8)),
        })
    return img, pred, lung, nodules

# ── 4-panel render ────────────────────────────────────────────────
def render_4panel(img_norm, pred, lung_mask, nodules):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#0a0a0a")
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0a0a0a")

    # ── Panel 1: Original CT ──────────────────────────────────────
    axes[0].imshow(img_norm, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original CT Scan", color="white", fontsize=11, pad=8, fontweight="bold")

    # ── Panel 2: Segmented Lungs (gray inside, black outside) ─────
    lung_display = np.zeros_like(img_norm)
    lung_display[lung_mask > 0] = img_norm[lung_mask > 0]
    axes[1].imshow(lung_display, cmap="gray", vmin=0, vmax=1)
    # Draw lung boundary in cyan
    contours, _ = cv2.findContours(lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        pts = cnt[:,0,:]
        axes[1].plot(pts[:,0], pts[:,1], color="#00e5ff", linewidth=1.2, alpha=0.8)
    axes[1].set_title("Segmented Lung Region", color="white", fontsize=11, pad=8, fontweight="bold")

    # ── Panel 3: Nodule location + juxta-pleural ──────────────────
    axes[2].imshow(lung_display, cmap="gray", vmin=0, vmax=1)
    # Draw pleural band (10px zone) in subtle yellow
    if lung_mask.sum() > 0:
        dist_map = distance_transform_edt(lung_mask)
        pleural_band = ((dist_map > 0) & (dist_map <= JUXTA_PX)).astype(np.float32)
        band_overlay = np.zeros((*img_norm.shape, 4), dtype=np.float32)
        band_overlay[..., 0] = pleural_band * 1.0   # R
        band_overlay[..., 1] = pleural_band * 0.85  # G
        band_overlay[..., 2] = 0.0
        band_overlay[..., 3] = pleural_band * 0.25  # alpha — subtle
        axes[2].imshow(band_overlay, zorder=3)
    # Draw each nodule bounding box + label
    for n in nodules:
        ym,yM,xm,xM = n["bbox"]
        cy,cx = n["centroid"]
        is_jx = n["juxta_pleural"]
        color = "#ffaa00" if is_jx else "#ffffff"
        rect = plt.Rectangle((xm, ym), xM-xm, yM-ym,
                              linewidth=1.5, edgecolor=color, facecolor="none", zorder=5)
        axes[2].add_patch(rect)
        lbl = f"⚠ Juxta\n⌀{n['diameter_mm']}mm" if is_jx else f"⌀{n['diameter_mm']}mm"
        x_text = cx+4 if cx < 180 else cx-70
        axes[2].text(x_text, cy, lbl, color=color, fontsize=7, zorder=6,
                     verticalalignment="center",
                     bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.2"))
    juxta_patch = mpatches.Patch(color="#ffaa00", label="Juxta-Pleural")
    normal_patch = mpatches.Patch(color="#ffffff", label="Non Juxta-Pleural")
    axes[2].legend(handles=[juxta_patch, normal_patch], loc="lower right",
                   fontsize=7, facecolor="black", labelcolor="white", framealpha=0.7)
    axes[2].set_title("Nodule Detection + Juxta-Pleural", color="white", fontsize=11, pad=8, fontweight="bold")

    # ── Panel 4: Malignancy result ────────────────────────────────
    axes[3].imshow(img_norm, cmap="gray", vmin=0, vmax=1)
    for n in nodules:
        ym,yM,xm,xM = n["bbox"]
        cy,cx = n["centroid"]
        is_mal = n["mal_pred"] == "Malignant"
        # Filled colour overlay on nodule region
        region = np.zeros((*img_norm.shape, 4), dtype=np.float32)
        mr = pred[ym:yM+1, xm:xM+1].astype(np.float32)
        if is_mal:
            region[ym:yM+1, xm:xM+1, 0] = mr          # red
            region[ym:yM+1, xm:xM+1, 3] = mr * 0.85
        else:
            region[ym:yM+1, xm:xM+1, 1] = mr          # green
            region[ym:yM+1, xm:xM+1, 3] = mr * 0.85
        axes[3].imshow(region, zorder=4)
        col = "#ff4444" if is_mal else "#44ff88"
        x_text = cx+4 if cx < 180 else cx-80
        lbl = f"{'MAL' if is_mal else 'BEN'}\n⌀{n['diameter_mm']}mm"
        if n["juxta_pleural"]: lbl += "\n⚠ Juxta"
        axes[3].text(x_text, cy, lbl, color=col, fontsize=7, zorder=6,
                     verticalalignment="center",
                     bbox=dict(facecolor="black", alpha=0.65, boxstyle="round,pad=0.3"))
    mal_patch = mpatches.Patch(color="#ff4444", label="Malignant")
    ben_patch  = mpatches.Patch(color="#44ff88", label="Benign")
    axes[3].legend(handles=[mal_patch, ben_patch], loc="lower right",
                   fontsize=7, facecolor="black", labelcolor="white", framealpha=0.7)
    axes[3].set_title("Malignancy Classification", color="white", fontsize=11, pad=8, fontweight="bold")

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#0a0a0a", dpi=150)
    plt.close(fig); buf.seek(0)
    return Image.open(buf).copy()

# ── Report ────────────────────────────────────────────────────────
def build_report(nodules, threshold):
    if not nodules:
        return "**No nodules detected.** Try lowering the threshold."
    any_jp = any(n["juxta_pleural"] for n in nodules)
    lines = [
        f"### 🫁 Analysis Report",
        f"**{len(nodules)} nodule(s) detected** · threshold = {threshold:.2f}",
        f"{'⚠️ **Juxta-pleural nodule present**' if any_jp else '✅ No juxta-pleural nodules'}",
        f"{'⚠️ Malignancy model not loaded' if not models_loaded['mal'] else ''}",
        f"{'⚠️ Using fallback lung segmentation' if not models_loaded['lung'] else '✅ Lung segmentation model active'}",
        "---",
    ]
    for i,n in enumerate(nodules):
        is_mal = n["mal_pred"] == "Malignant"
        icon   = "🔴" if is_mal else "🟢"
        jp_s   = " · ⚠️ **Juxta-pleural**" if n["juxta_pleural"] else ""
        lines += [
            f"#### {icon} Nodule {i+1} — {n['mal_pred']}{jp_s}",
            f"| Property | Value |",
            f"|---|---|",
            f"| Diameter | **{n['diameter_mm']} mm** |",
            f"| Area | {n['area_px']} px² |",
            f"| Centroid (row, col) | `({n['centroid'][0]}, {n['centroid'][1]})` |",
            f"| Bounding box | `r[{n['bbox'][0]}:{n['bbox'][1]}]  c[{n['bbox'][2]}:{n['bbox'][3]}]` |",
            f"| Juxta-pleural | {'Yes ⚠️' if n['juxta_pleural'] else 'No'} |",
            "",
        ]
    return "\n".join(lines)

# ── Gradio handlers ───────────────────────────────────────────────
def analyze(image, threshold):
    if image is None:
        return None, "Please upload a CT slice."
    try:
        arr = np.array(image.convert("L"), dtype=np.float32) / 255.0
        img_norm, pred, lung, nodules = run_analysis(arr, threshold=threshold)
        overlay = render_4panel(img_norm, pred, lung, nodules)
        report  = build_report(nodules, threshold)
        return overlay, report
    except Exception as e:
        import traceback
        return None, f"**Error:** {e}\n\n```\n{traceback.format_exc()}\n```"

def analyze_dcm(file, threshold):
    if file is None:
        return None, "Please upload a DICOM file."
    try:
        import pydicom
        ds  = pydicom.dcmread(file.name)
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * float(getattr(ds,"RescaleSlope",1)) + float(getattr(ds,"RescaleIntercept",0))
        img_norm, pred, lung, nodules = run_analysis(arr, threshold=threshold)
        overlay = render_4panel(img_norm, pred, lung, nodules)
        report  = build_report(nodules, threshold)
        return overlay, report
    except Exception as e:
        import traceback
        return None, f"**Error:** {e}\n\n```\n{traceback.format_exc()}\n```"

# ── CSS ───────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');
body, .gradio-container { background: #080c10 !important; font-family: 'DM Mono', monospace !important; }
.gr-panel, .gr-box, .gr-form { background: #0d1219 !important; border: 1px solid #1a2535 !important; border-radius: 12px !important; }
h1 { font-family: 'Syne', sans-serif !important; color: #00e5ff !important; }
h3 { font-family: 'Syne', sans-serif !important; color: #c8d8e8 !important; }
.gr-button-primary { background: #00e5ff !important; color: #000 !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; border: none !important; border-radius: 8px !important; }
.gr-button-primary:hover { background: #33ecff !important; box-shadow: 0 0 20px rgba(0,229,255,0.4) !important; }
label { color: #4a6070 !important; font-size: 11px !important; letter-spacing: 1px !important; }
.gr-input, .gr-slider { background: #080c10 !important; border-color: #1a2535 !important; color: #c8d8e8 !important; }
"""

# ── Gradio UI ─────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="LungScan — Nodule Analysis") as demo:

    gr.HTML("""
    <div style="text-align:center; padding:24px 0 8px">
      <h1 style="font-family:Syne,sans-serif; font-size:32px; font-weight:800;
                 color:#00e5ff; margin:0; letter-spacing:-1px">🫁 LungScan</h1>
      <p style="color:#4a6070; font-family:DM Mono,monospace; font-size:12px; margin:6px 0 0">
        LUNG NODULE SEGMENTATION · JUXTA-PLEURAL DETECTION · MALIGNANCY RISK SCORING
      </p>
    </div>
    """)

    with gr.Tabs():

        with gr.Tab("🖼️  PNG / JPG"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input   = gr.Image(type="pil", label="UPLOAD CT SLICE", image_mode="L", height=300)
                    threshold_1 = gr.Slider(minimum=0.10, maximum=0.65, value=0.35, step=0.01, label="SEGMENTATION THRESHOLD")
                    btn_1       = gr.Button("▶  Analyze", variant="primary")
            out_img_1    = gr.Image(label="ANALYSIS — 4 PANEL OUTPUT")
            out_report_1 = gr.Markdown(value="*Upload a CT slice and click Analyze.*")
            btn_1.click(fn=analyze, inputs=[img_input, threshold_1], outputs=[out_img_1, out_report_1])

        with gr.Tab("🏥  DICOM (.dcm)"):
            with gr.Row():
                with gr.Column(scale=1):
                    dcm_input   = gr.File(label="UPLOAD DICOM FILE", file_types=[".dcm"])
                    threshold_2 = gr.Slider(minimum=0.10, maximum=0.65, value=0.35, step=0.01, label="SEGMENTATION THRESHOLD")
                    btn_2       = gr.Button("▶  Analyze", variant="primary")
            out_img_2    = gr.Image(label="ANALYSIS — 4 PANEL OUTPUT")
            out_report_2 = gr.Markdown(value="*Upload a DICOM file and click Analyze.*")
            btn_2.click(fn=analyze_dcm, inputs=[dcm_input, threshold_2], outputs=[out_img_2, out_report_2])

    gr.HTML("""
    <div style="text-align:center; padding:16px 0; color:#1a2535; font-family:DM Mono,monospace; font-size:11px">
      ResUNet-34 segmentation · ResNet-18 malignancy classifier · U-Net lung segmentation · LIDC-IDRI dataset
    </div>
    """)

if __name__ == "__main__":
    demo.launch()