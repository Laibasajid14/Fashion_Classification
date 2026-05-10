"""
ui/app.py
Gradio UI — three sections in one page:
  1. Upload panel
  2. Top-5 predictions (baseline | improved) — side by side
  3. Grad-CAM heatmap overlay (baseline | improved) — side by side

Usage:
    cd fashion_classification        # project root
    python ui/app.py

Requirements:
    pip install gradio
Both model checkpoints must exist:
    baseline_model/outputs/checkpoints/best_model.pth
    improved_model/outputs/checkpoints/best_model.pth
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import io
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import torchvision.transforms as T
import gradio as gr
from PIL import Image

from configs.dataset_config import CLASS_NAMES, NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD
import configs.baseline_config as base_cfg
import configs.improved_config as imp_cfg
from baseline_model.src.model import build_baseline_model
from improved_model.src.model import build_improved_model


# ── Device ─────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model loader ───────────────────────────────────────────────────────────────

def _load_model(build_fn, checkpoint_path, device):
    model = build_fn()
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train both models before launching the UI.\n"
            "  Baseline:  cd baseline_model && python run_baseline.py\n"
            "  Improved:  cd improved_model && python run_improved.py"
        )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model


print("[UI] Loading models ...")
BASELINE_MODEL = _load_model(
    lambda: build_baseline_model(
        dropout_rate=base_cfg.DROPOUT_RATE,
        pretrained=False,
        freeze_backbone=False,
    ),
    base_cfg.BEST_MODEL_PATH,
    DEVICE,
)
IMPROVED_MODEL = _load_model(
    lambda: build_improved_model(
        dropout_rate=imp_cfg.DROPOUT_RATE,
        pretrained=False,
        freeze_backbone=False,
    ),
    imp_cfg.BEST_MODEL_PATH,
    DEVICE,
)
print(f"[UI] Both models loaded on {DEVICE}.")


# ── Transforms ─────────────────────────────────────────────────────────────────

IMAGE_SIZE = 224

EVAL_TRANSFORM = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Grad-CAM ───────────────────────────────────────────────────────────────────

class _GradCAM:
    """
    Lightweight Grad-CAM that hooks into model.layer4[-1].
    One instance per model; hooks persist for the lifetime of the app.
    """

    def __init__(self, model):
        self.model = model
        self._acts = None
        self._grads = None
        target = model.layer4[-1]
        target.register_forward_hook(self._fwd_hook)
        target.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self._acts = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self._grads = grad_out[0].detach()

    def __call__(self, tensor, class_idx=None):
        """
        Args:
            tensor:    (1, C, H, W) on DEVICE
            class_idx: int or None (uses argmax)
        Returns:
            heatmap  (H, W) float32 in [0, 1]
            pred_idx int
            conf     float
        """
        t = tensor.clone().requires_grad_(True)
        self.model.zero_grad()
        logits = self.model(t)
        probs  = F.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        conf = float(probs[0, class_idx].item())

        logits[0, class_idx].backward()

        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self._acts).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE),
                                mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()

        lo, hi = cam.min(), cam.max()
        cam = (cam - lo) / (hi - lo + 1e-8)
        return cam, class_idx, conf


_BASE_GRADCAM = _GradCAM(BASELINE_MODEL)
_IMP_GRADCAM  = _GradCAM(IMPROVED_MODEL)


def _blend_heatmap(pil_img, heatmap, alpha=0.45):
    """Blend jet heatmap (H x W float [0,1]) onto pil_img; return PIL RGB."""
    arr  = np.array(pil_img.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(float) / 255.0
    cmap = mpl_cm.get_cmap("jet")
    heat = cmap(heatmap)[..., :3]
    blended = np.clip((1 - alpha) * arr + alpha * heat, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))


def _make_gradcam_figure(orig_pil, overlay_pil, model_label, pred_name, conf):
    """
    Render a side-by-side (original | heatmap overlay) matplotlib figure
    and return it as a PIL Image for Gradio gr.Image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), dpi=110)
    fig.patch.set_facecolor("#f8f9fa")

    axes[0].imshow(orig_pil.resize((IMAGE_SIZE, IMAGE_SIZE)))
    axes[0].set_title("Original", fontsize=10, pad=5, color="#333")
    axes[0].axis("off")

    axes[1].imshow(overlay_pil)
    axes[1].set_title(
        f"Grad-CAM  ·  {model_label}\nPredicted: {pred_name}   ({conf:.1%})",
        fontsize=9, pad=5, color="#333",
    )
    axes[1].axis("off")

    # Colourbar for the heatmap channel
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Activation intensity", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout(pad=1.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── Main inference function ────────────────────────────────────────────────────

def run_inference(image):
    """
    Gradio callback — runs on every image upload / change.

    Returns 4 values:
        base_label  dict {class: prob}  → gr.Label (baseline)
        imp_label   dict {class: prob}  → gr.Label (improved)
        base_cam    PIL Image           → gr.Image (baseline Grad-CAM)
        imp_cam     PIL Image           → gr.Image (improved Grad-CAM)
    """
    empty_label = {c: 0.0 for c in CLASS_NAMES[:5]}
    blank_img   = Image.new("RGB", (700, 380), (245, 245, 245))

    if image is None:
        return empty_label, empty_label, blank_img, blank_img

    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    pil = pil.convert("RGB")

    tensor = EVAL_TRANSFORM(pil).unsqueeze(0).to(DEVICE)

    # ── Top-5 predictions (no grad needed) ────────────────────────────────────
    with torch.no_grad():
        base_probs = F.softmax(BASELINE_MODEL(tensor), dim=1).cpu().numpy()[0]
        imp_probs  = F.softmax(IMPROVED_MODEL(tensor),  dim=1).cpu().numpy()[0]

    base_top5  = np.argsort(base_probs)[::-1][:5]
    imp_top5   = np.argsort(imp_probs)[::-1][:5]
    base_label = {CLASS_NAMES[i]: float(base_probs[i]) for i in base_top5}
    imp_label  = {CLASS_NAMES[i]: float(imp_probs[i])  for i in imp_top5}

    # ── Grad-CAM (needs backward) ──────────────────────────────────────────────
    base_heatmap, base_pred_idx, base_conf = _BASE_GRADCAM(tensor)
    imp_heatmap,  imp_pred_idx,  imp_conf  = _IMP_GRADCAM(tensor)

    base_overlay = _blend_heatmap(pil, base_heatmap)
    imp_overlay  = _blend_heatmap(pil, imp_heatmap)

    base_cam_fig = _make_gradcam_figure(
        pil, base_overlay, "ResNet-18 Baseline",
        CLASS_NAMES[base_pred_idx], base_conf,
    )
    imp_cam_fig = _make_gradcam_figure(
        pil, imp_overlay, "ResNet-50 Improved",
        CLASS_NAMES[imp_pred_idx], imp_conf,
    )

    return base_label, imp_label, base_cam_fig, imp_cam_fig


# ── Gradio UI layout ───────────────────────────────────────────────────────────

CSS = """
body, .gradio-container { font-family: "Inter", "Segoe UI", sans-serif; }

#app-title {
    text-align: center;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 14px 0 4px;
    color: #1a1a2e;
}
#app-sub {
    text-align: center;
    color: #555;
    margin-bottom: 20px;
    font-size: 0.92rem;
    line-height: 1.5;
}

.sec-head {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1a1a2e;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 5px;
    margin: 4px 0 12px;
}

.card {
    border: 1px solid #dde1e7;
    border-radius: 10px;
    padding: 14px 16px;
    background: #fafbfc;
    height: 100%;
}
.card-blue { border-top: 3px solid #4C72B0; }
.card-orange { border-top: 3px solid #DD8452; }

.mname { font-size: 0.98rem; font-weight: 600; margin-bottom: 2px; }
.mname-blue   { color: #4C72B0; }
.mname-orange { color: #DD8452; }
.mdesc { font-size: 0.78rem; color: #888; margin-bottom: 10px; font-style: italic; }

.divider { border: none; border-top: 1px solid #e8e8e8; margin: 20px 0; }

#footer {
    text-align: center;
    font-size: 0.76rem;
    color: #bbb;
    margin-top: 22px;
    padding-top: 10px;
    border-top: 1px solid #eee;
}
"""

with gr.Blocks(css=CSS, title="Fashion Classifier — DeepFashion2") as demo:

    # ── Header ─────────────────────────────────────────────────────────────────
    gr.HTML('<div id="app-title">👗 Clothing Classification — Model Comparison</div>')
    gr.HTML(
        '<div id="app-sub">'
        "Upload a fashion image · compare <b>ResNet-18 Baseline</b> vs <b>ResNet-50 Improved</b><br>"
        "Top-5 confidence scores &nbsp;+&nbsp; Grad-CAM attention maps — side by side"
        "</div>"
    )

    # ── Upload row ─────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload a clothing image",
                height=300,
            )
        with gr.Column(scale=2):
            gr.HTML(
                "<div style='padding:16px 18px; background:#eef2ff; border-radius:9px;"
                " border-left:4px solid #4C72B0; font-size:0.87rem; color:#333; line-height:1.7;'>"
                "<b>How to use</b><br>"
                "① Upload a JPEG or PNG fashion photo.<br>"
                "② <b>Top-5 Predictions</b> — confidence bars appear instantly for each model.<br>"
                "③ <b>Grad-CAM</b> — heatmaps reveal which regions drove each model's decision.<br>"
                "&nbsp;&nbsp;&nbsp;&nbsp;"
                "<span style='color:#c0392b'>■</span> Warm/red = high activation &nbsp;"
                "<span style='color:#2980b9'>■</span> Cool/blue = low activation<br><br>"
                "<b>13 categories:</b> short/long sleeve top · short/long sleeve outwear · vest · "
                "sling · shorts · trousers · skirt · short/long sleeve dress · vest dress · sling dress"
                "</div>"
            )

    # ── Section 1 — Top-5 Predictions ──────────────────────────────────────────
    gr.HTML('<hr class="divider"><div class="sec-head">📊 Top-5 Predictions</div>')

    with gr.Row(equal_height=True):
        with gr.Column(elem_classes=["card", "card-blue"]):
            gr.HTML(
                '<div class="mname mname-blue">Baseline — ResNet-18</div>'
                '<div class="mdesc">Frozen backbone · fixed LR · CrossEntropy · single-layer head</div>'
            )
            baseline_label = gr.Label(num_top_classes=5, label="Top-5 confidence")

        with gr.Column(elem_classes=["card", "card-orange"]):
            gr.HTML(
                '<div class="mname mname-orange">Improved — ResNet-50</div>'
                '<div class="mdesc">Full fine-tuning · cosine LR · label smoothing (ε=0.1) · mixup</div>'
            )
            improved_label = gr.Label(num_top_classes=5, label="Top-5 confidence")

    # ── Section 2 — Grad-CAM ───────────────────────────────────────────────────
    gr.HTML('<hr class="divider"><div class="sec-head">🔍 Grad-CAM Attention Maps</div>')
    gr.HTML(
        "<div style='font-size:0.83rem; color:#666; margin-bottom:12px; line-height:1.6;'>"
        "Grad-CAM (Gradient-weighted Class Activation Mapping) uses gradients flowing into the "
        "final convolutional block (<code>layer4</code>) to localise the discriminative regions. "
        "Each panel shows the <b>original crop</b> (left) and the <b>heatmap overlay</b> (right). "
        "Download buttons let you save the figures."
        "</div>"
    )

    with gr.Row(equal_height=True):
        with gr.Column(elem_classes=["card", "card-blue"]):
            gr.HTML(
                '<div class="mname mname-blue">Baseline — ResNet-18</div>'
                '<div class="mdesc">Grad-CAM on layer4[-1] · jet colormap · α = 0.45</div>'
            )
            baseline_cam = gr.Image(
                type="pil",
                label="Grad-CAM — Baseline ResNet-18",
                height=340,
                show_download_button=True,
            )

        with gr.Column(elem_classes=["card", "card-orange"]):
            gr.HTML(
                '<div class="mname mname-orange">Improved — ResNet-50</div>'
                '<div class="mdesc">Grad-CAM on layer4[-1] · jet colormap · α = 0.45</div>'
            )
            improved_cam = gr.Image(
                type="pil",
                label="Grad-CAM — Improved ResNet-50",
                height=340,
                show_download_button=True,
            )

    # ── Wire event ─────────────────────────────────────────────────────────────
    image_input.change(
        fn=run_inference,
        inputs=image_input,
        outputs=[baseline_label, improved_label, baseline_cam, improved_cam],
    )

    # ── Footer ─────────────────────────────────────────────────────────────────
    gr.HTML(
        '<div id="footer">'
        "DeepFashion2 · Ge et al., CVPR 2019 · 264 K images · 13 clothing categories<br>"
        "ResNet-18 Baseline vs ResNet-50 Improved · Computer Vision Milestone 2"
        "</div>"
    )


# ── Launch ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,       # set True for a public Gradio tunnel link (useful on Kaggle)
        show_error=True,
    )
