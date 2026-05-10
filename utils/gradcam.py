"""
utils/gradcam.py
Grad-CAM implementation compatible with both ResNet-18 and ResNet-50.
Produces heatmap overlays on garment crops for qualitative analysis.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for ResNet architectures.
    Hooks into the final convolutional layer (layer4[-1]).
    """

    def __init__(self, model, target_layer=None):
        """
        Args:
            model:        PyTorch model (ResNet-18 or ResNet-50)
            target_layer: nn.Module to hook (default: model.layer4[-1])
        """
        self.model = model
        self.gradients = None
        self.activations = None

        if target_layer is None:
            target_layer = model.layer4[-1]

        self._register_hooks(target_layer)

    def _register_hooks(self, layer):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        """
        Generate a Grad-CAM heatmap.

        Args:
            input_tensor: (1, C, H, W) preprocessed image tensor
            class_idx:    target class index; if None, uses argmax of logits

        Returns:
            heatmap: (H, W) numpy array in [0, 1]
            pred_class: predicted class index
            confidence: softmax probability of predicted class
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        logits = self.model(input_tensor)
        probs  = F.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        confidence = probs[0, class_idx].item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global average pool gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam     = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx, confidence


def overlay_heatmap(pil_image, heatmap, alpha=0.45, colormap="jet"):
    """
    Overlay a Grad-CAM heatmap on a PIL image.

    Returns:
        PIL Image (RGB) with heatmap overlay
    """
    img_np = np.array(pil_image.resize((heatmap.shape[1], heatmap.shape[0]))).astype(float) / 255.0
    cmap   = cm.get_cmap(colormap)
    heat_c = cmap(heatmap)[..., :3]   # (H, W, 3), discard alpha
    blended = (1 - alpha) * img_np + alpha * heat_c
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def generate_gradcam_grid(model, dataset_df, device, transform, save_dir,
                           class_names, n_examples=3, image_size=224):
    """
    Generate and save a grid of Grad-CAM overlays for n_examples random images.

    Args:
        model:       trained PyTorch model
        dataset_df:  DataFrame with 'path', 'label', 'x1', 'y1', 'x2', 'y2'
        device:      torch.device
        transform:   eval transform pipeline
        save_dir:    directory to save PNG files
        class_names: list of class name strings
        n_examples:  number of examples to visualise
    """
    os.makedirs(save_dir, exist_ok=True)
    gradcam = GradCAM(model)

    sample = dataset_df.sample(n=min(n_examples * 3, len(dataset_df)),
                                random_state=42)

    saved = 0
    for _, row in sample.iterrows():
        if saved >= n_examples:
            break
        try:
            img = Image.open(row["path"]).convert("RGB")
            x1 = max(0, int(row["x1"]))
            y1 = max(0, int(row["y1"]))
            x2 = min(img.width,  int(row["x2"]))
            y2 = min(img.height, int(row["y2"]))
            crop = img.crop((x1, y1, x2, y2))
            crop_resized = crop.resize((image_size, image_size))

            input_tensor = transform(crop_resized).unsqueeze(0).to(device)
            heatmap, pred_idx, conf = gradcam.generate(input_tensor)

            overlay = overlay_heatmap(crop_resized, heatmap)

            true_name = class_names[int(row["label"])]
            pred_name = class_names[pred_idx]

            fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
            axes[0].imshow(crop_resized);  axes[0].set_title("Original crop"); axes[0].axis("off")
            axes[1].imshow(overlay);       axes[1].axis("off")
            axes[1].set_title(f"Grad-CAM\nGT: {true_name}\nPred: {pred_name} ({conf:.2f})")
            plt.tight_layout()
            fname = f"gradcam_{saved+1}_gt_{true_name.replace(' ','_')}.png"
            plt.savefig(os.path.join(save_dir, fname), dpi=120, bbox_inches="tight")
            plt.close()
            print(f"[GradCAM] Saved {fname}")
            saved += 1
        except Exception as e:
            print(f"[GradCAM] Skipping sample: {e}")
            continue
