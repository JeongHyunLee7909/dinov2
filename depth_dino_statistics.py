from time import time
import sys
import math
import itertools
from functools import partial
import torch
import torch.nn.functional as F
from dinov2.eval.depth.models import build_depther
import urllib
import mmcv
from mmcv.runner import load_checkpoint
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
from torchvision import transforms
import numpy as np
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, DPTForDepthEstimation


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m)
                    for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def create_depther(cfg, backbone_model, backbone_size, head_type):
    train_cfg = cfg.get("train_cfg")
    test_cfg = cfg.get("test_cfg")
    depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

    depther.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
        return_class_token=cfg.model.backbone.output_cls_token,
        norm=cfg.model.backbone.final_norm,
    )

    if hasattr(backbone_model, "patch_size"):
        depther.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))

    return depther


def make_depth_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        lambda x: 255.0 * x[:3],  # Discard alpha component and scale by 255
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])


def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True)  # ((1)xhxwx4)
    colors = colors[:, :, :3]  # Discard alpha component
    return Image.fromarray(colors)


def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")


BACKBONE_SIZE = "small"  # in ("small", "base", "large" or "giant")


backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
backbone_arch = backbone_archs[BACKBONE_SIZE]
backbone_name = f"dinov2_{backbone_arch}"

backbone_model = torch.hub.load(
    repo_or_dir="facebookresearch/dinov2", model=backbone_name)
backbone_model.eval()
backbone_model.cuda()


def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


HEAD_DATASET = "nyu"  # in ("nyu", "kitti")
HEAD_TYPE = "dpt"  # in ("linear", "linear4", "dpt")


DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

cfg_str = load_config_from_url(head_config_url)
cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

model = create_depther(
    cfg,
    backbone_model=backbone_model,
    backbone_size=BACKBONE_SIZE,
    head_type=HEAD_TYPE,
)

load_checkpoint(model, head_checkpoint_url, map_location="cpu")
model.eval()
model.cuda()


EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"


transform = make_depth_transform()

# load images
base_dir = Path(
    "~/navigation_ws/250617_seungdo_sensor_logging/sensor_logging_data/2025-06-17-18-07-12-trial2/"
).expanduser()
rgb_dir = base_dir / "rgb"
depth_dir = base_dir / "depth"

# -------------------------------
# Collect RGB + Depth image files
# -------------------------------
rgb_files = sorted([f for f in rgb_dir.glob("*.png")])
depth_files = sorted([
    f for f in depth_dir.glob("*.png")
    if f.name.startswith("d455_front")
])

assert len(rgb_files) == len(depth_files), \
    f"Mismatch between RGB {len(rgb_files)} and depth counts {len(depth_files)}!"

num_samples = 100

# Exclude first 400 and last 100
valid_range = range(400, len(rgb_files) - 100)

# Pick equally spaced indices across the valid range
indices = np.linspace(valid_range.start, valid_range.stop -
                      1, num_samples, dtype=int).tolist()

# Accumulators
all_rmse = []
all_absrel = []
bucket_rmse = {"0.6-1.0": [], "1.0-1.5": [],
               "1.5-2.0": [], "2.0-2.5": [], "2.5-3.0": []}
bucket_absrel = {"0.6-1.0": [], "1.0-1.5": [],
                 "1.5-2.0": [], "2.0-2.5": [], "2.5-3.0": []}
avg_time = []
for i in range(num_samples):
    idx = indices[i]

    rgb_path = rgb_files[idx]
    depth_path = depth_files[idx]

    rgb_img = Image.open(rgb_path).convert("RGB")
    depth_gt = np.array(Image.open(depth_path))

    transformed_image = transform(rgb_img)
    batch = transformed_image.unsqueeze(0).cuda()

    with torch.inference_mode():
        torch.cuda.synchronize()
        start_time = time()
        result = model.whole_inference(
            batch, img_meta=None, rescale=True)
        torch.cuda.synchronize()
        end_time = time()

        if i > 10:
            avg_time.append(end_time - start_time)

    depth_pred = result.squeeze().cpu().numpy()
    depth_gt = (depth_gt) / 255 * (6.0 - 0.3) + 0.3

    # 1) Mask valid pixels
    mask = (depth_gt > 0.6) & (depth_gt < 3.0)
    valid_pred = depth_pred[mask]
    valid_gt = depth_gt[mask]

    if valid_gt.size == 0:
        continue

    # 2) RMSE for this frame
    rmse = np.sqrt(np.mean((valid_pred - valid_gt) ** 2))
    all_rmse.append(rmse)

    # 3) AbsRel for this frame
    absrel = np.mean(np.abs(valid_pred - valid_gt) / valid_gt)
    all_absrel.append(absrel)

    # 4) Per-region stats
    for region, (low, high) in {
        "0.6-1.0": (0.6, 1.0),
        "1.0-1.5": (1.0, 1.5),
        "1.5-2.0": (1.5, 2.0),
        "2.0-2.5": (2.0, 2.5),
        "2.5-3.0": (2.5, 3.0),
    }.items():
        region_mask = (depth_gt >= low) & (depth_gt < high)
        region_mask = region_mask & mask
        if np.any(region_mask):
            region_rmse = np.sqrt(
                np.mean((depth_pred[region_mask] - depth_gt[region_mask]) ** 2))
            region_absrel = np.mean(
                np.abs(depth_pred[region_mask] - depth_gt[region_mask]) / depth_gt[region_mask])
            bucket_rmse[region].append(region_rmse)
            bucket_absrel[region].append(region_absrel)

# -----------------------------
# Print global averages
print("Average inference time:", np.mean(np.array(avg_time)))
print("Average RMSE :", np.mean(all_rmse), "m")
print("Average AbsRel:", np.mean(all_absrel))

# -----------------------------
# Plot RMSE and AbsRel per region
regions = list(bucket_rmse.keys())

# RMSE
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
means = [np.mean(bucket_rmse[r]) for r in regions]
stds = [np.std(bucket_rmse[r]) for r in regions]
plt.bar(regions, means, yerr=stds, capsize=5, color="steelblue", alpha=0.7)
plt.ylabel("RMSE (m)")
plt.title("Depth Error (RMSE) by Range")
plt.grid(axis="y", linestyle="--", alpha=0.6)

# AbsRel
plt.subplot(1, 2, 2)
means = [np.mean(bucket_absrel[r]) for r in regions]
stds = [np.std(bucket_absrel[r]) for r in regions]
plt.bar(regions, means, yerr=stds, capsize=5, color="darkorange", alpha=0.7)
plt.ylabel("AbsRel")
plt.title("Depth Error (AbsRel) by Range")
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
