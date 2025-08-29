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

num_rows = 6
num_blocks_per_row = 2   # rgb | pred | gt repeated
num_samples = num_rows * num_blocks_per_row

# Exclude first 400 and last 100
valid_range = range(400, len(rgb_files) - 100)

# Pick equally spaced indices across the valid range
indices = np.linspace(valid_range.start, valid_range.stop -
                      1, num_samples, dtype=int).tolist()

fig, axes = plt.subplots(num_rows, num_blocks_per_row *
                         4, figsize=(18, 2.5 * num_rows))

# Column titles (repeat across 3 blocks)
block_titles = ["RGB", "Predicted Depth", "D455 Depth", "Abs Diff"]
for b in range(num_blocks_per_row):
    for j, title in enumerate(block_titles):
        col = b * 4 + j
        axes[0, col].set_title(title, fontsize=14, weight="bold")

time_list = []
for row in range(num_rows):
    for b in range(num_blocks_per_row):
        idx = indices[row * num_blocks_per_row + b]

        rgb_path = rgb_files[idx]
        depth_path = depth_files[idx]

        rgb_img = Image.open(rgb_path).convert("RGB")
        depth_gt = np.array(Image.open(depth_path))

        transformed_image = transform(rgb_img)
        batch = transformed_image.unsqueeze(
            0).cuda()  # Make a batch of one image

        with torch.inference_mode():
            torch.cuda.synchronize()
            start_time = time()
            result = model.whole_inference(
                batch, img_meta=None, rescale=True)
            torch.cuda.synchronize()
            end_time = time()
            time_spend = end_time - start_time
            time_list.append(time_spend)
            print(end_time - start_time)

        depth_pred_raw = result.squeeze().cpu().numpy()
        depth_gt_raw = (depth_gt) / 255 * (6.0 - 0.3) + 0.3

        # Normalize prediction [0.3, 6.0] → [0, 255]
        depth_pred = np.clip(depth_pred_raw, 0.3, 6.0)
        depth_pred = ((depth_pred - 0.3) / (6.0 - 0.3)) * 255
        # depth_pred = depth_pred.astype("uint8")

        # Column offsets
        col_base = b * 4

        # RGB
        axes[row, col_base + 0].imshow(rgb_img)
        axes[row, col_base + 0].axis("off")

        # Pred
        axes[row, col_base + 1].imshow(depth_pred_raw, cmap="inferno")
        axes[row, col_base + 1].axis("off")

        # GT
        axes[row, col_base + 2].imshow(depth_gt_raw, cmap="inferno")
        axes[row, col_base + 2].axis("off")

        # Diff
        # diff = abs(depth_pred_raw - depth_gt_raw)

        diff = abs(depth_pred - depth_gt)

        print("diff min", np.min(diff), np.max(diff))

        # 2) Set diff to 0 where GT is invalid (== 0.0)
        diff[depth_gt == 0] = 0.0

        axes[row, col_base + 3].imshow(diff, cmap="inferno")
        axes[row, col_base + 3].axis("off")

print(np.mean(np.asarray(time_list)))
# Layout
plt.subplots_adjust(
    top=0.92, bottom=0.05, left=0.03, right=0.97, wspace=0.05, hspace=0.1
)
plt.show()
