import os
import random
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, DPTForDepthEstimation
import time

# -------------------------------
# Config
# -------------------------------
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


# -------------------------------
# Load model once
# -------------------------------
image_processor = AutoImageProcessor.from_pretrained(
    "facebook/dpt-dinov2-small-nyu")
model = DPTForDepthEstimation.from_pretrained("facebook/dpt-dinov2-small-nyu")

# -------------------------------
# Inference + Visualization
# -------------------------------

num_rows = 6
num_blocks_per_row = 3   # rgb | pred | gt repeated
num_samples = num_rows * num_blocks_per_row

# Exclude first 400 and last 100
valid_range = range(400, len(rgb_files) - 100)

# Pick equally spaced indices across the valid range
indices = np.linspace(valid_range.start, valid_range.stop -
                      1, num_samples, dtype=int).tolist()

fig, axes = plt.subplots(num_rows, num_blocks_per_row *
                         3, figsize=(18, 2.5 * num_rows))

# Column titles (repeat across 3 blocks)
block_titles = ["RGB", "Predicted Depth", "D455 Depth"]
for b in range(num_blocks_per_row):
    for j, title in enumerate(block_titles):
        col = b * 3 + j
        axes[0, col].set_title(title, fontsize=14, weight="bold")

time_list = []
for row in range(num_rows):
    for b in range(num_blocks_per_row):
        idx = indices[row * num_blocks_per_row + b]

        rgb_path = rgb_files[idx]
        depth_path = depth_files[idx]

        rgb_img = Image.open(rgb_path).convert("RGB")
        depth_gt = np.array(Image.open(depth_path))

        # Inference
        inputs = image_processor(images=rgb_img, return_tensors="pt")
        with torch.no_grad():
            start_time = time.time()
            outputs = model(**inputs)
            time_list.append(time.time() - start_time)
            predicted_depth = outputs.predicted_depth

        # Resize predicted depth to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=rgb_img.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        depth_pred = prediction.squeeze().cpu().numpy()

        # Normalize prediction [0.3, 6.0] → [0, 255]
        depth_pred = np.clip(depth_pred, 0.3, 6.0)
        depth_pred = ((depth_pred - 0.3) / (6.0 - 0.3)) * 255
        depth_pred = depth_pred.astype("uint8")

        # Column offsets
        col_base = b * 3

        # RGB
        axes[row, col_base + 0].imshow(rgb_img)
        axes[row, col_base + 0].axis("off")

        # Pred
        axes[row, col_base + 1].imshow(depth_pred, cmap="inferno")
        axes[row, col_base + 1].axis("off")

        # GT
        axes[row, col_base + 2].imshow(depth_gt, cmap="inferno")
        axes[row, col_base + 2].axis("off")

# Layout
plt.subplots_adjust(
    top=0.92, bottom=0.05, left=0.03, right=0.97, wspace=0.05, hspace=0.1
)
plt.show()
