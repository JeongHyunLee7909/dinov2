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
import re
import shutil

# -------------------------------
# Sorting key for npz files
# -------------------------------


def _fallback_time_key(p):
    # fallback: use file modification time
    return os.path.getmtime(p)


def extract_seq_time_key(p):
    """
    Return a tuple key for sorting:
      - If matches 'data_{token}_{time}.npz' and token contains sequence/seq number:
          (0, seq_num, time_float)
      - Else fallback:
          (1, fallback_time)
    """
    fname = os.path.basename(p)
    m = re.match(r"^data_([A-Za-z0-9_]+)_(\d+\.\d+)\.npz$", fname)
    if m:
        token = m.group(1)
        time_str = m.group(2)
        m_id = re.search(r"(?:sequence|seq)(\d+)", token, flags=re.IGNORECASE)
        if m_id:
            seq_id = int(m_id.group(1))
            try:
                t = float(time_str)
                return (0, seq_id, t)
            except ValueError:
                pass
    return (1, _fallback_time_key(p))


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


BACKBONE_SIZE = "giant"  # in ("small", "base", "large" or "giant")


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


transform = make_depth_transform()

# -------------------------------
# Paths
# -------------------------------
base_dir = Path(
    "/home/joshualee/navigation_ws/multidim-costmap/annotated_wo_sam_npzs")
train_dir = base_dir / "train"
val_dir = base_dir / "validation"

npz_files = sorted(
    list(train_dir.glob("*.npz")) + list(val_dir.glob("*.npz")),
    key=extract_seq_time_key
)

print(f"Found {len(npz_files)} npz files.")

# -------------------------------
# Config
# -------------------------------
visualize = False   # << change to False to skip plotting
num_rows = 2
num_cols = 2

time_list = []
vis_indices = []

if visualize:
    num_samples = num_rows * num_cols
    vis_indices = np.linspace(0, len(npz_files) - 1,
                              num_samples, dtype=int).tolist()
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 2.5 * num_rows))

# -------------------------------
# Process all npz files
# -------------------------------

total_files_num = len(npz_files)
counter = 0
for file_idx, npz_path in enumerate(npz_files):
    print(f"Start processing {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    image = data["image"].astype(np.uint8)

    # Convert to PIL for transforms
    rgb_img = Image.fromarray(image)
    transformed = transform(rgb_img)
    batch = transformed.unsqueeze(0).cuda()

    # Run DINO inference
    with torch.inference_mode():
        torch.cuda.synchronize()
        start_time = time()
        result = model.whole_inference(batch, img_meta=None, rescale=True)
        torch.cuda.synchronize()
        end_time = time()
        time_list.append(end_time - start_time)

    if len(time_list) == 100:
        print(f"Avg inference time: {np.mean(time_list):.3f} sec")

    depth_dino_raw = result.squeeze().cpu().numpy()
    depth_dino = np.clip(depth_dino_raw, 0.3, 6.0)
    depth_dino = ((depth_dino - 0.3) / (6.0 - 0.3)) * 255

    if depth_dino.ndim == 2:
        depth_dino = depth_dino[..., None]

    # Save back depth_dino safely
    out_data = dict(data)
    if "depth_dino" not in out_data:
        out_data["depth_dino"] = depth_dino

        tmp_path = str(npz_path).replace(".npz", ".tmp.npz")
        np.savez(tmp_path, **out_data)
        shutil.move(tmp_path, npz_path)

        print(f"Saved depth_dino into {npz_path}")
    else:
        print(f"Skipped {npz_path} (already has depth_dino)")

    counter += 1
    print(f"Index {counter} out of {total_files_num}")

    # -------------------------------
    # Visualization (optional)
    # -------------------------------
    if visualize and file_idx in vis_indices:
        pos = vis_indices.index(file_idx)
        row, col = divmod(pos, num_cols)

        # RGB
        axes[row, 0].imshow(rgb_img)
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("RGB", fontsize=14, weight="bold")

        # DINO Depth
        axes[row, 1].imshow(depth_dino.squeeze(), cmap="viridis")
        axes[row, 1].axis("off")
        if row == 0:
            axes[row, 1].set_title("DINO Depth", fontsize=14, weight="bold")


if visualize:
    plt.subplots_adjust(
        top=0.92, bottom=0.05, left=0.03, right=0.97, wspace=0.05, hspace=0.1
    )
    plt.show()
