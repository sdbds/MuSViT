"""COCO-format detection dataset for MuSViT object-detection training.

Reads a standard COCO ``instances`` JSON (``images`` / ``annotations`` /
``categories``) plus an images directory, and yields the ``(image, target)``
pairs torchvision detection models expect:

- ``image``  - a ``float`` tensor in ``[0, 1]`` at the *original* resolution
  (the model's ``GeneralizedRCNNTransform`` rescales it to the fixed input size).
- ``target`` - ``{"boxes": xyxy float tensor, "labels": int64 tensor,
  "image_id": int64 tensor}``. Labels are contiguous ids ``1..N``; id ``0`` is
  the background class Faster R-CNN reserves.

DeepScores and most music-object-detection datasets can be exported to this
format (COCO converters are widely available); point ``--train_ann`` /
``--val_ann`` at the resulting JSON files.
"""

import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def collate_fn(batch):
    """Detection collate: keep images and targets as parallel tuples."""
    return tuple(zip(*batch))


def _normalize_categories(raw):
    """Accept COCO's list-of-dicts or a ``{id: {...}}`` mapping (DeepScores-style)."""
    if isinstance(raw, dict):
        cats = [{"id": int(k), "name": v.get("name", str(k)) if isinstance(v, dict) else str(v)}
                for k, v in raw.items()]
    else:
        cats = [dict(c) for c in raw]
    return sorted(cats, key=lambda c: int(c["id"]))


class CocoDetectionDataset(Dataset):
    """Minimal COCO-style detection dataset (no ``pycocotools`` dependency)."""

    def __init__(self, images_dir, ann_file):
        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images_dir = Path(images_dir)
        self.images = {img["id"]: img for img in coco["images"]}

        # Map COCO category ids -> contiguous labels 1..N (0 == background).
        cats = _normalize_categories(coco["categories"])
        self.cat_id_to_label = {int(c["id"]): i + 1 for i, c in enumerate(cats)}
        self.label_to_name = {i + 1: c["name"] for i, c in enumerate(cats)}
        self.num_classes = len(cats) + 1  # + background

        anns_by_img = defaultdict(list)
        for ann in coco["annotations"]:
            anns_by_img[ann["image_id"]].append(ann)
        self.anns_by_img = anns_by_img

        self.ids = list(self.images.keys())
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.images[img_id]
        file_name = info.get("file_name") or info.get("filename")
        image = Image.open(self.images_dir / file_name).convert("RGB")

        boxes, labels = [], []
        for ann in self.anns_by_img.get(img_id, []):
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]  # COCO xywh
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])  # -> xyxy
            labels.append(self.cat_id_to_label[int(ann["category_id"])])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }
        return self.to_tensor(image), target
