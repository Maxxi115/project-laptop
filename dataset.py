import os

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

try:
    from . import augmentations as aug
except ImportError:
    import augmentations as aug


def _pick_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(f"Missing expected columns. Available columns: {list(df.columns)}")


def _resolve_path(path_value, default_dir):
    normalized = os.path.normpath(str(path_value))
    candidate_paths = [
        normalized,
        os.path.join(".", normalized),
        os.path.join(default_dir, os.path.basename(normalized)),
    ]

    if normalized.startswith(f"data{os.sep}"):
        stripped = normalized.split(os.sep, 1)[1]
        candidate_paths.extend(
            [
                stripped,
                os.path.join(".", stripped),
                os.path.join(default_dir, os.path.basename(stripped)),
            ]
        )

    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Could not resolve path: {path_value}")


class ObjDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_size, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.image_column = _pick_column(self.df, ["image_path", "images", "image"])
        self.label_column = _pick_column(self.df, ["label_path", "labels", "label"])

        if transform is None:
            self.transform = aug.NoTransform()
        else:
            self.transform = aug.Compose(transform)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = _resolve_path(row[self.image_column], "images")
        label_path = _resolve_path(row[self.label_column], "labels")

        img = Image.open(image_path).convert("RGB")
        original_w, original_h = img.size
        image = to_tensor(img)

        boxes = []
        labels = []
        with open(label_path, encoding="utf-8") as label_file:
            for line in label_file:
                line = line.strip()
                if not line:
                    continue

                cls, xc, yc, bw, bh = map(float, line.split())
                x1 = (xc - bw / 2) * original_w
                y1 = (yc - bh / 2) * original_h
                x2 = (xc + bw / 2) * original_w
                y2 = (yc + bh / 2) * original_h
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        image, target = self.transform(image, target)
        image = image.clone()
        target["boxes"] = target["boxes"].clone()
        target["labels"] = target["labels"].clone()
        return image, target
