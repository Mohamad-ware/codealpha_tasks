"""Shared pieces: datasets, models and image preprocessing for handwritten character recognition."""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import datasets, transforms

DATA_DIR = "data"
MEAN, STD = 0.1307, 0.3081  # MNIST statistics, close enough for EMNIST too

# Supported datasets: name -> (torchvision class, extra kwargs)
DATASETS = {
    "mnist": (datasets.MNIST, {}),                              # 10 digits
    "emnist-letters": (datasets.EMNIST, {"split": "letters"}),  # 26 letters (upper+lower merged)
    "emnist-balanced": (datasets.EMNIST, {"split": "balanced"}),  # 47 digits + letters
    "emnist-byclass": (datasets.EMNIST, {"split": "byclass"}),    # 62 digits + A-Z + a-z
}


def fix_emnist_orientation(img):
    """EMNIST images are stored transposed relative to MNIST; flip them upright."""
    return img.transpose(Image.TRANSPOSE)


def build_transform(name, train):
    steps = []
    if name.startswith("emnist"):
        steps.append(transforms.Lambda(fix_emnist_orientation))
    if train:
        steps.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=8))
    steps += [transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))]
    return transforms.Compose(steps)


def load_dataset(name, train, augment=None):
    """Return (dataset, class_names). Downloads on first use."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(DATASETS)}")
    cls, kwargs = DATASETS[name]
    tf = build_transform(name, train if augment is None else augment)
    ds = cls(DATA_DIR, train=train, download=True, transform=tf, **kwargs)
    classes = list(ds.classes)
    if name == "mnist":
        classes = [c.split(" - ")[0] for c in classes]  # "0 - zero" -> "0"
    return ds, classes


class CharCNN(nn.Module):
    """VGG-style CNN for 28x28 grayscale characters (~99.5% on MNIST, ~94% on EMNIST letters)."""

    def __init__(self, num_classes):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout(0.25),
            )

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))  # 28 -> 14 -> 7 -> 3
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CRNN(nn.Module):
    """CNN feature extractor + bidirectional LSTM + CTC head, for reading whole words/lines.

    Input: (B, 1, 32, W). Output: (T=W/4, B, num_classes+1) log-probs, index 0 is the CTC blank.
    """

    def __init__(self, num_classes, hidden=128):
        super().__init__()

        def conv(cin, cout):
            return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]

        self.cnn = nn.Sequential(
            *conv(1, 64), nn.MaxPool2d(2),            # H16, W/2
            *conv(64, 128), nn.MaxPool2d(2),          # H8,  W/4
            *conv(128, 256), *conv(256, 256), nn.MaxPool2d((2, 1)),  # H4
            *conv(256, 256), nn.MaxPool2d((2, 1)),    # H2
            nn.AdaptiveAvgPool2d((1, None)),          # H1
        )
        self.rnn = nn.LSTM(256, hidden, num_layers=2, bidirectional=True, dropout=0.25)
        self.fc = nn.Linear(hidden * 2, num_classes + 1)

    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(2, 0, 1)  # (T, B, C)
        out, _ = self.rnn(f)
        return self.fc(out).log_softmax(2)


def preprocess_char(img, size=28, box=20):
    """Turn an arbitrary photo/scan of one character into an MNIST-style 28x28 tensor.

    Grayscale -> make ink white on black -> crop to the ink -> fit into a 20x20 box -> centre on 28x28.
    """
    img = ImageOps.grayscale(img)
    arr = np.asarray(img, dtype=np.float32)
    if arr.mean() > 127:          # dark ink on light paper: invert
        arr = 255 - arr
    arr[arr < 50] = 0             # drop paper noise
    ys, xs = np.nonzero(arr)
    if len(xs):
        arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = arr.shape
    scale = box / max(h, w)
    small = Image.fromarray(arr.astype(np.uint8)).resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(small, ((size - small.width) // 2, (size - small.height) // 2))
    t = transforms.ToTensor()(canvas)
    return transforms.Normalize((MEAN,), (STD,))(t)


def segment_characters(img, min_gap=2, min_width=3):
    """Split a line of separated handwritten characters into crops using vertical ink projection."""
    arr = np.asarray(ImageOps.grayscale(img), dtype=np.float32)
    if arr.mean() > 127:
        arr = 255 - arr
    ink = arr > 80
    cols = ink.any(axis=0)
    crops, start, gap = [], None, 0
    for x, has_ink in enumerate(np.append(cols, False)):
        if has_ink:
            if start is None:
                start = x
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap or x == len(cols):
                end = x - gap + 1
                if end - start >= min_width:
                    crops.append(Image.fromarray(arr[:, start:end].astype(np.uint8)))
                start, gap = None, 0
    return crops


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
