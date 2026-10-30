"""Extension to word/sequence recognition: a CRNN (CNN + BiLSTM) trained with CTC loss.

There is no segmentation step: the network reads the whole image left to right and CTC aligns
its per-column predictions with the target string. Training words are synthesised on the fly by
stitching together random MNIST digits / EMNIST letters with random spacing and overlap.

Examples:
    python train_crnn.py --dataset mnist --epochs 5             # reads digit strings, e.g. "40719"
    python train_crnn.py --dataset emnist-letters --epochs 10   # reads letter strings, e.g. "HELLO"
"""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import CRNN, DATA_DIR, DATASETS, get_device

HEIGHT, MIN_LEN, MAX_LEN = 32, 3, 8
WIDTH = MAX_LEN * 28  # fixed canvas width -> 56 CTC time steps


class SyntheticWords(Dataset):
    """Random strings of handwritten characters rendered into one 32 x WIDTH image."""

    def __init__(self, name, train, size, seed=None):
        cls, kwargs = DATASETS[name]
        base = cls(DATA_DIR, train=train, download=True, **kwargs)
        imgs = base.data.numpy()
        if name.startswith("emnist"):
            imgs = imgs.transpose(0, 2, 1)  # same fix as fix_emnist_orientation, vectorised
        labels = base.targets.numpy()
        if name == "mnist":
            self.alphabet = "0123456789"
            chars = labels
        elif name == "emnist-letters":
            self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            chars = labels - 1  # EMNIST letters labels are 1..26
        else:
            self.alphabet = "".join(base.classes)
            chars = labels
        self.by_char = [imgs[chars == c] for c in range(len(self.alphabet))]
        self.size, self.seed = size, seed

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        rng = random.Random(self.seed * 1_000_003 + idx) if self.seed is not None else random
        length = rng.randint(MIN_LEN, MAX_LEN)
        text = [rng.randrange(len(self.alphabet)) for _ in range(length)]
        canvas = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        x, placed = rng.randint(0, 6), []
        for c in text:
            pool = self.by_char[c]
            glyph = pool[rng.randrange(len(pool))].astype(np.float32)
            glyph = glyph[:, glyph.any(0)] if glyph.any() else glyph  # trim empty columns
            w = glyph.shape[1]
            if x + w > WIDTH:
                break  # out of room: label only what was actually drawn
            y = rng.randint(0, HEIGHT - 28)
            canvas[y:y + 28, x:x + w] = np.maximum(canvas[y:y + 28, x:x + w], glyph)
            placed.append(c)
            x = max(x + w + rng.randint(-3, 6), 0)  # slight overlap or gap, like real handwriting
        img = torch.from_numpy(canvas / 255.0).unsqueeze(0)
        return (img - 0.5) / 0.5, torch.tensor([c + 1 for c in placed], dtype=torch.long)  # 0 = blank


def collate(batch):
    imgs, targets = zip(*batch)
    lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return torch.stack(imgs), torch.cat(targets), lengths


def greedy_decode(log_probs, alphabet):
    """Best-path CTC decoding: argmax per step, merge repeats, drop blanks."""
    out = []
    for seq in log_probs.argmax(2).T.tolist():
        chars, prev = [], 0
        for k in seq:
            if k != prev and k != 0:
                chars.append(alphabet[k - 1])
            prev = k
        out.append("".join(chars))
    return out


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def split_targets(flat, lengths, alphabet):
    texts, i = [], 0
    for n in lengths.tolist():
        texts.append("".join(alphabet[k - 1] for k in flat[i:i + n].tolist()))
        i += n
    return texts


@torch.no_grad()
def evaluate(model, loader, device, alphabet):
    model.eval()
    exact = total = errs = chars = 0
    for x, y, lengths in loader:
        preds = greedy_decode(model(x.to(device)).cpu(), alphabet)
        for p, t in zip(preds, split_targets(y, lengths, alphabet)):
            exact += p == t
            errs += edit_distance(p, t)
            chars += len(t)
            total += 1
    return exact / total, errs / chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "emnist-letters"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--train-size", type=int, default=30000, help="synthetic words per epoch")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = get_device()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    train_ds = SyntheticWords(args.dataset, train=True, size=args.train_size)
    test_ds = SyntheticWords(args.dataset, train=False, size=2000, seed=123)  # fixed test words
    alphabet = train_ds.alphabet
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=256, collate_fn=collate)

    model = CRNN(len(alphabet)).to(device)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_dl))
    print(f"CRNN on synthetic {args.dataset} words, alphabet={alphabet!r}, "
          f"{sum(p.numel() for p in model.parameters()):,} params, device={device}")

    ckpt_path = os.path.join(args.out, f"{args.dataset}_crnn.pt")
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y, lengths in train_dl:
            log_probs = model(x.to(device))
            input_lengths = torch.full((x.size(0),), log_probs.size(0), dtype=torch.long)
            loss = ctc(log_probs.cpu(), y, input_lengths, lengths)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        word_acc, cer = evaluate(model, test_dl, device, alphabet)
        print(f"epoch {epoch:2d}/{args.epochs}  CTC loss {running / len(train_dl):.4f}  "
              f"test word acc {word_acc:.4f}  CER {cer:.4f}")
        if word_acc >= best:
            best = word_acc
            torch.save({"state_dict": model.state_dict(), "alphabet": alphabet}, ckpt_path)
    print(f"\nBest word accuracy: {best:.4f} -> saved {ckpt_path}")

    # Show some test words and what the model read
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    x, y, lengths = collate([test_ds[i] for i in range(8)])
    preds = greedy_decode(model.eval()(x.to(device)).detach().cpu(), alphabet)
    truths = split_targets(y, lengths, alphabet)
    fig, axes = plt.subplots(8, 1, figsize=(8, 10))
    for ax, img, p, t in zip(axes, x, preds, truths):
        ax.imshow(img[0], cmap="gray"); ax.axis("off")
        ax.set_title(f"read: {p}   true: {t}", color="green" if p == t else "red", fontsize=11)
    fig.tight_layout(); fig.savefig(f"outputs/{args.dataset}_crnn_samples.png", dpi=110); plt.close(fig)
    print(f"Samples written to outputs/{args.dataset}_crnn_samples.png")


if __name__ == "__main__":
    main()
