"""Train a CNN to recognise single handwritten characters on MNIST or EMNIST.

Examples:
    python train_cnn.py --dataset mnist --epochs 5
    python train_cnn.py --dataset emnist-letters --epochs 10
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from common import CharCNN, DATASETS, MEAN, STD, get_device, load_dataset


def run_epoch(model, loader, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1 if training else 0.0)
    total_loss = correct = seen = 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate_details(model, loader, device, num_classes):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        confusion.index_put_((y, pred), torch.ones_like(y), accumulate=True)
    return confusion


def save_plots(history, model, test_ds, classes, device, out_dir, tag):
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(epochs, history["train_loss"], label="train"); a1.plot(epochs, history["test_loss"], label="test")
    a1.set_title("Loss"); a1.set_xlabel("epoch"); a1.legend()
    a2.plot(epochs, history["train_acc"], label="train"); a2.plot(epochs, history["test_acc"], label="test")
    a2.set_title("Accuracy"); a2.set_xlabel("epoch"); a2.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{tag}_curves.png"), dpi=120); plt.close(fig)

    # Grid of test samples with predictions (red = wrong)
    model.eval()
    idx = torch.randperm(len(test_ds))[:32]
    xs = torch.stack([test_ds[i][0] for i in idx])
    ys = [test_ds[i][1] for i in idx]
    with torch.no_grad():
        preds = model(xs.to(device)).argmax(1).cpu().tolist()
    fig, axes = plt.subplots(4, 8, figsize=(12, 6.5))
    for ax, img, y, p in zip(axes.flat, xs, ys, preds):
        ax.imshow(img[0] * STD + MEAN, cmap="gray"); ax.axis("off")
        ax.set_title(f"{classes[p]} ({classes[y]})", color="green" if p == y else "red", fontsize=10)
    fig.suptitle("prediction (true label)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{tag}_samples.png"), dpi=120); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist", choices=list(DATASETS))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--limit", type=int, default=0, help="use only N training samples (quick test)")
    p.add_argument("--out", default="models")
    args = p.parse_args()

    torch.manual_seed(0)
    device = get_device()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    train_ds, classes = load_dataset(args.dataset, train=True)
    test_ds, _ = load_dataset(args.dataset, train=False)
    if args.limit:
        train_ds = Subset(train_ds, torch.randperm(len(train_ds))[:args.limit].tolist())
    print(f"{args.dataset}: {len(train_ds)} train / {len(test_ds)} test, {len(classes)} classes, device={device}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=512, num_workers=0)

    model = CharCNN(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_dl))
    print(f"CharCNN parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = {k: [] for k in ("train_loss", "train_acc", "test_loss", "test_acc")}
    best = 0.0
    ckpt_path = os.path.join(args.out, f"{args.dataset}_cnn.pt")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, device, optimizer, scheduler)
        te_loss, te_acc = run_epoch(model, test_dl, device)
        for k, v in zip(history, (tr_loss, tr_acc, te_loss, te_acc)):
            history[k].append(v)
        print(f"epoch {epoch:2d}/{args.epochs}  train loss {tr_loss:.4f} acc {tr_acc:.4f}  "
              f"test loss {te_loss:.4f} acc {te_acc:.4f}  ({time.time() - t0:.0f}s)")
        if te_acc > best:
            best = te_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes, "dataset": args.dataset}, ckpt_path)

    print(f"\nBest test accuracy: {best:.4f}  -> saved {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])

    confusion = evaluate_details(model, test_dl, device, len(classes))
    per_class = confusion.diag().float() / confusion.sum(1).clamp(min=1).float()
    valid = confusion.sum(1) > 0
    worst = sorted(((per_class[i].item(), classes[i]) for i in range(len(classes)) if valid[i]))[:5]
    print("Hardest classes:", ", ".join(f"{c}={a:.3f}" for a, c in worst))
    off = confusion.clone(); off.fill_diagonal_(0)
    top = torch.topk(off.flatten(), 5)
    print("Most common confusions (true->pred):",
          ", ".join(f"{classes[i // len(classes)]}->{classes[i % len(classes)]} x{n}"
                    for n, i in zip(top.values.tolist(), top.indices.tolist()) if n))

    save_plots(history, model, test_ds, classes, device, "outputs", args.dataset)
    print(f"Plots written to outputs/{args.dataset}_curves.png and outputs/{args.dataset}_samples.png")


if __name__ == "__main__":
    main()
