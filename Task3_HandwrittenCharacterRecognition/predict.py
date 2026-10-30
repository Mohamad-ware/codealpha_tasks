"""Recognise handwritten characters in your own images with a trained CNN.

Examples:
    python predict.py --model models/mnist_cnn.pt my_digit.png
    python predict.py --model models/emnist-letters_cnn.pt --line word.png   # separated letters on one line
"""
import argparse

import torch
from PIL import Image

from common import CharCNN, get_device, preprocess_char, segment_characters


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = CharCNN(len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"]


@torch.no_grad()
def classify(model, classes, crops, device, top_k=3):
    """Return [(best_label, [(label, prob), ...]), ...] for a list of PIL crops."""
    x = torch.stack([preprocess_char(c) for c in crops]).to(device)
    probs = model(x).softmax(1).cpu()
    results = []
    for row in probs:
        p, i = row.topk(min(top_k, len(classes)))
        top = [(classes[j], v) for j, v in zip(i.tolist(), p.tolist())]
        results.append((top[0][0], top))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default="models/mnist_cnn.pt")
    ap.add_argument("--line", action="store_true", help="image holds several separated characters")
    args = ap.parse_args()

    device = get_device()
    model, classes = load_model(args.model, device)
    for path in args.images:
        img = Image.open(path)
        crops = segment_characters(img) if args.line else [img]
        if not crops:
            print(f"{path}: no ink found")
            continue
        results = classify(model, classes, crops, device)
        if args.line:
            print(f"{path}: {''.join(r[0] for r in results)}")
        for k, (_, top) in enumerate(results):
            alts = "  ".join(f"{lbl}:{p:.2%}" for lbl, p in top)
            print(f"{path}{f' [char {k}]' if args.line else ''}: {alts}")


if __name__ == "__main__":
    main()
