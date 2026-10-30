"""Tiny Tkinter drawing pad: draw a character with the mouse and the CNN guesses it live.

    python draw_app.py --model models/mnist_cnn.pt
"""
import argparse
import tkinter as tk

from PIL import Image, ImageDraw

from common import get_device
from predict import classify, load_model

SIZE, BRUSH = 280, 11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/mnist_cnn.pt")
    args = ap.parse_args()
    device = get_device()
    model, classes = load_model(args.model, device)

    root = tk.Tk()
    root.title("Handwritten Character Recognition")
    canvas = tk.Canvas(root, width=SIZE, height=SIZE, bg="black", cursor="cross")
    canvas.pack(padx=10, pady=10)
    result = tk.Label(root, text="Draw a character", font=("Segoe UI", 16))
    result.pack()
    image = Image.new("L", (SIZE, SIZE), 0)
    pen = ImageDraw.Draw(image)

    def paint(event):
        x, y = event.x, event.y
        canvas.create_oval(x - BRUSH, y - BRUSH, x + BRUSH, y + BRUSH, fill="white", outline="white")
        pen.ellipse([x - BRUSH, y - BRUSH, x + BRUSH, y + BRUSH], fill=255)

    def predict(_=None):
        if image.getbbox() is None:
            return
        _, top = classify(model, classes, [image], device)[0]
        result.config(text="   ".join(f"{lbl} {p:.0%}" for lbl, p in top))

    def clear():
        canvas.delete("all")
        pen.rectangle([0, 0, SIZE, SIZE], fill=0)
        result.config(text="Draw a character")

    canvas.bind("<B1-Motion>", paint)
    canvas.bind("<ButtonRelease-1>", predict)
    tk.Button(root, text="Clear", command=clear, width=12).pack(pady=8)
    root.mainloop()


if __name__ == "__main__":
    main()
