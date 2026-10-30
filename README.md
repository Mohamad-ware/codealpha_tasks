# Task 3 – Handwritten Character Recognition

Recognises handwritten digits (MNIST) and letters (EMNIST) with a Convolutional Neural Network,
and extends to whole-word recognition with a CRNN (CNN + BiLSTM + CTC loss).

## Files

| File | Purpose |
|------|---------|
| `common.py` | Dataset loading (MNIST / EMNIST splits), `CharCNN`, `CRNN`, image preprocessing and character segmentation |
| `train_cnn.py` | Trains the CNN on single characters; saves the best checkpoint, training curves, sample predictions and the most-confused classes |
| `predict.py` | Classifies your own images (one character, or a line of separated characters with `--line`) |
| `draw_app.py` | Tkinter drawing pad: draw with the mouse, the model guesses live |
| `train_crnn.py` | Sequence model: reads whole digit/letter strings without segmentation, reports word accuracy and character error rate (CER) |

## Setup

```bash
pip install -r requirements.txt
```

Datasets download automatically into `data/` on first run.

## Usage

```bash
# 1. Single characters
python train_cnn.py --dataset mnist --epochs 5             # digits 0-9
python train_cnn.py --dataset emnist-letters --epochs 10   # letters A-Z
python train_cnn.py --dataset emnist-balanced --epochs 10  # 47 digits + letters

# 2. Predict
python predict.py --model models/mnist_cnn.pt digit.png
python predict.py --model models/emnist-letters_cnn.pt --line word.png
python draw_app.py --model models/mnist_cnn.pt

# 3. Word / sequence recognition (CRNN + CTC)
python train_crnn.py --dataset mnist --epochs 5
python train_crnn.py --dataset emnist-letters --epochs 10
```

Outputs: checkpoints in `models/`, plots in `outputs/`.

## Approach

**Image processing** – user images are converted to the MNIST format: grayscale, ink inverted to
white-on-black, noise thresholded, cropped to the ink bounding box, fitted into a 20×20 box and
centred on a 28×28 canvas. EMNIST images are transposed to upright orientation.

**CharCNN** – three convolutional blocks (Conv-BN-ReLU ×2, MaxPool, Dropout) with 32/64/128
filters, followed by a 256-unit dense layer and softmax output. Trained with AdamW, a one-cycle
learning-rate schedule, label smoothing and random affine augmentation (rotation, shift, scale,
shear). Expected test accuracy: ~99.5% on MNIST, ~94% on EMNIST letters, ~90% on EMNIST balanced.

**CRNN** – a CNN collapses each 32-pixel-high image to a sequence of column features, a 2-layer
bidirectional LSTM models context left to right, and CTC loss aligns the per-column predictions
with the target text, so no character segmentation is needed. Training words are synthesised on
the fly by stitching random MNIST/EMNIST characters with random spacing, overlap and vertical
jitter. Decoding uses greedy best-path CTC. To move to real handwritten words or sentences, swap
`SyntheticWords` for a dataset such as IAM Handwriting and widen the alphabet.
