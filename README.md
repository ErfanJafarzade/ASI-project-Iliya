# ASL-100: Lightweight CPU-Based American Sign Language Recognition System

This repository contains a complete processing and modeling pipeline for recognizing up to 100 American Sign Language (ASL) lexical items from video. The approach is optimized for CPU-only environments and is suitable for research, prototyping, or deployment on low-resource hardware.

The system uses MediaPipe Holistic to extract pose and hand landmarks, transforms each video into a sequence of 225-dimensional feature vectors, and trains a bidirectional LSTM classifier on these features. All dataset configuration is provided through `Markup.json`, which defines the glosses, signer instances, frame ranges, and bounding boxes.

This project is designed to be easy to reproduce end-to-end and requires only local video files and Python.

---

## 1. Features

- Works entirely on CPU hardware (e.g., Intel i7-4790).
- Extracts pose and hand keypoints using MediaPipe Holistic.
- Automatic batch preprocessing with multithreaded extraction.
- Sequence-based classification using a bidirectional LSTM.
- Clear separation of dataset, extraction, training, and inference modules.
- Fully deterministic caching system for efficient reuse of extracted features.
- Supports `train`, `val`, and `test` splits through `Markup.json`.

---

## 2. Project Structure
project/
│── Markup.json
│── videos/
│── cache/
│── extract_keypoints.py
│── batch_extract.py
│── dataset.py
│── model.py
│── train.py
│── infer.py
└── requirements.txt


### Summary of Files

| File | Purpose |
|------|---------|
| `Markup.json` | Dataset definition: glosses, instances, bounding boxes, frame ranges |
| `videos/` | Local video files used for training and inference |
| `cache/` | Automatically generated keypoint sequences (`.npy` files) |
| `extract_keypoints.py` | Extracts 225-dimensional keypoint vectors per frame |
| `batch_extract.py` | Multithreaded preprocessing of all dataset videos |
| `dataset.py` | PyTorch dataset and collation logic |
| `model.py` | Bidirectional LSTM classifier |
| `train.py` | Training loop and validation routine |
| `infer.py` | Inference on new video samples |
| `debug_visualize.py` | Renders MediaPipe landmarks for inspection |
| `requirements.txt` | Reproducible environment specification |

---

## 3. Dataset Configuration (`Markup.json`)

The dataset is defined entirely in `Markup.json`. Each gloss entry contains one or more signer instances:
Also I want to thank WLASL and their team for the finding of the videos and json file.
```json
{
  "gloss": "HELLO",
  "instances": [
    {
      "url": "videos/hello_001.mp4",
      "bbox": [100, 50, 400, 450],
      "frame_start": 10,
      "frame_end": 45,
      "fps": 30,
      "split": "train"
    }
  ]
}

