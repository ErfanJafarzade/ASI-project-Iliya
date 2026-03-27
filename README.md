# ASL-100: Lightweight CPU-Based American Sign Language Recognition Model

This project implements a full training pipeline for a lightweight American Sign Language (ASL) recognition model capable of learning up to 100 glosses from video. The pipeline is optimized for CPU-only systems such as Intel i7-4790 and avoids heavy video models in favor of skeletal keypoint extraction and sequence modeling.

The dataset structure is defined through a `Markup.json` file containing per-gloss video instances, bounding boxes, frame windows, and signer IDs. All keypoints are extracted with MediaPipe Holistic and cached locally to reduce processing time during training.

---

## Features

- Uses bounding-box–cropped frames based on the dataset JSON file.
- Extracts pose and hand landmarks using MediaPipe Holistic.
- Converts videos into fixed-length keypoint sequences.
- Employs a compact bidirectional LSTM classifier.
- Supports training, validation, and inference.
- CPU-friendly and suitable for individual research environments.

---

## Project Structure
project/
│── Markup.json
│── extract_keypoints.py
│── dataset.py
│── model.py
│── train.py
│── infer.py
│── utils.py
└── requirements.txt
