import torch
import numpy as np
from model import ASLLSTM
from extract_keypoints import extract_keypoints_from_video
from dataset import ASLDataset


"""
infer.py
--------
Runs inference on a single ASL video clip using the trained model.

Usage example:

    python infer.py

Inside the script, set:
    video_path = "videos/your_clip.mp4"
    bbox = [x1, y1, x2, y2]

Or import the function infer() from another file.
"""


def infer(video_path, bbox, frame_start=0, frame_end=-1, fps=30,
          json_path="Markup.json", model_path="asl_model.pth"):

    # -----------------------------------------------------
    # Load label map from dataset
    # -----------------------------------------------------
    ds = ASLDataset(json_path, split="all")
    id_to_gloss = ds.id_to_gloss
    num_classes = len(id_to_gloss)

    # -----------------------------------------------------
    # Extract keypoints for this video
    # -----------------------------------------------------
    seq = extract_keypoints_from_video(
        video_path,
        bbox=bbox,
        frame_start=frame_start,
        frame_end=frame_end,
        fps=fps
    )

    if seq is None:
        raise ValueError("No keypoints extracted from video.")

    # Convert to tensor
    X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, T, 225)

    # -----------------------------------------------------
    # Load trained model
    # -----------------------------------------------------
    model = ASLLSTM(
        input_dim=225,
        hidden_dim=256,
        num_layers=2,
        num_classes=num_classes
    ).cpu()

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # -----------------------------------------------------
    # Forward pass
    # -----------------------------------------------------
    with torch.no_grad():
        logits = model(X)
        pred_id = logits.argmax(dim=1).item()

    predicted_gloss = id_to_gloss[pred_id]
    return predicted_gloss


# ---------------------------------------------------------
# Stand-alone example usage
# ---------------------------------------------------------

if __name__ == "__main__":
    # You must set your test video here:

    video_path = "videos/test_clip.mp4"   # <-- replace with your video file
    bbox = [100, 50, 400, 450]            # <-- replace with your bounding box

    print("[INFO] Running inference...")
    pred = infer(video_path, bbox)
    print("Predicted gloss:", pred)