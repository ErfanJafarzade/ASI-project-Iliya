import os
import json
import numpy as np
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from extract_keypoints import extract_keypoints_from_video


"""
batch_extract.py
----------------
Reads Markup.json, loops through every video entry,
runs keypoint extraction using MediaPipe, and caches the
results inside:

    cache/<hash(video_path)>.npy

Multithreaded extraction significantly reduces preprocessing time.
"""


def stable_hash(s: str) -> str:
    """
    Create a safe deterministic hash for filenames.
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def process_one(sample):
    """
    Handles extraction for a single video instance.
    """

    url = sample["url"]                        # local path to video
    bbox = sample["bbox"]
    fs = sample["frame_start"]
    fe = sample["frame_end"]
    fps = sample["fps"]

    # Convert to absolute hashed cache filename
    cache_path = f"cache/{stable_hash(url)}.npy"

    # Skip if already processed
    if os.path.exists(cache_path):
        return f"SKIP {url}"

    # Check video availability
    if not os.path.exists(url):
        return f"MISS {url}"

    # Extract keypoints
    seq = extract_keypoints_from_video(url, bbox, fs, fe, fps)

    if seq is None:
        seq = np.zeros((1, 225), dtype=np.float32)

    np.save(cache_path, seq)
    return f"DONE {url}"


def batch_extract(json_path="Markup.json", workers=6):
    """
    Batch-process all videos with multithreading.

    Parameters
    ----------
    json_path : str
        Path to dataset JSON file.
    workers : int
        Number of CPU threads to use.
    """

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} not found.")

    os.makedirs("cache", exist_ok=True)

    # Load dataset
    with open(json_path, "r") as f:
        data = json.load(f)

    # Collect all instances
    samples = []
    for gloss_entry in data:
        for inst in gloss_entry.get("instances", []):
            samples.append(inst)

    print(f"[INFO] Found {len(samples)} video instances.")
    print(f"[INFO] Starting extraction with {workers} threads...\n")

    # Multithreaded extraction
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_one, s) for s in samples]

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"[{i}/{len(futures)}] {result}")

    print("\n[DONE] Keypoint extraction complete.")


if __name__ == "__main__":
    batch_extract()