import os
import json
import numpy as np
from torch.utils.data import Dataset
import hashlib


"""
dataset.py
----------
Loads keypoint sequences stored in cache/<hash>.npy based on Markup.json,
assigns class labels for each gloss, and returns (sequence, label) pairs.

Each sequence is a float32 array of shape (T, 225), where T varies.
A collate function pads sequences inside a batch for LSTM training.
"""


def stable_hash(s: str) -> str:
    """
    Deterministic hash used to identify cache files.
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


class ASLDataset(Dataset):
    """
    PyTorch dataset for ASL gloss classification.

    Parameters
    ----------
    json_path : str
        Path to Markup.json.
    split : str
        One of ["train", "val", "test", "all"].
    """

    def __init__(self, json_path="Markup.json", split="train"):
        if split not in ["train", "val", "test", "all"]:
            raise ValueError("split must be one of: train, val, test, all")

        self.json_path = json_path
        self.split = split

        # Load dataset metadata
        self.data = self._load_json(json_path)

        # Build gloss → ID map
        self.gloss_to_id = {entry["gloss"]: idx for idx, entry in enumerate(self.data)}
        self.id_to_gloss = {v: k for k, v in self.gloss_to_id.items()}

        # Collect instances matching the requested split
        self.samples = self._collect_samples(split)

        os.makedirs("cache", exist_ok=True)

    # -----------------------------------------------------
    # Helper methods
    # -----------------------------------------------------

    def _load_json(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")

        with open(path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Markup.json must contain a list of gloss entries.")
        return data

    def _collect_samples(self, split):
        samples = []

        for gloss_entry in self.data:
            gloss = gloss_entry["gloss"]

            for inst in gloss_entry.get("instances", []):
                inst_split = inst.get("split", "train")

                if split == "all" or inst_split == split:
                    samples.append({
                        "gloss": gloss,
                        "url": inst["url"],
                        "bbox": inst["bbox"],
                        "frame_start": inst["frame_start"],
                        "frame_end": inst["frame_end"],
                        "fps": inst["fps"]
                    })

        return samples

    def _load_cached_sequence(self, sample):
        """
        Loads or extracts keypoints for one instance.
        """
        url = sample["url"]
        cache_path = f"cache/{stable_hash(url)}.npy"

        # If cache exists → load it
        if os.path.exists(cache_path):
            arr = np.load(cache_path)
            return arr.astype(np.float32)

        # Cache missing → try extraction
        if not os.path.exists(url):
            print(f"[WARN] Video not found: {url}")
            return np.zeros((1, 225), dtype=np.float32)

        try:
            from extract_keypoints import extract_keypoints_from_video

            arr = extract_keypoints_from_video(
                url,
                sample["bbox"],
                sample["frame_start"],
                sample["frame_end"],
                sample["fps"]
            )

            if arr is None:
                arr = np.zeros((1, 225), dtype=np.float32)

            np.save(cache_path, arr)
            return arr.astype(np.float32)

        except Exception as e:
            print(f"[ERROR] Extraction failed for {url}: {e}")
            return np.zeros((1, 225), dtype=np.float32)

    # -----------------------------------------------------
    # PyTorch Dataset interface
    # -----------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq = self._load_cached_sequence(sample)
        label = self.gloss_to_id[sample["gloss"]]
        return seq, label


# ---------------------------------------------------------
# Collate Function for DataLoader
# ---------------------------------------------------------

def asl_collate_fn(batch):
    """
    Pads variable-length sequences inside a batch.

    Parameters
    ----------
    batch : list of (seq, label)

    Returns
    -------
    X : float32 array of shape (B, T_max, 225)
    y : int64 array of shape (B,)
    """

    sequences, labels = zip(*batch)

    max_len = max(seq.shape[0] for seq in sequences)
    dim = sequences[0].shape[1]

    padded = []
    for seq in sequences:
        pad_len = max_len - seq.shape[0]
        if pad_len > 0:
            pad = np.zeros((pad_len, dim), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
        padded.append(seq)

    X = np.stack(padded).astype(np.float32)
    y = np.array(labels, dtype=np.int64)

    return X, y