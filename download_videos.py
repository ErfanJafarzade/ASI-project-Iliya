import os
import json
import subprocess
import hashlib


"""
download_videos.py
------------------
Downloads every video referenced in Markup.json using yt-dlp
and saves them into the 'videos/' directory.

Output filenames are deterministic and collision-free:
    videos/<hash(url)>.mp4
"""


def safe_hash(s: str) -> str:
    """
    Stable hash for filenames.
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def download_one(url: str, output_path: str):
    """
    Downloads a single video with yt-dlp.
    """
    cmd = [
        "yt-dlp",
        "-f", "mp4",
        "--no-warnings",
        "-o", output_path,
        url
    ]

    print(f"[INFO] Downloading: {url}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if result.returncode != 0:
        print(f"[ERROR] Failed to download: {url}")
    else:
        print(f"[OK] Saved → {output_path}")


def download_all(json_path="Markup.json"):
    """
    Scans Markup.json and downloads all unique URLs.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} not found.")

    os.makedirs("videos", exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    seen_urls = set()

    for gloss_entry in data:
        for inst in gloss_entry.get("instances", []):
            url = inst.get("url", None)
            if not url:
                continue

            # Avoid duplicate downloading
            if url in seen_urls:
                continue
            seen_urls.add(url)

            # Output path
            filename = safe_hash(url) + ".mp4"
            output_path = os.path.join("videos", filename)

            # Skip if already downloaded
            if os.path.exists(output_path):
                print(f"[SKIP] Already exists: {output_path}")
                continue

            # Download
            download_one(url, output_path)

    print("\n[DONE] All videos checked and downloaded where needed.")


if __name__ == "__main__":
    download_all()