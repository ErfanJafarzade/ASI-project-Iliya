import cv2
import mediapipe as mp
import numpy as np


"""
extract_keypoints.py
--------------------
Extracts pose + left hand + right hand keypoints from a video.

Output for each frame:
    33 pose landmarks
    21 left hand landmarks
    21 right hand landmarks
Total = 75 landmarks, each with (x, y, z)
→ 75 * 3 = 225-dimensional vector

Final output shape:
    (T, 225) float32
where T = number of processed frames
"""


# Global MediaPipe model for efficiency
mp_holistic = mp.solutions.holistic


def _extract_hand(landmarks):
    """
    Extracts 21×3 hand keypoints.
    Returns zeros if the hand is not detected.
    """
    if landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
                        dtype=np.float32)
    return np.zeros((21, 3), dtype=np.float32)


def extract_keypoints_from_video(
    video_path: str,
    bbox: list,
    frame_start: int,
    frame_end: int,
    fps: int
):
    """
    Extracts keypoints from a single video.

    Parameters
    ----------
    video_path : str
        Local path to a video file.
    bbox : list
        [x1, y1, x2, y2] crop region inside the frame.
        If None: full frame is used.
    frame_start : int
        First frame to process.
    frame_end : int
        Last frame to process. -1 means "until end".
    fps : int
        Included for consistency with dataset metadata.

    Returns
    -------
    numpy.ndarray
        Float32 array: (T, 225)
        or None if the video cannot be opened.
    """

    # Validate bounding box
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError("Bounding box must be [x1, y1, x2, y2].")
        x1, y1, x2, y2 = bbox
    else:
        x1 = y1 = x2 = y2 = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return None

    keypoints = []
    frame_idx = 0
    last_frame = frame_end if frame_end != -1 else 10**12

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx < frame_start:
                frame_idx += 1
                continue

            if frame_idx > last_frame:
                break

            # Crop if necessary
            if bbox is not None:
                frame = frame[y1:y2, x1:x2]

            # Convert BGR → RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run inference
            results = holistic.process(rgb)

            # Pose (33 landmarks)
            if results.pose_landmarks:
                pose = np.array(
                    [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
                    dtype=np.float32
                )
            else:
                pose = np.zeros((33, 3), dtype=np.float32)

            # Hands (21 landmarks)
            left_hand = _extract_hand(results.left_hand_landmarks)
            right_hand = _extract_hand(results.right_hand_landmarks)

            # Combine → (75, 3)
            merged = np.concatenate([pose, left_hand, right_hand], axis=0)

            # Flatten → (225,)
            keypoints.append(merged.flatten())

            frame_idx += 1

    cap.release()

    if not keypoints:
        print(f"[WARN] No keypoints extracted from: {video_path}")
        return None

    return np.array(keypoints, dtype=np.float32)