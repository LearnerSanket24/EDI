from typing import Dict
from .head_pose_model import infer_head_pose_trained
from .head_pose_mediapipe import infer_head_pose_mediapipe


def infer_head_pose(image_bytes: bytes) -> Dict:
    res = infer_head_pose_trained(image_bytes)
    if res.get("using_trained"):
        return res
    # fallback to CPU MediaPipe
    return infer_head_pose_mediapipe(image_bytes)


