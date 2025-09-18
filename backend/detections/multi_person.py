from typing import Dict
from .multi_person_improved import infer_multi_person_improved


def infer_multi_person(image_bytes: bytes) -> Dict:
    """Improved multi-person detection with better accuracy and debugging"""
    return infer_multi_person_improved(image_bytes)


