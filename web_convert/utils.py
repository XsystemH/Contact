import json
import os
from typing import Any, Dict


def load_json_maybe_list(path: str) -> Dict[str, Any]:
    """Load JSON file and normalize list-wrapped payloads.

    Some datasets store a list with a single element; normalize to the first dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"Empty JSON list in: {path}")
        payload = payload[0]

    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict (or list[dict]) in {path}, got {type(payload)}")

    return payload


def write_camera_files_from_smplx_params(data_dir: str) -> None:
    """Generate calibration.json and extrinsic.json in data_dir.

    This mirrors Ca3OH1/HOIGaussian/prepare/camera.py.
    """
    smplx_path = os.path.join(data_dir, "smplx_parameters.json")
    smplx_param = load_json_maybe_list(smplx_path)

    focal = smplx_param.get("focal")
    princpt = smplx_param.get("princpt")
    if focal is None or princpt is None:
        raise KeyError("smplx_parameters.json missing 'focal' or 'princpt'")

    K = {"K": [[focal[0], 0, princpt[0]], [0, focal[0], princpt[1]], [0, 0, 1]]}
    I = {
        "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "translation": [0, 0, 0],
    }

    with open(os.path.join(data_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(K, f)

    with open(os.path.join(data_dir, "extrinsic.json"), "w", encoding="utf-8") as f:
        json.dump(I, f)


def ensure_object_mask_name(data_dir: str) -> None:
    """Rename obj_mask.png to object_mask.png if needed."""
    old_path = os.path.join(data_dir, "obj_mask.png")
    new_path = os.path.join(data_dir, "object_mask.png")
    if os.path.exists(new_path):
        return
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
