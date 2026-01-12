#!/usr/bin/env python3
"""Flask HTTP inference service for ContactNet.

Only supports raw training-like inputs (server-side preprocessing).

POST /predict as multipart/form-data with files:
  - image (.jpg/.png)
  - object_mask (.png)
  - smplx_parameters (.json)
  - calibration (.json)
  - extrinsic (.json)
  - box_annotation (.json)
  - contact (.json, optional)

Query params:
  - threshold: float in [0,1], default 0.5
  - return_probs: 0/1, default 0
"""

import argparse
import os
import tempfile
import traceback
import threading
import time
from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from flask import Flask, jsonify, request

from data.dataset import SmplContactPreprocessor
from models.contact_net import ContactNet


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_project_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(PROJECT_ROOT, path))


def _load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Return (state_dict, metadata)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint

    if isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint, None

    raise ValueError(f"Unrecognized checkpoint format at: {checkpoint_path}")


def _save_raw_uploads_to_tempdir() -> str:
    mapping = {
        "image": "image.jpg",
        "object_mask": "object_mask.png",
        "smplx_parameters": "smplx_parameters.json",
        "calibration": "calibration.json",
        "extrinsic": "extrinsic.json",
        "box_annotation": "box_annotation.json",
        "contact": "contact.json",
    }

    required_fields = [
        "image",
        "object_mask",
        "smplx_parameters",
        "calibration",
        "extrinsic",
        "box_annotation",
    ]
    missing = [k for k in required_fields if k not in request.files]
    if missing:
        raise KeyError(f"Missing required file fields: {missing}")

    tmpdir = tempfile.mkdtemp(prefix="contact_raw_")
    for field, filename in mapping.items():
        if field not in request.files:
            continue
        f = request.files[field]
        f.save(os.path.join(tmpdir, filename))

    return tmpdir


def _cleanup_dir(path: str) -> None:
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(path)
    except Exception:
        pass


def _prepare_tensors_from_torch(sample_torch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    required = ("image", "vertices", "normals", "pose_params", "K", "object_bbox", "mask_dist_field")
    missing = [k for k in required if k not in sample_torch]
    if missing:
        raise KeyError(f"Preprocessor output missing keys: {missing}")

    return {
        "images": sample_torch["image"].unsqueeze(0).to(device),
        "vertices": sample_torch["vertices"].unsqueeze(0).to(device),
        "normals": sample_torch["normals"].unsqueeze(0).to(device),
        "pose_params": sample_torch["pose_params"].unsqueeze(0).to(device),
        "K": sample_torch["K"].unsqueeze(0).to(device),
        "object_bbox": sample_torch["object_bbox"].unsqueeze(0).to(device),
        "mask_dist_field": sample_torch["mask_dist_field"].unsqueeze(0).to(device),
    }


def create_app(config_path: str, checkpoint_path: Optional[str], device_str: Optional[str]) -> Flask:
    config_path = _resolve_project_path(config_path)
    if config_path is None or not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    ckpt_dir = _resolve_project_path(config["training"]["save_dir"])
    if ckpt_dir is None:
        raise ValueError("config.training.save_dir is missing")

    resolved_ckpt = _resolve_project_path(checkpoint_path) if checkpoint_path else os.path.join(ckpt_dir, "best_model.pth")
    if not os.path.exists(resolved_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {resolved_ckpt}")

    device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ContactNet(config).to(device)
    state_dict, meta = _load_checkpoint(resolved_ckpt, device)
    model.load_state_dict(state_dict)
    model.eval()

    preprocessor = SmplContactPreprocessor(
        smplx_model_path=config["data"]["smplx_model_path"],
        smplx_model_type=config["data"].get("smplx_model_type", "neutral"),
        img_size=tuple(config["data"]["img_size"]),
        split="test",
        augment=False,
    )

    app = Flask(__name__)
    app.config["CONTACT_DEVICE"] = str(device)
    app.config["CONTACT_CONFIG_PATH"] = config_path
    app.config["CONTACT_CHECKPOINT_PATH"] = resolved_ckpt
    app.config["CONTACT_CHECKPOINT_META"] = meta
    app.config["CONTACT_LOADED_AT"] = int(time.time())

    app.model = model  # type: ignore[attr-defined]
    app.device = device  # type: ignore[attr-defined]
    app.preprocessor = preprocessor  # type: ignore[attr-defined]
    app.reload_lock = threading.Lock()  # type: ignore[attr-defined]

    @app.get("/healthz")
    def healthz():
        meta_local = app.config.get("CONTACT_CHECKPOINT_META")
        epoch = None
        best_val_loss = None
        if isinstance(meta_local, dict):
            epoch = meta_local.get("epoch")
            best_val_loss = meta_local.get("best_val_loss")
        return jsonify(
            {
                "status": "ok",
                "device": app.config["CONTACT_DEVICE"],
                "config": app.config["CONTACT_CONFIG_PATH"],
                "checkpoint": app.config["CONTACT_CHECKPOINT_PATH"],
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "loaded_at": app.config.get("CONTACT_LOADED_AT"),
            }
        )

    @app.post("/reload")
    def reload_model():
        """Hot-reload model weights (and optionally config) without restarting the process.

        Body (JSON) or query params:
          - checkpoint: path to checkpoint (.pth). If omitted, reload current checkpoint path.
          - config: path to YAML config. If provided, will recreate model+preprocessor.
        """
        try:
            payload = request.get_json(silent=True) or {}
            ckpt_in = payload.get("checkpoint") or request.args.get("checkpoint")
            cfg_in = payload.get("config") or request.args.get("config")

            new_cfg_path = _resolve_project_path(cfg_in) if cfg_in else app.config["CONTACT_CONFIG_PATH"]
            if not new_cfg_path or not os.path.exists(new_cfg_path):
                raise FileNotFoundError(f"Config not found: {new_cfg_path}")

            with open(new_cfg_path, "r") as f:
                new_cfg = yaml.safe_load(f)

            # If checkpoint is not specified, use best_model under config.training.save_dir
            if ckpt_in:
                new_ckpt_path = _resolve_project_path(str(ckpt_in))
            else:
                ckpt_dir = _resolve_project_path(new_cfg["training"]["save_dir"])
                new_ckpt_path = os.path.join(ckpt_dir, "best_model.pth")

            if not new_ckpt_path or not os.path.exists(new_ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {new_ckpt_path}")

            with app.reload_lock:  # type: ignore[attr-defined]
                # Recreate model/preprocessor if config path changed; otherwise just load weights.
                if cfg_in:
                    app.model = ContactNet(new_cfg).to(app.device)  # type: ignore[attr-defined]
                    app.preprocessor = SmplContactPreprocessor(  # type: ignore[attr-defined]
                        smplx_model_path=new_cfg["data"]["smplx_model_path"],
                        smplx_model_type=new_cfg["data"].get("smplx_model_type", "neutral"),
                        img_size=tuple(new_cfg["data"]["img_size"]),
                        split="test",
                        augment=False,
                    )
                    app.config["CONTACT_CONFIG_PATH"] = new_cfg_path

                state_dict, meta_local = _load_checkpoint(new_ckpt_path, app.device)  # type: ignore[attr-defined]
                app.model.load_state_dict(state_dict)  # type: ignore[attr-defined]
                app.model.eval()  # type: ignore[attr-defined]

                app.config["CONTACT_CHECKPOINT_PATH"] = new_ckpt_path
                app.config["CONTACT_CHECKPOINT_META"] = meta_local
                app.config["CONTACT_LOADED_AT"] = int(time.time())

            return jsonify(
                {
                    "status": "ok",
                    "config": app.config["CONTACT_CONFIG_PATH"],
                    "checkpoint": app.config["CONTACT_CHECKPOINT_PATH"],
                }
            )
        except Exception as e:
            debug = request.args.get("debug", default="0") == "1"
            payload = {"error": str(e), "type": e.__class__.__name__}
            if debug:
                payload["traceback"] = traceback.format_exc()
            return jsonify(payload), 400

    @app.post("/predict")
    def predict():
        cleanup_tmpdir: Optional[str] = None
        try:
            threshold = request.args.get("threshold", default=0.5, type=float)
            if threshold is None or not (0.0 <= float(threshold) <= 1.0):
                raise ValueError("threshold must be within [0,1]")
            threshold = float(threshold)

            return_probs_q = request.args.get("return_probs", default="0", type=str)
            return_probs = return_probs_q.strip().lower() in {"1", "true", "yes", "y"}

            cleanup_tmpdir = _save_raw_uploads_to_tempdir()
            sample_torch = app.preprocessor.process_dir(cleanup_tmpdir, include_contact_labels=False)  # type: ignore[attr-defined]
            tensors = _prepare_tensors_from_torch(sample_torch, app.device)  # type: ignore[attr-defined]

            with torch.no_grad():
                logits = app.model(
                    tensors["images"],
                    tensors["vertices"],
                    tensors["normals"],
                    tensors["pose_params"],
                    tensors["K"],
                    tensors["object_bbox"],
                    tensors["mask_dist_field"],
                )
                probs_t = torch.sigmoid(logits)[0]  # (N,)
                contact_t = (probs_t > threshold).to(torch.int32)

            num_vertices = int(contact_t.numel())
            num_contact = int(contact_t.sum().item())
            resp: Dict[str, Any] = {
                "contact": contact_t.detach().cpu().tolist(),
                "num_vertices": num_vertices,
                "num_contact": num_contact,
                "contact_ratio": (num_contact / num_vertices) if num_vertices > 0 else 0.0,
                "threshold": threshold,
            }
            if return_probs:
                resp["probs"] = probs_t.detach().cpu().float().tolist()

            return jsonify(resp)

        except Exception as e:
            debug = request.args.get("debug", default="0") == "1"
            payload: Dict[str, Any] = {"error": str(e), "type": e.__class__.__name__}
            if debug:
                payload["traceback"] = traceback.format_exc()
            return jsonify(payload), 400

        finally:
            if cleanup_tmpdir:
                _cleanup_dir(cleanup_tmpdir)

    return app


def main():
    parser = argparse.ArgumentParser(description="ContactNet Flask inference server (raw inputs only)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (default: save_dir/best_model.pth)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. 'cuda' or 'cpu'")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(args.config, args.checkpoint, args.device)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
