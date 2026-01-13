#!/usr/bin/env python3
"""
Auto-training loop for Contact web_convert.

Goal (MVP):
- When the web UI successfully produces NEW labeled samples (new contact.json written),
  accumulate a counter.
- Once counter reaches N, run `train.py` in the background for K additional epochs.
- After training, ask the inference server (`server.py`) to hot-reload the new checkpoint.

This module is intentionally dependency-light and does not assume multi-user concurrency.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import yaml


# Defaults (can be overridden via base_train_config YAML: autotrain.small_update_freq / big_update_freq)
_DEFAULT_SMALL_UPDATE_FREQ = 10
_DEFAULT_BIG_UPDATE_FREQ = 80

# Small-loop defaults (epochs via AutoTrainConfig.additional_epochs; LR scale here)
_SMALL_LR_SCALE = 0.2

def _project_root() -> str:
    # Contact/web_convert/auto_train.py -> Contact/
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_project_path(path: str) -> str:
    if path is None:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_project_root(), path))


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _collect_labeled_sample_dirs(root_dir: str) -> List[str]:
    """Collect labeled sample directories under root_dir (absolute paths).

    Mirrors SmplContactDataset._collect_samples() / split_dataset() required-files filtering.
    """
    root_dir = os.path.abspath(root_dir)
    required_files = [
        "image.jpg",
        "smplx_parameters.json",
        "contact.json",
        "box_annotation.json",
        "calibration.json",
        "extrinsic.json",
    ]

    out: List[str] = []
    if not os.path.isdir(root_dir):
        return out

    for category in sorted(os.listdir(root_dir)):
        category_path = os.path.join(root_dir, category)
        if not os.path.isdir(category_path):
            continue
        for sample_id in sorted(os.listdir(category_path)):
            sample_path = os.path.join(category_path, sample_id)
            if not os.path.isdir(sample_path):
                continue
            if all(os.path.exists(os.path.join(sample_path, f)) for f in required_files):
                out.append(os.path.abspath(sample_path))
    return out


class _Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"


def _fmt_kv(k: str, v: Any) -> str:
    return f"{_Ansi.DIM}{k}{_Ansi.RESET}={v}"


def _log_autotrain(level: str, msg: str, **kv: Any) -> None:
    color = _Ansi.CYAN
    if level == "ok":
        color = _Ansi.GREEN
    elif level == "warn":
        color = _Ansi.YELLOW
    elif level == "err":
        color = _Ansi.RED
    elif level == "run":
        color = _Ansi.MAGENTA

    parts = [f"{_Ansi.BOLD}{color}[AutoTrain]{_Ansi.RESET} {msg}"]
    if kv:
        parts.append(" ".join(_fmt_kv(k, v) for k, v in kv.items()))
    print(" ".join(parts), flush=True)


@dataclass
class AutoTrainConfig:
    enabled: bool = False
    # Kept for backward compatibility (viewer progress bar), but the scheduler reads frequencies from YAML config.
    every_n_new_labels: int = _DEFAULT_SMALL_UPDATE_FREQ
    # Used as "small update epochs" (clamped to 10..20 for responsiveness).
    additional_epochs: int = 15
    base_train_config: str = "configs/default.yaml"
    # Where to write derived config + logs + checkpoints
    work_dir: Optional[str] = None  # default: <target_dir>/_autotrain
    # Initial checkpoint to resume from (if None, will auto-detect best_model.pth in work_dir)
    initial_checkpoint: Optional[str] = None
    # Inference server reload endpoint
    reload_host: str = "127.0.0.1"
    reload_port: int = 8000


class AutoTrainManager:
    def __init__(
        self,
        *,
        target_dir: str,
        cfg: AutoTrainConfig,
        socketio=None,
    ) -> None:
        self.target_dir = os.path.abspath(target_dir)
        self.cfg = cfg
        self.socketio = socketio

        self._lock = threading.Lock()
        self._running = False

        self.work_dir = os.path.abspath(cfg.work_dir) if cfg.work_dir else os.path.join(self.target_dir, "_autotrain")
        self.state_path = os.path.join(self.work_dir, "state.json")
        self.log_path = os.path.join(self.work_dir, "train.log")
        self.derived_config_path = os.path.join(self.work_dir, "train_config.yaml")
        self.ckpt_dir = os.path.join(self.work_dir, "checkpoints")
        self.viz_dir = os.path.join(self.work_dir, "visualizations")

        self._state = self._load_state()
        self.small_update_freq, self.big_update_freq = self._load_scheduler_freqs()
        self._reconcile_state_with_disk()

    def _load_scheduler_freqs(self) -> Tuple[int, int]:
        """Load SMALL/BIG update frequencies from base_train_config YAML."""
        small = int(_DEFAULT_SMALL_UPDATE_FREQ)
        big = int(_DEFAULT_BIG_UPDATE_FREQ)
        try:
            base_cfg_path = _resolve_project_path(self.cfg.base_train_config)
            if os.path.exists(base_cfg_path):
                with open(base_cfg_path, "r", encoding="utf-8") as f:
                    cfg_dict = yaml.safe_load(f) or {}
                if isinstance(cfg_dict, dict):
                    at = cfg_dict.get("autotrain", {}) or {}
                    if isinstance(at, dict):
                        if at.get("small_update_freq") is not None:
                            small = int(at["small_update_freq"])
                        if at.get("big_update_freq") is not None:
                            big = int(at["big_update_freq"])
        except Exception:
            # fallback to defaults
            pass

        # Basic validation
        if small <= 0:
            small = int(_DEFAULT_SMALL_UPDATE_FREQ)
        if big <= 0:
            big = int(_DEFAULT_BIG_UPDATE_FREQ)
        if big < small:
            # keep big meaningful; fallback
            big = int(_DEFAULT_BIG_UPDATE_FREQ)
        return small, big

    def _reconcile_state_with_disk(self) -> None:
        """Best-effort reconcile counters with what exists on disk.

        This prevents schedule drift after process restarts (state.json may be missing/stale).
        """
        try:
            disk_dirs = _collect_labeled_sample_dirs(self.target_dir)
            disk_total = int(len(disk_dirs))
            with self._lock:
                cur_total = int(self._state.get("total_annotated_images", 0) or 0)
                if disk_total > cur_total:
                    self._state["total_annotated_images"] = disk_total
                keep = int(self.small_update_freq)
                if not isinstance(self._state.get("recent_labeled_sample_dirs"), list) or not self._state.get("recent_labeled_sample_dirs"):
                    self._state["recent_labeled_sample_dirs"] = disk_dirs[-keep:]
                self._save_state()
        except Exception:
            # Best-effort only
            return

    def _emit(self, payload: Dict[str, Any]) -> None:
        # Best-effort broadcast (viewer.js will show toast messages)
        try:
            if self.socketio is not None:
                self.socketio.emit("training_status", payload)
        except Exception:
            pass

    def _load_state(self) -> Dict[str, Any]:
        default_state: Dict[str, Any] = {
            # Legacy counters kept for compatibility
            "pending_new_labels": 0,
            "inflight_new_labels": 0,
            # Hybrid scheduler state
            "total_annotated_images": 0,
            "recent_labeled_sample_dirs": [],
            "queued_jobs": [],
            "last_job_type": None,
            "last_job_trigger_total": None,
            # Train bookkeeping
            "last_train_started_at": None,
            "last_train_finished_at": None,
            "last_train_ok": None,
            "last_error": None,
            "last_checkpoint": None,
        }

        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for k, v in default_state.items():
                        data.setdefault(k, v)
                    # Basic type normalization
                    if not isinstance(data.get("recent_labeled_sample_dirs"), list):
                        data["recent_labeled_sample_dirs"] = []
                    if not isinstance(data.get("queued_jobs"), list):
                        data["queued_jobs"] = []
                    return data
            except Exception:
                pass
        return default_state

    def _save_state(self) -> None:
        _atomic_write_json(self.state_path, self._state)

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._running)

    def get_state_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snap = dict(self._state)
            snap["running"] = bool(self._running)
            snap["enabled"] = bool(self.cfg.enabled)
            # Backward-compatible fields (viewer.js expects these)
            snap["every_n_new_labels"] = int(self.small_update_freq)
            snap["additional_epochs"] = int(self.cfg.additional_epochs)
            # Hybrid scheduler fields
            snap["small_update_freq"] = int(self.small_update_freq)
            snap["big_update_freq"] = int(self.big_update_freq)
            snap["queued_jobs_len"] = int(len(self._state.get("queued_jobs", []) or []))
            snap["small_lr_scale"] = float(_SMALL_LR_SCALE)
        return snap

    def note_new_label(self, *, sample_dir: Optional[str] = None) -> None:
        """Called when a NEW labeled sample is created (first-time contact.json for that sample)."""
        if not self.cfg.enabled:
            return

        job: Optional[Dict[str, Any]] = None
        enqueue_only = False
        total = None
        with self._lock:
            # 1) Update global total
            self._state["total_annotated_images"] = int(self._state.get("total_annotated_images", 0)) + 1
            total = int(self._state["total_annotated_images"])

            # 2) Update recent FIFO for "recent 10"
            if sample_dir:
                sdir = os.path.abspath(sample_dir)
                recent = list(self._state.get("recent_labeled_sample_dirs", []) or [])
                recent.append(sdir)
                if len(recent) > int(self.small_update_freq):
                    recent = recent[-int(self.small_update_freq) :]
                self._state["recent_labeled_sample_dirs"] = recent

            # 3) Backward-compatible progress counter (0..SMALL_UPDATE_FREQ-1)
            self._state["pending_new_labels"] = int(total % int(self.small_update_freq))

            # 4) Decide trigger type (big has priority)
            job_type: Optional[str] = None
            if total > 0 and (total % int(self.big_update_freq) == 0):
                job_type = "big"
            elif total > 0 and (total % int(self.small_update_freq) == 0):
                job_type = "small"

            if job_type:
                recent_snapshot = list(self._state.get("recent_labeled_sample_dirs", []) or [])
                job = {
                    "type": job_type,
                    "trigger_total": int(total),
                    "created_at": int(time.time()),
                    "recent_dirs": recent_snapshot,
                }

                if self._running:
                    q = list(self._state.get("queued_jobs", []) or [])
                    q.append(job)
                    self._state["queued_jobs"] = q
                    enqueue_only = True
                    job = None

            self._save_state()

        if enqueue_only:
            _log_autotrain(
                "warn",
                "new label recorded (queued job while running)",
                total=total,
                queued=int(len(self._state.get("queued_jobs", []) or [])),
            )
        else:
            _log_autotrain("ok", "new label recorded", total=total, next_job=(job.get("type") if job else None))

        # Push a lightweight snapshot to clients so the UI badge stays up-to-date
        try:
            self._emit(self.get_state_snapshot())
        except Exception:
            pass

        if job is not None:
            self.start_async(job=job)

    def _build_derived_config(
        self, *, override_training: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Return (derived_config_path, resume_checkpoint_path_or_empty, derived_cfg_dict)."""
        base_cfg_path = _resolve_project_path(self.cfg.base_train_config)
        if not os.path.exists(base_cfg_path):
            raise FileNotFoundError(f"Base train config not found: {base_cfg_path}")

        with open(base_cfg_path, "r") as f:
            base_cfg = yaml.safe_load(f)

        # Ensure required keys exist
        if "data" not in base_cfg or "training" not in base_cfg:
            raise ValueError(f"Invalid base config (missing data/training): {base_cfg_path}")

        # Override to train on the web target_dir dataset
        base_cfg["data"]["root_dir"] = self.target_dir

        # Make SMPL-X path robust (relative -> project root)
        if "smplx_model_path" in base_cfg["data"]:
            base_cfg["data"]["smplx_model_path"] = _resolve_project_path(str(base_cfg["data"]["smplx_model_path"]))

        # Route outputs to work_dir
        base_cfg["training"]["save_dir"] = self.ckpt_dir
        if base_cfg.get("visualization", {}).get("enabled"):
            base_cfg["visualization"]["save_dir"] = self.viz_dir

        # Optional overrides (used for small-loop hyperparams)
        if override_training:
            base_cfg.setdefault("training", {})
            for k, v in override_training.items():
                base_cfg["training"][k] = v

        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)

        with open(self.derived_config_path, "w") as f:
            yaml.safe_dump(base_cfg, f, sort_keys=False)

        # Determine resume checkpoint: use initial_checkpoint if provided, otherwise auto-detect best_model.pth
        resume = ""
        if self.cfg.initial_checkpoint:
            # Use explicitly provided checkpoint
            resolved_initial = _resolve_project_path(self.cfg.initial_checkpoint)
            if os.path.exists(resolved_initial):
                resume = resolved_initial
            else:
                _log_autotrain("warn", f"initial_checkpoint not found, will start from scratch", path=resolved_initial)
        else:
            # Auto-detect: resume from last best_model if present
            auto_resume = os.path.join(self.ckpt_dir, "best_model.pth")
            if os.path.exists(auto_resume):
                resume = auto_resume

        return self.derived_config_path, resume, base_cfg

    def _reload_inference_server(self, checkpoint_path: str) -> None:
        import http.client
        import json as _json

        conn = http.client.HTTPConnection(self.cfg.reload_host, int(self.cfg.reload_port), timeout=30)
        body = _json.dumps({"checkpoint": checkpoint_path}).encode("utf-8")
        headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
        conn.request("POST", "/reload", body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status < 200 or resp.status >= 300:
            raise RuntimeError(f"/reload failed: HTTP {resp.status}: {raw[:2000]!r}")

    def start_async(self, *, job: Optional[Dict[str, Any]] = None) -> None:
        if not self.cfg.enabled:
            return

        if job is None:
            # Backward-compatible fallback: treat as small job at current total.
            with self._lock:
                total = int(self._state.get("total_annotated_images", 0))
                recent_snapshot = list(self._state.get("recent_labeled_sample_dirs", []) or [])
            job = {"type": "small", "trigger_total": int(total), "created_at": int(time.time()), "recent_dirs": recent_snapshot}

        job_type = str(job.get("type") or "small")
        trigger_total = int(job.get("trigger_total") or 0)

        with self._lock:
            if self._running:
                return
            self._running = True
            # UI compatibility: show a full bar (inflight/every_n == 1.0)
            self._state["inflight_new_labels"] = int(self.big_update_freq if job_type == "big" else self.small_update_freq)
            self._state["pending_new_labels"] = 0
            self._state["last_job_type"] = job_type
            self._state["last_job_trigger_total"] = trigger_total
            self._state["last_train_started_at"] = int(time.time())
            self._state["last_error"] = None
            self._save_state()

        # Compute message-friendly epochs for UI (small: additional; big: full num_epochs)
        ui_epochs = int(self.cfg.additional_epochs)
        try:
            if job_type == "big":
                _, _, cfg_dict = self._build_derived_config()
                ui_epochs = int(cfg_dict.get("training", {}).get("num_epochs", ui_epochs))
        except Exception:
            pass

        _log_autotrain(
            "run",
            "training started",
            job_type=job_type,
            trigger_total=trigger_total,
            base_config=self.cfg.base_train_config,
            work_dir=self.work_dir,
            reload=f"{self.cfg.reload_host}:{int(self.cfg.reload_port)}",
        )

        self._emit(
            {
                "state": "started",
                "job_type": job_type,
                "trigger_total": trigger_total,
                "inflight_new_labels": int(self.big_update_freq if job_type == "big" else self.small_update_freq),
                "every_n": int(self.big_update_freq if job_type == "big" else self.small_update_freq),
                "additional_epochs": int(ui_epochs),
            }
        )

        t = threading.Thread(target=self._run_train_job, kwargs={"job": job}, daemon=True)
        t.start()

    def _run_train_job(self, *, job: Dict[str, Any]) -> None:
        ok = False
        err: Optional[str] = None
        started_at = time.time()
        best_path = os.path.join(self.ckpt_dir, "best_model.pth")
        latest_path = os.path.join(self.ckpt_dir, "latest_model.pth")
        prev_best_epoch = None
        next_job: Optional[Dict[str, Any]] = None
        job_type = str(job.get("type") or "small")
        trigger_total = int(job.get("trigger_total") or 0)
        try:
            # Small-loop hyperparams
            small_epochs = int(self.cfg.additional_epochs)
            small_epochs = max(10, min(20, small_epochs))

            subset_json_path = ""

            # Base derived config
            derived_cfg, resume_ckpt, cfg_dict = self._build_derived_config()

            train_py = os.path.join(_project_root(), "train.py")

            if job_type == "small":
                # Lower LR for quick adaptation, replay to suppress forgetting
                base_lr = float(cfg_dict.get("training", {}).get("learning_rate", 0.0) or 0.0)
                small_lr = float(base_lr * float(_SMALL_LR_SCALE))
                derived_cfg, resume_ckpt, _ = self._build_derived_config(override_training={"learning_rate": small_lr})

                # Build experience-replay mixed subset
                all_dirs = _collect_labeled_sample_dirs(self.target_dir)
                recent_dirs: Sequence[str] = job.get("recent_dirs") or []
                new_dirs = [os.path.abspath(p) for p in list(recent_dirs)[-int(self.small_update_freq) :]]
                new_dirs = [p for p in new_dirs if os.path.isdir(p)]
                new_set = set(new_dirs)
                old_pool = [d for d in all_dirs if d not in new_set]

                old_k = max(30, 3 * len(new_dirs))
                seed = trigger_total if trigger_total > 0 else int(time.time())
                import random

                rng = random.Random(int(seed))
                if old_k >= len(old_pool):
                    old_sample = list(old_pool)
                else:
                    old_sample = rng.sample(old_pool, k=int(old_k))

                mixed: List[str] = []
                seen = set()
                for p in list(new_dirs) + list(old_sample):
                    ap = os.path.abspath(p)
                    if ap in seen:
                        continue
                    seen.add(ap)
                    mixed.append(ap)

                subset_json_path = os.path.join(self.work_dir, f"subset_small_{trigger_total or int(time.time())}.json")
                _atomic_write_json(
                    subset_json_path,
                    {"sample_dirs": mixed, "new_dirs": new_dirs, "old_dirs": old_sample, "seed": int(seed)},
                )

                _log_autotrain(
                    "ok",
                    "small-loop replay batch built",
                    trigger_total=trigger_total,
                    new=len(new_dirs),
                    old=len(old_sample),
                    total=len(mixed),
                    small_epochs=small_epochs,
                    small_lr=small_lr,
                )

                cmd = [
                    sys.executable,
                    train_py,
                    "--config",
                    derived_cfg,
                    "--epochs",
                    str(int(small_epochs)),
                    "--no_val",
                    "--train_subset_json",
                    subset_json_path,
                ]
                if resume_ckpt:
                    cmd += ["--resume", resume_ckpt]

            elif job_type == "big":
                # Big loop: from scratch, full dataset, full config training
                cmd = [sys.executable, train_py, "--config", derived_cfg]

            else:
                raise ValueError(f"Unknown job type: {job_type}")

            # Snapshot best epoch before training (for "did best update?" signal)
            try:
                if os.path.exists(best_path):
                    prev_ckpt = torch.load(best_path, map_location="cpu")  # type: ignore[name-defined]
                    if isinstance(prev_ckpt, dict) and "epoch" in prev_ckpt:
                        prev_best_epoch = int(prev_ckpt["epoch"])
            except Exception:
                prev_best_epoch = None

            os.makedirs(self.work_dir, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as logf:
                logf.write("\n" + "=" * 80 + "\n")
                logf.write(f"[AutoTrain] cmd: {' '.join(cmd)}\n")
                logf.write(f"[AutoTrain] job: {job}\n")
                if subset_json_path:
                    logf.write(f"[AutoTrain] subset_json: {subset_json_path}\n")
                logf.write(f"[AutoTrain] started_at: {time.ctime()}\n")
                logf.flush()
                proc = subprocess.run(
                    cmd,
                    cwd=_project_root(),
                    stdout=logf,
                    stderr=logf,
                    check=False,
                    env=os.environ.copy(),
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"train.py failed with exit code {proc.returncode}. See log: {self.log_path}")

            if not os.path.exists(latest_path):
                raise FileNotFoundError(f"Training finished but latest_model.pth not found: {latest_path}")
            if not os.path.exists(best_path):
                raise FileNotFoundError(f"Training finished but best_model.pth not found: {best_path}")

            # Ask inference server to reload (default to best_model to avoid regressions)
            self._reload_inference_server(best_path)
            ok = True
        except Exception as e:
            err = str(e)
            ok = False
        finally:
            duration_s = round(time.time() - started_at, 2)
            best_updated = None
            try:
                if os.path.exists(best_path):
                    cur_ckpt = torch.load(best_path, map_location="cpu")  # type: ignore[name-defined]
                    if isinstance(cur_ckpt, dict) and "epoch" in cur_ckpt:
                        cur_epoch = int(cur_ckpt["epoch"])
                        best_updated = (prev_best_epoch is None) or (cur_epoch != prev_best_epoch)
            except Exception:
                best_updated = None
            with self._lock:
                self._running = False
                self._state["last_train_finished_at"] = int(time.time())
                self._state["last_train_ok"] = bool(ok)
                self._state["last_error"] = err
                # Keep backward-compatible "last_checkpoint" as latest_model.pth
                self._state["last_checkpoint"] = latest_path if ok else self._state.get("last_checkpoint")
                self._state["last_best_checkpoint"] = best_path if ok else self._state.get("last_best_checkpoint")
                self._state["last_best_updated"] = best_updated
                self._state["inflight_new_labels"] = 0
                self._save_state()

                # Optional: auto-chain next queued job (only after a successful run)
                if ok and self.cfg.enabled:
                    q = list(self._state.get("queued_jobs", []) or [])
                    if q:
                        next_job = q.pop(0)
                        self._state["queued_jobs"] = q
                        self._save_state()

            self._emit(
                {
                    "state": "finished" if ok else "failed",
                    "ok": bool(ok),
                    "job_type": job_type,
                    "trigger_total": trigger_total,
                    "checkpoint": latest_path if ok else None,
                    "best_checkpoint": best_path if ok else None,
                    "best_updated": best_updated if ok else None,
                    "error": err,
                    "log_path": self.log_path,
                    "duration_s": duration_s,
                    "total_annotated_images": int(self._state.get("total_annotated_images", 0)),
                    "queued_jobs_len": int(len(self._state.get("queued_jobs", []) or [])),
                }
            )

            if ok:
                _log_autotrain(
                    "ok",
                    "training finished",
                    duration_s=duration_s,
                    latest=latest_path,
                    best=best_path,
                    best_updated=best_updated,
                )
            else:
                _log_autotrain("err", "training failed", duration_s=duration_s, error=err, log=self.log_path)

            if ok and next_job is not None:
                try:
                    _log_autotrain(
                        "run",
                        "chaining queued job",
                        next_type=str(next_job.get("type")),
                        queued_left=int(len(self._state.get("queued_jobs", []) or [])),
                    )
                    self.start_async(job=next_job)
                except Exception:
                    pass

