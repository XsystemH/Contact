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
from typing import Any, Dict, Optional, Tuple

import torch
import yaml


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
    every_n_new_labels: int = 20
    additional_epochs: int = 10
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

    def _emit(self, payload: Dict[str, Any]) -> None:
        # Best-effort broadcast (viewer.js will show toast messages)
        try:
            if self.socketio is not None:
                self.socketio.emit("training_status", payload)
        except Exception:
            pass

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {
            "pending_new_labels": 0,
            "inflight_new_labels": 0,
            "last_train_started_at": None,
            "last_train_finished_at": None,
            "last_train_ok": None,
            "last_error": None,
            "last_checkpoint": None,
        }

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
            snap["every_n_new_labels"] = int(self.cfg.every_n_new_labels)
            snap["additional_epochs"] = int(self.cfg.additional_epochs)
        return snap

    def note_new_label(self) -> None:
        """Called when a NEW labeled sample is created (first-time contact.json for that sample)."""
        if not self.cfg.enabled:
            return

        should_start = False
        pending = None
        with self._lock:
            self._state["pending_new_labels"] = int(self._state.get("pending_new_labels", 0)) + 1
            pending = int(self._state["pending_new_labels"])
            self._save_state()
            should_start = (not self._running) and (int(self._state["pending_new_labels"]) >= int(self.cfg.every_n_new_labels))

        _log_autotrain(
            "warn" if pending and pending >= int(self.cfg.every_n_new_labels) else "ok",
            "new label recorded",
            pending=pending,
            trigger_every=int(self.cfg.every_n_new_labels),
            add_epochs=int(self.cfg.additional_epochs),
        )

        # Push a lightweight snapshot to clients so the UI badge stays up-to-date
        try:
            self._emit(self.get_state_snapshot())
        except Exception:
            pass

        if should_start:
            self.start_async()

    def _build_derived_config(self) -> Tuple[str, str]:
        """Return (derived_config_path, resume_checkpoint_path_or_empty)."""
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
        
        return self.derived_config_path, resume

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

    def start_async(self) -> None:
        if not self.cfg.enabled:
            return

        with self._lock:
            if self._running:
                return
            self._running = True
            inflight = int(self._state.get("pending_new_labels", 0))
            self._state["pending_new_labels"] = 0
            self._state["inflight_new_labels"] = inflight
            self._state["last_train_started_at"] = int(time.time())
            self._state["last_error"] = None
            self._save_state()

        _log_autotrain(
            "run",
            "training started",
            inflight=inflight,
            trigger_every=int(self.cfg.every_n_new_labels),
            add_epochs=int(self.cfg.additional_epochs),
            base_config=self.cfg.base_train_config,
            work_dir=self.work_dir,
            reload=f"{self.cfg.reload_host}:{int(self.cfg.reload_port)}",
        )

        self._emit(
            {
                "state": "started",
                "inflight_new_labels": inflight,
                "every_n": int(self.cfg.every_n_new_labels),
                "additional_epochs": int(self.cfg.additional_epochs),
            }
        )

        t = threading.Thread(target=self._run_train_job, daemon=True)
        t.start()

    def _run_train_job(self) -> None:
        ok = False
        err: Optional[str] = None
        started_at = time.time()
        best_path = os.path.join(self.ckpt_dir, "best_model.pth")
        latest_path = os.path.join(self.ckpt_dir, "latest_model.pth")
        prev_best_epoch = None
        try:
            derived_cfg, resume_ckpt = self._build_derived_config()

            train_py = os.path.join(_project_root(), "train.py")
            cmd = [
                sys.executable,
                train_py,
                "--config",
                derived_cfg,
                "--epochs",
                str(int(self.cfg.additional_epochs)),
            ]
            if resume_ckpt:
                cmd += ["--resume", resume_ckpt]

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
                # If failed, restore inflight labels back to pending so it can be retried later.
                inflight = int(self._state.get("inflight_new_labels", 0))
                self._state["inflight_new_labels"] = 0
                if not ok and inflight > 0:
                    self._state["pending_new_labels"] = int(self._state.get("pending_new_labels", 0)) + inflight
                self._save_state()

                # Optional: auto-chain if new labels arrived while training
                should_chain = (
                    ok
                    and int(self._state.get("pending_new_labels", 0)) >= int(self.cfg.every_n_new_labels)
                    and self.cfg.enabled
                )

            self._emit(
                {
                    "state": "finished" if ok else "failed",
                    "ok": bool(ok),
                    "checkpoint": latest_path if ok else None,
                    "best_checkpoint": best_path if ok else None,
                    "best_updated": best_updated if ok else None,
                    "error": err,
                    "log_path": self.log_path,
                    "duration_s": duration_s,
                    "pending_new_labels": int(self._state.get("pending_new_labels", 0)),
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

            if should_chain:
                self.start_async()

