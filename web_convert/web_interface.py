#!/usr/bin/env python3
"""
Web-based Multi-User Data Convert Tool

This Flask application provides:
1. Browser-based 3D visualization with Plotly (interactive rotation/zoom)
2. Multi-user support with task distribution
3. Real-time progress tracking
4. Session-based state management
"""

import os
import json
import random
import numpy as np
import open3d as o3d
import plotly.graph_objs as go
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from pathlib import Path
import uuid
import threading
import base64
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import time
import http.client
import mimetypes
import shutil
import tempfile
from typing import List, Optional

# Local converter utilities (no HOIGaussian dependency)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from .contact_geometry import (
        load_obj_mesh,
        load_smplx_vertices,
        calculate_contact_area_interior,
        calculate_contact_area_proximity,
    )
    from .utils import load_json_maybe_list, write_camera_files_from_smplx_params
except Exception:
    from contact_geometry import (
        load_obj_mesh,
        load_smplx_vertices,
        calculate_contact_area_interior,
        calculate_contact_area_proximity,
    )
    from utils import load_json_maybe_list, write_camera_files_from_smplx_params

try:
    from .auto_train import AutoTrainManager, AutoTrainConfig
except Exception:
    from auto_train import AutoTrainManager, AutoTrainConfig


app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static'),
)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Thread pool for background processing
executor = ThreadPoolExecutor(max_workers=4)

# Optional auto-trainer (initialized in initialize_app)
auto_trainer = None

# ---------------------------
# Model status helpers (viewer header)
# ---------------------------

def _basename(path: Optional[str]) -> Optional[str]:
    try:
        return os.path.basename(path) if path else None
    except Exception:
        return None


def _contactnet_host_port() -> tuple[str, int]:
    """Best-effort inference server address for status checks."""
    try:
        if auto_trainer is not None and getattr(auto_trainer, "cfg", None) is not None:
            host = getattr(auto_trainer.cfg, "reload_host", "127.0.0.1")  # type: ignore[attr-defined]
            port = int(getattr(auto_trainer.cfg, "reload_port", 8000))  # type: ignore[attr-defined]
            return str(host), int(port)
    except Exception:
        pass
    return "127.0.0.1", 8000


def _fetch_contactnet_health() -> dict:
    """Best-effort fetch from ContactNet inference server /healthz."""
    host, port = _contactnet_host_port()
    try:
        conn = http.client.HTTPConnection(host, int(port), timeout=5)
        conn.request("GET", "/healthz")
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status < 200 or resp.status >= 300:
            return {"ok": False, "error": f"HTTP {resp.status}: {raw[:2000]!r}"}
        payload = json.loads(raw.decode("utf-8"))
        ckpt = payload.get("checkpoint")
        return {
            "ok": True,
            "checkpoint": ckpt,
            "name": _basename(ckpt) if ckpt else None,
            "epoch": payload.get("epoch"),
            "loaded_at": payload.get("loaded_at"),
            "device": payload.get("device"),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _get_autotrain_last_checkpoint() -> Optional[dict]:
    try:
        if auto_trainer is None:
            return None
        st = auto_trainer.get_state_snapshot()
        last_ckpt = st.get("last_checkpoint")
        last_best = st.get("last_best_checkpoint")
        last_best_updated = st.get("last_best_updated")
        return {
            "last_checkpoint": last_ckpt,
            "last_name": _basename(last_ckpt) if last_ckpt else None,
            "last_ok": st.get("last_train_ok"),
            "last_best_checkpoint": last_best,
            "last_best_name": _basename(last_best) if last_best else None,
            "last_best_updated": last_best_updated,
        }
    except Exception:
        return None


def _build_model_status_payload() -> dict:
    return {"inference": _fetch_contactnet_health(), "autotrain": _get_autotrain_last_checkpoint()}


# Global state management
class TaskManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.tasks = []
        self.current_index = 0
        self.progress = {}
        self.user_sessions = {}  # session_id -> current_task
        self.manual_annotations = {}  # task_id -> selected_faces (temporary storage)
        self.manual_annotation_seeds = {}  # task_id -> selected_faces (seed from current preview)
        self.latest_viz_data = {}  # task_id -> last viz_data sent to clients
        self.contactnet_tmp_dirs = {}  # task_id -> temp dir containing calibration/extrinsic
        
    def load_tasks(self, source_dir, category=None, random_order=False):
        """Load all tasks from source directory."""
        source_path = Path(source_dir)
        
        if category:
            categories = [category]
        else:
            categories = [d.name for d in source_path.iterdir() if d.is_dir()]
            categories.sort()
        
        for cat in categories:
            cat_path = source_path / cat
            subdirs = [d for d in cat_path.iterdir() if d.is_dir()]
            subdirs.sort()
            
            for subdir in subdirs:
                relative_path = os.path.join(cat, subdir.name)
                self.tasks.append({
                    'path': str(subdir),
                    'relative_path': relative_path,
                    'category': cat,
                    'status': 'pending'
                })
        
        # Shuffle tasks if random order is requested
        if random_order:
            random.shuffle(self.tasks)
    
    def get_next_task(self, session_id):
        """Get next available task for a user session."""
        with self.lock:
            # Check if user already has a task in progress
            if session_id in self.user_sessions:
                task_id = self.user_sessions[session_id]
                if task_id < len(self.tasks):
                    task = self.tasks[task_id]
                    if task['status'] == 'in-progress':
                        return task_id, task
            
            # Find next pending task
            for i in range(len(self.tasks)):
                if self.tasks[i]['status'] == 'pending':
                    self.tasks[i]['status'] = 'in-progress'
                    self.user_sessions[session_id] = i
                    return i, self.tasks[i]
            
            return None, None
    
    def complete_task(self, session_id, task_id, decision, distance_ratio, error_msg=None):
        """Mark task as completed."""
        tmp_dir = None
        with self.lock:
            if task_id < len(self.tasks):
                self.tasks[task_id]['status'] = 'completed'
                self.tasks[task_id]['decision'] = decision
                self.tasks[task_id]['distance_ratio'] = distance_ratio
                self.tasks[task_id]['error'] = error_msg
                
                # Remove from user's current task
                if session_id in self.user_sessions:
                    del self.user_sessions[session_id]

            # Pop ContactNet tmp dir under the same lock to avoid races.
            tmp_dir = self.contactnet_tmp_dirs.pop(task_id, None)

        # Best-effort cleanup of any ContactNet temp dir for this task (outside lock)
        if tmp_dir:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    def set_latest_viz(self, task_id, viz_data, distance_ratio=None):
        with self.lock:
            self.latest_viz_data[task_id] = viz_data
            if distance_ratio is not None and task_id < len(self.tasks):
                self.tasks[task_id]['last_distance_ratio'] = float(distance_ratio)

    def get_latest_viz(self, task_id):
        with self.lock:
            return self.latest_viz_data.get(task_id)

    def set_seed_faces(self, task_id, face_indices):
        with self.lock:
            self.manual_annotation_seeds[task_id] = face_indices

    def get_seed_faces(self, task_id):
        with self.lock:
            return self.manual_annotation_seeds.get(task_id)

    def clear_seed_faces(self, task_id):
        with self.lock:
            if task_id in self.manual_annotation_seeds:
                del self.manual_annotation_seeds[task_id]

    def get_contactnet_tmp(self, task_id):
        with self.lock:
            return self.contactnet_tmp_dirs.get(task_id)

    def set_contactnet_tmp(self, task_id, tmp_dir: str):
        with self.lock:
            old = self.contactnet_tmp_dirs.get(task_id)
            self.contactnet_tmp_dirs[task_id] = tmp_dir
        if old and old != tmp_dir:
            try:
                shutil.rmtree(old, ignore_errors=True)
            except Exception:
                pass

    def cleanup_contactnet_tmp(self, task_id):
        tmp_dir = None
        with self.lock:
            tmp_dir = self.contactnet_tmp_dirs.pop(task_id, None)
        if tmp_dir:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
    
    def get_statistics(self):
        """Get current statistics."""
        with self.lock:
            total = len(self.tasks)
            completed = sum(1 for t in self.tasks if t['status'] == 'completed')
            in_progress = sum(1 for t in self.tasks if t['status'] == 'in-progress')
            accepted = sum(1 for t in self.tasks if t.get('decision') == 'accept')
            skipped = sum(1 for t in self.tasks if t.get('decision') == 'skip')
            
            return {
                'total': total,
                'completed': completed,
                'in_progress': in_progress,
                'pending': total - completed - in_progress,
                'accepted': accepted,
                'skipped': skipped,
                'active_users': len(self.user_sessions)
            }

task_manager = TaskManager()

# Configuration
config = {
    'source_dir': None,
    'target_dir': None,
    'object_dir': None,
    'category': None,
    'smplx_model_path': None,
}


def generate_visualization(data_path, relative_path, distance_ratio):
    """Generate Plotly 3D visualization data."""
    try:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        
        object_mesh_path = os.path.join(data_path, "object_mesh.obj")
        smplx_param_path = os.path.join(data_path, "smplx_parameters.json")
        image_path = os.path.join(data_path, "image.jpg")
        
        # Load reference image
        reference_image = None
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                img_data = f.read()
                reference_image = base64.b64encode(img_data).decode('utf-8')
        
        # Load cam translation (for view angle hint only)
        cam_trans = None
        try:
            smplx_param = load_json_maybe_list(smplx_param_path)
            cam_val = smplx_param.get('cam_trans')
            if cam_val is None:
                cam_val = smplx_param.get('transl')
            if cam_val is not None:
                cam_trans = np.asarray(cam_val, dtype=float)
        except Exception:
            cam_trans = None
        
        # Load meshes
        h_verts, h_faces = load_smplx_vertices(
            data_path,
            smplx_model_path=config['smplx_model_path'],
        )
        human_mesh = o3d.geometry.TriangleMesh()
        human_mesh.vertices = o3d.utility.Vector3dVector(h_verts)
        human_mesh.triangles = o3d.utility.Vector3iVector(h_faces)
        
        obj_mesh, obj_verts, obj_faces = load_obj_mesh(
            object_mesh_path,
            simplify=True,
            target_triangles=8000,
        )
        
        obj_mesh_o3d = o3d.geometry.TriangleMesh()
        obj_mesh_o3d.vertices = o3d.utility.Vector3dVector(obj_verts)
        obj_mesh_o3d.triangles = obj_mesh.triangles
        
        # Calculate contact
        human_interior, obj_interior = calculate_contact_area_interior(
            human_mesh, h_verts, obj_mesh_o3d, obj_verts
        )
        human_proximity, obj_proximity = calculate_contact_area_proximity(
            human_mesh,
            h_verts,
            obj_verts,
            distance_ratio=distance_ratio,
        )
        
        human_contact = human_interior | human_proximity
        obj_contact = obj_interior | obj_proximity
        
        # Calculate view angles
        if cam_trans is not None:
            x, y, z = cam_trans
            r = np.sqrt(x**2 + y**2 + z**2)
            elev = np.degrees(np.arcsin(z / r)) if r > 0 else 15
            azim = np.degrees(np.arctan2(y, x))
        else:
            elev, azim = 15, 45
        
        # Prepare data for Plotly
        viz_data = {
            'human_verts': h_verts.tolist(),
            'obj_verts': obj_verts.tolist(),
            'human_faces': h_faces.tolist(),  # Include faces for manual annotation conversion
            'human_contact': human_contact.tolist(),
            'obj_contact': obj_contact.tolist(),
            'human_interior': human_interior.tolist(),
            'human_proximity': human_proximity.tolist(),
            'obj_interior': obj_interior.tolist(),
            'obj_proximity': obj_proximity.tolist(),
            'camera': {
                'elevation': float(elev),
                'azimuth': float(azim)
            },
            'reference_image': reference_image,
            'stats': {
                'total_contact': int(np.sum(human_contact)) + int(np.sum(obj_contact)),
                'human_interior': int(np.sum(human_interior)),
                'human_proximity': int(np.sum(human_proximity)),
                'human_both': int(np.sum(human_interior & human_proximity)),
                'human_total': int(np.sum(human_contact)),
                'obj_interior': int(np.sum(obj_interior)),
                'obj_proximity': int(np.sum(obj_proximity)),
                'obj_both': int(np.sum(obj_interior & obj_proximity)),
                'obj_total': int(np.sum(obj_contact))
            }
        }
        
        return viz_data, None
        
    except Exception as e:
        import traceback
        return None, f"Visualization error: {str(e)}\n{traceback.format_exc()}"


def _write_camera_files_from_smplx_params(data_dir: str) -> None:
    """Generate calibration.json and extrinsic.json in data_dir.

    This mirrors HOIGaussian/prepare/camera.py but is resilient to
    smplx_parameters.json being a list.
    """
    # Delegate to shared implementation
    write_camera_files_from_smplx_params(data_dir)


def ensure_contactnet_camera_tmp(task_id: int, source_data_path: str) -> str:
    """Ensure a per-task temp directory exists containing calibration/extrinsic.

    Returns the temp dir path to be added to ContactNet search paths.
    """
    existing = task_manager.get_contactnet_tmp(task_id)
    if existing:
        calib = os.path.join(existing, 'calibration.json')
        extr = os.path.join(existing, 'extrinsic.json')
        smplx = os.path.join(existing, 'smplx_parameters.json')
        if os.path.exists(calib) and os.path.exists(extr) and os.path.exists(smplx):
            return existing

    if not source_data_path:
        raise ValueError('source_data_path is required')

    src_smplx = os.path.join(source_data_path, 'smplx_parameters.json')
    if not os.path.exists(src_smplx):
        raise FileNotFoundError(f"Missing smplx_parameters.json in source: {src_smplx}")

    base_root = None
    try:
        if config.get('target_dir'):
            base_root = os.path.join(config['target_dir'], '.contactnet_tmp')
            os.makedirs(base_root, exist_ok=True)
    except Exception:
        base_root = None

    if not base_root:
        base_root = tempfile.gettempdir()

    tmp_dir = tempfile.mkdtemp(prefix=f"contactnet_task{int(task_id)}_", dir=base_root)
    shutil.copy2(src_smplx, os.path.join(tmp_dir, 'smplx_parameters.json'))
    _write_camera_files_from_smplx_params(tmp_dir)

    task_manager.set_contactnet_tmp(task_id, tmp_dir)
    return tmp_dir


def _find_required_contactnet_files(search_paths):
    """Find required raw files for ContactNet server-side preprocessing.

    search_paths: list[str] or tuple[str]
    """
    candidates = {
        "image": ["image.jpg", "image.png"],
        "object_mask": ["object_mask.png", "obj_mask.png"],
        "smplx_parameters": ["smplx_parameters.json"],
        "calibration": ["calibration.json"],
        "extrinsic": ["extrinsic.json"],
        "box_annotation": ["box_annotation.json"],
    }

    resolved = {}
    missing = []
    for field, names in candidates.items():
        found = None
        for name in names:
            for base in search_paths:
                if not base:
                    continue
                p = os.path.join(base, name)
                if os.path.exists(p):
                    found = p
                    break
            if found:
                break
        if not found:
            missing.append(field)
        else:
            resolved[field] = found

    return resolved, missing


def _encode_multipart_formdata(files: dict, boundary: str) -> bytes:
    """files: {field: filepath}. Return multipart/form-data body."""
    body = BytesIO()
    for field, filepath in files.items():
        filename = os.path.basename(filepath)
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(
            f"Content-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\n".encode(
                "utf-8"
            )
        )
        body.write(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        with open(filepath, "rb") as f:
            body.write(f.read())
        body.write(b"\r\n")

    body.write(f"--{boundary}--\r\n".encode("utf-8"))
    return body.getvalue()


def predict_contactnet_human_contact(
    data_path: str,
    threshold: float = 0.5,
    host: str = "127.0.0.1",
    port: int = 8000,
    fallback_path: str = None,
    extra_search_paths: Optional[List[str]] = None,
):
    """Call ContactNet inference server and return list[bool] contact mask for human vertices."""
    search_paths = [data_path]
    if extra_search_paths:
        for p in extra_search_paths:
            if p and p not in search_paths:
                search_paths.append(p)
    if fallback_path:
        search_paths.append(fallback_path)
    files, missing = _find_required_contactnet_files(search_paths)
    if missing:
        raise FileNotFoundError(f"Missing required files for ContactNet: {missing}")

    boundary = "----ContactNetBoundary" + uuid.uuid4().hex
    body = _encode_multipart_formdata(files, boundary)

    conn = http.client.HTTPConnection(host, port, timeout=60)
    path = f"/predict?threshold={float(threshold)}&return_probs=0"
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }
    conn.request("POST", path, body=body, headers=headers)
    resp = conn.getresponse()
    raw = resp.read()
    if resp.status < 200 or resp.status >= 300:
        raise RuntimeError(f"ContactNet server error: HTTP {resp.status}: {raw[:2000]!r}")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse ContactNet response: {e}; raw={raw[:2000]!r}")

    if "contact" not in payload:
        raise KeyError(f"ContactNet response missing 'contact' key. Keys={list(payload.keys())}")

    contact = payload["contact"]
    if not isinstance(contact, list):
        raise TypeError(f"ContactNet 'contact' must be a list, got {type(contact)}")

    # Normalize to bool list
    return [bool(x) for x in contact]


def faces_from_vertex_contact_mask(h_faces: np.ndarray, contact_mask: list) -> list:
    """Convert per-vertex contact mask to face indices (face selected if any vertex is contact)."""
    mask = np.asarray(contact_mask, dtype=bool)
    selected = []
    for fi, face in enumerate(h_faces):
        try:
            if mask[face[0]] or mask[face[1]] or mask[face[2]]:
                selected.append(int(fi))
        except Exception:
            continue
    return selected


@app.route('/')
def index():
    """Main page."""
    stats = task_manager.get_statistics()
    return render_template('index.html', stats=stats, config=config)


@app.route('/viewer')
def viewer():
    """Task viewer page."""
    return render_template('viewer.html')


@app.route('/manual_annotate')
def manual_annotate():
    """Manual annotation page."""
    return render_template('manual_annotate.html')


def get_smplx_faces():
    """Get SMPLX face indices from the model."""
    try:
        import smplx
        import torch

        smplx_model_path = config.get('smplx_model_path')
        if not smplx_model_path:
            return None
        
        model = smplx.create(
            smplx_model_path,
            model_type='smplx',
            gender='neutral',
            ext='pkl',
            use_pca=False,
            num_betas=10
        )
        return model.faces.astype(int).tolist()
    except Exception as e:
        print(f"Error loading SMPLX faces: {e}")
        # Return None, will need to compute from mesh
        return None


def load_manual_annotation_data(data_path):
    """Load data for manual annotation."""
    try:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        
        smplx_param_path = os.path.join(data_path, "smplx_parameters.json")
        image_path = os.path.join(data_path, "image.jpg")
        
        # Load reference image
        reference_image = None
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                img_data = f.read()
                reference_image = base64.b64encode(img_data).decode('utf-8')
        
        # Load human mesh vertices and faces
        h_verts, h_faces = load_smplx_vertices(
            data_path,
            smplx_model_path=config['smplx_model_path'],
        )
        
        return {
            'vertices': h_verts.tolist(),
            'faces': h_faces.tolist(),
            'reference_image': reference_image
        }, None
        
    except Exception as e:
        import traceback
        return None, f"Error loading data: {str(e)}\n{traceback.format_exc()}"


@socketio.on('request_manual_annotation_data')
def handle_request_manual_annotation_data(data):
    """Handle request for manual annotation data."""
    task_id = data.get('task_id')
    
    if task_id is None or task_id >= len(task_manager.tasks):
        emit('error', {'message': 'Invalid task ID'})
        return
    
    task = task_manager.tasks[task_id]
    annotation_data, error = load_manual_annotation_data(task['path'])
    
    if error:
        emit('error', {'message': error})
        return
    
    # Get existing annotation if any; otherwise use seeded faces (from current preview)
    existing_annotation = task_manager.manual_annotations.get(task_id)
    if existing_annotation is None:
        existing_annotation = task_manager.get_seed_faces(task_id) or []
    
    emit('manual_annotation_data', {
        'task_id': task_id,
        'task': task,
        'vertices': annotation_data['vertices'],
        'faces': annotation_data['faces'],
        'reference_image': annotation_data['reference_image'],
        'existing_annotation': existing_annotation
    })


@socketio.on('seed_manual_annotation')
def handle_seed_manual_annotation(data):
    """Seed manual annotation from the current preview (vertex contact mask)."""
    task_id = data.get('task_id')
    contact_mask = data.get('contact_mask')

    if task_id is None or task_id >= len(task_manager.tasks):
        return {'success': False, 'error': 'Invalid task ID'}
    if not isinstance(contact_mask, list):
        return {'success': False, 'error': 'contact_mask must be a list'}

    try:
        task = task_manager.tasks[task_id]
        # Load faces from dataset
        _, h_faces = load_smplx_vertices(
            task['path'],
            smplx_model_path=config['smplx_model_path'],
        )
        seed_faces = faces_from_vertex_contact_mask(h_faces, contact_mask)
        task_manager.set_seed_faces(task_id, seed_faces)
        return {'success': True, 'seed_faces': len(seed_faces)}
    except Exception as e:
        return {'success': False, 'error': str(e)}


@socketio.on('request_stats')
def handle_request_stats():
    """Send current task stats (used by index.html)."""
    emit('stats_update', task_manager.get_statistics())


@socketio.on('save_manual_annotation')
def handle_save_manual_annotation(data):
    """Handle saving manual annotation - sends to main viewer first."""
    task_id = data.get('task_id')
    selected_faces = data.get('selected_faces', [])
    hand_regions = data.get('hand_regions', {})
    
    if task_id is None or task_id >= len(task_manager.tasks):
        emit('annotation_saved', {'success': False, 'error': 'Invalid task ID'})
        return
    
    try:
        # Store annotation temporarily for this task
        task_manager.manual_annotations[task_id] = selected_faces

        # Clear any existing seed for this task
        task_manager.clear_seed_faces(task_id)
        
        # Broadcast the manual annotation to all clients (main viewer will pick it up)
        socketio.emit('manual_annotation_received', {
            'task_id': task_id,
            'selected_faces': selected_faces,
            'hand_regions': hand_regions
        })
        
        emit('annotation_saved', {'success': True, 'error': None})
        
    except Exception as e:
        emit('annotation_saved', {'success': False, 'error': str(e)})


def get_hand_vertex_mapping():
    """
    Get mapping from hand region IDs to SMPLX vertex indices.
    
    SMPLX has 10475 vertices. Hand vertices are approximately:
    - Left hand: vertices 5443-5656 (palm) and 5657-5905 (fingers)
    - Right hand: vertices 8017-8230 (palm) and 8231-8479 (fingers)
    
    This is a simplified mapping - actual indices depend on SMPLX version.
    """
    # Approximate SMPLX hand vertex indices
    # These should be adjusted based on actual SMPLX model
    mapping = {}
    
    # Left hand finger segments (approximate ranges)
    left_fingers = {
        'thumb': {'base': range(5657, 5680), 'middle': range(5680, 5705), 'tip': range(5705, 5730)},
        'index': {'base': range(5730, 5760), 'middle': range(5760, 5790), 'tip': range(5790, 5820)},
        'middle': {'base': range(5820, 5845), 'middle': range(5845, 5870), 'tip': range(5870, 5895)},
        'ring': {'base': range(5895, 5920), 'middle': range(5920, 5945), 'tip': range(5945, 5970)},
        'pinky': {'base': range(5970, 5995), 'middle': range(5995, 6020), 'tip': range(6020, 6045)}
    }
    
    # Right hand (offset from left)
    right_offset = 2574  # Approximate offset between left and right hands
    
    for finger, segments in left_fingers.items():
        for segment, verts in segments.items():
            # Left hand - 4 sides (simplified: all vertices for now)
            for side in ['F', 'B', 'L', 'R']:
                region_id = f"left_{finger}_{segment}_{side}"
                mapping[region_id] = list(verts)
            
            # Right hand
            for side in ['F', 'B', 'L', 'R']:
                region_id = f"right_{finger}_{segment}_{side}"
                mapping[region_id] = [v + right_offset for v in verts]
    
    # Palm regions
    left_palm_base = 5443
    palm_size = 40
    for i, finger in enumerate(['thumb', 'index', 'middle', 'ring', 'pinky']):
        start = left_palm_base + i * palm_size
        mapping[f"left_palm_{finger}"] = list(range(start, start + palm_size))
        mapping[f"left_back_{finger}"] = list(range(start, start + palm_size))
        mapping[f"right_palm_{finger}"] = list(range(start + right_offset, start + right_offset + palm_size))
        mapping[f"right_back_{finger}"] = list(range(start + right_offset, start + right_offset + palm_size))
    
    return mapping


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    emit('connected', {'user_id': session['user_id']})
    emit('stats_update', task_manager.get_statistics())
    # Send current auto-train status (if enabled)
    try:
        if auto_trainer is not None:
            emit('training_status', auto_trainer.get_state_snapshot())
    except Exception:
        pass
    # Send current model status (inference checkpoint + latest autotrain checkpoint)
    try:
        emit('model_status', _build_model_status_payload())
    except Exception:
        pass


@socketio.on('request_model_status')
def handle_request_model_status():
    """Client requests the currently loaded inference checkpoint + latest autotrain checkpoint."""
    try:
        emit('model_status', _build_model_status_payload())
    except Exception as e:
        emit('model_status', {"inference": {"ok": False, "error": str(e)}, "autotrain": _get_autotrain_last_checkpoint()})


@socketio.on('request_task')
def handle_request_task():
    """Handle task request from client."""
    session_id = session.get('user_id')
    task_id, task = task_manager.get_next_task(session_id)
    
    if task is None:
        emit('no_tasks', {'message': 'No more tasks available'})
        return

    # Pre-generate ContactNet camera files into a per-task temp dir
    # (calibration.json + extrinsic.json), so ContactNet can run even if
    # raw dataset folder doesn't already contain them.
    try:
        ensure_contactnet_camera_tmp(task_id, task['path'])
    except Exception as e:
        # Don't block base visualization; ContactNet run will surface missing
        # files as an error if this fails.
        print(f"[ContactNet] Failed to prepare camera files for task {task_id}: {e}")
    
    # Generate visualization for this task
    viz_data, error = generate_visualization(
        task['path'], 
        task['relative_path'], 
        0.5  # Initial distance ratio
    )
    
    if error:
        emit('error', {'message': error})
        return
    
    emit('task_data', {
        'task_id': task_id,
        'task': task,
        'viz_data': viz_data
    })

    task_manager.set_latest_viz(task_id, viz_data, distance_ratio=0.5)
    
    # Broadcast stats update to all clients
    socketio.emit('stats_update', task_manager.get_statistics())


@socketio.on('update_visualization')
def handle_update_visualization(data):
    """Handle visualization update with new distance ratio."""
    task_id = data['task_id']
    distance_ratio = data['distance_ratio']
    
    if task_id >= len(task_manager.tasks):
        emit('error', {'message': 'Invalid task ID'})
        return
    
    task = task_manager.tasks[task_id]
    viz_data, error = generate_visualization(
        task['path'],
        task['relative_path'],
        distance_ratio
    )
    
    if error:
        emit('error', {'message': error})
        return
    
    emit('visualization_updated', {'viz_data': viz_data})

    task_manager.set_latest_viz(task_id, viz_data, distance_ratio=distance_ratio)


@socketio.on('run_contactnet')
def handle_run_contactnet(data):
    """Run ContactNet inference (localhost:8000) and update preview human contact."""
    task_id = data.get('task_id')
    threshold = data.get('threshold', 0.5)
    sid = request.sid

    if task_id is None or task_id >= len(task_manager.tasks):
        emit('error', {'message': 'Invalid task ID'})
        return

    def _work():
        try:
            task = task_manager.tasks[task_id]

            # Ensure calibration/extrinsic exist for this task
            tmp_cam_dir = None
            try:
                tmp_cam_dir = ensure_contactnet_camera_tmp(task_id, task['path'])
            except Exception as e:
                tmp_cam_dir = None
                print(f"[ContactNet] Failed to ensure camera files for task {task_id}: {e}")

            # Optional: try converted target dir as fallback for missing raw files
            fallback_dir = None
            try:
                if config.get('target_dir') and task.get('relative_path'):
                    cand = os.path.join(config['target_dir'], task['relative_path'])
                    if os.path.exists(cand):
                        fallback_dir = cand
            except Exception:
                fallback_dir = None

            # Base viz data (for verts/faces/object stats)
            base_viz = task_manager.get_latest_viz(task_id)
            if base_viz is None:
                ratio = task.get('last_distance_ratio', 0.5)
                base_viz, err = generate_visualization(task['path'], task['relative_path'], ratio)
                if err:
                    raise RuntimeError(err)

            extra_paths = [tmp_cam_dir] if tmp_cam_dir else None
            human_contact = predict_contactnet_human_contact(
                task['path'],
                threshold=float(threshold),
                fallback_path=fallback_dir,
                extra_search_paths=extra_paths,
            )

            # Safety: match human vertex count
            n = len(base_viz.get('human_verts', []))
            if n and len(human_contact) != n:
                # Try to clip/pad to avoid hard failure; but report mismatch
                if len(human_contact) > n:
                    human_contact = human_contact[:n]
                else:
                    human_contact = human_contact + [False] * (n - len(human_contact))

            # Build updated viz
            updated = json.loads(json.dumps(base_viz))
            updated['human_contact'] = list(human_contact)
            updated['human_interior'] = list(human_contact)
            updated['human_proximity'] = [False for _ in human_contact]

            # Update stats (keep object stats from base)
            human_total = int(sum(1 for x in human_contact if x))
            obj_total = int(updated.get('stats', {}).get('obj_total', 0))
            updated['stats']['human_total'] = human_total
            updated['stats']['human_interior'] = human_total
            updated['stats']['human_proximity'] = 0
            updated['stats']['human_both'] = 0
            updated['stats']['total_contact'] = human_total + obj_total

            task_manager.set_latest_viz(task_id, updated, distance_ratio=task.get('last_distance_ratio', 0.5))

            socketio.emit('contactnet_updated', {'task_id': task_id, 'viz_data': updated}, to=sid)
        except Exception as e:
            socketio.emit('error', {'message': f'ContactNet error: {str(e)}'}, to=sid)

    executor.submit(_work)


@socketio.on('submit_decision')
def handle_submit_decision(data):
    """Handle task decision from client."""
    session_id = session.get('user_id')
    task_id = data['task_id']
    decision = data['decision']  # 'accept' or 'skip'
    distance_ratio = data['distance_ratio']
    manual_annotation = data.get('manual_annotation')  # Face indices from manual annotation
    
    # Mark task as completed immediately
    task_manager.complete_task(session_id, task_id, decision, distance_ratio, None)
    
    # Send immediate response to client
    emit('decision_accepted', {'success': True, 'error': None})
    
    # Process the task in background if accepted
    if decision == 'accept':
        def process_in_background():
            error_msg = None
            try:
                try:
                    from .data_convert import process_single_dataset
                except ImportError:
                    from data_convert import process_single_dataset
                
                task = task_manager.tasks[task_id]

                # Track whether this task creates a NEW labeled sample in target_dir
                target_path = os.path.join(config['target_dir'], task['relative_path'])
                contact_output_path = os.path.join(target_path, "contact.json")
                contact_existed_before = os.path.exists(contact_output_path)
                
                # Process dataset (copy files, run camera.py, normal.py)
                result_decision, rel_path, ratio, error = process_single_dataset(
                    task['path'],
                    task['relative_path'],
                    config['source_dir'],
                    config['target_dir'],
                    config['object_dir'],
                    distance_ratio,
                    smplx_model_path=config.get('smplx_model_path'),
                    interactive=False
                )
                error_msg = error
                
                # If manual annotation provided, override contact.json
                if manual_annotation and not error:
                    try:
                        # target_path/contact_output_path already computed above
                        
                        # Load mesh data to get vertex count
                        h_verts, h_faces = load_smplx_vertices(
                            task['path'],
                            smplx_model_path=config['smplx_model_path'],
                        )
                        num_human_verts = len(h_verts)
                        
                        # Create contact mask from selected faces
                        human_contact = np.zeros(num_human_verts, dtype=bool)
                        for face_idx in manual_annotation:
                            if face_idx < len(h_faces):
                                for v_idx in h_faces[face_idx]:
                                    if v_idx < num_human_verts:
                                        human_contact[v_idx] = True
                        
                        # Load object mesh to create full contact list
                        object_mesh_path = os.path.join(target_path, "object_mesh.obj")
                        if os.path.exists(object_mesh_path):
                            obj_mesh, obj_verts, _ = load_obj_mesh(
                                object_mesh_path,
                                simplify=True,
                                target_triangles=8000
                            )
                            obj_contact = np.zeros(len(obj_verts), dtype=bool)
                        else:
                            obj_contact = np.zeros(0, dtype=bool)
                        
                        # Save contact.json with manual annotation
                        contact_list = np.concatenate([
                            human_contact.astype(bool),
                            obj_contact.astype(bool)
                        ]).tolist()
                        
                        with open(contact_output_path, 'w') as f:
                            json.dump(contact_list, f)
                            
                        print(f"Saved manual annotation with {np.sum(human_contact)} contact vertices")
                    except Exception as e:
                        error_msg = f"Error saving manual annotation: {str(e)}"
                        
            except Exception as e:
                error_msg = str(e)

            # If successful AND this is a new labeled sample, notify auto-trainer
            try:
                if (error_msg is None) and (not contact_existed_before) and os.path.exists(contact_output_path):
                    if auto_trainer is not None:
                        # Pass the absolute sample dir so auto-train can track "recent 10" precisely.
                        auto_trainer.note_new_label(sample_dir=target_path)
            except Exception:
                pass
            
            # Update task with error if any
            if error_msg:
                with task_manager.lock:
                    task_manager.tasks[task_id]['error'] = error_msg
            
            # Clear temporary manual annotation for this task
            if task_id in task_manager.manual_annotations:
                del task_manager.manual_annotations[task_id]
            
            # Save progress
            save_progress()
            
            # Broadcast stats update
            socketio.emit('stats_update', task_manager.get_statistics())
        
        # Submit to thread pool
        executor.submit(process_in_background)
    else:
        # Clear temporary manual annotation for this task
        if task_id in task_manager.manual_annotations:
            del task_manager.manual_annotations[task_id]
        
        # Save progress for skip
        save_progress()
        socketio.emit('stats_update', task_manager.get_statistics())


def save_progress():
    """Save progress to file."""
    progress_path = os.path.join(config['target_dir'], 'progress.json')
    progress = {}
    
    for i, task in enumerate(task_manager.tasks):
        if task['status'] == 'completed':
            progress[task['relative_path']] = {
                'decision': task.get('decision', 'skip'),
                'distance_ratio': task.get('distance_ratio', 0.5),
                'error': task.get('error'),
                'category': task['category']
            }
    
    with open(progress_path, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def load_progress():
    """Load existing progress."""
    progress_path = os.path.join(config['target_dir'], 'progress.json')
    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                
            # Mark tasks as completed based on progress
            for task in task_manager.tasks:
                if task['relative_path'] in progress:
                    task['status'] = 'completed'
                    prog = progress[task['relative_path']]
                    task['decision'] = prog['decision']
                    task['distance_ratio'] = prog['distance_ratio']
                    task['error'] = prog.get('error')
        except Exception as e:
            print(f"Error loading progress: {e}")


def initialize_app(
    source_dir,
    target_dir,
    object_dir,
    category=None,
    random_order=False,
    smplx_model_path=None,
    *,
    auto_train: bool = False,
    auto_train_every: int = 20,
    auto_train_epochs: int = 10,
    auto_train_base_config: str = "configs/default.yaml",
    auto_train_initial_checkpoint: Optional[str] = None,
    contactnet_reload_host: str = "127.0.0.1",
    contactnet_reload_port: int = 8000,
):
    """Initialize the application with configuration."""
    global auto_trainer
    config['source_dir'] = source_dir
    config['target_dir'] = target_dir
    config['object_dir'] = object_dir
    config['category'] = category
    if smplx_model_path is None:
        smplx_model_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'smplx_models'))
    config['smplx_model_path'] = smplx_model_path
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Load tasks
    task_manager.load_tasks(source_dir, category, random_order)
    
    # Load existing progress
    load_progress()

    # Optional: initialize auto-trainer (single-process, best-effort)
    if auto_train:
        try:
            auto_cfg = AutoTrainConfig(
                enabled=True,
                every_n_new_labels=int(auto_train_every),
                additional_epochs=int(auto_train_epochs),
                base_train_config=str(auto_train_base_config),
                work_dir=os.path.join(target_dir, "_autotrain"),
                initial_checkpoint=auto_train_initial_checkpoint,
                reload_host=str(contactnet_reload_host),
                reload_port=int(contactnet_reload_port),
            )
            auto_trainer = AutoTrainManager(target_dir=target_dir, cfg=auto_cfg, socketio=socketio)
            checkpoint_info = f"initial_checkpoint={auto_cfg.initial_checkpoint}" if auto_cfg.initial_checkpoint else "auto-detect best_model.pth"
            print(
                f"[AutoTrain] enabled: every {auto_cfg.every_n_new_labels} new labels -> +{auto_cfg.additional_epochs} epochs; "
                f"base_config={auto_cfg.base_train_config}; work_dir={auto_cfg.work_dir}; {checkpoint_info}"
            )
        except Exception as e:
            auto_trainer = None
            print(f"[AutoTrain] Failed to initialize: {e}")
    
    print(f"Initialized with {len(task_manager.tasks)} tasks")
    print(f"Order: {'Random' if random_order else 'By Category'}")
    print(f"Statistics: {task_manager.get_statistics()}")


def run_server(host='0.0.0.0', port=5000):
    """Run the Flask server."""
    import socket
    
    # Get local IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = '127.0.0.1'
    
    print(f"\n{'='*70}")
    print(f"Web-based Multi-User Data Convert Tool")
    print(f"{'='*70}")
    print(f"Local access:   http://127.0.0.1:{port}")
    print(f"Network access: http://{local_ip}:{port}")
    print(f"\nShare the network URL with team members on the same LAN")
    print(f"{'='*70}\n")
    
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Contact web-based dataset convert viewer')
    parser.add_argument('--source_dir', required=True, help='Source dataset root (e.g. Ca3OH1/data)')
    parser.add_argument('--target_dir', required=True, help='Target output root (e.g. Contact/data_contact)')
    parser.add_argument('--object_dir', default=None, help='Optional object artifact root (for obj_pcd_h_align/object_mesh)')
    parser.add_argument('--category', default=None, help='Optional category subfolder to process')
    parser.add_argument('--random_order', action='store_true', help='Shuffle task order')
    parser.add_argument('--smplx_model_path', default=None, help='Path to Contact/smplx_models (defaults to ../smplx_models)')
    # Auto-train (MVP): every N NEW labeled samples -> train + reload inference server
    parser.add_argument('--auto_train', action='store_true', help='Enable auto-training loop (every N new labels)')
    parser.add_argument('--auto_train_every', type=int, default=20, help='Trigger auto-train after N NEW labeled samples')
    parser.add_argument('--auto_train_epochs', type=int, default=10, help='Train for K additional epochs on each trigger')
    parser.add_argument('--auto_train_base_config', type=str, default='configs/default.yaml', help='Base train config YAML to derive from')
    parser.add_argument('--contactnet_reload_host', type=str, default='127.0.0.1', help='Host of ContactNet inference server for /reload')
    parser.add_argument('--contactnet_reload_port', type=int, default=8000, help='Port of ContactNet inference server for /reload')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    initialize_app(
        os.path.abspath(args.source_dir),
        os.path.abspath(args.target_dir),
        os.path.abspath(args.object_dir) if args.object_dir else None,
        category=args.category,
        random_order=args.random_order,
        smplx_model_path=os.path.abspath(args.smplx_model_path) if args.smplx_model_path else None,
        auto_train=bool(args.auto_train),
        auto_train_every=int(args.auto_train_every),
        auto_train_epochs=int(args.auto_train_epochs),
        auto_train_base_config=str(args.auto_train_base_config),
        contactnet_reload_host=str(args.contactnet_reload_host),
        contactnet_reload_port=int(args.contactnet_reload_port),
    )
    run_server(host=args.host, port=args.port)
