import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import open3d as o3d

try:
    from .utils import load_json_maybe_list
except Exception:
    # Script execution fallback (when not imported as a package)
    from utils import load_json_maybe_list


@dataclass(frozen=True)
class ContactResult:
    human_contact: np.ndarray  # (Nh,) bool
    obj_contact: np.ndarray  # (No,) bool
    human_interior: np.ndarray
    obj_interior: np.ndarray
    human_proximity: np.ndarray
    obj_proximity: np.ndarray


def load_obj_mesh(obj_path: str, simplify: bool = True, target_triangles: int = 8000):
    """Load an OBJ mesh using Open3D legacy mesh."""
    mesh = o3d.io.read_triangle_mesh(obj_path)
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh: {obj_path}")

    if simplify and len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_triangles))

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    return mesh, vertices, triangles


def _create_smplx_model(smplx_model_path: str, gender: str = "neutral"):
    import smplx

    return smplx.create(
        smplx_model_path,
        model_type="smplx",
        gender=gender,
        use_face_contour=False,
        num_betas=10,
        num_expression_coeffs=10,
        ext="pkl",
        use_pca=False,
    )


def load_smplx_vertices(
    data_dir: str,
    smplx_model_path: str,
    gender: str = "neutral",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load SMPL-X vertices/faces from smplx_parameters.json (supports Ca3OH1 key names)."""
    import torch

    smplx_param_path = os.path.join(data_dir, "smplx_parameters.json")
    smplx_param = load_json_maybe_list(smplx_param_path)

    model = _create_smplx_model(smplx_model_path, gender=gender)

    def _get(name_a: str, name_b: Optional[str] = None, default=None):
        if name_a in smplx_param:
            return smplx_param[name_a]
        if name_b and name_b in smplx_param:
            return smplx_param[name_b]
        return default

    betas = _get("betas", "shape", default=[0.0] * 10)
    body_pose = _get("body_pose", default=[0.0] * 63)
    global_orient = _get("global_orient", "root_pose", default=[0.0, 0.0, 0.0])
    # IMPORTANT: In Ca3OH1 data, `cam_trans` is the camera position, not the
    # SMPL-X body translation. Applying it as `transl` will push the human mesh
    # away from the object (often in the opposite direction), leading to empty
    # contact regions. Only use `transl` when explicitly present.
    transl = _get("transl", default=[0.0, 0.0, 0.0])

    left_hand_pose = _get("left_hand_pose", "lhand_pose", default=None)
    right_hand_pose = _get("right_hand_pose", "rhand_pose", default=None)
    jaw_pose = _get("jaw_pose", default=None)
    expression = _get("expression", "expr", default=None)

    kwargs = {
        "betas": torch.tensor(betas, dtype=torch.float32).reshape(1, -1),
        "body_pose": torch.tensor(body_pose, dtype=torch.float32).reshape(1, -1),
        "global_orient": torch.tensor(global_orient, dtype=torch.float32).reshape(1, -1),
        "transl": torch.tensor(transl, dtype=torch.float32).reshape(1, -1),
        "return_verts": True,
    }

    if left_hand_pose is not None:
        kwargs["left_hand_pose"] = torch.tensor(left_hand_pose, dtype=torch.float32).reshape(1, -1)
    if right_hand_pose is not None:
        kwargs["right_hand_pose"] = torch.tensor(right_hand_pose, dtype=torch.float32).reshape(1, -1)
    if jaw_pose is not None:
        kwargs["jaw_pose"] = torch.tensor(jaw_pose, dtype=torch.float32).reshape(1, -1)
    if expression is not None:
        kwargs["expression"] = torch.tensor(expression, dtype=torch.float32).reshape(1, -1)

    with torch.no_grad():
        output = model(**kwargs)

    vertices = output.vertices[0].detach().cpu().numpy()
    faces = model.faces.astype(np.int32)
    return vertices, faces


def _build_vertex_to_neighbors_map(vertices: np.ndarray, faces: np.ndarray):
    neighbors = {i: set() for i in range(int(vertices.shape[0]))}
    for f in faces:
        v0, v1, v2 = int(f[0]), int(f[1]), int(f[2])
        neighbors[v0].update([v1, v2])
        neighbors[v1].update([v0, v2])
        neighbors[v2].update([v0, v1])
    return neighbors


def calculate_contact_area_interior(
    human_mesh: o3d.geometry.TriangleMesh,
    human_verts: np.ndarray,
    obj_mesh: o3d.geometry.TriangleMesh,
    obj_verts: np.ndarray,
):
    human_scene = o3d.t.geometry.RaycastingScene()
    obj_scene = o3d.t.geometry.RaycastingScene()

    human_tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(human_mesh)
    obj_tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(obj_mesh)

    human_scene.add_triangles(human_tensor_mesh)
    obj_scene.add_triangles(obj_tensor_mesh)

    human_q = o3d.core.Tensor(human_verts.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    obj_q = o3d.core.Tensor(obj_verts.astype(np.float32), dtype=o3d.core.Dtype.Float32)

    human_occ = obj_scene.compute_occupancy(human_q).numpy() > 0.5
    obj_occ = human_scene.compute_occupancy(obj_q).numpy() > 0.5

    return human_occ.astype(bool), obj_occ.astype(bool)


def calculate_contact_area_proximity(
    human_mesh: o3d.geometry.TriangleMesh,
    human_verts: np.ndarray,
    obj_verts: np.ndarray,
    distance_ratio: float = 0.5,
):
    human_faces = np.asarray(human_mesh.triangles)
    human_neighbors = _build_vertex_to_neighbors_map(human_verts, human_faces)

    human_min_neighbor_dist = np.zeros(len(human_verts), dtype=np.float32)
    for i, neigh in human_neighbors.items():
        if neigh:
            dists = [np.linalg.norm(human_verts[i] - human_verts[j]) for j in neigh]
            human_min_neighbor_dist[i] = float(min(dists))
        else:
            human_min_neighbor_dist[i] = np.inf

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_verts)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    human_contact = np.zeros(len(human_verts), dtype=bool)
    obj_contact = np.zeros(len(obj_verts), dtype=bool)

    for i in range(len(human_verts)):
        thr = float(distance_ratio) * float(human_min_neighbor_dist[i])
        if not np.isfinite(thr) or thr <= 0:
            continue
        k, idx, dist2 = kdtree.search_knn_vector_3d(human_verts[i], 1)
        if k <= 0:
            continue
        d = float(np.sqrt(dist2[0]))
        if d < thr:
            human_contact[i] = True
            obj_contact[int(idx[0])] = True

    return human_contact, obj_contact


def calculate_contact_area(
    human_mesh: o3d.geometry.TriangleMesh,
    human_verts: np.ndarray,
    obj_mesh: o3d.geometry.TriangleMesh,
    obj_verts: np.ndarray,
    distance_ratio: float = 0.5,
) -> ContactResult:
    h_in, o_in = calculate_contact_area_interior(human_mesh, human_verts, obj_mesh, obj_verts)
    h_pr, o_pr = calculate_contact_area_proximity(human_mesh, human_verts, obj_verts, distance_ratio=distance_ratio)

    h_contact = h_in | h_pr
    o_contact = o_in | o_pr

    return ContactResult(
        human_contact=h_contact,
        obj_contact=o_contact,
        human_interior=h_in,
        obj_interior=o_in,
        human_proximity=h_pr,
        obj_proximity=o_pr,
    )


def generate_contact_json(
    data_dir: str,
    smplx_model_path: str,
    distance_ratio: float = 0.5,
    simplify_obj: bool = True,
    target_triangles: int = 8000,
) -> None:
    object_mesh_path = os.path.join(data_dir, "object_mesh.obj")
    if not os.path.exists(object_mesh_path):
        raise FileNotFoundError(f"Missing object_mesh.obj: {object_mesh_path}")

    h_verts, h_faces = load_smplx_vertices(data_dir, smplx_model_path=smplx_model_path)
    human_mesh = o3d.geometry.TriangleMesh()
    human_mesh.vertices = o3d.utility.Vector3dVector(h_verts)
    human_mesh.triangles = o3d.utility.Vector3iVector(h_faces)

    obj_mesh, obj_verts, _ = load_obj_mesh(object_mesh_path, simplify=simplify_obj, target_triangles=target_triangles)
    obj_mesh_o3d = o3d.geometry.TriangleMesh()
    obj_mesh_o3d.vertices = o3d.utility.Vector3dVector(obj_verts)
    obj_mesh_o3d.triangles = obj_mesh.triangles

    res = calculate_contact_area(human_mesh, h_verts, obj_mesh_o3d, obj_verts, distance_ratio=distance_ratio)

    contact_list = np.concatenate([res.human_contact.astype(bool), res.obj_contact.astype(bool)]).tolist()
    out_path = os.path.join(data_dir, "contact.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(contact_list, f)
