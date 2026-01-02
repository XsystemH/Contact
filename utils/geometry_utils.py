"""
Geometry utility functions for coordinate transformations and projections.
"""

import torch
import numpy as np


def project_3d_to_2d(vertices, K):
    """
    Project 3D vertices to 2D image coordinates using camera intrinsics.
    
    Args:
        vertices: (B, N, 3) - 3D vertices in camera space
        K: (B, 3, 3) - Camera intrinsic matrix
        
    Returns:
        uv: (B, N, 2) - 2D pixel coordinates
        depth: (B, N) - Depth values (z-coordinate)
    """
    B, N, _ = vertices.shape
    
    # Homogeneous coordinates
    vertices_homo = torch.cat([vertices, torch.ones(B, N, 1, device=vertices.device)], dim=-1)
    
    # Project: [B, 3, 3] @ [B, 3, N] -> [B, 3, N]
    vertices_2d = torch.bmm(K, vertices[:, :, :3].transpose(1, 2))  # [B, 3, N]
    
    # Normalize by depth
    depth = vertices_2d[:, 2, :]  # [B, N]
    u = vertices_2d[:, 0, :] / (depth + 1e-8)  # [B, N]
    v = vertices_2d[:, 1, :] / (depth + 1e-8)  # [B, N]
    
    uv = torch.stack([u, v], dim=-1)  # [B, N, 2]
    
    return uv, depth


def normalize_vertices(vertices):
    """
    Normalize vertices to [-1, 1] range for better numerical stability.
    
    Args:
        vertices: (B, N, 3) - 3D vertices
        
    Returns:
        normalized_vertices: (B, N, 3)
        stats: dict with mean and std for de-normalization
    """
    B, N, _ = vertices.shape
    
    # Compute per-batch statistics
    mean = vertices.mean(dim=1, keepdim=True)  # [B, 1, 3]
    std = vertices.std(dim=1, keepdim=True) + 1e-8  # [B, 1, 3]
    
    normalized = (vertices - mean) / std
    
    # Clamp to reasonable range
    normalized = torch.clamp(normalized, -3, 3)
    
    stats = {'mean': mean, 'std': std}
    
    return normalized, stats


def compute_vertex_normals(vertices, faces):
    """
    Compute per-vertex normals from mesh faces.
    
    Args:
        vertices: (B, N, 3) - Vertex positions
        faces: (F, 3) - Face indices (shared across batch)
        
    Returns:
        normals: (B, N, 3) - Per-vertex normals
    """
    B, N, _ = vertices.shape
    F = faces.shape[0]
    
    # Get face vertices
    v0 = vertices[:, faces[:, 0]]  # [B, F, 3]
    v1 = vertices[:, faces[:, 1]]  # [B, F, 3]
    v2 = vertices[:, faces[:, 2]]  # [B, F, 3]
    
    # Compute face normals
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = torch.cross(edge1, edge2, dim=-1)  # [B, F, 3]
    
    # Normalize face normals
    face_normals = face_normals / (torch.norm(face_normals, dim=-1, keepdim=True) + 1e-8)
    
    # Accumulate to vertices (average)
    vertex_normals = torch.zeros(B, N, 3, device=vertices.device)
    for i in range(3):
        vertex_normals.scatter_add_(1, 
                                     faces[:, i].unsqueeze(0).unsqueeze(-1).expand(B, -1, 3),
                                     face_normals)
    
    # Normalize vertex normals
    vertex_normals = vertex_normals / (torch.norm(vertex_normals, dim=-1, keepdim=True) + 1e-8)
    
    return vertex_normals


def world_to_camera(vertices, R, T):
    """
    Transform vertices from world space to camera space.
    
    Args:
        vertices: (B, N, 3) - Vertices in world space
        R: (B, 3, 3) - Rotation matrix
        T: (B, 3) - Translation vector
        
    Returns:
        vertices_cam: (B, N, 3) - Vertices in camera space
    """
    B, N, _ = vertices.shape
    
    # Apply rotation: [B, 3, 3] @ [B, 3, N] -> [B, 3, N]
    vertices_rot = torch.bmm(R, vertices.transpose(1, 2))  # [B, 3, N]
    
    # Apply translation
    vertices_cam = vertices_rot + T.unsqueeze(-1)  # [B, 3, N]
    
    return vertices_cam.transpose(1, 2)  # [B, N, 3]


def scale_intrinsics(K, original_size, target_size):
    """
    Scale camera intrinsics when image is resized.
    
    Args:
        K: (B, 3, 3) or (3, 3) - Original intrinsic matrix
        original_size: (H, W) - Original image size
        target_size: (H, W) - Target image size
        
    Returns:
        K_scaled: Same shape as K - Scaled intrinsic matrix
    """
    scale_x = target_size[1] / original_size[1]
    scale_y = target_size[0] / original_size[0]
    
    K_scaled = K.clone()
    
    if K.dim() == 2:  # Single matrix
        K_scaled[0, 0] *= scale_x  # fx
        K_scaled[1, 1] *= scale_y  # fy
        K_scaled[0, 2] *= scale_x  # cx
        K_scaled[1, 2] *= scale_y  # cy
    else:  # Batch of matrices
        K_scaled[:, 0, 0] *= scale_x
        K_scaled[:, 1, 1] *= scale_y
        K_scaled[:, 0, 2] *= scale_x
        K_scaled[:, 1, 2] *= scale_y
    
    return K_scaled


def compute_bbox_features(uv, bbox, img_size):
    """
    Compute features related to object bounding box.
    
    Args:
        uv: (B, N, 2) - 2D pixel coordinates
        bbox: (B, 4) - Bounding box [x_min, y_min, x_max, y_max]
        img_size: (H, W) - Image size
        
    Returns:
        is_inside_box: (B, N, 1) - Binary flag if point is inside bbox
        dist_to_center: (B, N, 1) - Normalized distance to bbox center
    """
    B, N, _ = uv.shape
    H, W = img_size
    
    # Extract bbox coordinates
    x_min = bbox[:, 0].unsqueeze(1)  # [B, 1]
    y_min = bbox[:, 1].unsqueeze(1)
    x_max = bbox[:, 2].unsqueeze(1)
    y_max = bbox[:, 3].unsqueeze(1)
    
    # Check if inside bbox
    u = uv[:, :, 0]  # [B, N]
    v = uv[:, :, 1]
    
    inside_x = (u >= x_min) & (u <= x_max)
    inside_y = (v >= y_min) & (v <= y_max)
    is_inside_box = (inside_x & inside_y).float().unsqueeze(-1)  # [B, N, 1]
    
    # Compute distance to bbox center
    bbox_center_x = (x_min + x_max) / 2  # [B, 1]
    bbox_center_y = (y_min + y_max) / 2
    
    dist_x = (u - bbox_center_x) / W
    dist_y = (v - bbox_center_y) / H
    dist_to_center = torch.sqrt(dist_x**2 + dist_y**2).unsqueeze(-1)  # [B, N, 1]
    
    # Normalize distance
    dist_to_center = torch.clamp(dist_to_center, 0, 1)
    
    return is_inside_box, dist_to_center


def uv_to_grid_sample_coords(uv, img_size):
    """
    Convert pixel coordinates to grid_sample normalized coordinates [-1, 1].
    
    Args:
        uv: (B, N, 2) - Pixel coordinates (u, v)
        img_size: (H, W) - Image size
        
    Returns:
        grid_coords: (B, N, 2) - Normalized coordinates for grid_sample
    """
    H, W = img_size
    
    u = uv[:, :, 0]  # [B, N]
    v = uv[:, :, 1]
    
    # Normalize to [-1, 1]
    # grid_sample expects (x, y) where x is width, y is height
    x_norm = 2.0 * u / (W - 1) - 1.0
    y_norm = 2.0 * v / (H - 1) - 1.0
    
    grid_coords = torch.stack([x_norm, y_norm], dim=-1)  # [B, N, 2]
    
    return grid_coords
