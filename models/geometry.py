"""
Geometry processor for pixel-aligned feature extraction.
Handles projection, sampling, and corner cases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.geometry_utils import (
    project_3d_to_2d,
    normalize_vertices,
    uv_to_grid_sample_coords,
    compute_bbox_features
)


class GeometryProcessor(nn.Module):
    """
    Core module for geometry processing:
    1. Project 3D vertices to 2D
    2. Compute visibility masks (is_inside_image)
    3. Generate grid sampling coordinates
    4. Compute bbox-related features
    5. Prepare geometric features (normalized coords, normals)
    """
    
    def __init__(self):
        super(GeometryProcessor, self).__init__()
        
    def forward(self, vertices, normals, K, img_size, object_bbox):
        """
        Process geometry and prepare features for sampling.
        
        Args:
            vertices: (B, N, 3) - 3D vertices in camera space
            normals: (B, N, 3) - Vertex normals
            K: (B, 3, 3) - Camera intrinsic matrix
            img_size: tuple (H, W) - Image size
            object_bbox: (B, 4) - Object bounding box [x_min, y_min, x_max, y_max]
            
        Returns:
            grid_coords: (B, N, 1, 2) - Grid sample coordinates (for bilinear sampling)
            geom_feats: dict with:
                - xyz_norm: (B, N, 3) - Normalized vertex coordinates
                - normals: (B, N, 3) - Vertex normals
                - is_inside_img: (B, N, 1) - Visibility flag
                - is_inside_box: (B, N, 1) - Inside bbox flag
                - dist_to_center: (B, N, 1) - Distance to bbox center
        """
        B, N, _ = vertices.shape
        H, W = img_size
        
        # 1. Project 3D to 2D
        uv, depth = project_3d_to_2d(vertices, K)  # [B, N, 2], [B, N]
        
        # 2. Compute is_inside_image mask (critical for corner cases)
        # Check if projection is within image bounds AND depth > 0
        u = uv[:, :, 0]  # [B, N]
        v = uv[:, :, 1]
        
        mask_x = (u >= 0) & (u < W)
        mask_y = (v >= 0) & (v < H)
        mask_z = depth > 0
        
        is_inside_img = (mask_x & mask_y & mask_z).float().unsqueeze(-1)  # [B, N, 1]
        
        # 3. Convert to grid_sample coordinates [-1, 1]
        grid_coords = uv_to_grid_sample_coords(uv, img_size)  # [B, N, 2]
        
        # Reshape for grid_sample (requires 4D input)
        # grid_sample expects: (B, H, W, 2), but we treat N vertices as "pixels"
        # So we reshape to (B, N, 1, 2) to sample N points
        grid_coords = grid_coords.unsqueeze(2)  # [B, N, 1, 2]
        
        # 4. Normalize vertex coordinates (for geometric features)
        xyz_norm, _ = normalize_vertices(vertices)  # [B, N, 3]
        
        # 5. Compute bbox-related features
        is_inside_box, dist_to_center = compute_bbox_features(uv, object_bbox, img_size)
        
        # 6. Package geometric features
        geom_feats = {
            'xyz_norm': xyz_norm,            # [B, N, 3]
            'normals': normals,              # [B, N, 3]
            'is_inside_img': is_inside_img,  # [B, N, 1]
            'is_inside_box': is_inside_box,  # [B, N, 1]
            'dist_to_center': dist_to_center # [B, N, 1]
        }
        
        return grid_coords, geom_feats
    
    def sample_features(self, feature_maps, grid_coords):
        """
        Sample features from feature maps using grid_sample.
        
        Args:
            feature_maps: list of [(B, C1, H1, W1), (B, C2, H2, W2), ...]
            grid_coords: (B, N, 1, 2) - Sampling coordinates in [-1, 1]
            
        Returns:
            sampled_feats: (B, N, C_total) - Concatenated sampled features
        """
        sampled_list = []
        
        for feat_map in feature_maps:
            # F.grid_sample expects: input=(B,C,H,W), grid=(B,H_out,W_out,2)
            # Our grid_coords is (B, N, 1, 2), so output will be (B, C, N, 1)
            sampled = F.grid_sample(
                feat_map,
                grid_coords,
                mode='bilinear',
                padding_mode='zeros',  # CRITICAL: return 0 for out-of-bounds
                align_corners=False
            )
            
            # Reshape: (B, C, N, 1) -> (B, N, C)
            sampled = sampled.squeeze(-1).transpose(1, 2)  # [B, N, C]
            sampled_list.append(sampled)
        
        # Concatenate all features
        sampled_feats = torch.cat(sampled_list, dim=-1)  # [B, N, C_total]
        
        return sampled_feats


if __name__ == "__main__":
    # Test geometry processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processor = GeometryProcessor().to(device)
    
    # Mock data
    B, N = 2, 100
    vertices = torch.randn(B, N, 3).to(device)
    normals = torch.randn(B, N, 3).to(device)
    normals = normals / torch.norm(normals, dim=-1, keepdim=True)
    
    K = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(device)
    K[:, 0, 0] = 500  # fx
    K[:, 1, 1] = 500  # fy
    K[:, 0, 2] = 256  # cx
    K[:, 1, 2] = 256  # cy
    
    bbox = torch.tensor([[100, 100, 300, 400], [50, 50, 200, 300]]).float().to(device)
    
    img_size = (512, 512)
    
    # Forward
    grid_coords, geom_feats = processor(vertices, normals, K, img_size, bbox)
    
    print("Grid coords shape:", grid_coords.shape)
    for key, val in geom_feats.items():
        print(f"{key} shape:", val.shape)
    
    # Test sampling
    feat_maps = [
        torch.randn(B, 128, 64, 64).to(device),
        torch.randn(B, 256, 32, 32).to(device)
    ]
    
    sampled = processor.sample_features(feat_maps, grid_coords)
    print("\nSampled features shape:", sampled.shape)
