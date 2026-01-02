"""
Main contact prediction network.
Assembles all components: backbone, geometry processor, and MLP head.
"""

import torch
import torch.nn as nn

from models.backbone import FeatureExtractor
from models.geometry import GeometryProcessor


class ContactNet(nn.Module):
    """
    Complete SMPL-X contact prediction network.
    
    Architecture:
    1. Frozen ResNet18 backbone for visual features
    2. Geometry processor for projection and sampling
    3. Pose embedding MLP
    4. Fusion and classification MLP head
    """
    
    def __init__(self, config):
        super(ContactNet, self).__init__()
        
        self.config = config
        
        # 1. Frozen visual feature extractor
        self.backbone = FeatureExtractor(pretrained=config['model']['pretrained'])
        
        # 2. Geometry processor
        self.geometry_processor = GeometryProcessor()
        
        # 3. Pose embedding MLP
        pose_input_dim = 63  # body_pose only
        pose_embed_dim = config['model']['pose_embed_dim']
        
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_input_dim, pose_embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # 4. Classification MLP Head
        total_feat_dim = config['model']['total_feat_dim']
        hidden_dims = config['model']['hidden_dims']
        dropout = config['model']['dropout']
        
        layers = []
        input_dim = total_feat_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
        
        print(f"[ContactNet] Initialized with {self._count_parameters()} trainable parameters")
        
    def _count_parameters(self):
        """Count trainable parameters (excluding frozen backbone)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, images, vertices, normals, pose_params, K, object_bbox):
        """
        Forward pass.
        
        Args:
            images: (B, 3, H, W) - Input RGB images
            vertices: (B, N, 3) - SMPL-X vertices in camera space
            normals: (B, N, 3) - Vertex normals
            pose_params: (B, 63) - Body pose parameters
            K: (B, 3, 3) - Camera intrinsics
            object_bbox: (B, 4) - Object bounding box
            
        Returns:
            logits: (B, N) - Contact logits for each vertex (use sigmoid for probabilities)
        """
        B, N, _ = vertices.shape
        _, _, H, W = images.shape
        img_size = (H, W)
        
        # 1. Extract visual features (no gradients)
        with torch.no_grad():
            feature_maps = self.backbone(images)  # [feat_layer2, feat_layer3]
        
        # 2. Process geometry and get sampling coordinates
        grid_coords, geom_feats = self.geometry_processor(
            vertices, normals, K, img_size, object_bbox
        )
        
        # 3. Sample visual features at projected vertex locations
        visual_feats = self.geometry_processor.sample_features(feature_maps, grid_coords)
        # visual_feats: [B, N, 384] (128 + 256 from layer2 and layer3)
        
        # 4. Pose embedding
        pose_embed = self.pose_mlp(pose_params)  # [B, 32]
        pose_embed = pose_embed.unsqueeze(1).expand(-1, N, -1)  # [B, N, 32]
        
        # 5. Concatenate all features
        # Order: visual + xyz_norm + normals + is_inside_img + is_inside_box + dist_to_center + pose
        all_feats = torch.cat([
            visual_feats,                    # [B, N, 384]
            geom_feats['xyz_norm'],          # [B, N, 3]
            geom_feats['normals'],           # [B, N, 3]
            geom_feats['is_inside_img'],     # [B, N, 1]
            geom_feats['is_inside_box'],     # [B, N, 1]
            geom_feats['dist_to_center'],    # [B, N, 1]
            pose_embed                       # [B, N, 32]
        ], dim=-1)  # [B, N, 425]
        
        # 6. Reshape for BatchNorm (requires [B*N, C])
        all_feats_flat = all_feats.reshape(B * N, -1)  # [B*N, 425]
        
        # 7. Classification
        logits = self.classifier(all_feats_flat)  # [B*N, 1]
        logits = logits.reshape(B, N)  # [B, N]
        
        return logits  # Return logits directly for BCEWithLogitsLoss


if __name__ == "__main__":
    # Test the complete network
    import yaml
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    with open("configs/default.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    model = ContactNet(config).to(device)
    
    # Mock data
    B, N = 2, 10475
    images = torch.randn(B, 3, 512, 512).to(device)
    vertices = torch.randn(B, N, 3).to(device)
    normals = torch.randn(B, N, 3).to(device)
    normals = normals / torch.norm(normals, dim=-1, keepdim=True)
    pose_params = torch.randn(B, 63).to(device)
    
    K = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(device)
    K[:, 0, 0] = 500
    K[:, 1, 1] = 500
    K[:, 0, 2] = 256
    K[:, 1, 2] = 256
    
    bbox = torch.tensor([[100, 100, 300, 400], [50, 50, 200, 300]]).float().to(device)
    
    # Forward pass (returns logits)
    logits = model(images, vertices, normals, pose_params, K, bbox)
    probs = torch.sigmoid(logits)
    
    print("Logits shape:", logits.shape)
    print("Logits range:", logits.min().item(), "-", logits.max().item())
    print("Probs range:", probs.min().item(), "-", probs.max().item())
    
    # Check gradients
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}")
