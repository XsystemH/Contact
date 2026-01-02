"""
Visualization utilities for debugging and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F
import cv2
from mpl_toolkits.mplot3d import Axes3D

from utils.geometry_utils import project_3d_to_2d, uv_to_grid_sample_coords


def denormalize_image(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor to [0, 1] range for visualization.
    
    Args:
        img_tensor: (3, H, W) or (B, 3, H, W)
        
    Returns:
        Denormalized image in [0, 1]
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if img_tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    
    return img


def visualize_projection(image, vertices, K, contact_values=None, bbox=None, save_path=None, title=None):
    """
    Visualize projected vertices on image.
    
    Args:
        image: (3, H, W) - Normalized image tensor
        vertices: (N, 3) - Vertices in camera space
        K: (3, 3) - Camera intrinsics
        contact_values: (N,) - Optional values for coloring (e.g., predicted probs or GT labels)
        bbox: (4,) - Optional bbox [x_min, y_min, x_max, y_max] in pixel coords of the SAME resized image
        save_path: Path to save figure
        title: Optional plot title
    """
    # Denormalize image
    img = denormalize_image(image).permute(1, 2, 0).cpu().numpy()
    H, W = img.shape[:2]
    
    # Project vertices
    vertices_np = vertices.cpu().numpy()
    K_np = K.cpu().numpy()
    
    # Project: [3, 3] @ [3, N] -> [3, N]
    vertices_2d = K_np @ vertices_np.T  # [3, N]
    u = vertices_2d[0] / (vertices_2d[2] + 1e-8)
    v = vertices_2d[1] / (vertices_2d[2] + 1e-8)
    depth = vertices_2d[2]
    
    # Filter valid points (inside image and depth > 0)
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depth > 0)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    # Draw bbox if provided
    if bbox is not None:
        if isinstance(bbox, torch.Tensor):
            bbox_np = bbox.detach().cpu().numpy().astype(np.float32)
        else:
            bbox_np = np.asarray(bbox, dtype=np.float32)

        if bbox_np.shape[-1] == 4:
            x_min, y_min, x_max, y_max = bbox_np.tolist()
            # Clamp to image bounds for display
            x_min = float(np.clip(x_min, 0, W - 1))
            x_max = float(np.clip(x_max, 0, W - 1))
            y_min = float(np.clip(y_min, 0, H - 1))
            y_max = float(np.clip(y_max, 0, H - 1))

            rect = patches.Rectangle(
                (x_min, y_min),
                max(0.0, x_max - x_min),
                max(0.0, y_max - y_min),
                linewidth=2,
                edgecolor='lime',
                facecolor='none',
                alpha=0.9
            )
            ax.add_patch(rect)
            ax.text(
                x_min,
                max(0.0, y_min - 5.0),
                "bbox",
                color='lime',
                fontsize=10,
                bbox=dict(facecolor='black', alpha=0.4, pad=2, edgecolor='none')
            )
    
    if contact_values is not None:
        values_np = contact_values.detach().cpu().numpy()
        sc = ax.scatter(u[valid], v[valid], c=values_np[valid], cmap='hot', s=1, alpha=0.8, vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, shrink=0.7)
    else:
        ax.scatter(u[valid], v[valid], c='blue', s=1, alpha=0.6)

    ax.set_title(title or 'Projected Vertices on Image')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def visualize_contact_heatmap(vertices, contact_probs, contact_labels=None, save_path=None):
    """
    Visualize contact predictions as 3D heatmap.
    
    Args:
        vertices: (N, 3) - Vertex positions
        contact_probs: (N,) - Predicted contact probabilities
        contact_labels: (N,) - Optional ground truth labels
        save_path: Path to save figure
    """
    vertices_np = vertices.cpu().numpy()
    probs_np = contact_probs.cpu().numpy()
    
    fig = plt.figure(figsize=(15, 5))
    
    # Predicted heatmap
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(vertices_np[:, 0], vertices_np[:, 1], vertices_np[:, 2],
                          c=probs_np, cmap='hot', s=1, vmin=0, vmax=1)
    ax1.set_title('Predicted Contact Probability')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5)
    
    if contact_labels is not None:
        labels_np = contact_labels.cpu().numpy()
        
        # Ground truth
        ax2 = fig.add_subplot(132, projection='3d')
        scatter2 = ax2.scatter(vertices_np[:, 0], vertices_np[:, 1], vertices_np[:, 2],
                              c=labels_np, cmap='hot', s=1, vmin=0, vmax=1)
        ax2.set_title('Ground Truth Contact')
        plt.colorbar(scatter2, ax=ax2, shrink=0.5)
        
        # Error map
        ax3 = fig.add_subplot(133, projection='3d')
        error = np.abs(probs_np - labels_np)
        scatter3 = ax3.scatter(vertices_np[:, 0], vertices_np[:, 1], vertices_np[:, 2],
                              c=error, cmap='viridis', s=1, vmin=0, vmax=1)
        ax3.set_title('Absolute Error')
        plt.colorbar(scatter3, ax=ax3, shrink=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def visualize_batch_predictions(batch, predictions, num_samples=4, save_dir=None):
    """
    Visualize predictions for a batch of samples.
    
    Args:
        batch: Dictionary containing batch data
        predictions: (B, N) - Predicted contact probabilities
        num_samples: Number of samples to visualize
        save_dir: Directory to save figures
    """
    B = batch['image'].shape[0]
    num_samples = min(num_samples, B)
    
    for i in range(num_samples):
        # Extract sample data
        image = batch['image'][i]
        vertices = batch['vertices'][i]
        K = batch['K'][i]
        bbox = batch.get('object_bbox', None)
        bbox_i = bbox[i] if isinstance(bbox, torch.Tensor) else None
        contact_labels = batch['contact_labels'][i]
        contact_pred = predictions[i]
        sample_id = batch['sample_ids'][i]
        mask_dist_field = batch.get('mask_dist_field', None)
        mask_dist_field_i = mask_dist_field[i] if isinstance(mask_dist_field, torch.Tensor) else None
        
        # Projection visualization
        if save_dir:
            proj_path = f"{save_dir}/{sample_id}_projection.png"
        else:
            proj_path = None
        
        visualize_projection(
            image,
            vertices,
            K,
            contact_values=contact_pred,
            bbox=bbox_i,
            save_path=proj_path,
            title='Projected Vertices (Predicted Contact)'
        )

        # Extra debug: visualize per-vertex sampled mask distance feature f_mask_dist
        if mask_dist_field_i is not None:
            # mask_dist_field_i: (1, H, W) in [0, 1] (0=near object, 1=far)
            H, W = image.shape[-2], image.shape[-1]
            mask_dist_field_i = mask_dist_field_i.unsqueeze(0)  # (1, 1, H, W)

            with torch.no_grad():
                uv, depth = project_3d_to_2d(vertices.unsqueeze(0), K.unsqueeze(0))  # (1,N,2), (1,N)
                u = uv[:, :, 0]
                v = uv[:, :, 1]
                inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depth > 0)
                inside = inside.float().unsqueeze(-1)  # (1,N,1)

                grid = uv_to_grid_sample_coords(uv, (H, W)).unsqueeze(2)  # (1,N,1,2)
                sampled = F.grid_sample(
                    mask_dist_field_i,
                    grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False
                )  # (1,1,N,1)
                sampled = sampled.squeeze(-1).transpose(1, 2)  # (1,N,1)
                sampled = torch.where(inside > 0.5, sampled, torch.ones_like(sampled))
                f_mask_dist = sampled.squeeze(0).squeeze(-1).clamp(0, 1)  # (N,)

            if save_dir:
                maskdist_path = f"{save_dir}/{sample_id}_mask_dist.png"
            else:
                maskdist_path = None

            visualize_projection(
                image,
                vertices,
                K,
                contact_values=f_mask_dist,
                bbox=bbox_i,
                save_path=maskdist_path,
                title='Projected Vertices (f_mask_dist from Dilated Distance Field)'
            )
        
        # Heatmap visualization
        if save_dir:
            heatmap_path = f"{save_dir}/{sample_id}_heatmap.png"
        else:
            heatmap_path = None
        
        visualize_contact_heatmap(vertices, contact_pred, contact_labels, heatmap_path)


def plot_training_curves(train_losses, val_losses, save_path=None, val_epochs=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save figure
        val_epochs: Optional list/array of 1-based epoch indices for each val loss
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(train_epochs, train_losses, label='Train Loss', linewidth=2)

    if val_losses is not None and len(val_losses) > 0:
        if val_epochs is None or len(val_epochs) != len(val_losses):
            # Fallback: spread validation points across training epoch range
            if len(train_losses) > 0:
                val_x = np.linspace(1, len(train_losses), num=len(val_losses))
            else:
                val_x = np.arange(1, len(val_losses) + 1)
        else:
            val_x = np.asarray(val_epochs)

        ax.plot(val_x, val_losses, label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def compute_metrics(predictions, labels, threshold=0.5):
    """
    Compute classification metrics.
    
    Args:
        predictions: (N,) - Predicted probabilities
        labels: (N,) - Ground truth labels
        threshold: Classification threshold
        
    Returns:
        dict with precision, recall, f1, accuracy
    """
    preds_binary = (predictions > threshold).float()
    labels = labels.float()
    
    tp = ((preds_binary == 1) & (labels == 1)).sum().item()
    fp = ((preds_binary == 1) & (labels == 0)).sum().item()
    tn = ((preds_binary == 0) & (labels == 0)).sum().item()
    fn = ((preds_binary == 0) & (labels == 1)).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


if __name__ == "__main__":
    # Test visualization functions
    print("Visualization utilities ready.")
    
    # Mock data for testing
    H, W = 512, 512
    N = 10475
    
    image = torch.randn(3, H, W)
    vertices = torch.randn(N, 3)
    K = torch.eye(3)
    K[0, 0] = K[1, 1] = 500
    K[0, 2] = W / 2
    K[1, 2] = H / 2
    
    contact_labels = torch.bernoulli(torch.ones(N) * 0.1)
    contact_probs = torch.rand(N)
    
    print("Test visualization with mock data...")
    # visualize_projection(image, vertices, K, contact_labels)
    # visualize_contact_heatmap(vertices, contact_probs, contact_labels)
