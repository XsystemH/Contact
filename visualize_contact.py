#!/usr/bin/env python3
"""
Visualize contact predictions on SMPL-X mesh.
Load SMPL-X parameters and color vertices based on contact predictions.
"""

import os
import json
import yaml
import numpy as np
import torch
import smplx
import trimesh
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse


def load_smplx_mesh(smplx_params_path, smplx_model_path, smplx_model_type='neutral'):
    """Load SMPL-X mesh from parameters."""
    # Load parameters
    with open(smplx_params_path, 'r') as f:
        smplx_params = json.load(f)
    
    # Extract parameters
    body_pose = torch.tensor(smplx_params['body_pose'], dtype=torch.float32).flatten()
    
    if 'global_orient' in smplx_params:
        global_orient = torch.tensor(smplx_params['global_orient'], dtype=torch.float32).flatten()
    elif 'root_pose' in smplx_params:
        global_orient = torch.tensor(smplx_params['root_pose'], dtype=torch.float32).flatten()
    else:
        global_orient = torch.zeros(3, dtype=torch.float32)
    
    if 'transl' in smplx_params:
        transl = torch.tensor(smplx_params['transl'], dtype=torch.float32).flatten()
    elif 'cam_trans' in smplx_params:
        transl = torch.tensor(smplx_params['cam_trans'], dtype=torch.float32).flatten()
    else:
        transl = torch.zeros(3, dtype=torch.float32)
    
    if 'betas' in smplx_params:
        betas = torch.tensor(smplx_params['betas'], dtype=torch.float32).flatten()
    elif 'shape' in smplx_params:
        betas = torch.tensor(smplx_params['shape'], dtype=torch.float32).flatten()
    else:
        betas = torch.zeros(10, dtype=torch.float32)
    
    # Create SMPL-X model
    smplx_model = smplx.create(
        smplx_model_path,
        model_type='smplx',
        gender=smplx_model_type,
        use_face_contour=False,
        num_betas=10,
        num_expression_coeffs=10,
        ext='pkl',
        use_pca=False
    )
    
    # Generate mesh
    with torch.no_grad():
        output = smplx_model(
            body_pose=body_pose.unsqueeze(0),
            global_orient=global_orient.unsqueeze(0),
            transl=transl.unsqueeze(0),
            betas=betas.unsqueeze(0),
            return_verts=True
        )
        vertices = output.vertices[0].cpu().numpy()
        faces = smplx_model.faces
    
    return vertices, faces


def create_colored_mesh(vertices, faces, contact_labels, colormap='hot'):
    """Create a trimesh object with vertex colors based on contact labels."""
    # Create color array (RGBA)
    cmap = cm.get_cmap(colormap)
    
    # Normalize contact labels to [0, 1]
    contact_array = np.array(contact_labels[:len(vertices)])
    
    # Get colors from colormap
    colors = cmap(contact_array)
    
    # Convert to uint8 (trimesh expects RGBA in 0-255 range)
    vertex_colors = (colors * 255).astype(np.uint8)
    
    # Create mesh
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        process=False
    )
    
    return mesh


def visualize_contact_mesh(sample_dir, output_dir, smplx_model_path, data_root, show_interactive=False):
    """Visualize contact predictions on SMPL-X mesh."""
    # Extract category and sample_id from path
    # sample_dir format: output/category/sample_id
    parts = sample_dir.split(os.sep)
    if len(parts) >= 3:
        category = parts[-2]
        sample_id = parts[-1]
        
        # Original data path
        orig_sample_dir = os.path.join(data_root, category, sample_id)
    else:
        orig_sample_dir = sample_dir
    
    # File paths
    smplx_params_path = os.path.join(orig_sample_dir, 'smplx_parameters.json')
    contact_json_path = os.path.join(sample_dir, 'contact.json')
    image_path = os.path.join(orig_sample_dir, 'image.jpg')
    
    # Check files exist
    if not os.path.exists(smplx_params_path):
        print(f"✗ SMPL-X parameters not found: {smplx_params_path}")
        return False
    
    if not os.path.exists(contact_json_path):
        print(f"✗ Contact predictions not found: {contact_json_path}")
        return False
    
    # Load contact predictions
    with open(contact_json_path, 'r') as f:
        contact_pred = json.load(f)
    
    print(f"Processing: {sample_dir}")
    print(f"  Contact vertices: {sum(contact_pred)} / {len(contact_pred)} ({sum(contact_pred)/len(contact_pred)*100:.2f}%)")
    
    # Load SMPL-X mesh
    print("  Loading SMPL-X mesh...")
    vertices, faces = load_smplx_mesh(smplx_params_path, smplx_model_path)
    
    # Create colored mesh
    print("  Creating colored mesh...")
    mesh = create_colored_mesh(vertices, faces, contact_pred, colormap='hot')
    
    # Save mesh
    os.makedirs(output_dir, exist_ok=True)
    mesh_output_path = os.path.join(output_dir, 'contact_mesh.obj')
    mesh.export(mesh_output_path)
    print(f"  ✓ Saved mesh: {mesh_output_path}")
    
    # Create visualization figure
    print("  Creating visualization...")
    fig = plt.figure(figsize=(15, 5))
    
    # Subplot 1: Original image
    if os.path.exists(image_path):
        ax1 = fig.add_subplot(131)
        img = plt.imread(image_path)
        ax1.imshow(img)
        ax1.set_title('Original Image')
        ax1.axis('off')
    
    # Subplot 2: Front view
    ax2 = fig.add_subplot(132, projection='3d')
    contact_array = np.array(contact_pred[:len(vertices)])
    scatter = ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                         c=contact_array, cmap='hot', s=1, vmin=0, vmax=1)
    ax2.set_title('Contact Prediction (Front)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=0, azim=0)
    
    # Subplot 3: Side view
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
               c=contact_array, cmap='hot', s=1, vmin=0, vmax=1)
    ax3.set_title('Contact Prediction (Side)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.view_init(elev=0, azim=90)
    
    # Add colorbar
    plt.colorbar(scatter, ax=[ax2, ax3], label='Contact Probability', shrink=0.5)
    
    # Save figure
    fig_output_path = os.path.join(output_dir, 'contact_visualization.png')
    plt.tight_layout()
    plt.savefig(fig_output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved visualization: {fig_output_path}")
    
    if show_interactive:
        plt.show()
    else:
        plt.close()
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Visualize contact predictions on SMPL-X mesh')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Directory containing prediction results (default: from config or output/<run>)')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--show', action='store_true',
                       help='Show interactive visualization')
    parser.add_argument('--specific', type=str, default=None,
                       help='Visualize specific sample (e.g., "air cushion/floating_94")')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    smplx_model_path = config['data']['smplx_model_path']
    data_root = config['data']['root_dir']

    ckpt_dir = config.get('training', {}).get('save_dir', 'checkpoints')
    run_name = os.path.basename(os.path.normpath(ckpt_dir)) if ckpt_dir else 'run'
    default_input_dir = config.get('inference', {}).get('output_dir', os.path.join('output', run_name))
    input_dir = args.input_dir or default_input_dir

    vis_root = config.get('visualization', {}).get('save_dir', 'visualizations')
    
    print("=" * 70)
    print("Contact Prediction Visualization")
    print("=" * 70)
    print(f"Data root: {data_root}")
    print(f"SMPL-X model: {smplx_model_path}")
    
    # Find all contact.json files
    if args.specific:
        # Visualize specific sample
        sample_paths = [os.path.join(input_dir, args.specific)]
    else:
        # Find all samples
        sample_paths = []
        for root, dirs, files in os.walk(input_dir):
            if 'contact.json' in files:
                sample_paths.append(root)
    
    print(f"Found {len(sample_paths)} samples to visualize\n")
    
    # Process each sample
    success_count = 0
    for sample_dir in sample_paths:
        # Get relative path for output
        rel_path = os.path.relpath(sample_dir, input_dir)
        output_dir = os.path.join(vis_root, rel_path)
        
        try:
            success = visualize_contact_mesh(
                sample_dir, 
                output_dir, 
                smplx_model_path,
                data_root,
                show_interactive=args.show
            )
            if success:
                success_count += 1
            print()
        except Exception as e:
            print(f"✗ Error processing {sample_dir}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Summary
    print("=" * 70)
    print(f"Visualization complete!")
    print(f"  Successfully processed: {success_count}/{len(sample_paths)}")
    print(f"  Output directory: {vis_root}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
