#!/usr/bin/env python3
"""
Generate synthetic test dataset for SMPL-X contact prediction.
Creates mock samples in the expected directory structure.
"""

import os
import json
import numpy as np
from PIL import Image
import argparse


def generate_mock_sample(output_dir, sample_id, category='test_category'):
    """Generate a single mock sample with all required files."""
    
    # Create sample directory
    sample_path = os.path.join(output_dir, category, sample_id)
    os.makedirs(sample_path, exist_ok=True)
    
    # 1. Generate mock image (512x512 RGB)
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    Image.fromarray(img).save(os.path.join(sample_path, 'image.jpg'))
    
    # 2. Generate SMPL-X parameters
    smplx_params = {
        'body_pose': np.random.randn(1, 63).tolist(),  # Body pose
        'global_orient': np.random.randn(1, 3).tolist(),  # Root orientation
        'transl': [0.0, 0.5, 2.0],  # Translation (z=2m in front of camera)
        'betas': np.random.randn(10).tolist(),  # Shape parameters
        'left_hand_pose': np.random.randn(1, 45).tolist(),
        'right_hand_pose': np.random.randn(1, 45).tolist(),
    }
    
    with open(os.path.join(sample_path, 'smplx_parameters.json'), 'w') as f:
        json.dump(smplx_params, f, indent=2)
    
    # 3. Generate contact labels (10475 vertices for SMPL-X body)
    # Simulate ~10% contact (typical for standing/sitting poses)
    contact_labels = (np.random.rand(10475) < 0.1).tolist()
    
    contact_data = {
        'contact': contact_labels
    }
    
    with open(os.path.join(sample_path, 'contact.json'), 'w') as f:
        json.dump(contact_data, f)
    
    # 4. Generate bounding box annotation (object in image)
    # Random bbox within image bounds
    x1 = np.random.randint(50, 200)
    y1 = np.random.randint(50, 200)
    x2 = x1 + np.random.randint(100, 250)
    y2 = y1 + np.random.randint(100, 250)
    
    bbox_data = {
        'bbox': [float(x1), float(y1), float(x2), float(y2)]
    }
    
    with open(os.path.join(sample_path, 'box_annotation.json'), 'w') as f:
        json.dump(bbox_data, f, indent=2)
    
    # 5. Generate camera calibration (intrinsics)
    # Typical camera parameters for 512x512 image
    focal_length = 500.0
    cx, cy = 256.0, 256.0
    
    K = [
        [focal_length, 0.0, cx],
        [0.0, focal_length, cy],
        [0.0, 0.0, 1.0]
    ]
    
    calib_data = {
        'K': K
    }
    
    with open(os.path.join(sample_path, 'calibration.json'), 'w') as f:
        json.dump(calib_data, f, indent=2)
    
    # 6. Generate camera extrinsics (rotation + translation)
    # Identity rotation (camera looking at origin)
    R = np.eye(3).tolist()
    T = [0.0, 0.0, 0.0]
    
    extrinsic_data = {
        'R': R,
        'T': T
    }
    
    with open(os.path.join(sample_path, 'extrinsic.json'), 'w') as f:
        json.dump(extrinsic_data, f, indent=2)
    
    # 7. (Optional) Generate normals
    # Random unit normals for 10475 vertices
    normals = np.random.randn(10475, 3)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    np.save(os.path.join(sample_path, 'normals_smplx.npy'), normals.astype(np.float32))
    
    print(f"✓ Generated sample: {category}/{sample_id}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic test dataset')
    parser.add_argument('--output_dir', type=str, default='/home/xhsystem/Code/Term7/Ca3OH1/data_contact',
                       help='Output directory for dataset')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples to generate')
    parser.add_argument('--num_categories', type=int, default=2,
                       help='Number of categories')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Generating Synthetic Test Dataset")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Number of categories: {args.num_categories}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate samples across categories
    samples_per_category = args.num_samples // args.num_categories
    
    for cat_idx in range(args.num_categories):
        category = f"category_{cat_idx + 1}"
        
        for sample_idx in range(samples_per_category):
            sample_id = f"sample_{sample_idx + 1:04d}"
            generate_mock_sample(args.output_dir, sample_id, category)
    
    print()
    print("=" * 60)
    print(f"✓ Dataset generation complete!")
    print(f"✓ Created {args.num_samples} samples in {args.num_categories} categories")
    print(f"✓ Location: {args.output_dir}")
    print("=" * 60)
    
    # List the structure
    print("\nDataset structure:")
    for root, dirs, files in os.walk(args.output_dir):
        level = root.replace(args.output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        if level < 2:  # Only show first two levels
            subindent = ' ' * 2 * (level + 1)
            for file in sorted(files)[:3]:  # Show first 3 files
                print(f'{subindent}{file}')
            if len(files) > 3:
                print(f'{subindent}... ({len(files)} files total)')


if __name__ == "__main__":
    main()
