#!/usr/bin/env python3
"""
Test script: Run inference on one sample per category and save contact predictions.
"""

import os
import json
import yaml
import argparse
import torch
from tqdm import tqdm
from collections import defaultdict
import numpy as np

from data.dataset import SmplContactDataset
from data.dataset import split_dataset
from models.contact_net import ContactNet


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_project_path(path: str) -> str:
    if path is None:
        return path
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(PROJECT_ROOT, path))


def _default_run_name_from_save_dir(save_dir: str) -> str:
    save_dir = os.path.normpath(save_dir)
    return os.path.basename(save_dir) if save_dir else "run"


def load_or_create_splits(config, save_dir):
    """Load train/val/test indices saved in checkpoint dir; create them if missing."""
    save_dir = _resolve_project_path(save_dir)
    split_npz_path = os.path.join(save_dir, 'dataset_splits.npz')
    split_json_path = os.path.join(save_dir, 'dataset_splits.json')

    if os.path.exists(split_npz_path):
        data = np.load(split_npz_path)
        return data['train_indices'], data['val_indices'], data['test_indices']

    if os.path.exists(split_json_path):
        with open(split_json_path, 'r') as f:
            payload = json.load(f)
        return (
            np.asarray(payload['train_indices'], dtype=np.int64),
            np.asarray(payload['val_indices'], dtype=np.int64),
            np.asarray(payload['test_indices'], dtype=np.int64),
        )

    # Fallback: recreate splits deterministically and persist for later runs
    train_indices, val_indices, test_indices = split_dataset(
        config['data']['root_dir'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        seed=config['data']['random_seed']
    )

    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        split_npz_path,
        train_indices=np.asarray(train_indices, dtype=np.int64),
        val_indices=np.asarray(val_indices, dtype=np.int64),
        test_indices=np.asarray(test_indices, dtype=np.int64)
    )
    with open(split_json_path, 'w') as f:
        json.dump(
            {
                'train_indices': np.asarray(train_indices, dtype=np.int64).tolist(),
                'val_indices': np.asarray(val_indices, dtype=np.int64).tolist(),
                'test_indices': np.asarray(test_indices, dtype=np.int64).tolist(),
                'meta': {
                    'root_dir': config['data']['root_dir'],
                    'train_ratio': config['data']['train_ratio'],
                    'val_ratio': config['data']['val_ratio'],
                    'test_ratio': config['data']['test_ratio'],
                    'random_seed': config['data']['random_seed'],
                    'note': 'Splits were recreated by test_inference because no saved split file was found.'
                }
            },
            f,
            indent=2
        )

    print(
        "[Inference] No saved splits found; recreated and saved. "
        f"Expected one of: {split_npz_path} or {split_json_path}"
    )
    return train_indices, val_indices, test_indices


def collect_samples_by_category(dataset):
    """Collect one sample per category."""
    category_samples = defaultdict(list)
    
    # Group samples by category
    for idx, sample_info in enumerate(dataset.samples):
        category = sample_info['category']
        category_samples[category].append((idx, sample_info))
    
    # Select first sample from each category
    selected_samples = {}
    for category, samples in category_samples.items():
        idx, sample_info = samples[0]
        selected_samples[category] = {
            'idx': idx,
            'category': category,
            'sample_id': sample_info['id'],
            'path': sample_info['path']
        }
    
    return selected_samples


def run_inference(model, dataset, sample_idx, device):
    """Run inference on a single sample."""
    model.eval()
    
    # Get sample
    sample = dataset[sample_idx]
    
    # Move to device and add batch dimension
    images = sample['image'].unsqueeze(0).to(device)
    vertices = sample['vertices'].unsqueeze(0).to(device)
    normals = sample['normals'].unsqueeze(0).to(device)
    pose_params = sample['pose_params'].unsqueeze(0).to(device)
    K = sample['K'].unsqueeze(0).to(device)
    object_bbox = sample['object_bbox'].unsqueeze(0).to(device)
    mask_dist_field = sample['mask_dist_field'].unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(images, vertices, normals, pose_params, K, object_bbox, mask_dist_field)
        probs = torch.sigmoid(logits)
    
    # Convert to binary predictions
    contact_pred = (probs > 0.5).float().squeeze(0).cpu().numpy()
    
    return contact_pred.tolist()


def save_contact_json(contact_pred, output_path):
    """Save contact predictions to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(contact_pred, f)
    
    print(f"✓ Saved: {output_path}")


def main():
    print("=" * 60)
    print("Contact Prediction Inference - One Sample Per Category")
    print("=" * 60)

    parser = argparse.ArgumentParser(description='Run inference on one sample per category (test split only)')
    parser.add_argument('--config', type=str, default='configs/251225.yaml', help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, default=None, help='Override checkpoint path (best_model.pth)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ckpt_dir = _resolve_project_path(config['training']['save_dir'])
    checkpoint_path = _resolve_project_path(args.checkpoint) if args.checkpoint else os.path.join(ckpt_dir, 'best_model.pth')
    
    # Load dataset (use all samples to get all categories)
    print("\nLoading dataset...")

    _, _, test_indices = load_or_create_splits(config, ckpt_dir)

    dataset = SmplContactDataset(
        root_dir=config['data']['root_dir'],
        smplx_model_path=config['data']['smplx_model_path'],
        smplx_model_type=config['data']['smplx_model_type'],
        img_size=tuple(config['data']['img_size']),
        split='test',
        augment=False,
        indices=test_indices
    )
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Collect one sample per category
    print("\nCollecting samples by category...")
    selected_samples = collect_samples_by_category(dataset)
    print(f"Found {len(selected_samples)} categories")
    
    # Load best model
    print("\nLoading best model...")
    model = ContactNet(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    # Create output directory
    run_name = _default_run_name_from_save_dir(ckpt_dir)
    output_root = config.get('inference', {}).get('output_dir', os.path.join('output', run_name))
    output_root = _resolve_project_path(output_root)
    os.makedirs(output_root, exist_ok=True)
    
    # Run inference on selected samples
    print(f"\nRunning inference on {len(selected_samples)} samples...")
    results = []
    
    for category, info in tqdm(sorted(selected_samples.items())):
        # Run inference
        contact_pred = run_inference(model, dataset, info['idx'], device)
        
        # Prepare output path: output/category/sample_id/contact.json
        output_dir = os.path.join(output_root, category, info['sample_id'])
        output_path = os.path.join(output_dir, 'contact.json')
        
        # Save prediction
        save_contact_json(contact_pred, output_path)
        
        # Record result
        num_contact = sum(contact_pred)
        results.append({
            'category': category,
            'sample_id': info['sample_id'],
            'num_vertices': len(contact_pred),
            'num_contact': num_contact,
            'contact_ratio': num_contact / len(contact_pred),
            'output_path': output_path
        })
    
    # Print summary
    print("\n" + "=" * 60)
    print("Inference Summary")
    print("=" * 60)
    print(f"Total samples processed: {len(results)}")
    print(f"Output directory: {output_root}/")
    print("\nContact statistics by category:")
    print(f"{'Category':<30} {'Contact Vertices':<20} {'Contact Ratio':<15}")
    print("-" * 65)
    
    for result in results:
        print(f"{result['category']:<30} {result['num_contact']:<20} {result['contact_ratio']:<15.2%}")
    
    # Save summary
    summary_path = os.path.join(output_root, 'inference_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'total_samples': len(results),
            'checkpoint_epoch': checkpoint['epoch'],
            'best_val_loss': checkpoint['best_val_loss'],
            'results': results
        }, f, indent=2)
    
    print(f"\n✓ Summary saved to: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
