#!/usr/bin/env python3
"""
Test script to verify dataset loading and model functionality.
"""

import torch
import yaml
from data.dataset import SmplContactDataset
from models.contact_net import ContactNet


def test_dataset():
    """Test dataset loading."""
    print("=" * 60)
    print("Testing Dataset...")
    print("=" * 60)
    
    try:
        # Load config
        with open("configs/default.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        dataset = SmplContactDataset(
            root_dir=config['data']['root_dir'],
            smplx_model_path=config['data']['smplx_model_path'],
            smplx_model_type=config['data']['smplx_model_type'],
            img_size=tuple(config['data']['img_size']),
            split='train',
            augment=False
        )
        
        print(f"✓ Dataset loaded successfully with {len(dataset)} samples")
        
        # Try loading a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print("\n✓ Sample loaded successfully:")
            print(f"  - Image shape: {sample['image'].shape}")
            print(f"  - Vertices shape: {sample['vertices'].shape}")
            print(f"  - Normals shape: {sample['normals'].shape}")
            print(f"  - Pose params shape: {sample['pose_params'].shape}")
            print(f"  - K shape: {sample['K'].shape}")
            print(f"  - BBox shape: {sample['object_bbox'].shape}")
            print(f"  - Mask dist field shape: {sample['mask_dist_field'].shape}")
            print(f"  - Contact labels shape: {sample['contact_labels'].shape}")
            print(f"  - Contact ratio: {sample['contact_labels'].mean():.4f}")
            print(f"  - Sample ID: {sample['sample_id']}")
        else:
            print("✗ Dataset is empty! Check data_contact directory.")
            return False
            
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_model():
    """Test model initialization and forward pass."""
    print("\n" + "=" * 60)
    print("Testing Model...")
    print("=" * 60)
    
    try:
        # Load config
        with open("configs/default.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Initialize model
        model = ContactNet(config).to(device)
        print("✓ Model initialized successfully")
        
        # Create dummy input
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
        mask_dist_field = torch.rand(B, 1, 512, 512).to(device)
        
        # Forward pass
        with torch.no_grad():
            logits = model(images, vertices, normals, pose_params, K, bbox, mask_dist_field)
            probs = torch.sigmoid(logits)
        
        print("✓ Forward pass successful:")
        print(f"  - Output shape: {logits.shape}")
        print(f"  - Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        print(f"  - Probs range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
        
        # Check trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n✓ Parameter count:")
        print(f"  - Total: {total_params:,}")
        print(f"  - Trainable: {trainable_params:,}")
        print(f"  - Frozen: {total_params - trainable_params:,}")
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SMPL-X Contact Prediction - System Test")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test dataset
    results.append(("Dataset", test_dataset()))
    
    # Test model
    results.append(("Model", test_model()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! Ready to train.")
    else:
        print("Some tests failed. Please fix the issues before training.")
    print("=" * 60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
