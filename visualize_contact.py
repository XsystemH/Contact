#!/usr/bin/env python3
"""
Visualize contact predictions using the same pipeline as training-time visualizations.

中文用法(简述):
- **用途**: 加载你在 `--config` 指定的 YAML(用来构建模型并定位 best_model), 对 `--data_root` 指定的数据集目录做推理, 并按训练时同样的方式输出可视化图片.
- **典型命令**:
  - `python visualize_contact.py --config configs/260102_1.yaml --data_root data_example_net --split all --tag data_example_net`
- **输出内容**: 每个样本会保存 `*_projection.png`(投影到图像上, 含 bbox), `*_heatmap.png`(3D 热力图), 以及在有 `mask_dist_field` 时保存 `*_mask_dist.png`.
- **输出目录**: 默认写到 `config.visualization.save_dir/<tag>/`(例如 `visualizations/260102_1/data_example_net/`), 也可用 `--save_dir` 手动指定.
- **常用参数**:
  - `--checkpoint`: 手动指定权重(默认 `<training.save_dir>/best_model.pth`, 不填则默认 <training.save_dir>/best_model.pth)
  - `--specific "wine glass/HICO_train2015_00009097"`: 只可视化单个样本
  - `--max_samples`: 限制最多输出多少个样本
  - `--batch_size`: 可视化建议设小一点(比如 1), 避免一次生成太多图

This script runs **inference + visualization on a dataset folder**:
- Provide `--config` (for model + best checkpoint resolution)
- Optionally override dataset root with `--data_root`
- It will run the model and write `*_projection.png`, `*_heatmap.png` (and `*_mask_dist.png` if available)

example:
python /home/rtwang/workspace/Contact/visualize_contact.py \
--config /home/rtwang/workspace/Contact/configs/260102_1.yaml \
--data_root /home/rtwang/workspace/Contact/data_example_net \
--split all \
--batch_size 1 --max_samples 1 --tag data_example_net_smoke   
"""

import os
import yaml
import torch
import argparse
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Subset

from data.dataset import SmplContactDataset, collate_fn, split_dataset
from models.contact_net import ContactNet
from utils.visualization import visualize_batch_predictions


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_project_path(path: str) -> str:
    if path is None:
        return path
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(PROJECT_ROOT, path))


def _default_run_name_from_save_dir(save_dir: str) -> str:
    save_dir = os.path.normpath(save_dir)
    return os.path.basename(save_dir) if save_dir else "run"


def _load_or_create_splits(config, root_dir_for_split: str, save_dir: str):
    """
    Load train/val/test indices saved in checkpoint dir; create them if missing.
    If `root_dir_for_split` differs from config['data']['root_dir'], we do NOT reuse the saved indices.
    """
    save_dir = _resolve_project_path(save_dir)

    # If dataset root differs, recreate splits (they're only meaningful for the original dataset).
    if os.path.normpath(root_dir_for_split) != os.path.normpath(config['data']['root_dir']):
        train_indices, val_indices, test_indices = split_dataset(
            root_dir_for_split,
            train_ratio=config['data']['train_ratio'],
            val_ratio=config['data']['val_ratio'],
            test_ratio=config['data']['test_ratio'],
            seed=config['data']['random_seed'],
        )
        return train_indices, val_indices, test_indices

    split_npz_path = os.path.join(save_dir, "dataset_splits.npz")
    split_json_path = os.path.join(save_dir, "dataset_splits.json")

    if os.path.exists(split_npz_path):
        data = np.load(split_npz_path)
        return data["train_indices"], data["val_indices"], data["test_indices"]

    if os.path.exists(split_json_path):
        import json

        with open(split_json_path, "r") as f:
            payload = json.load(f)
        return (
            np.asarray(payload["train_indices"], dtype=np.int64),
            np.asarray(payload["val_indices"], dtype=np.int64),
            np.asarray(payload["test_indices"], dtype=np.int64),
        )

    # Fallback: recreate splits deterministically and persist for later runs
    train_indices, val_indices, test_indices = split_dataset(
        root_dir_for_split,
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        seed=config["data"]["random_seed"],
    )

    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        split_npz_path,
        train_indices=np.asarray(train_indices, dtype=np.int64),
        val_indices=np.asarray(val_indices, dtype=np.int64),
        test_indices=np.asarray(test_indices, dtype=np.int64),
    )
    with open(split_json_path, "w") as f:
        import json

        json.dump(
            {
                "train_indices": np.asarray(train_indices, dtype=np.int64).tolist(),
                "val_indices": np.asarray(val_indices, dtype=np.int64).tolist(),
                "test_indices": np.asarray(test_indices, dtype=np.int64).tolist(),
                "meta": {
                    "root_dir": root_dir_for_split,
                    "train_ratio": config["data"]["train_ratio"],
                    "val_ratio": config["data"]["val_ratio"],
                    "test_ratio": config["data"]["test_ratio"],
                    "random_seed": config["data"]["random_seed"],
                    "note": "Splits were recreated by visualize_contact because no saved split file was found.",
                },
            },
            f,
            indent=2,
        )

    print(
        "[Visualize] No saved splits found; recreated and saved. "
        f"Expected one of: {split_npz_path} or {split_json_path}"
    )
    return train_indices, val_indices, test_indices


def _select_indices_for_split(split: str, train_idx, val_idx, test_idx):
    if split == "train":
        return train_idx
    if split == "val":
        return val_idx
    if split == "test":
        return test_idx
    if split == "all":
        return None
    raise ValueError(f"Unknown split: {split}")


def _subset_by_specific(dataset: SmplContactDataset, specific: str):
    """
    `specific` can be:
    - "category/sample_id" (recommended; works with spaces in category)
    - or an exact dataset sample_id like "{category}_{id}" (must match exactly)
    """
    spec = specific.strip()
    if "/" in spec:
        category, sid = spec.split("/", 1)
        target = f"{category}_{sid}"
    else:
        target = spec

    indices = []
    for i, s in enumerate(dataset.samples):
        sample_id = f"{s['category']}_{s['id']}"
        if sample_id == target:
            indices.append(i)
            break

    if not indices:
        raise ValueError(
            f"Specific sample not found: {specific}. "
            "Try the 'category/sample_id' format, and ensure it exists under --data_root."
        )

    return Subset(dataset, indices)


def main():
    parser = argparse.ArgumentParser(description='Visualize contact predictions (train.py-style)')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path (default: <training.save_dir>/best_model.pth)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override dataset root dir for visualization (e.g., data_example_net)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test', 'all'],
                        help='Which split to visualize. If --data_root differs from config data root, splits are recreated.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Output directory for images (default: <visualization.save_dir>/<tag>)')
    parser.add_argument('--tag', type=str, default='inference',
                        help='Subfolder name under visualization.save_dir (default: inference)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for inference (default: from training.batch_size)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader workers (default: from data.num_workers)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Stop after visualizing this many samples (default: all)')
    parser.add_argument('--specific', type=str, default=None,
                        help='Visualize one specific sample (e.g., "wine glass/HICO_train2015_00009097")')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve core paths
    config['data']['smplx_model_path'] = _resolve_project_path(config['data']['smplx_model_path'])
    config['training']['save_dir'] = _resolve_project_path(config['training']['save_dir'])
    if config.get('visualization', {}).get('save_dir'):
        config['visualization']['save_dir'] = _resolve_project_path(config['visualization']['save_dir'])

    data_root = args.data_root or config['data']['root_dir']
    data_root = _resolve_project_path(data_root)
    config['data']['root_dir'] = data_root

    ckpt_dir = config['training']['save_dir']
    checkpoint_path = _resolve_project_path(args.checkpoint) if args.checkpoint else os.path.join(ckpt_dir, 'best_model.pth')

    run_name = _default_run_name_from_save_dir(ckpt_dir)
    vis_base = config.get('visualization', {}).get('save_dir', os.path.join(PROJECT_ROOT, 'visualizations', run_name))
    vis_base = _resolve_project_path(vis_base)
    save_dir = _resolve_project_path(args.save_dir) if args.save_dir else os.path.join(vis_base, args.tag)
    os.makedirs(save_dir, exist_ok=True)

    # Device
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("Contact Prediction Visualization (train.py-style)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data root: {data_root}")
    print(f"Output dir: {save_dir}")

    # Build dataset
    train_idx, val_idx, test_idx = _load_or_create_splits(config, data_root, ckpt_dir)
    indices = _select_indices_for_split(args.split, train_idx, val_idx, test_idx)

    dataset = SmplContactDataset(
        root_dir=data_root,
        smplx_model_path=config['data']['smplx_model_path'],
        smplx_model_type=config['data']['smplx_model_type'],
        img_size=tuple(config['data']['img_size']),
        split=args.split if args.split != 'all' else 'test',
        augment=False,
        indices=indices,
    )

    if args.specific:
        dataset = _subset_by_specific(dataset, args.specific)
        print(f"[Visualize] Using specific sample: {args.specific}")

    # Load model
    print("\nLoading model...")
    model = ContactNet(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")

    # DataLoader
    batch_size = int(args.batch_size) if args.batch_size is not None else int(config['training']['batch_size'])
    num_workers = int(args.num_workers) if args.num_workers is not None else int(config['data'].get('num_workers', 0))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
    )

    # Inference + visualization
    print("\nRunning inference + saving visualizations...")
    seen = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Visualize"):
            images = batch['image'].to(device)
            vertices = batch['vertices'].to(device)
            normals = batch['normals'].to(device)
            pose_params = batch['pose_params'].to(device)
            K = batch['K'].to(device)
            object_bbox = batch['object_bbox'].to(device)
            mask_dist_field = batch['mask_dist_field'].to(device)

            logits = model(images, vertices, normals, pose_params, K, object_bbox, mask_dist_field)
            probs = torch.sigmoid(logits).detach().cpu()

            batch_cpu = {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            visualize_batch_predictions(
                batch_cpu,
                probs,
                num_samples=probs.shape[0],
                save_dir=save_dir,
            )

            seen += probs.shape[0]
            if args.max_samples is not None and seen >= int(args.max_samples):
                break

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print(f"Saved to: {save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
