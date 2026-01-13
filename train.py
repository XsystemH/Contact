"""
Training script for SMPL-X contact prediction.
"""

import os
import json
import yaml
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import SmplContactDataset, collate_fn, split_dataset
from models.contact_net import ContactNet
from utils.visualization import (
    visualize_batch_predictions, 
    plot_training_curves, 
    compute_metrics
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_project_path(path: str) -> str:
    if path is None:
        return path
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(PROJECT_ROOT, path))


def save_dataset_splits(save_dir, train_indices, val_indices, test_indices, meta=None):
    """Save dataset split indices for reproducible train/val/test usage."""
    save_dir = _resolve_project_path(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    split_npz_path = os.path.join(save_dir, 'dataset_splits.npz')
    np.savez(
        split_npz_path,
        train_indices=np.asarray(train_indices, dtype=np.int64),
        val_indices=np.asarray(val_indices, dtype=np.int64),
        test_indices=np.asarray(test_indices, dtype=np.int64)
    )

    split_json_path = os.path.join(save_dir, 'dataset_splits.json')
    payload = {
        'train_indices': np.asarray(train_indices, dtype=np.int64).tolist(),
        'val_indices': np.asarray(val_indices, dtype=np.int64).tolist(),
        'test_indices': np.asarray(test_indices, dtype=np.int64).tolist(),
    }
    if meta is not None:
        payload['meta'] = meta
    with open(split_json_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"[Trainer] Saved dataset splits: {split_npz_path}")


class Trainer:
    """Training manager for contact prediction model."""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        
        # Create directories
        os.makedirs(config['training']['save_dir'], exist_ok=True)
        if config['visualization']['enabled']:
            os.makedirs(config['visualization']['save_dir'], exist_ok=True)
        
        # Initialize model
        self.model = ContactNet(config).to(device)
        
        # Setup optimizer (only for trainable parameters)
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        if config['training']['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
        else:
            self.optimizer = torch.optim.Adam(
                trainable_params,
                lr=config['training']['learning_rate']
            )
        
        # Setup learning rate scheduler
        self.scheduler = None
        if config['training']['scheduler']['type'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config['training']['scheduler']['step_size'],
                gamma=config['training']['scheduler']['gamma']
            )
        elif config['training']['scheduler']['type'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['training']['num_epochs']
            )
        
        # Setup loss function
        pos_weight = torch.tensor([config['training']['pos_weight']]).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_epochs = []
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        
        print(f"[Trainer] Initialized on {device}")
        print(f"[Trainer] Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            vertices = batch['vertices'].to(self.device)
            normals = batch['normals'].to(self.device)
            pose_params = batch['pose_params'].to(self.device)
            K = batch['K'].to(self.device)
            object_bbox = batch['object_bbox'].to(self.device)
            mask_dist_field = batch['mask_dist_field'].to(self.device)
            contact_labels = batch['contact_labels'].to(self.device)
            
            # Forward pass (returns logits)
            logits = self.model(images, vertices, normals, pose_params, K, object_bbox, mask_dist_field)
            
            # Compute loss
            loss = self.criterion(logits, contact_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Record (convert logits to probs for metrics)
            contact_probs = torch.sigmoid(logits)
            epoch_loss += loss.item()
            all_preds.append(contact_probs.detach().cpu())
            all_labels.append(contact_labels.detach().cpu())
            
            # Update progress bar
            if batch_idx % self.config['training']['log_interval'] == 0:
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0).flatten()
        all_labels = torch.cat(all_labels, dim=0).flatten()
        metrics = compute_metrics(all_preds, all_labels)
        
        return avg_loss, metrics
    
    def validate(self, val_loader, epoch):
        """Validate the model."""
        # DataLoader length is number of batches; if dataset is empty, len(val_loader)==0.
        # Auto-train can hit this early when there are only a handful of labeled samples.
        if val_loader is None or len(val_loader) == 0:
            # Return a sentinel loss + empty metrics; caller should typically skip logging/best-model logic.
            return float("nan"), {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        self.model.eval()
        
        epoch_loss = 0.0
        all_preds = []
        all_labels = []
        
        viz_batch_cpu = None
        viz_predictions_cpu = None

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                # Move data to device
                images = batch['image'].to(self.device)
                vertices = batch['vertices'].to(self.device)
                normals = batch['normals'].to(self.device)
                pose_params = batch['pose_params'].to(self.device)
                K = batch['K'].to(self.device)
                object_bbox = batch['object_bbox'].to(self.device)
                mask_dist_field = batch['mask_dist_field'].to(self.device)
                contact_labels = batch['contact_labels'].to(self.device)
                
                # Forward pass (returns logits)
                logits = self.model(images, vertices, normals, pose_params, K, object_bbox, mask_dist_field)
                
                # Compute loss
                loss = self.criterion(logits, contact_labels)
                
                # Convert to probabilities for metrics
                contact_probs = torch.sigmoid(logits)

                # Cache first validation batch for visualization
                if self.config['visualization']['enabled'] and batch_idx == 0:
                    viz_batch_cpu = {
                        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    viz_predictions_cpu = contact_probs.detach().cpu()
                
                epoch_loss += loss.item()
                all_preds.append(contact_probs.cpu())
                all_labels.append(contact_labels.cpu())

        # If no batches were produced (e.g., empty dataset), avoid division by zero / cat on empty lists.
        if len(all_preds) == 0:
            return float("nan"), {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        avg_loss = epoch_loss / len(val_loader)
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0).flatten()
        all_labels = torch.cat(all_labels, dim=0).flatten()
        metrics = compute_metrics(all_preds, all_labels)
        
        # Visualization (only save at specified frequency)
        viz_frequency = self.config['visualization'].get('save_frequency', 5)
        if (self.config['visualization']['enabled'] and 
            epoch % viz_frequency == 0 and 
            viz_batch_cpu is not None and 
            viz_predictions_cpu is not None):
            save_dir = os.path.join(
                self.config['visualization']['save_dir'],
                f"epoch_{epoch}"
            )
            os.makedirs(save_dir, exist_ok=True)

            visualize_batch_predictions(
                viz_batch_cpu,
                viz_predictions_cpu,
                num_samples=self.config['visualization']['num_samples'],
                save_dir=save_dir
            )
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint.

        Files:
        - checkpoint_epoch_{epoch}.pth: versioned checkpoint
        - latest_model.pth: always overwritten to the most recently saved checkpoint
        - best_model.pth: overwritten only when `is_best` is True (val improves, or no-val final epoch)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_epochs': self.val_epochs,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config['training']['save_dir'],
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"[Trainer] Saved checkpoint: {checkpoint_path}")

        # Save "latest" model pointer (always updated when we save a checkpoint)
        latest_path = os.path.join(self.config['training']['save_dir'], 'latest_model.pth')
        try:
            torch.save(checkpoint, latest_path)
            print(f"[Trainer] Updated latest model: {latest_path}")
        except Exception as e:
            print(f"[Trainer] Warning: failed to update latest model: {e}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                self.config['training']['save_dir'],
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"[Trainer] Saved best model: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_epochs = checkpoint.get('val_epochs', [])
        self.best_val_loss = checkpoint['best_val_loss']

        # Backward compatibility for older checkpoints
        if not self.val_epochs and self.val_losses:
            val_frequency = int(self.config['validation']['val_frequency'])
            self.val_epochs = [i * val_frequency + 1 for i in range(len(self.val_losses))]
        
        print(f"[Trainer] Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        print(f"[Trainer] Starting training from epoch {self.start_epoch}")

        has_val = (val_loader is not None) and (len(val_loader) > 0)
        warned_no_val = False
        
        for epoch in range(self.start_epoch, self.config['training']['num_epochs']):
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
            print(f"  Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}")
            
            # Validate (optional). When val split is empty, skip validation and
            # save best_model at the final epoch so AutoTrain can reload it.
            if has_val and epoch % self.config['validation']['val_frequency'] == 0:
                val_loss, val_metrics = self.validate(val_loader, epoch)
                self.val_losses.append(val_loss)
                self.val_epochs.append(epoch + 1)
                
                print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}")
                print(f"  Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
                
                # Check if best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print(f"  New best validation loss: {val_loss:.4f}")
            elif not has_val:
                if not warned_no_val:
                    print("[Trainer] Warning: val split is empty (0 batches). Skipping validation.")
                    warned_no_val = True
                # In no-val mode, define "best" as the final epoch to ensure best_model.pth exists.
                is_best = (epoch == self.config['training']['num_epochs'] - 1)
                if is_best:
                    self.best_val_loss = train_loss
                    print(f"[Trainer] No-val mode: saving final checkpoint as best_model (train_loss={train_loss:.4f}).")
            else:
                is_best = False
            
            # Save checkpoint
            is_last = (epoch == self.config['training']['num_epochs'] - 1)
            if epoch % self.config['training']['save_frequency'] == 0 or is_best or is_last:
                self.save_checkpoint(epoch, is_best)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Plot training curves
        plot_path = os.path.join(self.config['training']['save_dir'], 'training_curves.png')
        plot_training_curves(self.train_losses, self.val_losses, plot_path, val_epochs=self.val_epochs)
        print(f"\n[Trainer] Training complete! Loss curves saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Train SMPL-X contact prediction model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument(
        '--train_subset_json',
        type=str,
        default=None,
        help='Optional JSON file to train on a subset. Format: {"sample_dirs": ["/abs/sample_dir", ...]}.',
    )
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='Disable validation (useful for small/fast online updates).',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help=(
            'Train for N additional epochs. '
            'If used with --resume, will set training.num_epochs = (resume_epoch + 1 + N). '
            'If used without --resume, will set training.num_epochs = N.'
        ),
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Determine resume checkpoint path (CLI overrides config)
    resume_path = args.resume or config.get('training', {}).get('resume_from')
    if resume_path:
        resume_path = _resolve_project_path(resume_path)

    # If requested, translate "additional epochs" into an absolute num_epochs
    if args.epochs is not None:
        extra_epochs = int(args.epochs)
        if extra_epochs <= 0:
            raise ValueError("--epochs must be a positive integer")

        if resume_path:
            # Peek epoch from checkpoint to avoid init+load just to compute end epoch
            ckpt = torch.load(resume_path, map_location="cpu")
            if not isinstance(ckpt, dict) or "epoch" not in ckpt:
                raise ValueError(f"Resume checkpoint missing 'epoch': {resume_path}")
            base_epoch = int(ckpt["epoch"])
            config["training"]["num_epochs"] = base_epoch + 1 + extra_epochs
        else:
            config["training"]["num_epochs"] = extra_epochs

    # Resolve run output paths relative to project root
    config['training']['save_dir'] = _resolve_project_path(config['training']['save_dir'])
    if config.get('visualization', {}).get('enabled'):
        config['visualization']['save_dir'] = _resolve_project_path(config['visualization']['save_dir'])
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Loading datasets...")

    # Optional: train on explicit subset (experience replay / online updates)
    if args.train_subset_json:
        subset_path = _resolve_project_path(str(args.train_subset_json))
        with open(subset_path, "r", encoding="utf-8") as f:
            subset_payload = json.load(f)
        if isinstance(subset_payload, dict):
            sample_dirs = subset_payload.get("sample_dirs", [])
        elif isinstance(subset_payload, list):
            sample_dirs = subset_payload
        else:
            raise ValueError(f"--train_subset_json must be a dict or list, got: {type(subset_payload)}")

        sample_dirs = [os.path.abspath(str(p)) for p in list(sample_dirs)]
        if len(sample_dirs) == 0:
            raise ValueError(f"--train_subset_json contains 0 sample_dirs: {subset_path}")

        # Optionally create a small val split from the subset (unless --no_val)
        if args.no_val:
            train_dirs = sample_dirs
            val_dirs = []
        else:
            rng = np.random.RandomState(int(config["data"]["random_seed"]))
            perm = rng.permutation(len(sample_dirs))
            n_train = max(1, int(len(sample_dirs) * float(config["data"]["train_ratio"])))
            train_idx = perm[:n_train]
            val_idx = perm[n_train:]
            train_dirs = [sample_dirs[int(i)] for i in train_idx]
            val_dirs = [sample_dirs[int(i)] for i in val_idx]

        train_dataset = SmplContactDataset(
            root_dir=config['data']['root_dir'],
            smplx_model_path=config['data']['smplx_model_path'],
            smplx_model_type=config['data']['smplx_model_type'],
            img_size=tuple(config['data']['img_size']),
            split='train',
            augment=True,
            sample_dirs=train_dirs,
        )

        val_dataset = None
        if (not args.no_val) and len(val_dirs) > 0:
            val_dataset = SmplContactDataset(
                root_dir=config['data']['root_dir'],
                smplx_model_path=config['data']['smplx_model_path'],
                smplx_model_type=config['data']['smplx_model_type'],
                img_size=tuple(config['data']['img_size']),
                split='val',
                augment=False,
                sample_dirs=val_dirs,
            )

        print(f"Train samples (subset): {len(train_dataset)}")
        print(f"Val samples (subset): {len(val_dataset) if val_dataset is not None else 0}")

        test_indices = []

    else:
        # Full dataset mode
        train_indices, val_indices, test_indices = split_dataset(
            config['data']['root_dir'],
            train_ratio=config['data']['train_ratio'],
            val_ratio=config['data']['val_ratio'],
            test_ratio=config['data']['test_ratio'],
            seed=config['data']['random_seed']
        )

        # Persist split indices alongside checkpoints
        save_dataset_splits(
            config['training']['save_dir'],
            train_indices,
            val_indices,
            test_indices,
            meta={
                'root_dir': config['data']['root_dir'],
                'train_ratio': config['data']['train_ratio'],
                'val_ratio': config['data']['val_ratio'],
                'test_ratio': config['data']['test_ratio'],
                'random_seed': config['data']['random_seed'],
            }
        )

        train_dataset = SmplContactDataset(
            root_dir=config['data']['root_dir'],
            smplx_model_path=config['data']['smplx_model_path'],
            smplx_model_type=config['data']['smplx_model_type'],
            img_size=tuple(config['data']['img_size']),
            split='train',
            augment=True,
            indices=train_indices
        )

        val_dataset = None
        if not args.no_val:
            val_dataset = SmplContactDataset(
                root_dir=config['data']['root_dir'],
                smplx_model_path=config['data']['smplx_model_path'],
                smplx_model_type=config['data']['smplx_model_type'],
                img_size=tuple(config['data']['img_size']),
                split='val',
                augment=False,
                indices=val_indices
            )

        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset) if val_dataset is not None else 0}")
        print(f"Test samples: {len(test_indices)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    # Initialize trainer
    trainer = Trainer(config, device)
    
    # Resume from checkpoint if specified
    if resume_path:
        trainer.load_checkpoint(resume_path)
        # Ensure the currently requested learning rate (e.g., small-loop LR) takes effect.
        desired_lr = float(config['training']['learning_rate'])
        for g in trainer.optimizer.param_groups:
            g['lr'] = desired_lr
            # Some schedulers rely on initial_lr for state_dict roundtrips
            g['initial_lr'] = desired_lr
        if trainer.scheduler is not None and hasattr(trainer.scheduler, "base_lrs"):
            try:
                trainer.scheduler.base_lrs = [desired_lr for _ in trainer.optimizer.param_groups]
            except Exception:
                pass
    
    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
