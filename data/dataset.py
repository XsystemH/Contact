"""
Dataset for SMPL-X contact prediction.
Loads images, SMPL-X parameters, contact labels, and camera parameters.
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import smplx
import cv2

from utils.geometry_utils import world_to_camera, scale_intrinsics, compute_vertex_normals


def build_image_transform(img_size, augment=False, split='test'):
    """Build the same image transform pipeline used by SmplContactDataset.

    Args:
        img_size: (H, W)
        augment: whether to apply train-time augmentation
        split: 'train'|'val'|'test'
    """
    if augment and split == 'train':
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class SmplContactPreprocessor:
    """Preprocess a single raw sample directory into model inputs.

    This is the same processing as SmplContactDataset.__getitem__, but designed for
    inference services (contact labels optional).

    Expected files in sample_dir:
      - image.jpg
      - object_mask.png
      - smplx_parameters.json
      - calibration.json
      - extrinsic.json
      - box_annotation.json
      - contact.json (optional; only used if include_contact_labels=True)
    """

    def __init__(self, smplx_model_path, smplx_model_type='neutral', img_size=(512, 512), split='test', augment=False):
        self.img_size = img_size
        self.split = split
        self.augment = augment

        self.smplx_model = smplx.create(
            smplx_model_path,
            model_type='smplx',
            gender=smplx_model_type,
            use_face_contour=False,
            num_betas=10,
            num_expression_coeffs=10,
            ext='pkl',
            use_pca=False
        )

        self.faces = torch.from_numpy(self.smplx_model.faces.astype(np.int64))
        self.transform = build_image_transform(self.img_size, augment=self.augment, split=self.split)

    def process_dir(self, sample_dir, include_contact_labels=False):
        # 1. Load image
        img_path = os.path.join(sample_dir, 'image.jpg')
        image_pil = Image.open(img_path).convert('RGB')
        original_size = image_pil.size  # (W, H)
        image = self.transform(image_pil)  # (3, H, W)

        # 1b. Load object mask and build distance field (in resized image space)
        mask_path = os.path.join(sample_dir, 'object_mask.png')
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"[Preprocessor] Missing required file: {mask_path}")

        mask_img = Image.open(mask_path).convert('L')
        mask_img = mask_img.resize((self.img_size[1], self.img_size[0]), resample=Image.NEAREST)
        mask_np = np.array(mask_img)
        dist_np = SmplContactDataset._compute_dilated_distance_field(mask_np, dilation_kernel=31, dilation_iters=1)
        mask_dist_field = torch.from_numpy(dist_np).unsqueeze(0).float()  # (1, H, W)

        # 2. Load SMPL-X parameters
        with open(os.path.join(sample_dir, 'smplx_parameters.json'), 'r') as f:
            smplx_params = json.load(f)

        body_pose = torch.tensor(smplx_params['body_pose'], dtype=torch.float32).flatten()  # (63,)

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

        # 3. Generate SMPL-X mesh vertices
        with torch.no_grad():
            smplx_output = self.smplx_model(
                body_pose=body_pose.unsqueeze(0),
                global_orient=global_orient.unsqueeze(0),
                transl=transl.unsqueeze(0),
                betas=betas.unsqueeze(0),
                return_verts=True
            )
            vertices_world = smplx_output.vertices[0]

        # 4. Load camera extrinsics
        with open(os.path.join(sample_dir, 'extrinsic.json'), 'r') as f:
            extrinsic = json.load(f)

        if 'R' in extrinsic:
            R = torch.tensor(extrinsic['R'], dtype=torch.float32).reshape(3, 3)
        elif 'rotation' in extrinsic:
            R = torch.tensor(extrinsic['rotation'], dtype=torch.float32).reshape(3, 3)
        else:
            R = torch.eye(3, dtype=torch.float32)

        if 'T' in extrinsic:
            T = torch.tensor(extrinsic['T'], dtype=torch.float32).flatten()
        elif 'translation' in extrinsic:
            T = torch.tensor(extrinsic['translation'], dtype=torch.float32).flatten()
        else:
            T = torch.zeros(3, dtype=torch.float32)

        vertices_cam = world_to_camera(vertices_world.unsqueeze(0), R.unsqueeze(0), T.unsqueeze(0))[0]

        # 5. Compute normals
        normals = compute_vertex_normals(vertices_cam.unsqueeze(0), self.faces)[0]

        # 6. Load camera intrinsics
        with open(os.path.join(sample_dir, 'calibration.json'), 'r') as f:
            calibration = json.load(f)

        K = torch.tensor(calibration['K'], dtype=torch.float32).reshape(3, 3)
        original_img_size = (original_size[1], original_size[0])  # (H, W)
        K = scale_intrinsics(K, original_img_size, self.img_size)

        # 7. Load object bounding box
        with open(os.path.join(sample_dir, 'box_annotation.json'), 'r') as f:
            bbox_data = json.load(f)

        if 'bbox' in bbox_data:
            bbox = torch.tensor(bbox_data['bbox'], dtype=torch.float32)
        elif 'obj' in bbox_data:
            bbox = torch.tensor(bbox_data['obj'], dtype=torch.float32)
        else:
            bbox = torch.tensor([100.0, 100.0, 400.0, 400.0], dtype=torch.float32)

        scale_x = self.img_size[1] / original_size[0]
        scale_y = self.img_size[0] / original_size[1]
        bbox[0] *= scale_x
        bbox[1] *= scale_y
        bbox[2] *= scale_x
        bbox[3] *= scale_y

        out = {
            'image': image,
            'vertices': vertices_cam,
            'normals': normals,
            'pose_params': body_pose,
            'K': K,
            'object_bbox': bbox,
            'mask_dist_field': mask_dist_field,
        }

        if include_contact_labels:
            contact_path = os.path.join(sample_dir, 'contact.json')
            if not os.path.exists(contact_path):
                raise FileNotFoundError(f"[Preprocessor] Missing required file: {contact_path}")
            with open(contact_path, 'r') as f:
                contact_data = json.load(f)
            if isinstance(contact_data, dict):
                contact_labels = torch.tensor(contact_data['contact'][:10475], dtype=torch.float32)
            elif isinstance(contact_data, list):
                contact_labels = torch.tensor(contact_data[:10475], dtype=torch.float32)
            else:
                raise ValueError(f"Unknown contact data format: {type(contact_data)}")
            if contact_labels.dtype == torch.bool:
                contact_labels = contact_labels.float()
            out['contact_labels'] = contact_labels

        return out


class SmplContactDataset(Dataset):
    """
    Dataset for SMPL-X contact prediction.
    
    Directory structure:
        data_contact/
            category1/
                id1/
                    image.jpg
                    object_mask.png
                    smplx_parameters.json
                    contact.json
                    box_annotation.json
                    calibration.json
                    extrinsic.json
                    normals_smplx.npy (optional)
                id2/
                ...
            category2/
            ...
    """
    
    def __init__(
        self,
        root_dir,
        smplx_model_path,
        smplx_model_type='neutral',
        img_size=(512, 512),
        split='train',
        augment=False,
        indices=None,
        sample_dirs=None,
    ):
        """
        Args:
            root_dir: Root directory of dataset
            smplx_model_path: Path to SMPL-X model files
            smplx_model_type: 'neutral', 'male', or 'female'
            img_size: Target image size (H, W)
            split: 'train', 'val', or 'test'
            augment: Whether to apply data augmentation
            indices: Optional list of indices to use (for train/val/test split)
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        self.augment = augment
        self.indices = indices
        
        # Initialize SMPL-X model
        self.smplx_model = smplx.create(
            smplx_model_path,
            model_type='smplx',
            gender=smplx_model_type,
            use_face_contour=False,
            num_betas=10,
            num_expression_coeffs=10,
            ext='pkl',
            use_pca=False
        )
        
        # Get SMPL-X faces for normal computation
        self.faces = torch.from_numpy(self.smplx_model.faces.astype(np.int64))
        
        # Collect all sample paths (either full scan, or explicit subset)
        if sample_dirs is not None:
            required_files = [
                'image.jpg',
                'smplx_parameters.json',
                'contact.json',
                'box_annotation.json',
                'calibration.json',
                'extrinsic.json'
            ]
            samples = []
            for d in list(sample_dirs):
                if d is None:
                    continue
                sample_path = os.path.abspath(str(d))
                if not os.path.isdir(sample_path):
                    continue
                if not all(os.path.exists(os.path.join(sample_path, f)) for f in required_files):
                    continue

                # Best-effort parse category/id from .../<category>/<id>
                sample_id = os.path.basename(sample_path)
                category = os.path.basename(os.path.dirname(sample_path))
                samples.append({'path': sample_path, 'category': category, 'id': sample_id})
            self.samples = samples
        else:
            all_samples = self._collect_samples()

            # Filter by indices if provided
            if indices is not None:
                self.samples = [all_samples[i] for i in indices if i < len(all_samples)]
            else:
                self.samples = all_samples

        # Fail-fast: mask file is required by the model (do NOT silently skip samples)
        self._validate_object_masks()
        
        # Image transforms
        self.transform = self._get_transforms()
        
        print(f"[Dataset] Loaded {len(self.samples)} samples for {split} split")
        
    def _collect_samples(self):
        """Collect all sample directories."""
        samples = []
        
        # Iterate through categories
        for category in sorted(os.listdir(self.root_dir)):
            category_path = os.path.join(self.root_dir, category)
            if not os.path.isdir(category_path):
                continue
            
            # Iterate through sample IDs
            for sample_id in sorted(os.listdir(category_path)):
                sample_path = os.path.join(category_path, sample_id)
                if not os.path.isdir(sample_path):
                    continue
                
                # Check if all required files exist
                required_files = [
                    'image.jpg',
                    'smplx_parameters.json',
                    'contact.json',
                    'box_annotation.json',
                    'calibration.json',
                    'extrinsic.json'
                ]
                
                if all(os.path.exists(os.path.join(sample_path, f)) for f in required_files):
                    samples.append({
                        'path': sample_path,
                        'category': category,
                        'id': sample_id
                    })
        
        return samples
    
    def _get_transforms(self):
        """Get image transforms."""
        if self.augment and self.split == 'train':
            # Training augmentation
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Validation/test: no augmentation
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _validate_object_masks(self):
        """Ensure required object mask file exists for all samples in this dataset view."""
        missing = []
        for s in self.samples:
            mask_path = os.path.join(s['path'], 'object_mask.png')
            if not os.path.exists(mask_path):
                missing.append(mask_path)
                # Show only a few to keep error readable
                if len(missing) >= 5:
                    break
        if missing:
            preview = "\n".join(f"  - {p}" for p in missing)
            raise FileNotFoundError(
                "[Dataset] Missing required file object_mask.png for one or more samples.\n"
                f"{preview}\n"
                "Fix the dataset or remove those samples."
            )

    @staticmethod
    def _compute_dilated_distance_field(mask_np, dilation_kernel=31, dilation_iters=1):
        """
        Compute dilated distance field from an object mask.

        Args:
            mask_np: (H, W) uint8/boolean mask where foreground is object (non-zero/True)
            dilation_kernel: Kernel size for morphological dilation (odd integer recommended)
            dilation_iters: Dilation iterations

        Returns:
            dist_norm: (H, W) float32 in [0, 1], where 0 is on/inside dilated mask and increases with distance.
        """
        if mask_np.dtype != np.uint8:
            mask_u8 = (mask_np > 0).astype(np.uint8)
        else:
            mask_u8 = (mask_np > 0).astype(np.uint8)

        # Morphological dilation to fill occlusion gaps
        k = int(dilation_kernel)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        iters = max(1, int(dilation_iters))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_dilated = cv2.dilate(mask_u8, kernel, iterations=iters)

        # Distance to nearest object pixel: distanceTransform computes distance to nearest zero pixel,
        # so invert such that object pixels become zeros.
        inv = (1 - mask_dilated).astype(np.uint8)  # 0 on object, 1 elsewhere
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)  # float32

        H, W = dist.shape
        diag = float(np.sqrt(H * H + W * W) + 1e-8)
        dist_norm = np.clip(dist / diag, 0.0, 1.0).astype(np.float32)
        return dist_norm
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            dict with keys:
                - image: (3, H, W) - Normalized RGB image
                - vertices: (N, 3) - SMPL-X vertices in camera space
                - normals: (N, 3) - Vertex normals
                - pose_params: (63,) - Body pose parameters
                - K: (3, 3) - Camera intrinsic matrix (scaled for resized image)
                - object_bbox: (4,) - Object bounding box [x_min, y_min, x_max, y_max]
                - mask_dist_field: (1, H, W) - Dilated distance field from object mask in [0, 1]
                - contact_labels: (N,) - Ground truth contact labels (0 or 1)
                - sample_id: str - Sample identifier
        """
        sample_info = self.samples[idx]
        sample_path = sample_info['path']
        
        # 1. Load image
        img_path = os.path.join(sample_path, 'image.jpg')
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (W, H)
        image = self.transform(image)  # (3, H, W)

        # 1b. Load object mask and build distance field (in resized image space)
        mask_path = os.path.join(sample_path, 'object_mask.png')
        if not os.path.exists(mask_path):
            # This should already be caught in __init__, but keep an explicit guard here too.
            raise FileNotFoundError(f"[Dataset] Missing required file: {mask_path}")

        mask_img = Image.open(mask_path).convert('L')
        # Resize to target (W, H) using nearest neighbor to keep mask edges crisp
        mask_img = mask_img.resize((self.img_size[1], self.img_size[0]), resample=Image.NEAREST)
        mask_np = np.array(mask_img)  # (H, W), 0..255

        dist_np = self._compute_dilated_distance_field(mask_np, dilation_kernel=31, dilation_iters=1)
        mask_dist_field = torch.from_numpy(dist_np).unsqueeze(0).float()  # (1, H, W)
        
        # 2. Load SMPL-X parameters
        with open(os.path.join(sample_path, 'smplx_parameters.json'), 'r') as f:
            smplx_params = json.load(f)
        
        # Extract parameters (handle both formats)
        body_pose = torch.tensor(smplx_params['body_pose'], dtype=torch.float32).flatten()  # (63,)
        
        # Handle different naming conventions
        if 'global_orient' in smplx_params:
            global_orient = torch.tensor(smplx_params['global_orient'], dtype=torch.float32).flatten()
        elif 'root_pose' in smplx_params:
            global_orient = torch.tensor(smplx_params['root_pose'], dtype=torch.float32).flatten()
        else:
            global_orient = torch.zeros(3, dtype=torch.float32)
        
        # Handle translation
        if 'transl' in smplx_params:
            transl = torch.tensor(smplx_params['transl'], dtype=torch.float32).flatten()
        elif 'cam_trans' in smplx_params:
            transl = torch.tensor(smplx_params['cam_trans'], dtype=torch.float32).flatten()
        else:
            transl = torch.zeros(3, dtype=torch.float32)
        
        # Handle shape parameters
        if 'betas' in smplx_params:
            betas = torch.tensor(smplx_params['betas'], dtype=torch.float32).flatten()
        elif 'shape' in smplx_params:
            betas = torch.tensor(smplx_params['shape'], dtype=torch.float32).flatten()
        else:
            betas = torch.zeros(10, dtype=torch.float32)
        
        # 3. Generate SMPL-X mesh vertices using parametric model
        with torch.no_grad():
            smplx_output = self.smplx_model(
                body_pose=body_pose.unsqueeze(0),
                global_orient=global_orient.unsqueeze(0),
                transl=transl.unsqueeze(0),
                betas=betas.unsqueeze(0),
                return_verts=True
            )
            vertices_world = smplx_output.vertices[0]  # (N, 3) in world space
        
        # 4. Load camera extrinsics
        with open(os.path.join(sample_path, 'extrinsic.json'), 'r') as f:
            extrinsic = json.load(f)
        
        # Handle different naming conventions
        if 'R' in extrinsic:
            R = torch.tensor(extrinsic['R'], dtype=torch.float32).reshape(3, 3)
        elif 'rotation' in extrinsic:
            R = torch.tensor(extrinsic['rotation'], dtype=torch.float32).reshape(3, 3)
        else:
            R = torch.eye(3, dtype=torch.float32)
        
        if 'T' in extrinsic:
            T = torch.tensor(extrinsic['T'], dtype=torch.float32).flatten()
        elif 'translation' in extrinsic:
            T = torch.tensor(extrinsic['translation'], dtype=torch.float32).flatten()
        else:
            T = torch.zeros(3, dtype=torch.float32)
        
        # Transform vertices to camera space
        vertices_cam = world_to_camera(
            vertices_world.unsqueeze(0), 
            R.unsqueeze(0), 
            T.unsqueeze(0)
        )[0]  # (N, 3)
        
        # 5. Compute normals from vertices (normals_smplx.npy in dataset is actually contact data)
        normals = compute_vertex_normals(vertices_cam.unsqueeze(0), self.faces)[0]
        
        # 6. Load camera intrinsics
        with open(os.path.join(sample_path, 'calibration.json'), 'r') as f:
            calibration = json.load(f)
        
        K = torch.tensor(calibration['K'], dtype=torch.float32).reshape(3, 3)
        
        # Scale intrinsics if image was resized
        original_img_size = (original_size[1], original_size[0])  # (H, W)
        K = scale_intrinsics(K, original_img_size, self.img_size)
        
        # 7. Load object bounding box
        with open(os.path.join(sample_path, 'box_annotation.json'), 'r') as f:
            bbox_data = json.load(f)
        
        # bbox format: [x_min, y_min, x_max, y_max]
        # Handle different naming conventions
        if 'bbox' in bbox_data:
            bbox = torch.tensor(bbox_data['bbox'], dtype=torch.float32)
        elif 'obj' in bbox_data:
            bbox = torch.tensor(bbox_data['obj'], dtype=torch.float32)
        else:
            # Default to center region if no bbox found
            bbox = torch.tensor([100.0, 100.0, 400.0, 400.0], dtype=torch.float32)
        
        # Scale bbox coordinates for resized image
        scale_x = self.img_size[1] / original_size[0]
        scale_y = self.img_size[0] / original_size[1]
        bbox[0] *= scale_x  # x_min
        bbox[1] *= scale_y  # y_min
        bbox[2] *= scale_x  # x_max
        bbox[3] *= scale_y  # y_max
        
        # 8. Load contact labels
        with open(os.path.join(sample_path, 'contact.json'), 'r') as f:
            contact_data = json.load(f)
        
        # Extract first 10475 labels (human body vertices)
        # Handle both dict and list formats
        if isinstance(contact_data, dict):
            contact_labels = torch.tensor(contact_data['contact'][:10475], dtype=torch.float32)
        elif isinstance(contact_data, list):
            contact_labels = torch.tensor(contact_data[:10475], dtype=torch.float32)
        else:
            raise ValueError(f"Unknown contact data format: {type(contact_data)}")
        
        # Convert boolean to float if necessary
        if contact_labels.dtype == torch.bool:
            contact_labels = contact_labels.float()
        
        return {
            'image': image,
            'vertices': vertices_cam,
            'normals': normals,
            'pose_params': body_pose,
            'K': K,
            'object_bbox': bbox,
            'mask_dist_field': mask_dist_field,
            'contact_labels': contact_labels,
            'sample_id': f"{sample_info['category']}_{sample_info['id']}"
        }


def collate_fn(batch):
    """
    Custom collate function to handle batching.
    """
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'vertices': torch.stack([item['vertices'] for item in batch]),
        'normals': torch.stack([item['normals'] for item in batch]),
        'pose_params': torch.stack([item['pose_params'] for item in batch]),
        'K': torch.stack([item['K'] for item in batch]),
        'object_bbox': torch.stack([item['object_bbox'] for item in batch]),
        'mask_dist_field': torch.stack([item['mask_dist_field'] for item in batch]),
        'contact_labels': torch.stack([item['contact_labels'] for item in batch]),
        'sample_ids': [item['sample_id'] for item in batch]
    }


def split_dataset(root_dir, train_ratio=0.85, val_ratio=0.10, test_ratio=0.05, seed=42):
    """
    Split dataset into train/val/test sets.
    
    Returns:
        train_indices, val_indices, test_indices
    """
    # Collect valid samples (must match SmplContactDataset._collect_samples ordering)
    required_files = [
        'image.jpg',
        'smplx_parameters.json',
        'contact.json',
        'box_annotation.json',
        'calibration.json',
        'extrinsic.json'
    ]

    all_samples = []
    for category in sorted(os.listdir(root_dir)):
        category_path = os.path.join(root_dir, category)
        if not os.path.isdir(category_path):
            continue

        for sample_id in sorted(os.listdir(category_path)):
            sample_path = os.path.join(category_path, sample_id)
            if not os.path.isdir(sample_path):
                continue

            if all(os.path.exists(os.path.join(sample_path, f)) for f in required_files):
                all_samples.append(sample_path)
    
    # Shuffle
    np.random.seed(seed)
    indices = np.random.permutation(len(all_samples))
    
    # Split
    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    return train_indices, val_indices, test_indices


if __name__ == "__main__":
    # Test dataset
    dataset = SmplContactDataset(
        root_dir="data_contact",
        smplx_model_path="smplx_models",
        img_size=(512, 512),
        split='train',
        augment=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    
    print("\nSample keys:", sample.keys())
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"{key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"{key}: {val}")
