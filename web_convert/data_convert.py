#!/usr/bin/env python3
"""Web-only dataset converter for Contact.

This is the migrated converter for the Contact repo. It intentionally avoids
any dependency on Ca3OH1/HOIGaussian scripts.

Per dataset, it:
- Copies source files into the target sample directory
- Ensures object mask naming is `object_mask.png` (renames `obj_mask.png`)
- Copies `obj_pcd_h_align.obj` and `object_mesh.obj` if available
- Writes `calibration.json` / `extrinsic.json` from `smplx_parameters.json`
- Generates `contact.json` using geometric contact (Open3D)

The web UI imports and calls `process_single_dataset()`.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional, Tuple


try:
    from .contact_geometry import generate_contact_json
    from .utils import ensure_object_mask_name, write_camera_files_from_smplx_params
except Exception:
    # Script execution fallback (python data_convert.py ...)
    from contact_geometry import generate_contact_json
    from utils import ensure_object_mask_name, write_camera_files_from_smplx_params


def _default_smplx_model_path() -> str:
    # Contact repo layout: Contact/web_convert/ -> Contact/smplx_models/
    here = Path(__file__).resolve()
    return str(here.parents[1] / "smplx_models")


def _copytree_files(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for entry in os.scandir(src_dir):
        if not entry.is_file():
            continue
        shutil.copy2(entry.path, os.path.join(dst_dir, entry.name))


def _resolve_object_artifact(object_dir: Optional[str], relative_path: str, filename: str) -> Optional[str]:
    if not object_dir:
        return None
    candidates: Iterable[str] = (
        os.path.join(object_dir, relative_path, filename),
        os.path.join(object_dir, os.path.basename(relative_path), filename),
    )
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def process_single_dataset(
    dataset_dir: str,
    relative_path: str,
    source_dir: str,
    target_dir: str,
    object_dir: Optional[str],
    distance_ratio: float,
    *,
    smplx_model_path: Optional[str] = None,
    interactive: bool = False,
) -> Tuple[str, str, float, Optional[str]]:
    """Convert one dataset.

    Returns:
        (decision, relative_path, distance_ratio, error_message)

    Notes:
        - `interactive` is kept only for API compatibility with the legacy tool.
    """
    del source_dir
    del interactive

    if smplx_model_path is None:
        smplx_model_path = _default_smplx_model_path()

    try:
        dst_dir = os.path.join(target_dir, relative_path)
        _copytree_files(dataset_dir, dst_dir)

        ensure_object_mask_name(dst_dir)

        # Copy required object artifacts (per requirement)
        for name in ("obj_pcd_h_align.obj", "object_mesh.obj"):
            dst_path = os.path.join(dst_dir, name)
            if os.path.exists(dst_path):
                continue

            src_path = os.path.join(dataset_dir, name)
            if not os.path.exists(src_path):
                src_path = _resolve_object_artifact(object_dir, relative_path, name) or src_path

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)

        write_camera_files_from_smplx_params(dst_dir)
        generate_contact_json(dst_dir, smplx_model_path=smplx_model_path, distance_ratio=float(distance_ratio))

        return "accept", relative_path, float(distance_ratio), None
    except Exception as e:
        return "skip", relative_path, float(distance_ratio), str(e)


def _iter_tasks(source_dir: str, category: Optional[str] = None):
    source_path = Path(source_dir)
    if category:
        categories = [category]
    else:
        categories = [d.name for d in source_path.iterdir() if d.is_dir()]
        categories.sort()

    for cat in categories:
        cat_path = source_path / cat
        if not cat_path.exists():
            continue
        subdirs = [d for d in cat_path.iterdir() if d.is_dir()]
        subdirs.sort()
        for subdir in subdirs:
            yield str(subdir), os.path.join(cat, subdir.name)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Web-only dataset converter (Contact)")
    p.add_argument("--source_dir", required=True)
    p.add_argument("--target_dir", required=True)
    p.add_argument("--object_dir", default=None)
    p.add_argument("--category", default=None)
    p.add_argument("--distance_ratio", type=float, default=0.5)
    p.add_argument("--smplx_model_path", default=None)
    args = p.parse_args(argv)

    os.makedirs(args.target_dir, exist_ok=True)
    failures = 0
    for dataset_dir, rel in _iter_tasks(args.source_dir, category=args.category):
        _, _, _, err = process_single_dataset(
            dataset_dir,
            rel,
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            object_dir=args.object_dir,
            distance_ratio=args.distance_ratio,
            smplx_model_path=args.smplx_model_path,
            interactive=False,
        )
        if err:
            failures += 1
            print(f"[FAIL] {rel}: {err}")
        else:
            print(f"[OK]   {rel}")

    print(f"Done. failures={failures}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

# --- Disabled legacy/duplicated code below (kept only to avoid patch churn) ---
'''

#!/usr/bin/env python3
"""Web-only dataset converter for Contact.

This module is a slimmed-down replacement for the original Ca3OH1/HOIGaussian
converter. It is designed to live inside the Contact repo and run in the
"contact" conda environment.

What it does for each dataset directory:
1) Copy source sample files to target sample dir
2) Ensure mask name is object_mask.png (rename obj_mask.png if needed)
3) Copy required object artifacts (obj_pcd_h_align.obj and object_mesh.obj)
4) Generate camera files: calibration.json / extrinsic.json
5) Generate contact.json via geometric contact (Open3D)

The web UI imports and calls `process_single_dataset()` in background.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional, Tuple


try:
    from .contact_geometry import generate_contact_json
    from .utils import ensure_object_mask_name, write_camera_files_from_smplx_params
except Exception:
    # Script execution fallback (python data_convert.py ...)
    from contact_geometry import generate_contact_json
    from utils import ensure_object_mask_name, write_camera_files_from_smplx_params


def _default_smplx_model_path() -> str:
    # Contact repo layout: Contact/web_convert/ -> Contact/smplx_models/
    here = Path(__file__).resolve()
    return str(here.parents[1] / "smplx_models")


def _copytree_files(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for entry in os.scandir(src_dir):
        if not entry.is_file():
            continue
        src_path = entry.path
        dst_path = os.path.join(dst_dir, entry.name)
        shutil.copy2(src_path, dst_path)


def _resolve_object_artifact(object_dir: Optional[str], relative_path: str, filename: str) -> Optional[str]:
    if not object_dir:
        return None
    candidates: Iterable[str] = (
        os.path.join(object_dir, relative_path, filename),
        os.path.join(object_dir, os.path.basename(relative_path), filename),
    )
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def process_single_dataset(
    dataset_dir: str,
    relative_path: str,
    source_dir: str,
    target_dir: str,
    object_dir: Optional[str],
    distance_ratio: float,
    *,
    smplx_model_path: Optional[str] = None,
    interactive: bool = False,
) -> Tuple[str, str, float, Optional[str]]:
    """Convert one dataset.

    Returns:
        (decision, relative_path, distance_ratio, error_message)

    Notes:
        - `interactive` is kept for API compatibility with the legacy tool; web
          mode always passes interactive=False.
    """

    del interactive

    if smplx_model_path is None:
        smplx_model_path = _default_smplx_model_path()

    try:
        # Compute target path and copy sample files
        dst_dir = os.path.join(target_dir, relative_path)
        _copytree_files(dataset_dir, dst_dir)

        # Ensure object mask naming
        ensure_object_mask_name(dst_dir)

        # Copy object artifacts (per user requirement)
        for name in ("obj_pcd_h_align.obj", "object_mesh.obj"):
            dst_path = os.path.join(dst_dir, name)
            if os.path.exists(dst_path):
                continue

            # Prefer artifact in source dataset
            src_path = os.path.join(dataset_dir, name)
            if not os.path.exists(src_path):
                src_path = _resolve_object_artifact(object_dir, relative_path, name) or src_path

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)

        # Generate camera intrinsics/extrinsics (required by Contact dataset)
        write_camera_files_from_smplx_params(dst_dir)

        # Generate geometric contact labels
        generate_contact_json(
            dst_dir,
            smplx_model_path=smplx_model_path,
            distance_ratio=float(distance_ratio),

            # Calculate view angles based on cam_trans
            # cam_trans gives the camera position, we want to look from there towards origin
            if self.cam_trans is not None:
                # Convert cam_trans to spherical coordinates for view angles
                x, y, z = self.cam_trans
                r = np.sqrt(x**2 + y**2 + z**2)
                # Elevation: angle from xy-plane
                elev = np.degrees(np.arcsin(z / r)) if r > 0 else 15
                # Azimuth: angle in xy-plane
                azim = np.degrees(np.arctan2(y, x))
            else:
                elev, azim = 15, 45
            
            # Create figure with reference image
            if self.fig is not None:
                plt.close(self.fig)
            
            # Adjust layout based on whether we have reference image
            if self.reference_image is not None:
                self.fig = plt.figure(figsize=(20, 9))
                # Use GridSpec for better control
                from matplotlib.gridspec import GridSpec
                gs = GridSpec(2, 4, figure=self.fig, height_ratios=[4, 1], hspace=0.35)
                ax1 = self.fig.add_subplot(gs[0, 0], projection='3d')
                ax2 = self.fig.add_subplot(gs[0, 1], projection='3d')
                ax3 = self.fig.add_subplot(gs[0, 2], projection='3d')
                ax_img = self.fig.add_subplot(gs[0, 3])
            else:
                self.fig = plt.figure(figsize=(18, 7))
                from matplotlib.gridspec import GridSpec
                gs = GridSpec(2, 3, figure=self.fig, height_ratios=[4, 1], hspace=0.35)
                ax1 = self.fig.add_subplot(gs[0, 0], projection='3d')
                ax2 = self.fig.add_subplot(gs[0, 1], projection='3d')
                ax3 = self.fig.add_subplot(gs[0, 2], projection='3d')
            
            # Plot 1: Human with contact (color-coded by detection method)
            # Non-contact vertices
            ax1.scatter(self.h_verts[~human_contact, 0], self.h_verts[~human_contact, 1], 
                       self.h_verts[~human_contact, 2], c='lightblue', s=1, alpha=0.3)
            # Interior-only contact (orange)
            human_interior_only = human_interior & ~human_proximity
            ax1.scatter(self.h_verts[human_interior_only, 0], self.h_verts[human_interior_only, 1], 
                       self.h_verts[human_interior_only, 2], c='orange', s=3, label='Interior')
            # Proximity-only contact (yellow)
            human_proximity_only = human_proximity & ~human_interior
            ax1.scatter(self.h_verts[human_proximity_only, 0], self.h_verts[human_proximity_only, 1], 
                       self.h_verts[human_proximity_only, 2], c='yellow', s=3, label='Proximity')
            # Both methods (red)
            human_both = human_interior & human_proximity
            ax1.scatter(self.h_verts[human_both, 0], self.h_verts[human_both, 1], 
                       self.h_verts[human_both, 2], c='red', s=4, label='Both')
            ax1.set_xlim([h_x_min, h_x_max])
            ax1.set_ylim([h_y_min, h_y_max])
            ax1.set_zlim([h_z_min, h_z_max])
            ax1.set_title('Human Mesh')
            ax1.legend(loc='upper right', fontsize=8)
            ax1.view_init(elev=elev, azim=azim)
            ax1.set_box_aspect([1,1,1])
            
            # Plot 2: Object with contact (color-coded by detection method)
            # Non-contact vertices
            ax2.scatter(obj_verts[~obj_contact, 0], obj_verts[~obj_contact, 1], 
                       obj_verts[~obj_contact, 2], c='lightgreen', s=1, alpha=0.3)
            # Interior-only contact (orange)
            obj_interior_only = obj_interior & ~obj_proximity
            ax2.scatter(obj_verts[obj_interior_only, 0], obj_verts[obj_interior_only, 1], 
                       obj_verts[obj_interior_only, 2], c='orange', s=3, label='Interior')
            # Proximity-only contact (yellow)
            obj_proximity_only = obj_proximity & ~obj_interior
            ax2.scatter(obj_verts[obj_proximity_only, 0], obj_verts[obj_proximity_only, 1], 
                       obj_verts[obj_proximity_only, 2], c='yellow', s=3, label='Proximity')
            # Both methods (red)
            obj_both = obj_interior & obj_proximity
            ax2.scatter(obj_verts[obj_both, 0], obj_verts[obj_both, 1], 
                       obj_verts[obj_both, 2], c='red', s=4, label='Both')
            ax2.set_xlim([o_x_min, o_x_max])
            ax2.set_ylim([o_y_min, o_y_max])
            ax2.set_zlim([o_z_min, o_z_max])
            ax2.set_title('Object Mesh')
            ax2.legend(loc='upper right', fontsize=8)
            ax2.view_init(elev=elev, azim=azim)
            ax2.set_box_aspect([1,1,1])
            
            # Plot 3: Combined view with color-coded contacts
            ax3.scatter(self.h_verts[:, 0], self.h_verts[:, 1], self.h_verts[:, 2], 
                       c='lightblue', s=1, alpha=0.2, label='Human')
            ax3.scatter(obj_verts[:, 0], obj_verts[:, 1], obj_verts[:, 2], 
                       c='lightgreen', s=1, alpha=0.2, label='Object')
            # Show all contact points with combined color
            ax3.scatter(self.h_verts[human_contact, 0], self.h_verts[human_contact, 1], 
                       self.h_verts[human_contact, 2], c='red', s=3, label='H-Contact', alpha=0.6)
            ax3.scatter(obj_verts[obj_contact, 0], obj_verts[obj_contact, 1], 
                       obj_verts[obj_contact, 2], c='darkred', s=3, label='O-Contact', alpha=0.6)
            ax3.set_xlim([c_x_min, c_x_max])
            ax3.set_ylim([c_y_min, c_y_max])
            ax3.set_zlim([c_z_min, c_z_max])
            ax3.set_title('Combined View')
            ax3.legend(loc='upper right', fontsize=8)
            ax3.view_init(elev=elev, azim=azim)
            ax3.set_box_aspect([1,1,1])
            
            # Display reference image if available
            if self.reference_image is not None:
                ax_img.imshow(self.reference_image)
                ax_img.axis('off')
                ax_img.set_title('Reference Image')
            
            # Calculate statistics
            total_contact = np.sum(human_contact) + np.sum(obj_contact)
            total_verts = len(human_contact) + len(obj_contact)
            contact_ratio = total_contact / total_verts * 100
            
            human_interior_count = np.sum(human_interior)
            human_proximity_count = np.sum(human_proximity)
            human_both_count = np.sum(human_interior & human_proximity)
            
            obj_interior_count = np.sum(obj_interior)
            obj_proximity_count = np.sum(obj_proximity)
            obj_both_count = np.sum(obj_interior & obj_proximity)
            
            # Title with detailed statistics
            cam_info = f' | View: elev={elev:.1f}°, azim={azim:.1f}°' if self.cam_trans is not None else ''
            self.fig.suptitle(
                f'Dataset: {self.relative_path} | Distance Ratio: {self.distance_ratio:.3f} | '
                f'Total Contact: {total_contact} ({contact_ratio:.2f}%){cam_info}\n'
                f'Human: Interior={human_interior_count}, Proximity={human_proximity_count}, '
                f'Both={human_both_count}, Total={np.sum(human_contact)}\n'
                f'Object: Interior={obj_interior_count}, Proximity={obj_proximity_count}, '
                f'Both={obj_both_count}, Total={np.sum(obj_contact)}',
                fontsize=11
            )
            
            # Add controls at bottom with better positioning
            # Distance ratio input box
            ax_ratio_box = plt.axes([0.15, 0.08, 0.12, 0.05])
            self.ratio_box = TextBox(ax_ratio_box, 'Ratio:', initial=f'{self.distance_ratio:.3f}')
            self.ratio_box.on_submit(self.on_ratio_change)
            
            # Preview button
            ax_preview = plt.axes([0.30, 0.08, 0.08, 0.05])
            self.btn_preview = Button(ax_preview, 'Preview')
            self.btn_preview.on_clicked(self.on_preview)
            
            # Accept button
            ax_accept = plt.axes([0.42, 0.08, 0.08, 0.05])
            self.btn_accept = Button(ax_accept, 'Accept')
            self.btn_accept.on_clicked(self.on_accept)
            
            # Skip button
            ax_skip = plt.axes([0.54, 0.08, 0.08, 0.05])
            self.btn_skip = Button(ax_skip, 'Give Up')
            self.btn_skip.on_clicked(self.on_skip)
            
            # Abort button
            ax_abort = plt.axes([0.66, 0.08, 0.08, 0.05])
            self.btn_abort = Button(ax_abort, 'Abort All')
            self.btn_abort.on_clicked(self.on_abort)
            
            # Adjust layout - reduce top margin and bottom margin
            plt.subplots_adjust(top=0.92, bottom=0.18)
            
        def on_ratio_change(self, text):
            """Handle distance ratio text input."""
            try:
                new_ratio = float(text)
                if 0.01 <= new_ratio <= 3.0:
                    self.distance_ratio = new_ratio
                else:
                    print(f"Distance ratio must be between 0.01 and 3.0")
            except ValueError:
                print(f"Invalid distance ratio: {text}")
        
        def on_preview(self, event):
            """Handle preview button click."""
            self.decision = 'preview'
            plt.close(self.fig)
        
        def on_accept(self, event):
            """Handle accept button click."""
            self.decision = 'accept'
            plt.close(self.fig)
        
        def on_skip(self, event):
            """Handle skip button click."""
            self.decision = 'skip'
            plt.close(self.fig)
        
        def on_abort(self, event):
            """Handle abort all button click."""
            self.decision = 'abort'
            plt.close(self.fig)
        
        def run(self):
            """Run interactive visualization loop."""
            while True:
                self.calculate_and_visualize()
                plt.show()
                
                if self.decision == 'preview':
                    self.decision = None
                    continue
                elif self.decision in ['accept', 'skip', 'abort']:
                    return self.decision, self.distance_ratio
                else:
                    # Window closed without decision
                    return 'skip', self.distance_ratio
    
    try:
        visualizer = InteractiveVisualizer(data_path, relative_path, initial_distance_ratio)
        return visualizer.run()
    except Exception as e:
        print(f"Visualization error for {relative_path}: {e}")
        return 'skip', initial_distance_ratio





def generate_contact_json(data_path, distance_ratio=0.5):
    """
    Generate contact.json for a dataset (efficient version without PLY output).
    
    Args:
        data_path: Path to directory containing smplx_parameters.json and object_mesh.obj
        distance_ratio: Distance ratio for proximity detection (relative to mesh resolution)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Suppress Open3D warnings
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        
        object_mesh_path = os.path.join(data_path, "object_mesh.obj")
        smplx_param_path = os.path.join(data_path, "smplx_parameters.json")
        
        # Check if files exist
        if not os.path.exists(smplx_param_path) or not os.path.exists(object_mesh_path):
            return False
        
        # Load human mesh
        h_verts, h_faces = load_smplx_vertices(data_path)
        
        # Create human mesh from SMPLX vertices and faces
        human_mesh = o3d.geometry.TriangleMesh()
        human_mesh.vertices = o3d.utility.Vector3dVector(h_verts)
        human_mesh.triangles = o3d.utility.Vector3iVector(h_faces)
        
        # Load and simplify object mesh (same as create_contact_area.py)
        obj_mesh, obj_verts, obj_faces = load_obj_mesh(
            object_mesh_path,
            cam_trans=None,
            simplify=True, 
            target_triangles=8000
        )
        
        # Create object mesh
        obj_mesh_o3d = o3d.geometry.TriangleMesh()
        obj_mesh_o3d.vertices = o3d.utility.Vector3dVector(obj_verts)
        obj_mesh_o3d.triangles = obj_mesh.triangles
        
        # Calculate contact area using combined method (interior + proximity)
        # This will merge results from both detection methods
        human_contact, obj_contact = calculate_contact_area(
            human_mesh, h_verts, obj_mesh_o3d, obj_verts, 
            distance_ratio=distance_ratio
        )
        
        # Save contact.json (merged results from both methods)
        contact_list = np.concatenate([
            human_contact.astype(bool),
            obj_contact.astype(bool)
        ]).tolist()
        
        contact_output_path = os.path.join(data_path, "contact.json")
        with open(contact_output_path, 'w') as f:
            json.dump(contact_list, f)
        
        return True
        
    except Exception as e:
        print(f"Error generating contact.json for {data_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_single_dataset(subdir_path, relative_path, source_base, target_base, object_base, 
                          initial_distance_ratio=0.5, interactive=True):
    """
    Process a single dataset with interactive visualization.
    
    Args:
        subdir_path: Full path to subdirectory
        relative_path: Relative path (including category and subdirectory name)
        source_base: Source base directory (data)
        target_base: Target base directory (data_sandwich)
        object_base: Object base directory (data_object)
        initial_distance_ratio: Initial distance ratio for proximity detection
        interactive: If True, show interactive GUI; if False, use default ratio
    
    Returns:
        (decision: str, relative_path: str, distance_ratio: float, error_msg: str or None)
        decision: 'accept', 'skip'
    """
    target_path = os.path.join(target_base, relative_path)
    
    try:
        # First, get user decision with interactive visualization
        if interactive:
            decision, distance_ratio = visualize_and_get_decision(subdir_path, relative_path, initial_distance_ratio)
            
            if decision == 'skip':
                return ('skip', relative_path, distance_ratio, 'User gave up')
        else:
            decision = 'accept'
            distance_ratio = initial_distance_ratio
        
        # If accepted, proceed with processing
        if decision == 'accept':
            target_dir = os.path.dirname(target_path)
            
            # 1. Create target directory and copy
            os.makedirs(target_dir, exist_ok=True)
            
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            
            shutil.copytree(subdir_path, target_path)
            
            # 2. Rename obj_mask.png to object_mask.png
            old_mask_path = os.path.join(target_path, "obj_mask.png")
            new_mask_path = os.path.join(target_path, "object_mask.png")
            if os.path.exists(old_mask_path):
                os.rename(old_mask_path, new_mask_path)
            
            # 3. Copy obj_pcd_h_align.obj from data_object
            object_source_path = os.path.join(object_base, relative_path, "obj_pcd_h_align.obj")
            object_target_path = os.path.join(target_path, "obj_pcd_h_align.obj")
            
            if os.path.exists(object_source_path):
                shutil.copy2(object_source_path, object_target_path)
            
            # 4. Run camera.py
            camera_script = os.path.join(os.path.dirname(__file__), "..", "prepare", "camera.py")
            camera_cmd = ["python", camera_script, "--data_dir", target_path]
            result = subprocess.run(camera_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                return ('accept', relative_path, distance_ratio, f"camera.py failed")
            
            # 5. Run normal.py
            normal_script = os.path.join(os.path.dirname(__file__), "..", "prepare", "normal.py")
            normal_cmd = ["python", normal_script, "--data_dir", target_path]
            result = subprocess.run(normal_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                return ('accept', relative_path, distance_ratio, f"normal.py failed")
            
            # 6. Generate contact.json with custom distance ratio
            if not generate_contact_json(target_path, distance_ratio):
                return ('accept', relative_path, distance_ratio, f"contact.json generation failed")
            
            return ('accept', relative_path, distance_ratio, None)
        
        return (decision, relative_path, distance_ratio, None)
        
    except Exception as e:
        return ('skip', relative_path, initial_distance_ratio, str(e))


def check_directory_structure(source_dir, object_dir):
    """Check if directories have the expected structure and required files."""
    source_path = Path(source_dir)
    object_path = Path(object_dir)
    
    issues = []
    
    if not source_path.exists():
        issues.append(f"Source directory does not exist: {source_dir}")
    if not object_path.exists():
        issues.append(f"Object directory does not exist: {object_dir}")
    
    if issues:
        return issues
    
    # Check a few samples
    categories = [d for d in source_path.iterdir() if d.is_dir()]
    if not categories:
        issues.append(f"No category directories found in {source_dir}")
        return issues
    
    # Sample check
    sample_cat = categories[0]
    subdirs = [d for d in sample_cat.iterdir() if d.is_dir()]
    if subdirs:
        sample_subdir = subdirs[0]
        required_files = ["smplx_parameters.json", "image.jpg"]
        for req_file in required_files:
            if not (sample_subdir / req_file).exists():
                issues.append(f"Sample dataset missing {req_file}: {sample_subdir}")
        
        # Check if obj_pcd_h_align.obj is in object_dir
        relative = sample_subdir.relative_to(source_path)
        obj_file = object_path / relative / "obj_pcd_h_align.obj"
        if not obj_file.exists():
            issues.append(f"Object file not found in object_dir: {obj_file}")
    
    return issues


def load_progress(target_dir):
    """Load progress from target directory."""
    progress_path = os.path.join(target_dir, "progress.json")
    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_progress(target_dir, progress):
    """Save progress to target directory."""
    progress_path = os.path.join(target_dir, "progress.json")
    with open(progress_path, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

def scan_and_process(source_dir, target_dir, object_dir, category=None, auto_scale=False, random_order=False):
    """
    Scan and process all data directories with interactive visualization for each dataset.
    
    Args:
        source_dir: Source data directory
        target_dir: Target data directory
        object_dir: Object data directory
        category: Specified category name, if None process all categories
        auto_scale: If True, skip interactive visualization and use default scale
        random_order: If True, process datasets in random order instead of by category
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return
    
    # Create target directory if not exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Load existing progress
    progress = load_progress(target_dir)
    if progress:
        print(f"\n{'='*70}")
        print(f"Found existing progress: {len(progress)} datasets already processed")
        print(f"{'='*70}\n")
    
    # Check directory structure
    issues = check_directory_structure(source_dir, object_dir)
    if issues:
        print(f"\n{'='*70}")
        print("Directory Structure Issues Detected:")
        print(f"{'='*70}")
        for issue in issues:
            print(f"  ⚠ {issue}")
        print(f"{'='*70}\n")
        
        if "does not exist" in issues[0]:
            return
        
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return
    
    # Determine categories to process
    if category:
        categories = [category]
        category_path = source_path / category
        if not category_path.exists():
            print(f"Error: Category directory does not exist: {category_path}")
            return
    else:
        # Get all category directories
        categories = [d.name for d in source_path.iterdir() if d.is_dir()]
        categories.sort()
    
    print(f"\n{'='*70}")
    print(f"Data Conversion Tool - Interactive Mode")
    print(f"{'='*70}")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Categories: {len(categories)}")
    print(f"Interactive: {not auto_scale}")
    print(f"Order: {'Random' if random_order else 'By Category'}")
    print(f"{'='*70}\n")
    
    # Collect all datasets
    all_datasets = []
    for cat in categories:
        cat_path = source_path / cat
        subdirs = [d for d in cat_path.iterdir() if d.is_dir()]
        subdirs.sort()
        
        for subdir in subdirs:
            relative_path = os.path.join(cat, subdir.name)
            all_datasets.append((str(subdir), relative_path, cat))
    
    # Shuffle if random order is requested
    if random_order:
        random.shuffle(all_datasets)
    
    total_datasets = len(all_datasets)
    print(f"Total datasets to process: {total_datasets}\n")
    
    if total_datasets == 0:
        print("No datasets found.")
        return
    
    # Process each dataset sequentially with visualization
    success_count = 0
    skipped_count = 0
    failed_count = 0
    resumed_count = 0
    given_up_list = []
    dataset_info = {}  # {relative_path: distance_ratio}
    aborted = False
    
    for idx, (subdir_path, relative_path, cat) in enumerate(all_datasets, 1):
        # Check if already processed
        if relative_path in progress:
            resumed_count += 1
            prog = progress[relative_path]
            if prog['decision'] == 'accept':
                success_count += 1
                dataset_info[relative_path] = prog['distance_ratio']
                print(f"\n[{idx}/{total_datasets}] {relative_path}: ✓ Already processed (ratio={prog['distance_ratio']:.3f})")
            else:
                skipped_count += 1
                print(f"\n[{idx}/{total_datasets}] {relative_path}: ⊗ Already skipped")
            continue
        
        print(f"\n[{idx}/{total_datasets}] Processing: {relative_path}")
        
        decision, rel_path, distance_ratio, error_msg = process_single_dataset(
            subdir_path, relative_path, source_dir, target_dir, object_dir,
            initial_distance_ratio=0.5, interactive=(not auto_scale)
        )
        
        # Check for abort
        if decision == 'abort':
            print(f"⊗ User aborted processing")
            aborted = True
            break
        
        # Record progress
        progress[relative_path] = {
            'decision': decision,
            'distance_ratio': distance_ratio,
            'error': error_msg,
            'category': cat
        }
        save_progress(target_dir, progress)
        
        if decision == 'accept':
            if error_msg is None:
                success_count += 1
                dataset_info[relative_path] = distance_ratio
                print(f"✓ Accepted with ratio={distance_ratio:.3f}")
            else:
                failed_count += 1
                print(f"✗ Failed: {error_msg}")
        elif decision == 'skip':
            skipped_count += 1
            given_up_list.append({
                'path': relative_path,
                'category': cat,
                'reason': error_msg or 'User gave up',
                'distance_ratio_attempted': distance_ratio
            })
            print(f"⊗ Skipped/Given up")
    
    # Save give_up.json
    give_up_path = os.path.join(target_dir, "give_up.json")
    with open(give_up_path, 'w', encoding='utf-8') as f:
        json.dump(given_up_list, f, indent=2, ensure_ascii=False)
    
    # Print final summary
    print(f"\n{'='*70}")
    if aborted:
        print(f"Processing Aborted by User")
    else:
        print(f"Conversion Complete")
    print(f"{'='*70}")
    print(f"Total: {total_datasets}")
    print(f"  ✓ Success: {success_count}")
    print(f"  ⊗ Skipped: {skipped_count}")
    print(f"  ✗ Failed: {failed_count}")
    if resumed_count > 0:
        print(f"  ↻ Resumed: {resumed_count}")
    print(f"\nProgress saved to: {os.path.join(target_dir, 'progress.json')}")
    print(f"Given up records saved to: {give_up_path}")
    
    if dataset_info:
        print(f"\nDistance ratios used:")
        cat_ratios = {}
        for path, ratio in dataset_info.items():
            cat = path.split(os.sep)[0]
            if cat not in cat_ratios:
                cat_ratios[cat] = []
            cat_ratios[cat].append(ratio)
        
        for cat, ratios in sorted(cat_ratios.items()):
            avg_ratio = sum(ratios) / len(ratios)
            print(f"  {cat}: avg={avg_ratio:.3f} (min={min(ratios):.3f}, max={max(ratios):.3f}, n={len(ratios)})")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Data Convert Tool - Batch convert datasets with interactive scale parameter tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode - adjust scale parameters for each dataset
  python data_convert.py
  
  # Web mode - browser-based multi-user interface
  python data_convert.py --web-mode --host 0.0.0.0 --port 5000
  
  # Auto mode - use default parameters without interaction
  python data_convert.py --auto-scale
  
  # Convert only specific category with interaction
  python data_convert.py --category sandwich
  
  # Specify custom paths
  python data_convert.py --source ./data --target ./data_contact --object ./data_object

Interactive GUI Controls (shown for each dataset):
  - Scale Factor input box: Enter desired scale factor (1.0-2.0)
  - Preview button: Regenerate visualization with new scale
  - Accept button: Process dataset with current scale
  - Give Up button: Skip this dataset (recorded in give_up.json)

Web Mode:
  - Browser-based interface with Plotly 3D visualization
  - Multi-user support with automatic task distribution
  - Real-time progress tracking across all users
  - Access from any device on the same network

Note: Distance threshold is fixed at 5.0% (same as create_contact_area.py)
Given up datasets are recorded in give_up.json in the target directory
        """
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='/home/xhsystem/Code/Term7/Ca3OH1/data',
        help='Source data directory (default: /home/xhsystem/Code/Term7/Ca3OH1/data)'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='/home/xhsystem/Code/Term7/Ca3OH1/data_contact',
        help='Target data directory (default: /home/xhsystem/Code/Term7/Ca3OH1/data_contact)'
    )
    
    parser.add_argument(
        '--object',
        type=str,
        default='/home/xhsystem/Code/Term7/Ca3OH1/data_object',
        help='Object data directory (default: /home/xhsystem/Code/Term7/Ca3OH1/data_object)'
    )
    
    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='Specify category name to process (e.g., bottle), if not specified will process all categories'
    )
    
    parser.add_argument(
        '--auto-scale',
        action='store_true',
        help='Use default scale parameters without interactive preview (default: False)'
    )
    
    parser.add_argument(
        '--random-order',
        action='store_true',
        help='Process datasets in random order instead of by category (default: False)'
    )
    
    parser.add_argument(
        '--web-mode',
        action='store_true',
        help='Run in web mode with browser-based interface for multi-user collaboration (default: False)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host address for web server (default: 0.0.0.0 for LAN access)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port for web server (default: 5000)'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    source_dir = os.path.abspath(args.source)
    target_dir = os.path.abspath(args.target)
    object_dir = os.path.abspath(args.object)
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Check if web mode
    if args.web_mode:
        # Import and run web interface
        try:
            from web_interface import initialize_app, run_server
            
            print(f"\n{'='*70}")
            print(f"Starting Web Interface for Data Convert Tool")
            print(f"{'='*70}")
            print(f"Source: {source_dir}")
            print(f"Target: {target_dir}")
            print(f"Object: {object_dir}")
            print(f"Category: {args.category or 'All categories'}")
            print(f"Order: {'Random' if args.random_order else 'By Category'}")
            print(f"{'='*70}\n")
            
            initialize_app(source_dir, target_dir, object_dir, args.category, args.random_order)
            run_server(host=args.host, port=args.port)
            
        except ImportError as e:
            print(f"\nError: Web interface dependencies not installed.")
            print(f"Please install: pip install flask flask-socketio plotly")
            print(f"Details: {e}\n")
            sys.exit(1)
    else:
        # Start traditional processing
        scan_and_process(source_dir, target_dir, object_dir, args.category, args.auto_scale, args.random_order)


if __name__ == "__main__":
    main()

'''
