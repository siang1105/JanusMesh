#!/usr/bin/env python3
"""
CLIP helpers for voxel occupancy: multi-view rendering, scoring, and azimuth selection.
"""

import os
import math
import csv
from typing import List, Tuple
import numpy as np
import torch
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

# --- CLIP model / encoding ---
def _load_clip():
    """Load CLIP (open_clip preferred, OpenAI clip fallback)."""
    try:
        import open_clip
        print("[INFO] Using open-clip-torch")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        backend = "open_clip"
        return model, preprocess, tokenizer, backend
    except Exception as e:
        print("[WARN] open-clip not available:", e)
        try:
            import clip
            print("[INFO] Falling back to openai/clip")
            model, preprocess = clip.load("ViT-B/32", device="cpu")
            tokenizer = None
            backend = "openai_clip"
            return model, preprocess, tokenizer, backend
        except Exception as e2:
            print("[ERROR] Could not import any CLIP backend:", e2)
            raise

def _clip_encode_text(model, tokenizer, backend, prompt: str, device="cpu"):
    """L2-normalized text embedding for the given prompt."""
    if backend == "open_clip":
        with torch.no_grad():
            text = tokenizer([prompt]).to(device)
            text_features = model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    else:
        import clip
        with torch.no_grad():
            text = clip.tokenize([prompt]).to(device)
            text_features = model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

def _clip_encode_images(model, preprocess, backend, image_paths: List[str], device="cpu"):
    """L2-normalized image embeddings for a batch of image paths."""
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        imgs.append(preprocess(img))
    image_input = torch.stack(imgs, dim=0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features

# --- Voxel occupancy -> PLY point cloud ---
def voxel_to_ply(occ: torch.Tensor, out_path: str, resolution=64):
    """
    Export occupied voxels (threshold > 0.5) to a PLY point cloud.

    Args:
        occ: Tensor [B, C, D, H, W], [B, D, H, W], or [D, H, W].
        out_path: Output PLY path.
        resolution: Grid resolution used to map indices to [-0.5, 0.5].
    """
    # Collapse batch / channel if present.
    if occ.ndim == 5:  # [B, C, D, H, W]
        occ = occ[0, 0]  # First batch and channel.
    elif occ.ndim == 4:  # [B, D, H, W]
        occ = occ[0]
    
    # Occupied cell centers as 3D points.
    coords = torch.argwhere(occ > 0.5).cpu().numpy()
    
    if len(coords) == 0:
        print(f"[WARN] Empty voxel, creating single point")
        coords = np.array([[resolution//2, resolution//2, resolution//2]])
    
    points = (coords + 0.5) / resolution - 0.5
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"[INFO] Saved voxel as PLY: {out_path} ({len(points)} points)")
    
    return out_path

def spherical_to_cartesian(radius: float, azimuth_deg: float, elevation_deg: float):
    """Spherical (radius, azimuth, elevation in degrees) to Cartesian camera position."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = radius * math.cos(el) * math.sin(az)
    y = radius * math.sin(el)
    z = radius * math.cos(el) * math.cos(az)
    return np.array([x, y, z], dtype=np.float32)

def normalize_geometry_to_unit_sphere(geom):
    """Center geometry and scale so its AABB fits inside the unit ball."""
    bbox = geom.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    geom.translate(-center)
    extent = bbox.get_extent()
    scale = 0.5 / float(np.max(extent)) if float(np.max(extent)) > 0 else 1.0
    geom.scale(scale, center=(0, 0, 0))
    return geom

def render_voxel_views(ply_path: str, out_dir: str, azimuths: List[float], 
                       width: int = 512, height: int = 512,
                       elevation: float = 0.0, radius: float = 1.8,
                       point_size: float = 4.0) -> List[Tuple[str, float]]:
    """
    Offscreen-render the voxel point cloud at each azimuth (degrees).

    Returns:
        List of (image_path, azimuth) pairs.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd is None or pcd.is_empty():
        raise RuntimeError(f"Failed to read point cloud from {ply_path}")
    
    pcd = normalize_geometry_to_unit_sphere(pcd)
    if not pcd.has_normals():
        pcd.estimate_normals()
    
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([1, 1, 1, 1])

    try:
        scene.scene.enable_sun_light(False)
        scene.scene.set_indirect_light_intensity(0.0)
    except Exception:
        pass
    
    # Unlit point cloud material.
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = float(point_size)
    
    geom_name = "pcd"
    scene.add_geometry(geom_name, pcd, mat)
    
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fov = 50.0
    FovType = o3d.visualization.rendering.Camera.FovType
    fov_type = FovType.Vertical
    near_plane, far_plane = 0.001, 100.0
    
    images = []
    for az in azimuths:
        cam_pos = spherical_to_cartesian(radius, az, elevation)
        renderer.scene.camera.look_at(target, cam_pos, up)
        renderer.scene.camera.set_projection(
            fov, float(width) / float(height), near_plane, far_plane, fov_type
        )
        
        img_o3d = renderer.render_to_image()
        img = Image.fromarray(np.asarray(img_o3d))
        out_path = os.path.join(out_dir, f"view_a{az:03.0f}.png")
        img.save(out_path)
        images.append((out_path, az))
    
    # Tear down renderer / scene geometry.
    try:
        scene.remove_geometry(geom_name)
    except Exception:
        pass
    del renderer
    
    return images

def create_score_composite(views: List[Tuple[str, float]], scores: List[float], 
                          out_dir: str, voxel_name: str):
    """Save a 2-row grid of views with per-view CLIP scores overlaid."""
    if len(views) == 0:
        return
    
    first_img = Image.open(views[0][0])
    img_width, img_height = first_img.size
    
    n_images = len(views)
    n_cols = (n_images + 1) // 2
    n_rows = 2
    
    composite_width = n_cols * img_width
    composite_height = n_rows * img_height
    composite = Image.new('RGB', (composite_width, composite_height), 'white')
    
    try:
        score_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    except:
        score_font = ImageFont.load_default()
    
    for i, ((img_path, az), score) in enumerate(zip(views, scores)):
        row = i // n_cols
        col = i % n_cols
        
        img = Image.open(img_path)
        paste_x = col * img_width
        paste_y = row * img_height
        composite.paste(img, (paste_x, paste_y))
        
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        is_best = (score == max(scores))
        bg_color = (0, 200, 0, 220) if is_best else (0, 100, 255, 200)
        bg = Image.new('RGBA', (250, 50), bg_color)
        overlay.paste(bg, (img.width - 260, img.height - 60))
        
        text = f"Az{az:.0f}°: {score:.3f}"
        if is_best:
            text += " [best]"
        bbox = overlay_draw.textbbox((0, 0), text, font=score_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = img.width - 260 + (250 - text_width) // 2
        text_y = img.height - 60 + (50 - text_height) // 2
        overlay_draw.text((text_x, text_y), text, fill='white', font=score_font)
        
        composite.paste(overlay, (paste_x, paste_y), overlay)
    
    out_path = os.path.join(out_dir, f"{voxel_name}_composite.png")
    composite.save(out_path)
    print(f"[INFO] Score composite saved to: {out_path}")

def evaluate_voxel_with_clip(occ: torch.Tensor, prompt: str, output_dir: str, 
                            voxel_name: str, step: int,
                            azimuths: List[float] = None,
                            device: str = "cuda") -> Tuple[float, float, List[float]]:
    """
    Render multi-view images, score against the text prompt with CLIP, return best azimuth.

    Returns:
        (best_azimuth_deg, best_score, scores_per_view).
    """
    if azimuths is None:
        azimuths = [0, 45, 90, 135, 180, 225, 270, 315]  # Default: eight azimuths, 45° apart.

    step_dir = os.path.join(output_dir, f"step_{step:03d}_{voxel_name}")
    os.makedirs(step_dir, exist_ok=True)
    
    ply_path = os.path.join(step_dir, f"{voxel_name}.ply")
    voxel_to_ply(occ, ply_path)

    print(f"[CLIP] Rendering {len(azimuths)} views for {voxel_name}...")
    views = render_voxel_views(ply_path, step_dir, azimuths)
    
    print(f"[CLIP] Evaluating with CLIP for prompt: '{prompt}'...")
    model, preprocess, tokenizer, backend = _load_clip()
    model = model.to(device).eval()
    
    image_paths = [v[0] for v in views]
    image_features = _clip_encode_images(model, preprocess, backend, image_paths, device=device)
    text_features = _clip_encode_text(model, tokenizer, backend, prompt, device=device)
    
    with torch.no_grad():
        scores = (image_features @ text_features.T).squeeze(1).cpu().numpy()

    best_idx = int(np.argmax(scores))
    best_azimuth = azimuths[best_idx]
    best_score = float(scores[best_idx])
    
    create_score_composite(views, scores.tolist(), step_dir, voxel_name)
    
    csv_path = os.path.join(step_dir, f"{voxel_name}_scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["azimuth_deg", "clip_score"])
        for az, score in zip(azimuths, scores):
            writer.writerow([f"{az:.0f}", f"{float(score):.6f}"])
    
    print(f"[CLIP] {voxel_name} best angle: {best_azimuth}° (score: {best_score:.4f})")
    
    return best_azimuth, best_score, scores.tolist()
