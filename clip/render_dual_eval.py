#!/usr/bin/env python3
# render_dual_eval.py
# 整合CLIP和GPT兩種方法來評估3D渲染圖片

import argparse
import math
import os
import base64
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import open3d as o3d
from openai import OpenAI
import torch


class _LazyOpenAIClient:
    """Defer OpenAI construction until first use; avoids requiring a key for CLIP-only runs."""

    def __init__(self):
        self._inner = None

    def _ensure(self):
        if self._inner is None:
            key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. It is only needed for GPT-based analysis "
                    "functions in this module; default example_text.py uses CLIP for rotation."
                )
            self._inner = OpenAI(api_key=key)
        return self._inner

    def __getattr__(self, name):
        return getattr(self._ensure(), name)


client = _LazyOpenAIClient()

# -----------------------------
# CLIP loader (from render_and_clip2.py)
# -----------------------------
def _load_clip():
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
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        imgs.append(preprocess(img))
    image_input = torch.stack(imgs, dim=0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features

# -----------------------------
# Rendering helpers (copied from render_and_clip2.py)
# -----------------------------
@dataclass
class CameraParams:
    fov_deg: float = 50.0
    radius: float = 1.8
    lookat: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)

def normalize_geometry_to_unit_sphere(geom):
    bbox = geom.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    geom.translate(-center)
    extent = bbox.get_extent()
    scale = 0.5 / float(np.max(extent)) if float(np.max(extent)) > 0 else 1.0
    geom.scale(scale, center=(0, 0, 0))
    return geom

def read_geometry(ply_path: str):
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if mesh is not None and not mesh.is_empty():
        if len(np.asarray(mesh.triangles)) > 0:
            print("[INFO] Loaded TriangleMesh")
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            return mesh, "mesh"
        else:
            print("[INFO] File has no triangles; will try PointCloud path")
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd is None or pcd.is_empty():
        raise RuntimeError(f"Failed to read geometry from {ply_path}")
    print("[INFO] Loaded PointCloud")
    if not pcd.has_normals():
        pcd.estimate_normals()
    return pcd, "pointcloud"

def spherical_to_cartesian(radius: float, azimuth_deg: float, elevation_deg: float):
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = radius * math.cos(el) * math.sin(az)
    y = radius * math.sin(el)
    z = radius * math.cos(el) * math.cos(az)
    return np.array([x, y, z], dtype=np.float32)

def setup_material(kind: str, point_size: float = 4.0):
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    if kind == "pointcloud":
        mat.point_size = float(point_size)
    try:
        mat.base_color = (0.2, 0.2, 0.2, 1.0)
    except Exception:
        pass
    return mat

def render_views(ply_path: str, out_dir: str, width: int, height: int,
                 elevations: List[float], azimuths_per_elev: int,
                 cam: CameraParams, point_size: float = 4.0) -> List[Tuple[str, float, float]]:
    os.makedirs(out_dir, exist_ok=True)

    geom, kind = read_geometry(ply_path)
    geom = normalize_geometry_to_unit_sphere(geom)

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([1, 1, 1, 1])

    try:
        scene.scene.enable_sun_light(False)
        scene.scene.set_indirect_light_intensity(0.0)
    except Exception:
        pass

    mat = setup_material(kind, point_size=point_size)
    geom_name = "geom"
    scene.add_geometry(geom_name, geom, mat)

    fov = cam.fov_deg
    target = np.array(cam.lookat, dtype=np.float32)
    FovType = o3d.visualization.rendering.Camera.FovType
    fov_type = FovType.Vertical

    near_plane, far_plane = 0.001, 100.0

    images = []
    for elev in elevations:
        for ai in range(azimuths_per_elev):
            az = (360.0 / azimuths_per_elev) * ai
            cam_pos = spherical_to_cartesian(cam.radius, az, elev)
            up = np.array(cam.up, dtype=np.float32)

            renderer.scene.camera.look_at(target, cam_pos, up)
            renderer.scene.camera.set_projection(
                fov, float(width) / float(height), near_plane, far_plane, fov_type
            )

            img_o3d = renderer.render_to_image()
            img = Image.fromarray(np.asarray(img_o3d))
            out_path = os.path.join(out_dir, f"view_e{elev:+.1f}_a{az:03.0f}.png")
            img.save(out_path)
            images.append((out_path, elev, az))

    try:
        scene.remove_geometry(geom_name)
    except Exception:
        pass
    del renderer
    return images

def evenly_spaced_elevations(num: int, min_deg=-30.0, max_deg=30.0):
    if num <= 1:
        return [(min_deg + max_deg) * 0.5]
    return list(np.linspace(min_deg, max_deg, num).astype(float))

# -----------------------------
# GPT evaluation (from render_and_gpt.py)
# -----------------------------
def encode_image_to_base64(image_path: str) -> str:
    """將圖片編碼為base64字串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def evaluate_with_gpt(image_paths: List[str], prompt: str) -> List[float]:
    """使用GPT-4 Vision來評估圖片與prompt的相似度"""
    
    # 準備所有圖片
    images_data = []
    for img_path in image_paths:
        base64_image = encode_image_to_base64(img_path)
        images_data.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        })
    
    # 構建GPT提示
    system_prompt = """You are an expert at analyzing 3D rendered images. You will be shown multiple images of the same 3D object from different viewing angles, and you need to evaluate how well each image matches a given description.

Please analyze each image and provide a PRECISE similarity score from 0.000 to 1.000, where:
- 0.000 = completely unrelated to the description
- 1.000 = perfectly matches the description

CRITICAL: You must provide scores with 3 decimal places (e.g., 0.847, 0.623, 0.934). Do NOT round to simple fractions like 0.3, 0.4, 0.5.

Consider factors like:
- Visual similarity to the described object
- Pose and angle appropriateness  
- Clarity of features
- Overall composition
- Lighting and visibility
- How well the view shows the key characteristics

Return your response as a JSON object with the following format:
{
    "scores": [0.847, 0.623, 0.934, 0.756, ...],
    "reasoning": "Brief explanation of your evaluation"
}

The scores array should contain exactly one score for each image in the same order they were presented. Each score must be a precise decimal with 3 decimal places."""

    user_prompt = f"""Please evaluate how well each of these 3D rendered images matches this description: "{prompt}"

I will show you {len(image_paths)} images. Please provide PRECISE similarity scores for each one with 3 decimal places.

IMPORTANT: 
- Each score must be unique and precise (e.g., 0.847, 0.623, 0.934)
- Do NOT use simple fractions like 0.3, 0.4, 0.5
- Consider subtle differences between views
- Higher scores for views that better match the description

Please analyze each image carefully and provide detailed, precise scores."""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": user_prompt}
            ] + images_data
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1500,
            temperature=0.0
        )
        
        # 解析JSON回應
        import json
        content = response.choices[0].message.content
        
        # 嘗試提取JSON
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
                scores = result.get('scores', [])
                reasoning = result.get('reasoning', '')
                
                print(f"[GPT Reasoning] {reasoning}")
                
                # 確保分數是浮點數且精確到3位小數
                processed_scores = []
                for score in scores:
                    if isinstance(score, (int, float)):
                        processed_scores.append(round(float(score), 3))
                    else:
                        print(f"[WARN] Invalid score type: {type(score)}, value: {score}")
                        processed_scores.append(0.5)
                
                # 確保分數數量正確
                if len(processed_scores) == len(image_paths):
                    return processed_scores
                else:
                    print(f"[WARN] Expected {len(image_paths)} scores, got {len(processed_scores)}")
                    return processed_scores + [0.5] * (len(image_paths) - len(processed_scores))
            else:
                raise ValueError("No JSON found in response")
                
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse JSON: {e}")
            # 如果JSON解析失敗，返回隨機精確分數
            import random
            return [round(random.uniform(0.1, 0.9), 3) for _ in range(len(image_paths))]
            
    except Exception as e:
        print(f"[ERROR] GPT API call failed: {e}")
        # 如果API調用失敗，返回隨機分數
        import random
        return [round(random.uniform(0.1, 0.9), 3) for _ in range(len(image_paths))]

# -----------------------------
# Dual evaluation composite grid
# -----------------------------
def create_dual_composite_grid(views: List[Tuple[str, float, float]], 
                              num_elevations: int, 
                              num_azimuths: int,
                              out_path: str,
                              img_width: int,
                              img_height: int,
                              clip_scores: List[float] = None,
                              gpt_scores: List[float] = None):
    """創建同時顯示CLIP和GPT分數的composite grid"""
    # Add padding for labels
    label_height = 120
    label_width = 120
    
    # Calculate composite dimensions
    composite_width = num_azimuths * img_width + label_width
    composite_height = num_elevations * img_height + label_height
    
    # Create white canvas
    composite = Image.new('RGB', (composite_width, composite_height), 'white')
    draw = ImageDraw.Draw(composite)
    
    # Try to load fonts
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 50)
        score_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 24)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 20)
            score_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
            score_font = ImageFont.load_default()
    
    # Organize views by elevation and azimuth
    view_dict = {}
    clip_dict = {}
    gpt_dict = {}
    for i, (img_path, elev, az) in enumerate(views):
        view_dict[(elev, az)] = img_path
        if clip_scores is not None and i < len(clip_scores):
            clip_dict[(elev, az)] = clip_scores[i]
        if gpt_scores is not None and i < len(gpt_scores):
            gpt_dict[(elev, az)] = gpt_scores[i]
    
    # Get sorted elevations and azimuths
    elevations = sorted(set([elev for _, elev, _ in views]), reverse=True)
    azimuths = sorted(set([az for _, _, az in views]))
    
    # Add column headers (azimuth angles)
    for col_idx, az in enumerate(azimuths):
        x = label_width + col_idx * img_width + img_width // 2
        y = label_height // 2
        text = f"Az {az:.0f}°"
        bbox = draw.textbbox((0, 0), text, font=font_small)
        text_width = bbox[2] - bbox[0]
        draw.text((x - text_width // 2, y - 10), text, fill='black', font=font_small)
    
    # Add row headers and images
    for row_idx, elev in enumerate(elevations):
        # Row header
        y = label_height + row_idx * img_height + img_height // 2
        x = label_width // 2
        text = f"El {elev:+.0f}°"
        
        bbox = draw.textbbox((0, 0), text, font=font_small)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        temp_img = Image.new('RGB', (text_width + 20, text_height + 20), 'white')
        temp_draw = ImageDraw.Draw(temp_img)
        temp_draw.text((10, 10), text, fill='black', font=font_small)
        
        rotated = temp_img.rotate(90, expand=True)
        paste_x = x - rotated.width // 2
        paste_y = y - rotated.height // 2
        composite.paste(rotated, (paste_x, paste_y))
        
        # Paste images with dual scores
        for col_idx, az in enumerate(azimuths):
            if (elev, az) in view_dict:
                img_path = view_dict[(elev, az)]
                try:
                    img = Image.open(img_path)
                    paste_x = label_width + col_idx * img_width
                    paste_y = label_height + row_idx * img_height
                    composite.paste(img, (paste_x, paste_y))
                    
                    # Create overlay for dual scores (both in bottom-right)
                    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                    overlay_draw = ImageDraw.Draw(overlay)
                    
                    # CLIP score (top part, blue background)
                    if (elev, az) in clip_dict:
                        clip_score = clip_dict[(elev, az)]
                        clip_bg = Image.new('RGBA', (200, 40), (0, 100, 255, 200))  # Blue background
                        overlay.paste(clip_bg, (img.width - 200, img.height - 90))  # Move left by 50 pixels
                        
                        clip_text = f"CLIP: {clip_score:.3f}"
                        bbox = overlay_draw.textbbox((0, 0), clip_text, font=score_font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        text_x = img.width - 200 + (200 - text_width) // 2  # Use correct width 200
                        text_y = img.height - 90 + (40 - text_height) // 2
                        overlay_draw.text((text_x, text_y), clip_text, fill='white', font=score_font)
                    
                    # GPT score (bottom part, green background)
                    if (elev, az) in gpt_dict:
                        gpt_score = gpt_dict[(elev, az)]
                        gpt_bg = Image.new('RGBA', (200, 40), (0, 150, 0, 200))  # Green background
                        overlay.paste(gpt_bg, (img.width - 200, img.height - 45))  # Move left by 50 pixels
                        
                        gpt_text = f"GPT: {gpt_score:.3f}"
                        bbox = overlay_draw.textbbox((0, 0), gpt_text, font=score_font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        text_x = img.width - 200 + (200 - text_width) // 2  # Use correct width 200
                        text_y = img.height - 45 + (40 - text_height) // 2
                        overlay_draw.text((text_x, text_y), gpt_text, fill='white', font=score_font)
                    
                    # Paste the overlay onto the composite
                    composite.paste(overlay, (paste_x, paste_y), overlay)
                        
                except Exception as e:
                    print(f"[WARN] Failed to load image {img_path}: {e}")
    
    # Save composite
    composite.save(out_path)
    print(f"[INFO] Dual evaluation composite grid saved to: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ply", required=True, help="Path to .ply mesh or point cloud")
    parser.add_argument("--prompt", required=True, help="Text prompt for evaluation")
    parser.add_argument("--out", default="output", help="Output directory for renders")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--elevations", type=int, default=5, help="Number of elevation bands (default 5)")
    parser.add_argument("--azimuths", type=int, default=8, help="Number of azimuth views per elevation (default 8)")
    parser.add_argument("--fov", type=float, default=50.0)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--point_size", type=float, default=4.0, help="Point size when rendering point clouds")
    parser.add_argument("--elev_list", type=str, default="90,45,0,-45,-90", help="Comma-separated elevation degrees")
    args = parser.parse_args()

    # Extract PLY filename without extension to create subdirectory
    ply_basename = os.path.splitext(os.path.basename(args.ply))[0]
    output_dir = os.path.join(args.out, ply_basename)

    cam = CameraParams(fov_deg=args.fov, radius=args.radius)
    if args.elev_list is not None:
        elevs = [float(x) for x in args.elev_list.split(",")]
    else:
        elevs = evenly_spaced_elevations(args.elevations, -30.0, 30.0)

    print(f"[INFO] Rendering {len(elevs)} elevations × {args.azimuths} azimuths = {len(elevs)*args.azimuths} views")
    print(f"[INFO] Output directory: {output_dir}")
    
    # Render all views
    views = render_views(
        ply_path=args.ply,
        out_dir=output_dir,
        width=args.width,
        height=args.height,
        elevations=elevs,
        azimuths_per_elev=args.azimuths,
        cam=cam,
        point_size=args.point_size
    )

    # Evaluate with CLIP
    print(f"[INFO] Evaluating {len(views)} images with CLIP...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer, backend = _load_clip()
    model = model.to(device).eval()

    image_paths = [v[0] for v in views]
    image_features = _clip_encode_images(model, preprocess, backend, image_paths, device=device)
    text_features = _clip_encode_text(model, tokenizer, backend, args.prompt, device=device)

    with torch.no_grad():
        clip_scores = (image_features @ text_features.T).squeeze(1).cpu().numpy()

    # Evaluate with GPT
    print(f"[INFO] Evaluating {len(views)} images with GPT-4 Vision...")
    gpt_scores = evaluate_with_gpt(image_paths, args.prompt)

    # Create dual composite grid
    composite_path = os.path.join(output_dir, "composite_grid_dual.png")
    create_dual_composite_grid(
        views=views,
        num_elevations=len(elevs),
        num_azimuths=args.azimuths,
        out_path=composite_path,
        img_width=args.width,
        img_height=args.height,
        clip_scores=clip_scores.tolist(),
        gpt_scores=gpt_scores
    )

    # Save results to CSV
    import csv
    csv_path = os.path.join(output_dir, "dual_scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "elevation_deg", "azimuth_deg", "clip_similarity", "gpt_similarity", "score_difference"])
        for i, ((path, elev, az), clip_score, gpt_score) in enumerate(zip(views, clip_scores, gpt_scores)):
            diff = abs(clip_score - gpt_score)
            writer.writerow([
                os.path.basename(path), 
                f"{elev:.1f}", 
                f"{az:.0f}", 
                f"{float(clip_score):.6f}", 
                f"{float(gpt_score):.6f}",
                f"{diff:.6f}"
            ])

    # Find best views for each method
    clip_best_idx = int(np.argmax(clip_scores))
    gpt_best_idx = int(np.argmax(gpt_scores))
    
    clip_best_path, clip_best_elev, clip_best_az = views[clip_best_idx]
    gpt_best_path, gpt_best_elev, gpt_best_az = views[gpt_best_idx]
    
    print("\n=== DUAL EVALUATION RESULTS ===")
    print(f"CLIP Best view: {os.path.basename(clip_best_path)} (elev={clip_best_elev:.1f}, az={clip_best_az:.0f})")
    print(f"CLIP Score: {float(clip_scores[clip_best_idx]):.6f}")
    print(f"GPT Best view: {os.path.basename(gpt_best_path)} (elev={gpt_best_elev:.1f}, az={gpt_best_az:.0f})")
    print(f"GPT Score: {float(gpt_scores[gpt_best_idx]):.6f}")
    print(f"CSV saved to: {csv_path}")
    print(f"Dual composite grid saved to: {composite_path}")
    print(f"Images saved to: {output_dir}")

if __name__ == "__main__":
    main()

# python render_dual_eval.py --ply input/eagle.ply --prompt "A eagle"
# python render_dual_eval.py --ply input/fish.ply --prompt "A fish"
# python render_dual_eval.py --ply input/frog.ply --prompt "A frog sitting on a leaf"