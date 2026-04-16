#!/usr/bin/env python3
# render_single_clip.py — Four azimuths (0/90/180/270°) + CLIP scoring for single-object eval.

import os
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
import base64
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trellis.utils import render_utils

# Import OpenAI client from render_dual_eval.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_dual_eval import client

# -----------------------------
# CLIP loader
# -----------------------------
def _load_clip():
    """Load CLIP (open_clip preferred; OpenAI clip fallback)."""
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
    """L2-normalized text embedding for ``prompt``."""
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


def _clip_encode_images(model, preprocess, backend, images: List[np.ndarray], device="cpu"):
    """L2-normalized image embeddings from uint8/RGB numpy frames."""
    imgs = []
    for img_array in images:
        # Convert numpy array to PIL Image
        img = Image.fromarray(img_array)
        imgs.append(preprocess(img))
    image_input = torch.stack(imgs, dim=0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features


# -----------------------------
# Rendering functions
# -----------------------------
def render_four_views(gaussian_obj, resolution=512, bg_color=(255, 255, 255), r=2, fov=40):
    """
    Render four turntable azimuths: 0°, 90°, 180°, 270°.

    Args:
        gaussian_obj: ``output['gaussian'][0]`` from the Trellis pipeline.
        resolution: Output resolution in pixels.
        bg_color: Background RGB (white is best for CLIP).
        r: Camera radius.
        fov: Vertical field of view (degrees).

    Returns:
        List of ``(rgb_uint8_array, angle_degrees)``.
    """
    angles_degrees = [0, 90, 180, 270]
    angles_radians = [np.deg2rad(angle) for angle in angles_degrees]
    
    rendered_views = []
    
    print(f"[INFO] Rendering 4 views at angles: {angles_degrees}")
    
    for angle_deg, yaw in zip(angles_degrees, angles_radians):
        pitch = 0.0  # Elevation: level camera.

        extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            yaw, pitch, r, fov
        )

        result = render_utils.render_frames(
            gaussian_obj,
            [extrinsics], 
            [intrinsics],
            {'resolution': resolution, 'bg_color': bg_color},
            verbose=False
        )
        
        img = result['color'][0]  # RGB frame
        rendered_views.append((img, angle_deg))
        print(f"  [INFO] Rendered view at {angle_deg}°")
    
    return rendered_views


# -----------------------------
# Visualization
# -----------------------------
def create_composite_image(views: List[Tuple[np.ndarray, float]], 
                          clip_scores: List[float],
                          prompt: str,
                          output_path: str):
    """
    Save a 2×2 grid of views with per-view CLIP scores and prompt title.

    Args:
        views: ``(image_array, angle_degrees)`` pairs.
        clip_scores: Cosine similarity per view.
        prompt: Conditioning text (shown in title).
        output_path: Output PNG path.
    """
    # Get image dimensions
    img_height, img_width = views[0][0].shape[:2]
    
    # Layout: 2x2 grid
    padding = 20
    title_height = 80  # Reduced from 100 to 80
    
    composite_width = 2 * img_width + 3 * padding
    composite_height = 2 * img_height + 3 * padding + title_height
    
    # Create white canvas
    composite = Image.new('RGB', (composite_width, composite_height), 'white')
    draw = ImageDraw.Draw(composite)
    
    # Load fonts
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)  # Reduced from 36 to 28
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        score_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except:
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 28)  # Reduced from 36 to 28
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 32)
            score_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 28)
        except:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            score_font = ImageFont.load_default()
    
    # Draw title with text wrapping for long prompts
    title_text = f"CLIP Evaluation: \"{prompt}\""
    
    # Check if text is too long and needs wrapping
    bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_width = bbox[2] - bbox[0]
    max_width = composite_width - 40  # Leave 20px margin on each side
    
    if title_width > max_width:
        # Split text into lines
        words = title_text.split()
        lines = []
        current_line = "CLIP Evaluation:"
        
        for word in words[2:]:  # Skip "CLIP Evaluation:"
            test_line = current_line + " " + word
            bbox = draw.textbbox((0, 0), test_line, font=title_font)
            if bbox[2] - bbox[0] <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = f'"{word}'
        
        if current_line:
            lines.append(current_line)
        
        # Draw each line
        line_height = 35
        start_y = 15
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=title_font)
            line_width = bbox[2] - bbox[0]
            line_x = (composite_width - line_width) // 2
            draw.text((line_x, start_y + i * line_height), line, fill='black', font=title_font)
    else:
        # Single line title
        title_x = (composite_width - title_width) // 2
        draw.text((title_x, 25), title_text, fill='black', font=title_font)
    
    # Find best view
    best_idx = np.argmax(clip_scores)
    
    # Paste images in 2x2 grid
    positions = [
        (0, 0),  # Top-left (0°)
        (1, 0),  # Top-right (90°)
        (0, 1),  # Bottom-left (180°)
        (1, 1),  # Bottom-right (270°)
    ]
    
    for i, ((img_array, angle_deg), clip_score) in enumerate(zip(views, clip_scores)):
        col, row = positions[i]
        
        # Calculate paste position
        paste_x = padding + col * (img_width + padding)
        paste_y = title_height + padding + row * (img_height + padding)
        
        # Convert to PIL Image
        img = Image.fromarray(img_array)
        
        # Create overlay for angle label and score
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Angle label (top-left corner)
        label_text = f"{angle_deg}°"
        label_bg = Image.new('RGBA', (120, 50), (50, 50, 50, 220))
        overlay.paste(label_bg, (10, 10))
        
        bbox = overlay_draw.textbbox((0, 0), label_text, font=label_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = 10 + (120 - text_width) // 2
        text_y = 10 + (50 - text_height) // 2
        overlay_draw.text((text_x, text_y), label_text, fill='white', font=label_font)
        
        # CLIP score (bottom, full width)
        # Use gradient color based on score
        is_best = (i == best_idx)
        if is_best:
            # Gold/yellow for best view
            bg_color = (255, 215, 0, 230)  # Gold
        else:
            # Blue gradient based on score
            intensity = int(100 + 155 * clip_score)
            bg_color = (0, intensity, 255, 200)
        
        score_bg = Image.new('RGBA', (img.width, 60), bg_color)
        overlay.paste(score_bg, (0, img.height - 60))
        
        score_text = f"CLIP: {clip_score:.4f}"
        if is_best:
            score_text += " * BEST"
        
        bbox = overlay_draw.textbbox((0, 0), score_text, font=score_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = (img.width - text_width) // 2
        text_y = img.height - 60 + (60 - text_height) // 2
        overlay_draw.text((text_x, text_y), score_text, fill='white', font=score_font)
        
        # Paste image and overlay
        composite.paste(img, (paste_x, paste_y))
        composite.paste(overlay, (paste_x, paste_y), overlay)
    
    # Save composite
    composite.save(output_path)
    print(f"[INFO] Composite image saved to: {output_path}")


# -----------------------------
# Image similarity comparison function
# -----------------------------
def compare_image_similarity_clip(single_image_path, four_images_dir, output_dir):
    """
    CLIP cosine similarity: one query image vs four fixed azimuths (0/90/180/270°).

    Args:
        single_image_path: Path to the query PNG.
        four_images_dir: Directory containing ``view_000.png`` … ``view_270.png``.
        output_dir: Unused (kept for API compatibility).

    Returns:
        Dict with best angle, scores, and paths; ``None`` on failure.
    """
    try:
        four_images = [
            os.path.join(four_images_dir, "view_000.png"),
            os.path.join(four_images_dir, "view_090.png"),
            os.path.join(four_images_dir, "view_180.png"),
            os.path.join(four_images_dir, "view_270.png")
        ]

        if not os.path.exists(single_image_path):
            print(f"[ERROR] Single image not found: {single_image_path}")
            return None
        
        for img_path in four_images:
            if not os.path.exists(img_path):
                print(f"[ERROR] Image not found: {img_path}")
                return None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess, tokenizer, backend = _load_clip()
        model = model.to(device).eval()

        single_img = Image.open(single_image_path).convert("RGB")
        single_input = preprocess(single_img).unsqueeze(0).to(device)
        
        four_imgs = []
        for img_path in four_images:
            img = Image.open(img_path).convert("RGB")
            four_imgs.append(preprocess(img))
        four_input = torch.stack(four_imgs, dim=0).to(device)

        with torch.no_grad():
            single_features = model.encode_image(single_input)
            single_features = single_features / single_features.norm(dim=-1, keepdim=True)
            
            four_features = model.encode_image(four_input)
            four_features = four_features / four_features.norm(dim=-1, keepdim=True)
            
            similarities = (single_features @ four_features.T).squeeze(0).cpu().numpy()

        best_idx = np.argmax(similarities)
        angles = [0, 90, 180, 270]
        best_angle = angles[best_idx]
        best_similarity = similarities[best_idx]
        best_path = four_images[best_idx]
        
        print(f"\n[IMAGE SIMILARITY] Best match: {best_angle}° with similarity {best_similarity:.6f}")
        
        return {
            'best_angle': best_angle,
            'best_similarity': best_similarity,
            'best_path': best_path,
            'all_similarities': similarities.tolist(),
            'all_angles': angles
        }
        
    except Exception as e:
        print(f"[ERROR] Image similarity comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_obj2_best_with_obj1_all_28(single_image_path, obj1_all_rotations_dir, output_dir):
    """
    CLIP similarity: one query PNG vs all views under ``rotation_*/view_*.png`` (7 poses × 4 azimuths).

    Args:
        single_image_path: Reference view path.
        obj1_all_rotations_dir: Parent of ``rotation_<name>/`` (legacy name; may hold obj2 variants).
        output_dir: Unused; reserved for API compatibility.

    Returns:
        Best match, per-view scores, and per-pose averages; ``None`` on failure.
    """
    try:
        rotation_names = ['x_1', 'x_2', 'x_3', 'y_1', 'y_2', 'y_3', 'z']
        all_images = []
        all_angles = []
        all_rotations = []
        
        for rotation in rotation_names:
            rotation_dir = os.path.join(obj1_all_rotations_dir, f"rotation_{rotation}")
            for angle in [0, 90, 180, 270]:
                img_path = os.path.join(rotation_dir, f"view_{angle:03d}.png")
                if os.path.exists(img_path):
                    all_images.append(img_path)
                    all_angles.append(angle)
                    all_rotations.append(rotation)
                else:
                    print(f"[WARNING] Image not found: {img_path}")

        if not os.path.exists(single_image_path):
            print(f"[ERROR] Single image not found: {single_image_path}")
            return None
        
        if len(all_images) == 0:
            print(f"[ERROR] No images found in {obj1_all_rotations_dir}")
            return None
        
        print(f"[INFO] Found {len(all_images)} images to compare")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess, tokenizer, backend = _load_clip()
        model = model.to(device).eval()

        single_img = Image.open(single_image_path).convert("RGB")
        single_input = preprocess(single_img).unsqueeze(0).to(device)
        
        all_imgs = []
        for img_path in all_images:
            img = Image.open(img_path).convert("RGB")
            all_imgs.append(preprocess(img))
        all_input = torch.stack(all_imgs, dim=0).to(device)

        with torch.no_grad():
            single_features = model.encode_image(single_input)
            single_features = single_features / single_features.norm(dim=-1, keepdim=True)
            
            all_features = model.encode_image(all_input)
            all_features = all_features / all_features.norm(dim=-1, keepdim=True)
            
            similarities = (single_features @ all_features.T).squeeze(0).cpu().numpy()

        best_idx = np.argmax(similarities)
        best_angle = all_angles[best_idx]
        best_rotation = all_rotations[best_idx]
        best_similarity = similarities[best_idx]
        best_path = all_images[best_idx]
        
        print(f"\n[OBJ2-TO-OBJ1-ALL-28] Best match: {best_rotation} rotation, {best_angle}° with similarity {best_similarity:.6f}")

        rotation_stats = {}
        for i, (sim, rot, angle) in enumerate(zip(similarities, all_rotations, all_angles)):
            if rot not in rotation_stats:
                rotation_stats[rot] = {'similarities': [], 'angles': []}
            rotation_stats[rot]['similarities'].append(sim)
            rotation_stats[rot]['angles'].append(angle)

        rotation_avg_similarities = {}
        for rot, stats in rotation_stats.items():
            rotation_avg_similarities[rot] = np.mean(stats['similarities'])

        best_rotation_avg = max(rotation_avg_similarities.items(), key=lambda x: x[1])
        
        return {
            'best_angle': best_angle,
            'best_rotation': best_rotation,
            'best_similarity': best_similarity,
            'best_path': best_path,
            'all_similarities': similarities.tolist(),
            'all_angles': all_angles,
            'all_rotations': all_rotations,
            'rotation_avg_similarities': rotation_avg_similarities,
            'best_rotation_avg': best_rotation_avg
        }
        
    except Exception as e:
        print(f"[ERROR] Obj2-to-Obj1-all-28 similarity comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# -----------------------------
# Main evaluation function
# -----------------------------
def evaluate_single_with_clip(gaussian_obj, prompt: str, output_dir: str, 
                              resolution=512, bg_color=(255, 255, 255)):
    """
    End-to-end single-object CLIP eval: four renders, cosine scores vs prompt, composite + CSV.

    Args:
        gaussian_obj: ``output['gaussian'][0]`` from Trellis.
        prompt: Text used for CLIP alignment.
        output_dir: Directory for PNGs, composite, and ``clip_scores.csv``.
        resolution: Render resolution.
        bg_color: Background RGB tuple.

    Returns:
        ``views``, ``scores``, ``best_view`` tuple ``(image, angle_deg, score)``.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Render 4 views
    print("\n" + "="*60)
    print("STEP 1: Rendering 4 views")
    print("="*60)
    views = render_four_views(gaussian_obj, resolution, bg_color)
    
    # Save individual views
    for img, angle in views:
        img_path = os.path.join(output_dir, f"view_{int(angle):03d}.png")
        Image.fromarray(img).save(img_path)
        print(f"  [INFO] Saved view: {img_path}")
    
    # Step 2: Evaluate with CLIP
    print("\n" + "="*60)
    print("STEP 2: Evaluating with CLIP")
    print("="*60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    model, preprocess, tokenizer, backend = _load_clip()
    model = model.to(device).eval()
    
    # Encode images and text
    images = [img for img, _ in views]
    image_features = _clip_encode_images(model, preprocess, backend, images, device=device)
    text_features = _clip_encode_text(model, tokenizer, backend, prompt, device=device)
    
    # Calculate similarity scores
    with torch.no_grad():
        clip_scores = (image_features @ text_features.T).squeeze(1).cpu().numpy()
    
    # Print results
    print(f"\nPrompt: \"{prompt}\"")
    print("\nCLIP Scores:")
    for (_, angle), score in zip(views, clip_scores):
        print(f"  {int(angle):3d}°: {score:.6f}")
    
    # Find best view
    best_idx = np.argmax(clip_scores)
    best_img, best_angle = views[best_idx]
    best_score = clip_scores[best_idx]
    
    print(f"\n[INFO] Best view (CLIP): azimuth={int(best_angle)}°, score={best_score:.6f}")
    
    # Step 3: Create composite image
    print("\n" + "="*60)
    print("STEP 3: Creating composite image")
    print("="*60)
    composite_path = os.path.join(output_dir, "clip_evaluation.png")
    create_composite_image(views, clip_scores.tolist(), prompt, composite_path)
    
    # Save scores to CSV
    import csv
    csv_path = os.path.join(output_dir, "clip_scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_degrees", "clip_score"])
        for (_, angle), score in zip(views, clip_scores):
            writer.writerow([int(angle), f"{float(score):.6f}"])
    print(f"[INFO] Scores saved to: {csv_path}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    
    return {
        'views': views,
        'scores': clip_scores.tolist(),
        'best_view': (best_img, best_angle, best_score)
    }


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    """
    Example (e.g. from ``example_text.py`` single-prompt branch after ``pipeline.run``):

        from clip.render_single_clip import evaluate_single_with_clip

        results = evaluate_single_with_clip(
            gaussian_obj=output["gaussian"][0],
            prompt="A small tropical bird with wide wings",
            output_dir=os.path.join(output_dir, "clip_eval"),
            resolution=512,
            bg_color=(255, 255, 255),
        )
    """
    print(__doc__)


# -----------------------------
# Geometric Analysis Function
# -----------------------------

def analyze_geometric_compatibility_from_images(image1_path: str, image2_path: str, prompt1: str, prompt2: str):
    """
    GPT-4V geometric pairing: two 4-view CLIP composite PNGs → best azimuth per object.

    Args:
        image1_path, image2_path: Paths to 2×2 composite renders.
        prompt1, prompt2: Captions (sent in the API prompt).

    Returns:
        ``angle1``, ``angle2``, ``rotation_needed``, ``rotation_step``; or ``None``.
    """
    print("\n" + "="*70)
    print("GEOMETRIC COMPATIBILITY ANALYSIS")
    print("="*70)

    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    def encode_image_to_base64(image):
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    
    img1_base64 = encode_image_to_base64(img1)
    img2_base64 = encode_image_to_base64(img2)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Analyze these two images showing 3D objects from 4 different angles (0°, 90°, 180°, 270°).

Image 1: "{prompt1}" (4 angles: 0°, 90°, 180°, 270°)
Image 2: "{prompt2}" (4 angles: 0°, 90°, 180°, 270°)

Ignore CLIP scores completely. Focus purely on geometric shape compatibility:
1. Which angle from image 1 and which angle from image 2 would be easiest to connect/merge geometrically?
2. Which combination would create the most seamless 3D object when blended?
3. Consider shape contours, edges, and overall geometry.

Please respond ONLY in this exact format:
prompt1: [angle_number]
prompt2: [angle_number]

For example:
prompt1: 0
prompt2: 180"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img1_base64}"
                            }
                        },
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/png;base64,{img2_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=50
        )
        
        result_text = response.choices[0].message.content.strip()
        print(f"GPT-4V Analysis Result: {result_text}")

        lines = result_text.strip().split('\n')
        angle1 = None
        angle2 = None
        
        for line in lines:
            if 'prompt1:' in line.lower():
                try:
                    angle1 = int(line.split(':')[1].strip())
                except:
                    pass
            elif 'prompt2:' in line.lower():
                try:
                    angle2 = int(line.split(':')[1].strip())
                except:
                    pass
        
        if angle1 is not None and angle2 is not None:
            rotation_needed = angle1 - angle2
            if rotation_needed < 0:
                rotation_needed += 360
            
            rotation_step = int(rotation_needed / 90)
            
            print(f"\n[INFO] Geometric analysis (GPT-4V) summary:")
            print(f"       Best azimuth for '{prompt1}': {angle1}°")
            print(f"       Best azimuth for '{prompt2}': {angle2}°")
            print(f"       Rotation delta: {rotation_needed}°")
            print(f"       Rotation steps (90°): {rotation_step}")
            
            return {
                'angle1': angle1,
                'angle2': angle2,
                'rotation_needed': rotation_needed,
                'rotation_step': rotation_step
            }
        else:
            print(f"[ERROR] Failed to parse GPT-4V response: {result_text}")
            return None
            
    except Exception as e:
        print(f"[ERROR] GPT-4V API error: {e}")
        return None


def analyze_geometric_compatibility_4x7_images(image1_path: str, image2_path: str, prompt1: str, prompt2: str):
    """
    GPT-4V on two 4×7 CLIP composites; return dict shaped like ``compare_obj2_best_with_obj1_all_28``.

    Args:
        image1_path, image2_path: Full-grid composite PNG paths.
        prompt1, prompt2: Object captions for the vision prompt.

    Returns:
        Parsed fields plus placeholder CLIP-like keys; ``None`` if parsing fails.
    """
    print("\n" + "="*70)
    print("GEOMETRIC COMPATIBILITY ANALYSIS (4×7 IMAGES)")
    print("="*70)

    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    def encode_image_to_base64(image):
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    
    img1_base64 = encode_image_to_base64(img1)
    img2_base64 = encode_image_to_base64(img2)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Analyze these two composite images showing 3D objects.

Image 1: "{prompt1}" (4 angles only)
- Rows: 4 angles (0°, 90°, 180°, 270°)
- Single object with 4 different viewing angles

Image 2: "{prompt2}" (7 rotations × 4 angles = 28 total views)
- Rows: 4 angles (0°, 90°, 180°, 270°)
- Columns: 7 rotations (x_1, x_2, x_3, y_1, y_2, y_3, z)

Ignore CLIP scores and texture completely. Focus purely on geometric shape compatibility:
1. Which angle from image 1 and which specific rotation and angle from image 2 would be easiest to connect/merge geometrically?
2. Which combination would create the most seamless 3D object when blended?
3. Consider shape contours, edges, and overall geometry only.

Please respond ONLY in this exact format:
obj1_angle: [angle_number]
obj2_rotation: [rotation_name]
obj2_angle: [angle_number]

For example:
obj1_angle: 0
obj2_rotation: y_2
obj2_angle: 180"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img1_base64}"
                            }
                        },
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/png;base64,{img2_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100
        )
        
        result_text = response.choices[0].message.content.strip()
        print(f"GPT-4V Analysis Result: {result_text}")

        lines = result_text.strip().split('\n')
        obj1_angle = None
        obj2_rotation = None
        obj2_angle = None
        
        for line in lines:
            if 'obj1_angle:' in line.lower():
                try:
                    obj1_angle = int(line.split(':')[1].strip())
                except:
                    pass
            elif 'obj2_rotation:' in line.lower():
                try:
                    obj2_rotation = line.split(':')[1].strip()
                except:
                    pass
            elif 'obj2_angle:' in line.lower():
                try:
                    obj2_angle = int(line.split(':')[1].strip())
                except:
                    pass
        
        if all(x is not None for x in [obj1_angle, obj2_rotation, obj2_angle]):
            rotation_needed = obj1_angle - obj2_angle
            if rotation_needed < 0:
                rotation_needed += 360
            
            rotation_step = int(rotation_needed / 90)
            
            print(f"\n[INFO] Geometric analysis (GPT-4V) summary:")
            print(f"       Best match for '{prompt1}': {obj1_angle}° (single object)")
            print(f"       Best match for '{prompt2}': rotation={obj2_rotation}, azimuth={obj2_angle}°")
            print(f"       Rotation delta: {rotation_needed}°")
            print(f"       Rotation steps (90°): {rotation_step}")

            return {
                'best_angle': obj1_angle,
                'best_rotation': 'single',  # obj1 is a single 4-view strip
                'best_similarity': 1.0,  # no numeric score from GPT; placeholder
                'best_path': f"view_{obj1_angle:03d}.png",
                'all_similarities': [1.0],
                'all_angles': [obj1_angle],
                'all_rotations': ['single'],
                'rotation_avg_similarities': {'single': 1.0},
                'best_rotation_avg': ('single', 1.0),
                'obj2_rotation': obj2_rotation,
                'obj2_angle': obj2_angle,
                'rotation_needed': rotation_needed,
                'rotation_step': rotation_step
            }
        else:
            print(f"[ERROR] Failed to parse GPT-4V response: {result_text}")
            return None
            
    except Exception as e:
        print(f"[ERROR] GPT-4V API error: {e}")
        return None


def create_all_rotations_composite_image(obj_results: List[Tuple[dict, str]], 
                              prompt: str,
                              output_path: str):
    """
    Build a 4 (azimuth) × 7 (pose variant) grid from seven ``evaluate_single_with_clip`` results.

    Args:
        obj_results: ``(clip_result_dict, rotation_name)`` in column order.
        prompt: Title string on the composite.
        output_path: Output PNG path.
    """
    angles_degrees = [0, 90, 180, 270]

    first_result = obj_results[0][0]
    first_views = first_result['views']
    img_height, img_width = first_views[0][0].shape[:2]
    
    # Layout: 4×7 grid (4 angles × 7 rotations)
    rows, cols = 4, 7
    padding = 10
    title_height = 60
    
    composite_width = cols * img_width + (cols + 1) * padding
    composite_height = rows * img_height + (rows + 1) * padding + title_height
    
    # Create white canvas
    composite = Image.new('RGB', (composite_width, composite_height), 'white')
    draw = ImageDraw.Draw(composite)
    
    # Load fonts
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        score_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 24)
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 20)
            score_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 16)
        except:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            score_font = ImageFont.load_default()
    
    # Draw title
    title_text = f"4×7 CLIP Evaluation: \"{prompt}\""
    bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_width = bbox[2] - bbox[0]
    title_x = (composite_width - title_width) // 2
    draw.text((title_x, 15), title_text, fill='black', font=title_font)

    all_scores = []
    for result, _ in obj_results:
        all_scores.extend(result['scores'])
    best_global_score = max(all_scores)

    for row, angle_deg in enumerate(angles_degrees):
        for col, (result, rotation_name) in enumerate(obj_results):
            views = result['views']
            scores = result['scores']
            best_view_idx = result['best_view'][1]  # best CLIP azimuth (deg) from this rotation's eval

            paste_x = padding + col * (img_width + padding)
            paste_y = title_height + padding + row * (img_height + padding)

            img_array = views[row][0]  # (rgb_array, angle) for this row's azimuth
            score = scores[row]
            img = Image.fromarray(img_array)

            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)

            rot_label = rotation_name.upper()
            label_bg = Image.new('RGBA', (80, 30), (50, 50, 50, 220))
            overlay.paste(label_bg, (5, 5))
            
            bbox = overlay_draw.textbbox((0, 0), rot_label, font=label_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = 5 + (80 - text_width) // 2
            text_y = 5 + (30 - text_height) // 2
            overlay_draw.text((text_x, text_y), rot_label, fill='white', font=label_font)

            angle_label = f"{angle_deg}°"
            label_bg = Image.new('RGBA', (50, 30), (50, 50, 50, 220))
            overlay.paste(label_bg, (img.width - 55, 5))
            
            bbox = overlay_draw.textbbox((0, 0), angle_label, font=label_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = img.width - 55 + (50 - text_width) // 2
            text_y = 5 + (30 - text_height) // 2
            overlay_draw.text((text_x, text_y), angle_label, fill='white', font=label_font)

            is_best_angle = (row == best_view_idx)
            is_best_global = (score == best_global_score)
            
            if is_best_global:
                bg_color = (255, 215, 0, 230)  # gold: global max CLIP in grid
                score_text = f"BEST: {score:.4f}"
            # elif is_best_angle:
            #     bg_color = (0, 255, 0, 200)  # per-column best (optional)
            #     score_text = f"BEST_ROT: {score:.4f}"
            else:
                intensity = int(100 + 155 * score)
                bg_color = (0, intensity, 255, 200)  # blue ramp by score
                score_text = f"{score:.4f}"
            
            score_bg = Image.new('RGBA', (img.width, 40), bg_color)
            overlay.paste(score_bg, (0, img.height - 40))
            
            bbox = overlay_draw.textbbox((0, 0), score_text, font=score_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = (img.width - text_width) // 2
            text_y = img.height - 40 + (40 - text_height) // 2
            overlay_draw.text((text_x, text_y), score_text, fill='white', font=score_font)

            composite.paste(img, (paste_x, paste_y))
            composite.paste(overlay, (paste_x, paste_y), overlay)

    composite.save(output_path)
    print(f"[INFO] Wrote 4x7 CLIP composite: {output_path}")

