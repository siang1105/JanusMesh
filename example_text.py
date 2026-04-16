import os
import sys
# Repo root on sys.path (trellis/, clip/, dataset_toolkits/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.pipelines.trellis_text_to_3d import generate_output_dir_name
from trellis.utils import render_utils, postprocessing_utils

from clip.render_single_clip import (
    evaluate_single_with_clip,
    compare_obj2_best_with_obj1_all_28,
    create_composite_image,
    create_all_rotations_composite_image,
)

import subprocess
import shlex
import shutil
import tempfile
from pathlib import Path
import torch
import argparse

# Axis rotation indices for case 3.
ROTATION_AXIS_MAP = {
    'x_1': {'axis': 'x', 'k': 1, 'dims': [3, 4]},
    'x_2': {'axis': 'x', 'k': 2, 'dims': [3, 4]},
    'x_3': {'axis': 'x', 'k': 3, 'dims': [3, 4]},
    'y_1': {'axis': 'y', 'k': 1, 'dims': [2, 4]},
    'y_2': {'axis': 'y', 'k': 2, 'dims': [2, 4]},
    'y_3': {'axis': 'y', 'k': 3, 'dims': [2, 4]},
    'z':   {'axis': 'z', 'k': 0, 'dims': [2, 3]},
}

# View angle (90° step) indices for case 3.
ROTATION_ANGLE_MAP = {
    0:   {'k': 0, 'dims': [2, 3]},
    90:  {'k': 1, 'dims': [2, 3]},
    180: {'k': 2, 'dims': [2, 3]},
    270: {'k': 3, 'dims': [2, 3]},
}

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipeline.cuda()

def run_synctweedies_mesh(
    glb_path: str,
    prompt1: str,
    prompt2: str,
    output_dir: str,
    tag: str = "mesh",
    conda_env_name: str = "synctweedies",
    azim_split_mode: str = "quadrant",
):
    """
    Texture Trellis `sample1.glb` via the SyncTweedies mesh app.

    glb_path: Trellis-exported GLB (typically sample1.glb).
    prompt1 / prompt2: Same dual prompts as the Trellis run.
    output_dir: Trellis output folder; passed to SyncTweedies as save_top_dir.
    """
    
    glb_path_abs = str(Path(glb_path).resolve())
    save_top_dir_abs = str(Path(output_dir).resolve())

    # JanusMesh: SyncTweedies/ is next to this script in the release repo.
    script_dir = Path(__file__).resolve().parent
    synctweedies_root = script_dir / "SyncTweedies"

    main_prompt = prompt1
    secondary_prompt = prompt2
    azim_split_mode = azim_split_mode

    cmd = [
        "conda", "run", "-n", conda_env_name,
        "python", "main.py",
        "--app", "mesh",
        "--prompt", main_prompt,
        "--secondary_prompt", secondary_prompt,
        "--save_top_dir", save_top_dir_abs,
        "--tag", tag,
        "--save_dir_now",
        "--case_num", "2",
        "--mesh", glb_path_abs,
        "--seed", "0",
        "--sampling_method", "ddim",
        "--initialize_xt_from_zt",
        "--save_gif",
        "--azim_split_mode", azim_split_mode,
    ]

    print("\n" + "=" * 70)
    print("[INFO] Running SyncTweedies mesh texturing...")
    print("CWD:", synctweedies_root)
    print("COMMAND:")
    print(" ".join(shlex.quote(c) for c in cmd))
    print("=" * 70 + "\n")

    try:
        subprocess.run(cmd, check=True, cwd=str(synctweedies_root))
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 70)
        print("[ERROR] SyncTweedies command failed with return code:", e.returncode)
        print("Please check the traceback ABOVE this message for the real error from SyncTweedies.")
        print("=" * 70 + "\n")
        # Comment out `raise` below to keep this script running after a SyncTweedies failure.
        raise

    print("\n[INFO] SyncTweedies finished. Textured mesh & renders should be saved under:")
    print(f"       {output_dir}")
    print("=" * 70 + "\n")


# ==================== Parse Command Line Arguments ====================
def parse_args():
    parser = argparse.ArgumentParser(description='JanusMesh: TRELLIS dual-prompt text-to-3D (Janus illusion)')

    parser.add_argument('--prompt1', type=str, default="A Sofa",
                        help='First prompt')
    parser.add_argument('--prompt2', type=str, default="Open Book",
                        help='Second prompt')

    parser.add_argument('--manual_rotation_step', type=int, default=0,
                        help='Case 2 only: azimuth alignment in 90° steps (default: 0). Case 3 uses CLIP instead.')
    parser.add_argument('--guidance', type=str, default='false', choices=['false', 'noise_guidance', 'space_control'],
                        help='Guidance: false, noise_guidance, or space_control (default: false)')
    parser.add_argument('--case', type=int, default=1, choices=[1, 2, 3],
                        help='1/2: generate only (fixed split; use --manual_rotation_step for case 2). '
                             '3: CLIP pose search then generate.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--output_base', type=str, default='outputs',
                        help='Base directory for outputs')
    parser.add_argument('--t0_idx_value', type=int, default=15,
                        help='Time index for SPACE CONTROL τ₀')
    parser.add_argument('--guided_structure_weight', type=float, default=0.3,
                        help='Weight for noise guidance structure blend')
    parser.add_argument('--blend_strategy', type=str, default='sdf_avg',
                        choices=['union', 'DistanceTransform', 'sdf_avg', 'union_closing', 'minkowski', 'polar_blend', 'blur_average_threshold', 'DistanceTransform_median', 'DistanceTransform_light'],
                        help='Voxel blend strategy (default: sdf_avg)')
    return parser.parse_args()

args = parse_args()

MANUAL_ROTATION_STEP = args.manual_rotation_step
prompt1 = args.prompt1
prompt2 = args.prompt2
guidance = args.guidance
case = args.case
seed = args.seed
output_base = args.output_base
t0_idx_value = args.t0_idx_value
guided_structure_weight = args.guided_structure_weight
blend_strategy = args.blend_strategy

output_dir = generate_output_dir_name(prompt1, prompt2, base_dir=output_base, case=case)
os.makedirs(output_dir, exist_ok=True)
print(f"[INFO] Output directory: {output_dir}")
print(f"[INFO] Arguments: prompt1='{prompt1}', prompt2='{prompt2}', guidance={guidance}, case={case}, seed={seed}")
if case == 3:
    print("[INFO] case=3: CLIP alignment (step 1) then dual generation (step 2).")
else:
    print(f"[INFO] case={case}: dual generation only (--manual_rotation_step={MANUAL_ROTATION_STEP} for case 2).")

rotation_step = MANUAL_ROTATION_STEP
rotation_info = None

if case == 3:
    print("\n" + "="*70)
    print("STEP 1: Angle Check (CLIP, case 3 only)")
    print("="*70)

    clip_scratch_root = tempfile.mkdtemp(prefix="janusmesh_clip_case3_")
    clip_eval1_dir = os.path.join(clip_scratch_root, "clip_eval1")
    clip_eval2_dir = os.path.join(clip_scratch_root, "clip_eval2")
    print("[INFO] CLIP intermediates use a temporary directory (not saved under output_dir).")

    try:
        outputs1, \
        outputs2_x_1, outputs2_x_2, outputs2_x_3, \
        outputs2_y_1, outputs2_y_2, outputs2_y_3, \
        outputs2_z = pipeline.run(
            prompt1=prompt1,
            prompt2=prompt2,
            seed=seed,
            angle_check=True,
            output_dir=output_dir,
        )

        print("[INFO] Skipping sample_single_*.mp4 angle-check previews (case 3).")

        print("\n" + "="*70)
        print("CLIP EVALUATION: Rendering 4 views and evaluating with CLIP")
        print("="*70)

        obj1_result = evaluate_single_with_clip(
            gaussian_obj=outputs1['gaussian'][0],
            prompt=prompt1,
            output_dir=clip_eval1_dir,
            resolution=512,
            bg_color=(255, 255, 255)
        )

        obj2_results = []
        rotation_names = ['x_1', 'x_2', 'x_3', 'y_1', 'y_2', 'y_3', 'z']

        for outputs, name in zip([outputs2_x_1, outputs2_x_2, outputs2_x_3, outputs2_y_1, outputs2_y_2, outputs2_y_3, outputs2_z], rotation_names):
            print(f"\n[INFO] CLIP evaluation: obj2 pose variant '{name}'")
            result = evaluate_single_with_clip(
                gaussian_obj=outputs['gaussian'][0],
                prompt=prompt2,
                output_dir=os.path.join(clip_eval2_dir, f"rotation_{name}"),
                resolution=512,
                bg_color=(255, 255, 255)
            )
            obj2_results.append((result, name))

        obj1_composite_path = os.path.join(clip_eval1_dir, "clip_evaluation.png")
        create_composite_image(obj1_result['views'], obj1_result['scores'], prompt1, obj1_composite_path)

        obj2_composite_path = os.path.join(clip_eval2_dir, "clip_evaluation.png")
        create_all_rotations_composite_image(obj2_results, prompt2, obj2_composite_path)

        best_obj1_result = obj1_result
        best_obj2_result = max(obj2_results, key=lambda x: max(x[0]['scores']))

        R1P1Angle = best_obj1_result['best_view'][1]
        R2P2Angle = best_obj2_result[0]['best_view'][1]
        R2P2Rotation = best_obj2_result[1]

        print(f"\n[INFO] Best obj1 (CLIP): azimuth={R1P1Angle}°, score={max(best_obj1_result['scores']):.6f}")
        print(f"[INFO] Best obj2 (CLIP): rotation={R2P2Rotation}, azimuth={R2P2Angle}°, score={max(best_obj2_result[0]['scores']):.6f}")

        R1P1_image_path = os.path.join(clip_eval1_dir, f"view_{int(R1P1Angle):03d}.png")

        similarity_result = compare_obj2_best_with_obj1_all_28(
            single_image_path=R1P1_image_path,
            obj1_all_rotations_dir=clip_eval2_dir,
            output_dir=output_dir
        )

        clip_rotation_step = 0

        if similarity_result:
            best_match_angle = similarity_result['best_angle']
            best_match_rotation = similarity_result['best_rotation']
            best_match_similarity = similarity_result['best_similarity']
            best_rotation_avg = similarity_result['best_rotation_avg']

            print(f"\n" + "="*70)
            print("CLIP SIMILARITY COMPARISON RESULTS")
            print("="*70)
            print(f"R1P1 best angle ({int(R1P1Angle)}°) compared with R2P2 all 28 images:")
            print(f"Best match: R2P2 {best_match_rotation} rotation, {best_match_angle}° with similarity {best_match_similarity:.6f}")
            print(f"Best rotation average: {best_rotation_avg[0]} rotation with avg similarity {best_rotation_avg[1]:.6f}")

            rotation_needed = best_match_angle - R1P1Angle
            if rotation_needed < 0:
                rotation_needed += 360

            clip_rotation_step = int(rotation_needed / 90)

            similarity_output_file = os.path.join(output_dir, "similarity_comparison.txt")
            with open(similarity_output_file, "w") as f:
                f.write("IMAGE SIMILARITY COMPARISON RESULTS (R1P1 vs R2P2 ALL 28)\n")
                f.write("="*70 + "\n")
                f.write(f"R1P1: single result, {int(R1P1Angle)}°\n")
                f.write(f"Best match R2P2: {best_match_rotation} rotation, {best_match_angle}°\n")
                f.write(f"Similarity score: {best_match_similarity:.6f}\n")
                f.write(f"Best rotation average: {best_rotation_avg[0]} rotation with avg similarity {best_rotation_avg[1]:.6f}\n")
                f.write(f"Rotation needed: {rotation_needed}°\n")
                f.write(f"Rotation steps: {clip_rotation_step}\n\n")

                f.write("Rotation average similarities:\n")
                for rot, avg_sim in sorted(similarity_result['rotation_avg_similarities'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {rot:4s}: {avg_sim:.6f}\n")

                f.write(f"\nAll 28 similarity scores:\n")
                for i, (rot, angle, sim) in enumerate(zip(similarity_result['all_rotations'], similarity_result['all_angles'], similarity_result['all_similarities'])):
                    f.write(f"  {i+1:2d}. R2P2 {rot:4s} {angle:3d}°: {sim:.6f}\n")

            print(f"\n[INFO] CLIP similarity report written to: {similarity_output_file}")

            rotation_step = clip_rotation_step
            best_rotation_name = best_match_rotation
            best_angle = best_match_angle

            rotation_info = {
                'axis_rotation': ROTATION_AXIS_MAP[best_rotation_name],
                'angle_rotation': ROTATION_ANGLE_MAP[best_angle],
                'axis_name': best_rotation_name,
                'angle': best_angle,
            }

            print(f"\n[AUTO] CLIP rotation (for case 3):")
            print(f"  Axis rotation: {best_rotation_name} (k={rotation_info['axis_rotation']['k']}, dims={rotation_info['axis_rotation']['dims']})")
            print(f"  Angle rotation: {best_angle}° (k={rotation_info['angle_rotation']['k']}, dims={rotation_info['angle_rotation']['dims']})")
        else:
            print("[ERROR] CLIP image similarity comparison failed")
    finally:
        if clip_scratch_root and os.path.isdir(clip_scratch_root):
            shutil.rmtree(clip_scratch_root, ignore_errors=True)
            print(f"[INFO] Removed temporary CLIP directory: {clip_scratch_root}")

print("\n" + "="*70)
print("STEP 2: Dual-prompt generation (Janus / illusion asset)")
print("="*70)
outputs1, outputs2 = pipeline.run(
    prompt1=prompt1,
    prompt2=prompt2,
    seed=seed,
    rotation_step=rotation_step,
    rotation_info=rotation_info,
    output_dir=output_dir,
    guidance=guidance,
    case=case,
    t0_idx_value=t0_idx_value,
    guided_structure_weight=guided_structure_weight,
    blend_strategy=blend_strategy,
)

print("[INFO] Rendering output stream 1...")
# video = render_utils.render_video(outputs1['gaussian'][0])['color']
# imageio.mimsave(os.path.join(output_dir, "sample_gs1.mp4"), video, fps=30)
# print(f"[INFO] Wrote: {os.path.join(output_dir, 'sample_gs1.mp4')}")

glb = postprocessing_utils.to_glb(
    outputs1['gaussian'][0],
    outputs1['mesh'][0],
    simplify=0.95,
    texture_size=1024,
)
glb.export(os.path.join(output_dir, "sample.glb"))
outputs1['gaussian'][0].save_ply(os.path.join(output_dir, "sample.ply"))

# print("[INFO] Rendering output stream 2...")
# video = render_utils.render_video(outputs2['gaussian'][0])['color']
# imageio.mimsave(os.path.join(output_dir, "sample_gs2.mp4"), video, fps=30)
# print(f"[INFO] Wrote: {os.path.join(output_dir, 'sample_gs2.mp4')}")

glb = postprocessing_utils.to_glb(
    outputs2['gaussian'][0],
    outputs2['mesh'][0],
    simplify=0.95,
    texture_size=1024,
)
# glb.export(os.path.join(output_dir, "sample2.glb"))
# outputs2['gaussian'][0].save_ply(os.path.join(output_dir, "sample2.ply"))

print("\n[INFO] Releasing Trellis GPU memory before running SyncTweedies...")

try:
    pipeline.to("cpu")
except Exception:
    pass

del outputs1
del outputs2

torch.cuda.empty_cache()
print("[INFO] torch.cuda.empty_cache() called.\n")

run_synctweedies_mesh(
    glb_path=os.path.join(output_dir, "sample.glb"),
    prompt1=prompt1,
    prompt2=prompt2,
    output_dir=output_dir,
    tag="mesh",
    conda_env_name="synctweedies",
    azim_split_mode="quadrant",
)

print("\n[INFO] Completed. Output directory:", output_dir)
