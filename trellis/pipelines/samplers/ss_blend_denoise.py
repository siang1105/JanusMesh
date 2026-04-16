import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import numpy as np
import torch
import json
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from typing import Optional

import trellis.models as models
import trellis.modules.sparse as sp
from dataset_toolkits.render import _render
from scipy.ndimage import binary_erosion, generate_binary_structure, binary_dilation, distance_transform_edt, gaussian_filter, binary_closing, binary_fill_holes, median_filter
from sklearn.decomposition import PCA

def blur_average_threshold_3d(a_mask: np.ndarray, b_mask: np.ndarray, sigma: float = 2.0, thresh: float = 0.5) -> np.ndarray:
    """Per-mask 3D Gaussian blur, mean, then threshold (simple tunable blend)."""
    if a_mask.shape != b_mask.shape:
        raise ValueError("Input masks must have the same shape")
    
    if a_mask.ndim == 5:  # (B, C, D, H, W)
        B, C, D, H, W = a_mask.shape
        result = np.zeros_like(a_mask, dtype=np.uint8)
        
        for b in range(B):
            for c in range(C):
                a_blurred = gaussian_filter(a_mask[b, c].astype(float), sigma=sigma)
                b_blurred = gaussian_filter(b_mask[b, c].astype(float), sigma=sigma)
                
                avg = 0.5 * (a_blurred + b_blurred)
                result[b, c] = (avg >= thresh).astype(np.uint8)
        
        return result
    
    elif a_mask.ndim == 3:  # (D, H, W) single volume
        a = gaussian_filter(a_mask.astype(float), sigma=sigma)
        b = gaussian_filter(b_mask.astype(float), sigma=sigma)
        
        avg = 0.5 * (a + b)
        
        return (avg >= thresh).astype(np.uint8)
    
    else:
        raise ValueError("Input masks must be 3D or 5D arrays")

def fill_small_holes_3d(volume, radius=1, iterations=1):
    """
    3D hole filling via binary closing.

    radius: structuring-element radius (typically 1-2).
    iterations: closing iterations.
    """
    struct = generate_binary_structure(3, 1)  # 3D connectivity-1 structuring element
    closed = binary_closing(volume, structure=struct, iterations=iterations)
    return closed.astype(np.uint8)

def distance_transform_blend_fixed(occ1, occ2_warped, hole_radius=1, hole_iters=1):
    """
    Midpoint bridge between occupied sets, then 2D hole fill per slice + 3D closing.
    """
    occ1_np = occ1.cpu().numpy()
    occ2_np = occ2_warped.cpu().numpy()
    
    B, C, D, H, W = occ1_np.shape
    blended_occ_np = np.zeros_like(occ1_np)

    for b in range(B):
        for c in range(C):
            vox1 = occ1_np[b, c]
            vox2 = occ2_np[b, c]
            
            coords1 = np.argwhere(vox1 > 0)
            coords2 = np.argwhere(vox2 > 0)
            
            if len(coords1) == 0:
                blended_occ_np[b, c] = vox2
                continue
            if len(coords2) == 0:
                blended_occ_np[b, c] = vox1
                continue
            
            midpoints = []
            # A→B
            for coord1 in coords1:
                distances = np.sqrt(np.sum((coords2 - coord1)**2, axis=1))
                closest_coord2 = coords2[np.argmin(distances)]
                midpoint = (coord1 + closest_coord2) // 2
                if (0 <= midpoint[0] < D and 0 <= midpoint[1] < H and 0 <= midpoint[2] < W):
                    midpoints.append(midpoint)
            # B→A
            for coord2 in coords2:
                distances = np.sqrt(np.sum((coords1 - coord2)**2, axis=1))
                closest_coord1 = coords1[np.argmin(distances)]
                midpoint = (coord2 + closest_coord1) // 2
                if (0 <= midpoint[0] < D and 0 <= midpoint[1] < H and 0 <= midpoint[2] < W):
                    midpoints.append(midpoint)

            midpoints = np.unique(midpoints, axis=0)
            for midpoint in midpoints:
                blended_occ_np[b, c, midpoint[0], midpoint[1], midpoint[2]] = 1

            # Post: 2D hole fill per depth slice, then 3D closing.
            for z in range(D):
                blended_occ_np[b, c, z] = binary_fill_holes(blended_occ_np[b, c, z]).astype(float)
            
            blended_occ_np[b, c] = fill_small_holes_3d(blended_occ_np[b, c], 
                                                       radius=hole_radius, 
                                                       iterations=hole_iters)

            num_blended_points = np.sum(blended_occ_np[b, c] > 0)
            print(f"[DIST_TRANS_FIX] batch={b} ch={c}: voxels_after_blend={num_blended_points}")

    blended_occ = torch.from_numpy(blended_occ_np).float().to(occ1.device)
    return blended_occ

def distance_transform_blend_median(occ1, occ2_warped, median_size=3, threshold=0.5):
    """
    Midpoint bridge + 2D hole fill, dilation, median smooth, then threshold.

    median_size: median window (typically 3 or 5).
    threshold: binarize after median.
    """
    occ1_np = occ1.cpu().numpy()
    occ2_np = occ2_warped.cpu().numpy()
    
    B, C, D, H, W = occ1_np.shape
    blended_occ_np = np.zeros_like(occ1_np)

    for b in range(B):
        for c in range(C):
            vox1 = occ1_np[b, c]
            vox2 = occ2_np[b, c]
            
            coords1 = np.argwhere(vox1 > 0)
            coords2 = np.argwhere(vox2 > 0)
            
            if len(coords1) == 0:
                blended_occ_np[b, c] = vox2
                continue
            if len(coords2) == 0:
                blended_occ_np[b, c] = vox1
                continue
            
            midpoints = []
            # A→B
            for coord1 in coords1:
                distances = np.sqrt(np.sum((coords2 - coord1)**2, axis=1))
                closest_coord2 = coords2[np.argmin(distances)]
                midpoint = (coord1 + closest_coord2) // 2
                if (0 <= midpoint[0] < D and 0 <= midpoint[1] < H and 0 <= midpoint[2] < W):
                    midpoints.append(midpoint)
            # B→A
            for coord2 in coords2:
                distances = np.sqrt(np.sum((coords1 - coord2)**2, axis=1))
                closest_coord1 = coords1[np.argmin(distances)]
                midpoint = (coord2 + closest_coord1) // 2
                if (0 <= midpoint[0] < D and 0 <= midpoint[1] < H and 0 <= midpoint[2] < W):
                    midpoints.append(midpoint)

            midpoints = np.unique(midpoints, axis=0)
            for midpoint in midpoints:
                blended_occ_np[b, c, midpoint[0], midpoint[1], midpoint[2]] = 1

            num_midpoints = len(midpoints)
            print(f"[DIST_TRANS_MEDIAN] batch={b} ch={c}: midpoints={num_midpoints}")

            # Like fixed variant: fill holes → closing-like; dilation + median instead of closing-only.
            for z in range(D):
                blended_occ_np[b, c, z] = binary_fill_holes(blended_occ_np[b, c, z]).astype(np.uint8)
            
            num_after_fill = np.sum(blended_occ_np[b, c] > 0)
            print(f"[DIST_TRANS_MEDIAN] batch={b} ch={c}: after_2d_fill={num_after_fill}")
            
            struct = generate_binary_structure(3, 1)  # 6-neighborhood in 3D
            blended_occ_np[b, c] = binary_dilation(blended_occ_np[b, c], structure=struct, iterations=1)
            
            num_after_dilation = np.sum(blended_occ_np[b, c] > 0)
            print(f"[DIST_TRANS_MEDIAN] batch={b} ch={c}: after_dilation={num_after_dilation}")
            
            blended_occ_np[b, c] = median_filter(
                blended_occ_np[b, c].astype(float), 
                size=median_size
            )
            
            blended_occ_np[b, c] = (blended_occ_np[b, c] >= threshold).astype(np.uint8)
            
            num_after_median = np.sum(blended_occ_np[b, c] > 0)
            print(f"[DIST_TRANS_MEDIAN] batch={b} ch={c}: after_median={num_after_median}")

            num_blended_points = np.sum(blended_occ_np[b, c] > 0)
            print(f"[DIST_TRANS_MEDIAN] batch={b} ch={c}: final_voxels={num_blended_points} median_size={median_size}")

    blended_occ = torch.from_numpy(blended_occ_np).float().to(occ1.device)
    return blended_occ

def distance_transform_blend_light(occ1, occ2_warped, dilation_iters=1, erosion_iters=1):
    """
    Midpoint bridge + light morphology: 2D hole fill, 3D dilation then erosion (no median).
    """
    occ1_np = occ1.cpu().numpy()
    occ2_np = occ2_warped.cpu().numpy()
    
    B, C, D, H, W = occ1_np.shape
    blended_occ_np = np.zeros_like(occ1_np)

    for b in range(B):
        for c in range(C):
            vox1 = occ1_np[b, c]
            vox2 = occ2_np[b, c]
            
            coords1 = np.argwhere(vox1 > 0)
            coords2 = np.argwhere(vox2 > 0)
            
            if len(coords1) == 0:
                blended_occ_np[b, c] = vox2
                continue
            if len(coords2) == 0:
                blended_occ_np[b, c] = vox1
                continue
            
            midpoints = []
            # A→B
            for coord1 in coords1:
                distances = np.sqrt(np.sum((coords2 - coord1)**2, axis=1))
                closest_coord2 = coords2[np.argmin(distances)]
                midpoint = (coord1 + closest_coord2) // 2
                if (0 <= midpoint[0] < D and 0 <= midpoint[1] < H and 0 <= midpoint[2] < W):
                    midpoints.append(midpoint)
            # B→A
            for coord2 in coords2:
                distances = np.sqrt(np.sum((coords1 - coord2)**2, axis=1))
                closest_coord1 = coords1[np.argmin(distances)]
                midpoint = (coord2 + closest_coord1) // 2
                if (0 <= midpoint[0] < D and 0 <= midpoint[1] < H and 0 <= midpoint[2] < W):
                    midpoints.append(midpoint)

            midpoints = np.unique(midpoints, axis=0)
            for midpoint in midpoints:
                blended_occ_np[b, c, midpoint[0], midpoint[1], midpoint[2]] = 1

            num_midpoints = len(midpoints)
            print(f"[DIST_TRANS_LIGHT] batch={b} ch={c}: midpoints={num_midpoints}")

            for z in range(D):
                blended_occ_np[b, c, z] = binary_fill_holes(blended_occ_np[b, c, z]).astype(np.uint8)
            
            num_after_fill = np.sum(blended_occ_np[b, c] > 0)
            print(f"[DIST_TRANS_LIGHT] batch={b} ch={c}: after_2d_fill={num_after_fill}")
            
            struct = generate_binary_structure(3, 1)
            blended_occ_np[b, c] = binary_dilation(blended_occ_np[b, c], structure=struct, iterations=dilation_iters)
            
            num_after_dilation = np.sum(blended_occ_np[b, c] > 0)
            print(f"[DIST_TRANS_LIGHT] batch={b} ch={c}: after_dilation={num_after_dilation}")
            
            blended_occ_np[b, c] = binary_erosion(blended_occ_np[b, c], structure=struct, iterations=erosion_iters)
            
            num_after_erosion = np.sum(blended_occ_np[b, c] > 0)
            print(f"[DIST_TRANS_LIGHT] batch={b} ch={c}: after_erosion={num_after_erosion}")

            num_blended_points = np.sum(blended_occ_np[b, c] > 0)
            print(f"[DIST_TRANS_LIGHT] batch={b} ch={c}: final_voxels={num_blended_points}")

    blended_occ = torch.from_numpy(blended_occ_np).float().to(occ1.device)
    return blended_occ

# --- Distance-based midpoint bridge ---
def distance_transform_blend(occ1, occ2_warped):
    """
    Bidirectional nearest-neighbor bridge: for each occupied voxel in A (resp. B),
    take the midpoint to the closest occupied voxel in B (resp. A), then fill holes.

    Args:
        occ1: Occupancy [B, C, D, H, W].
        occ2_warped: Warped second occupancy, same shape.

    Returns:
        Binary blended occupancy as float tensor on ``occ1.device``.
    """
    occ1_np = occ1.cpu().numpy()
    occ2_np = occ2_warped.cpu().numpy()
    
    B, C, D, H, W = occ1_np.shape
    
    blended_occ_np = np.zeros_like(occ1_np)
    
    for b in range(B):
        for c in range(C):
            vox1 = occ1_np[b, c]  # [D, H, W]
            vox2 = occ2_np[b, c]  # [D, H, W]
            
            coords1 = np.argwhere(vox1 > 0)  # [N, 3] indices (d, h, w)
            
            if len(coords1) == 0:
                blended_occ_np[b, c] = vox2
                continue
            
            coords2 = np.argwhere(vox2 > 0)  # [M, 3]
            
            if len(coords2) == 0:
                blended_occ_np[b, c] = vox1
                continue
            
            midpoints = []
            
            for coord1 in coords1:
                distances = np.sqrt(np.sum((coords2 - coord1)**2, axis=1))
                closest_idx = np.argmin(distances)
                closest_coord2 = coords2[closest_idx]
                
                midpoint = (coord1 + closest_coord2) // 2
                
                if (0 <= midpoint[0] < D and 
                    0 <= midpoint[1] < H and 
                    0 <= midpoint[2] < W):
                    midpoints.append(midpoint)
            
            for coord2 in coords2:
                distances = np.sqrt(np.sum((coords1 - coord2)**2, axis=1))
                closest_idx = np.argmin(distances)
                closest_coord1 = coords1[closest_idx]
                
                midpoint = (coord2 + closest_coord1) // 2
                
                if (0 <= midpoint[0] < D and 
                    0 <= midpoint[1] < H and 
                    0 <= midpoint[2] < W):
                    midpoints.append(midpoint)
            
            midpoints = np.unique(midpoints, axis=0)
            
            for midpoint in midpoints:
                blended_occ_np[b, c, midpoint[0], midpoint[1], midpoint[2]] = 1
            
            from scipy.ndimage import binary_fill_holes
            blended_occ_np[b, c] = binary_fill_holes(blended_occ_np[b, c]).astype(float)
            
            num_blended_points = np.sum(blended_occ_np[b, c] > 0)
            print(f"[DISTANCE_TRANSFORM] batch={b} ch={c}: |A|={len(coords1)} |B|={len(coords2)} voxels={num_blended_points}")
    
    blended_occ = torch.from_numpy(blended_occ_np).float().to(occ1.device)
    
    return blended_occ

# --- Polar (per-ray) slice blend ---
def polar_blend_slices(sliceA, sliceB, alpha=0.5):
    """
    Polar θ-bin blend on one 2D slice: max radius per wedge, linear mix, then fill disk.

    sliceA, sliceB: binary masks. alpha: 0=A only, 1=B only, 0.5=mean boundary.
    """
    H, W = sliceA.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    theta = np.arctan2(yy - cy, xx - cx)
    
    theta = (theta + 2 * np.pi) % (2 * np.pi)
    
    num_angles = 360  # angular bins
    angle_step = 2 * np.pi / num_angles
    
    rA_angles = np.zeros(num_angles)
    rB_angles = np.zeros(num_angles)
    
    for i in range(num_angles):
        angle_start = i * angle_step
        angle_end = (i + 1) * angle_step
        
        if i == num_angles - 1:  # wrap at 2π
            mask_angle = (theta >= angle_start) | (theta < angle_end)
        else:
            mask_angle = (theta >= angle_start) & (theta < angle_end)
        
        maskA_angle = mask_angle & (sliceA > 0)
        rA_angles[i] = np.max(r * maskA_angle) if maskA_angle.any() else 0
        
        maskB_angle = mask_angle & (sliceB > 0)
        rB_angles[i] = np.max(r * maskB_angle) if maskB_angle.any() else 0
    
    rBlend_angles = (1 - alpha) * rA_angles + alpha * rB_angles
    
    angle_indices = ((theta / angle_step) % num_angles).astype(int)
    rBlend = rBlend_angles[angle_indices]
    
    blended = (r <= rBlend).astype(np.uint8)
    return blended

def polar_blend_voxel(occ1_np, occ2_np, alpha=0.5):
    """
    Apply polar_blend_slices along depth D for each (B, C) slice of [B,C,D,H,W].
    """
    B, C, D, H, W = occ1_np.shape
    out = np.zeros_like(occ1_np, dtype=np.uint8)
    for b in range(B):
        for c in range(C):
            for z in range(D):
                out[b, c, z] = polar_blend_slices(occ1_np[b, c, z], occ2_np[b, c, z], alpha)
    return out

# --- Morphological blends ---
def ball_struct(radius: int) -> np.ndarray:
    """3D spherical binary structuring element of given integer radius."""
    L = np.arange(-radius, radius+1)
    zz, yy, xx = np.meshgrid(L, L, L, indexing='ij')
    return (xx*xx + yy*yy + zz*zz) <= radius*radius

def minkowski_blend_3d(
    occ1_np: np.ndarray, 
    occ2_np: np.ndarray, 
    r: int = 2,
    mode: str = "avg", 
    thresh: float = 0.5
    ) -> np.ndarray:
    """
    Approximate Minkowski-style blend: dilate A and B with a ball of radius r, then combine.

    mode:
      - "avg": mean of dilated masks, threshold at thresh.
      - "union": logical OR of dilated masks.
      - "intersect": logical AND (overlap core).
    """
    se = ball_struct(r)
    B, C, D, H, W = occ1_np.shape
    A = np.zeros_like(occ1_np)
    B_dilated = np.zeros_like(occ2_np)
    for b in range(B):
        for c in range(C):
            A[b, c] = binary_dilation(occ1_np[b, c], structure=se)
            B_dilated[b, c] = binary_dilation(occ2_np[b, c], structure=se)

    if mode == "union":
        out = np.logical_or(A, B_dilated)
    elif mode == "intersect":
        out = np.logical_and(A, B_dilated)
    elif mode == "avg":
        avg = (A.astype(np.float32) + B_dilated.astype(np.float32)) * 0.5
        out = avg >= thresh
    return out.astype(np.uint8)

def union_then_closing_3d(occ1_np: np.ndarray, occ2_np: np.ndarray, radius: int = 2) -> np.ndarray:
    """Union of occupancies then 3D binary closing with a spherical structuring element."""
    uni = np.logical_or(occ1_np, occ2_np)
    se  = ball_struct(radius)
    
    B, C, D, H, W = uni.shape
    clo = np.zeros_like(uni)
    for b in range(B):
        for c in range(C):
            clo[b, c] = binary_closing(uni[b, c], structure=se)
    
    return clo.astype(np.uint8)

# --- Signed-distance blend ---
def _sdf_3d_np(mask_3d: np.ndarray) -> np.ndarray:
    """
    Signed distance field for one 3D mask: negative inside, positive outside, zero on surface.

    mask_3d: [D, H, W] bool or {0,1}.
    """
    m = mask_3d.astype(bool)
    outside = distance_transform_edt(~m)
    inside  = distance_transform_edt(m)
    return outside - inside  # outside positive, inside negative

def sdf_average_blend_3d_np(
    occ1_np: np.ndarray,        # [B,C,D,H,W] bool/0-1
    occ2_np: np.ndarray,        # [B,C,D,H,W] aligned / warped second volume
    alpha: float = 0.5,         # blend weight on phi1 vs phi2 (0.5 = mean SDF)
    clip_s: float | None = 12.0,# TSDF clamp magnitude; None = no clamp
    threshold: float = -1.0,     # level-set threshold (negative leans union-like)
    step: Optional[int] = None,
    ) -> np.ndarray:
    assert occ1_np.shape == occ2_np.shape, "occ1/occ2 shape must match"
    B, C, D, H, W = occ1_np.shape
    out = np.zeros((B, C, D, H, W), dtype=np.uint8)

    for b in range(B):
        for c in range(C):
            m1 = occ1_np[b, c].astype(bool)
            m2 = occ2_np[b, c].astype(bool)

            if not m1.any() and not m2.any():
                out[b, c] = 0
                continue
            if not m1.any():
                # Only m2: threshold its (possibly truncated) SDF.
                phi = _sdf_3d_np(m2)
                if clip_s is not None:
                    phi = np.clip(phi, -clip_s, clip_s)
                out[b, c] = (phi <= threshold).astype(np.uint8)
                continue
            if not m2.any():
                phi = _sdf_3d_np(m1)
                if clip_s is not None:
                    phi = np.clip(phi, -clip_s, clip_s)
                out[b, c] = (phi <= threshold).astype(np.uint8)
                continue

            phi1 = _sdf_3d_np(m1)
            phi2 = _sdf_3d_np(m2)
            if clip_s is not None:
                phi1 = np.clip(phi1, -clip_s, clip_s)
                phi2 = np.clip(phi2, -clip_s, clip_s)

            phi = alpha * phi1 + (1.0 - alpha) * phi2
            print(f"[SDF_AVG] step: {step}")

            out[b, c] = (phi <= threshold).astype(np.uint8)

    return out

# --- Decode, blend voxels, re-encode ---
def blend_and_reencode_voxel_to_latent(
    pred_v_1,
    pred_v_2,
    count: int,
    resolution=64,
    encoder_ckpt="microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16",
    decoder_ckpt="microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
    blend_strategy="union",
    threshold=0.5,
    blend_weight=0.5,
    step: Optional[int] = None,
    # sdf_avg strategy
    sdf_alpha: float = 0.5,       # weight on first SDF vs second (0.5 = average)
    sdf_clip_s: float | None = 12.0,  # TSDF clamp; None disables
    sdf_threshold: float = 0.8,  # isosurface threshold after blend
    # other morph / polar / DT variants
    radius: int = 0,              # closing ball radius (union_closing)
    minkowski_r: int = 1,         # dilation ball radius (minkowski)
    minkowski_mode: str = "intersect",  # avg | union | intersect
    minkowski_thresh: float = 0.5, # for mode=="avg"
    polar_alpha: float = 0.5,     # polar slice blend alpha
    # DistanceTransform_median
    median_size: int = 3,         # median window (3 or 5 typical)
    median_threshold: float = 0.5, # post-median binarize
    angle_check: bool = False,
    rotation_step: int = 0,
    rotation_info: dict = None,
    output_dir: Optional[str] = None,
    case: int = 1,
    ): 
    
    # Load pretrained encoder/decoder
    encoder = models.from_pretrained(encoder_ckpt).eval().cuda()
    decoder = models.from_pretrained(decoder_ckpt).eval().cuda()

    with torch.no_grad():
        # Decode latent to voxel occupancy

        probs1 = torch.sigmoid(decoder(pred_v_1))
        probs2 = torch.sigmoid(decoder(pred_v_2))            
        
        occ1 = probs1 > threshold
        occ2 = probs2 > threshold

        occ1_original = occ1.clone()
        occ2_original = occ2.clone()
        
        if case == 1:
            occ2_warped = occ2
        elif case == 2:
            occ2_warped = torch.rot90(occ2, k=2, dims=[2, 3])
        elif case == 3 and rotation_info is not None:
            # Case 3: two rot90 steps from CLIP alignment (axis pose, then view azimuth).
            axis_rot = rotation_info['axis_rotation']
            occ2_warped = torch.rot90(
                occ2,
                k=axis_rot['k'],
                dims=axis_rot['dims']
            )
            print(f"[CASE 3] Applying axis rotation: {rotation_info['axis_name']} "
                  f"(k={axis_rot['k']}, dims={axis_rot['dims']})")

            angle_rot = rotation_info['angle_rotation']
            occ2_warped = torch.rot90(
                occ2_warped,
                k=angle_rot['k'],
                dims=angle_rot['dims']
            )
            if angle_rot['k'] != 0:
                print(f"[CASE 3] Applying angle rotation: {rotation_info['angle']}° "
                      f"(k={angle_rot['k']}, dims={angle_rot['dims']})")
        else:
            raise ValueError(f"Unknown case: {case}. Use case=1, 2, or 3")
        
        coords1 = torch.argwhere(occ1 > threshold)[:, [0, 2, 3, 4]].int()
        coords2 = torch.argwhere(occ2_warped > threshold)[:, [0, 2, 3, 4]].int()

        if output_dir is None:
            output_dir = "outputs/default"

        occ1_np = occ1.detach().cpu().numpy().astype(np.uint8)
        occ2_np = occ2_warped.detach().cpu().numpy().astype(np.uint8)
        
        if blend_strategy == "union":
            blended_occ = (occ1 | occ2_warped).float()
            
        elif blend_strategy == "DistanceTransform":
            blended_occ = distance_transform_blend_fixed(occ1, occ2_warped)
            
        elif blend_strategy == "DistanceTransform_median":
            blended_occ = distance_transform_blend_median(occ1, occ2_warped, median_size=median_size, threshold=median_threshold)
            
        elif blend_strategy == "DistanceTransform_light":
            # Lighter than median variant: dilation + erosion only.
            blended_occ = distance_transform_blend_light(occ1, occ2_warped, dilation_iters=1, erosion_iters=1)
            
        elif blend_strategy == "sdf_avg":
            blended_np = sdf_average_blend_3d_np(occ1_np, occ2_np, alpha=sdf_alpha, clip_s=sdf_clip_s, threshold=sdf_threshold, step=step)
            blended_occ = torch.from_numpy(blended_np).to(occ1.device).float()
        
        elif blend_strategy == "union_closing":
            blended_np = union_then_closing_3d(occ1_np, occ2_np, radius=radius)
            blended_occ = torch.from_numpy(blended_np).to(occ1.device).float()
            
        elif blend_strategy == "minkowski":
            blended_np = minkowski_blend_3d(occ1_np, occ2_np, r=minkowski_r, mode=minkowski_mode, thresh=minkowski_thresh)
            blended_occ = torch.from_numpy(blended_np).to(occ1.device).float()
            
        elif blend_strategy == "polar_blend":
            blended_np = polar_blend_voxel(occ1_np, occ2_np, alpha=polar_alpha)
            blended_occ = torch.from_numpy(blended_np).to(occ1.device).float()
            
        elif blend_strategy == "blur_average_threshold":
            blended_np = blur_average_threshold_3d(occ1_np, occ2_np, sigma=0.5, thresh=0.4)  # slightly lower thresh
            blended_occ = torch.from_numpy(blended_np).to(occ1.device).float()
            
        else:
            raise ValueError(f"Unknown blend_strategy: {blend_strategy}")

        if angle_check:
            occ_rewarped_1 = occ1.float()
            occ_rewarped_2 = occ2.float()
        else:
            # For stream 1: apply π_1 (identity in this case, or specific transformation)
            occ_rewarped_1 = blended_occ.clone()  # π_1 = Id for stream 1
            
            # For stream 2: apply π_2 (rotate back 180 degrees)
            if case == 1:
                occ_rewarped_2 = blended_occ.clone()
            elif case == 2:
                occ_rewarped_2 = torch.rot90(blended_occ, k=-2, dims=[2, 3])
            elif case == 3 and rotation_info is not None:
                # Case 3 inverse: undo view rotation, then undo axis rotation.
                angle_rot = rotation_info['angle_rotation']
                occ_rewarped_2 = torch.rot90(
                    blended_occ,
                    k=(-angle_rot['k']) % 4,
                    dims=angle_rot['dims']
                )
                if angle_rot['k'] != 0:
                    print(f"[CASE 3] Reverse angle rotation: {rotation_info['angle']}° "
                          f"(k={(-angle_rot['k']) % 4})")
                
                axis_rot = rotation_info['axis_rotation']
                occ_rewarped_2 = torch.rot90(
                    occ_rewarped_2,
                    k=(-axis_rot['k']) % 4,
                    dims=axis_rot['dims']
                )
                print(f"[CASE 3] Reverse axis rotation: {rotation_info['axis_name']} "
                      f"(k={(-axis_rot['k']) % 4})")
            else:
                raise ValueError(f"Unknown case: {case}. Use case=1, 2, or 3")

        latent_1 = encoder(occ_rewarped_1, sample_posterior=False)
        latent_2 = encoder(occ_rewarped_2, sample_posterior=False)
        
    return latent_1, latent_2