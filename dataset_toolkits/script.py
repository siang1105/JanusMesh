import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import open3d as o3d
import numpy as np
import utils3d
from render import _render
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import utils3d
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import trellis.models as models
import trellis.modules.sparse as sp


torch.set_grad_enabled(False)

BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = "/home/siangling/blender/blender-3.0.1-linux-x64/blender"

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')

def voxelize_single_mesh(mesh_path, output_dir, output_name="mysample", resolution=64):
    """
    將單一 mesh 進行 voxelization 並存成 .ply 格式。
    """
    os.makedirs(os.path.join(output_dir, "voxels"), exist_ok=True)

    # 讀入 mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    # 限制 mesh 在 [-0.5, 0.5] 內
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # voxelize（open3d 內建方法）
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=1 / resolution,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5)
    )

    # 取得 voxel grid index（64x64x64）
    coords = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])  # [N, 3]
    assert np.all(coords >= 0) and np.all(coords < resolution), "Some voxels out of bounds"

    # 儲存為 ply（每個 voxel 中心點位置）
    points = (coords + 0.5) / resolution - 0.5  # 將 voxel 格點轉為 [-0.5, 0.5] 空間座標
    ply_path = os.path.join(output_dir, "voxels", f"{output_name}.ply")
    utils3d.io.write_ply(ply_path, points)

    print(f"num_voxels: {len(vertices)}")
    print(f"voxelized mesh saved to: {ply_path}")
    return coords  # 你之後還能直接用這些座標來建構 occupancy tensor

def extract_feature_for_instance(instance_name, output_dir, model_name="dinov2_vitl14_reg", resolution=64):
    print(f"[INFO] Extracting DINO features for: {instance_name}")

    # Load DINOv2
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    model.eval().cuda()
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    n_patch = 518 // 14  # 14 是 DINO patch size
    

    # === Load transforms & image data ===
    render_dir = os.path.join(output_dir, 'renders', instance_name)
    voxel_path = os.path.join(output_dir, 'voxels', f'{instance_name}.ply')
    out_path = os.path.join(output_dir, 'features', model_name, f'{instance_name}.npz')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(os.path.join(render_dir, 'transforms.json')) as f:
        frames = json.load(f)['frames']
    
    images, extrinsics, intrinsics = [], [], []
    for view in frames:
        image_path = os.path.join(render_dir, view['file_path'])
        image = Image.open(image_path)
        image = image.resize((518, 518), Image.Resampling.LANCZOS)
        image = np.array(image).astype(np.float32) / 255
        image = image[:, :, :3] * image[:, :, 3:]  # apply alpha
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = transform(image)
        images.append(image)

        c2w = torch.tensor(view['transform_matrix'])
        c2w[:3, 1:3] *= -1
        extrinsics.append(torch.inverse(c2w))
        fov = view['camera_angle_x']
        intrinsics.append(utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov)))

    # === Load voxel positions ===
    positions = utils3d.io.read_ply(voxel_path)[0]
    positions = torch.from_numpy(positions).float().cuda()
    indices = ((positions + 0.5) * resolution).long()
    assert torch.all(indices >= 0) and torch.all(indices < resolution), "Some vertices are out of bounds"

    patchtokens_lst, uv_lst = [], []

    for i in tqdm(range(0, len(images), 16)):
        batch_images = torch.stack(images[i:i+16]).cuda()
        batch_ext = torch.stack(extrinsics[i:i+16]).cuda()
        batch_int = torch.stack(intrinsics[i:i+16]).cuda()

        features = model(batch_images, is_training=True)
        uv = utils3d.torch.project_cv(positions, batch_ext, batch_int)[0] * 2 - 1

        patchtokens = features['x_prenorm'][:, model.num_register_tokens + 1:]
        patchtokens = patchtokens.permute(0, 2, 1).reshape(len(batch_images), 1024, n_patch, n_patch)

        patchtokens_lst.append(patchtokens)
        uv_lst.append(uv)

    patchtokens = torch.cat(patchtokens_lst, dim=0)
    uv = torch.cat(uv_lst, dim=0)

    feat = F.grid_sample(patchtokens, uv.unsqueeze(1), mode='bilinear', align_corners=False)
    feat = feat.squeeze(2).permute(0, 2, 1).cpu().numpy()
    feat = np.mean(feat, axis=0).astype(np.float16)

    np.savez_compressed(out_path, indices=indices.cpu().numpy().astype(np.uint8), patchtokens=feat)
    print(f"[DONE] Saved to {out_path}")
    
    
def encode_slat_for_instance(
    instance_name,
    output_dir,
    feat_model="dinov2_vitl14_reg",
    enc_pretrained="microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16"
):
    print(f"[INFO] Encoding SLat for: {instance_name}")
    feat_path = os.path.join(output_dir, 'features', feat_model, f'{instance_name}.npz')
    assert os.path.exists(feat_path), f"Feature file not found: {feat_path}"
    
    out_dir = os.path.join(output_dir, 'latents', f"{feat_model}_{enc_pretrained.split('/')[-1]}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{instance_name}.npz')

    # === Load features ===
    data = np.load(feat_path)
    coords = torch.cat([
        torch.zeros(data['patchtokens'].shape[0], 1, dtype=torch.int32),  # batch=0
        torch.from_numpy(data['indices']).int()
    ], dim=1)
    feats = torch.from_numpy(data['patchtokens']).float()
    sparse = sp.SparseTensor(coords=coords, feats=feats).cuda()

    # === Load pretrained encoder ===
    encoder = models.from_pretrained(enc_pretrained).eval().cuda()

    # === Encode ===
    with torch.no_grad():
        slat = encoder(sparse, sample_posterior=False)

    # === Save ===
    np.savez_compressed(
        out_path,
        feats=slat.feats.cpu().numpy().astype(np.float32),
        coords=slat.coords[:, 1:].cpu().numpy().astype(np.uint8),  # 去掉 batch 維度
    )
    print(f"[DONE] SLat saved to: {out_path}")

if __name__ == "__main__":
    
    output_dir="/project2/siangling/TRELLIS/data"
    
    # ========== Step 1: Voxelize 3D Models ==========
    
    voxelize_single_mesh(
        mesh_path="/project2/siangling/TRELLIS/data/mesh.ply",
        output_dir="/project2/siangling/TRELLIS/data",
        output_name="mysample"
    )
    
    # ========== Step 2: Render Multiview Images ==========
    
    os.makedirs(os.path.join(output_dir, 'renders'), exist_ok=True)
    
    # install blender
    print('Checking blender...', flush=True)
    _install_blender()
    
    _render(
        file_path="/project2/siangling/TRELLIS/data/mesh.ply",
        sha256="mysample",  # 輸出會存到 .../renders/mysample/
        output_dir="/project2/siangling/TRELLIS/data",
        num_views=150
    )
    
    # ========== Step 3: Extract DINO Features ==========
    
    extract_feature_for_instance(
        instance_name="mysample",
        output_dir="/project2/siangling/TRELLIS/data",
        model_name="dinov2_vitl14_reg"
    )
    
    # path = "/project2/siangling/TRELLIS/data/features/dinov2_vitl14_reg/mysample.npz"
    # data = np.load(path)
    
    # print("=== keys ===")
    # print(data.files)
    
    # print("\n=== indices shape & example ===")
    # print(data['indices'].shape)
    # print(data['indices'][:5])
    
    # print("\n=== patchtokens shape & example ===")
    # print(data['patchtokens'].shape)
    # print(data['patchtokens'][:2])  # 顯示前兩個 voxel 的特徵
    
    
    # ========== Step 4: Encode SLat ==========
    
    encode_slat_for_instance(
        instance_name="mysample",
        output_dir="/project2/siangling/TRELLIS/data",
    )
    
    # path = "/project2/siangling/TRELLIS/data/latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16/mysample.npz"
    # data = np.load(path)
    
    # print("=== keys ===")
    # print(data.files)
    
    # print("\n=== feats shape & example ===")
    # print(data['feats'].shape)
    # print(data['feats'][:5])
    
    # print("\n=== coords shape & example ===")
    # print(data['coords'].shape)
    # print(data['coords'][:2])  # 顯示前兩個 voxel 的特徵
    
    
    
