import open3d as o3d
import torch
import numpy

def visualize_coords(coords: torch.Tensor):
    """
    coords: shape [N, 4] tensor, where [:, 1:] are [z, y, x]
    """

    xyz = coords[:, 1:].cpu().numpy().astype(float)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)

    o3d.io.write_point_cloud("slat_coords.ply", pc)