
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# 可視SG読み込み
visSGs_all = torch.load("visibility/visSGs_all.pt")  # [V, J, 7]
J = visSGs_all.shape[1]

# 法線読み込み（例：.npyなどの外部ファイル）
normals = np.load("visibility/normals.npy")  # [V, 3], 各頂点の法線ベクトル

# 出力先
out_dir = "visibility/sg_vis_local_images"
os.makedirs(out_dir, exist_ok=True)

# 回転行列作成（任意ベクトルをY軸へ）
def get_rotation_to_y(normal):
    normal = normal / np.linalg.norm(normal)
    target = np.array([0, 1, 0])
    axis = np.cross(normal, target)
    if np.linalg.norm(axis) < 1e-6:
        return np.eye(3)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(normal, target), -1.0, 1.0))
    return R.from_rotvec(axis * angle).as_matrix()

# グリッド生成（θ: 0~π/2, φ: 0~2π）
grid_size = 64
theta = np.linspace(0, 0.5 * np.pi, grid_size)
phi = np.linspace(0, 2 * np.pi, grid_size)
theta, phi = np.meshgrid(theta, phi)
dirs = np.stack([
    np.sin(theta) * np.cos(phi),
    np.cos(theta),
    np.sin(theta) * np.sin(phi)
], axis=-1)  # [H, W, 3]
dirs = dirs.reshape(-1, 3)  # [N, 3]
dirs_tensor = torch.tensor(dirs, dtype=torch.float32)  # [N, 3]

# 可視化
for i in tqdm(range(visSGs_all.shape[0]), desc="Rendering SG visibility"):
    visSG = visSGs_all[i].numpy()  # [J, 7]
    normal = normals[i]            # [3]
    R_y = get_rotation_to_y(normal)  # [3, 3]

    axes = visSG[:, :3] @ R_y.T  # SG軸をローカルY軸座標へ変換
    lambdas = visSG[:, 3:4]
    mus = visSG[:, 4:]  # [J, 3]

    dot = dirs @ axes.T  # [N, J]
    dot = np.clip(dot, -1, 1)
    sg_vals = np.exp(lambdas.T * (dot - 1.0))  # [N, J]
    rgb = sg_vals @ mus  # [N, 3]
    rgb = rgb.reshape(grid_size, grid_size, 3)
    rgb = np.clip(rgb, 0.0, 1.0)

    plt.figure(figsize=(5, 5))
    plt.imshow(rgb, origin='lower', extent=[0, 360, 0, 90])
    plt.xlabel("Azimuth φ (deg)")
    plt.ylabel("Elevation θ (deg)")
    plt.title(f"Vertex {i} SG Visibility (Local Y-up)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/sg_local_vis_{i:05d}.png")
    plt.close()
