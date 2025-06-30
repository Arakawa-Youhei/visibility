
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_sphere_directions(grid_size=100):
    phi = np.linspace(0, np.pi, grid_size)
    theta = np.linspace(0, 2 * np.pi, grid_size)
    phi, theta = np.meshgrid(phi, theta)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    dirs = np.stack([x, y, z], axis=-1)  # [H, W, 3]
    return dirs, phi, theta

def compute_visibility_map(visSG, dirs_flat):
    axes = visSG[:, :3]  # [J, 3]
    lambdas = visSG[:, 3:4]  # [J, 1]
    mus = visSG[:, 4:]  # [J, 3]
    axes = axes / (torch.norm(axes, dim=-1, keepdim=True) + 1e-6)

    dot = torch.sum(dirs_flat.unsqueeze(1) * axes.unsqueeze(0), dim=-1, keepdim=True)  # [N, J, 1]
    sg = mus.unsqueeze(0) * torch.exp(lambdas.unsqueeze(0) * (dot - 1.0))  # [N, J, 3]
    visibility = sg.sum(dim=1).mean(dim=-1)  # [N]
    return visibility

# 入力読み込み
visSGs_all = torch.load("visibility/visSGs_all.pt")  # [V, J, 7]
V = visSGs_all.shape[0]

# グリッド生成
grid_size = 100
dirs, phi, theta = generate_sphere_directions(grid_size)
dirs = torch.tensor(dirs, dtype=torch.float32)  # [H, W, 3]
H, W, _ = dirs.shape
dirs_flat = dirs.view(-1, 3)  # [N, 3]

# 出力ディレクトリ
output_dir = "visibility/sg_visibility_images"
os.makedirs(output_dir, exist_ok=True)

# 可視化
for i in tqdm(range(V), desc="Visualizing SG visibility"):
    visSG = visSGs_all[i]  # [J, 7]
    visibility = compute_visibility_map(visSG, dirs_flat)  # [H*W]
    visibility = visibility.view(H, W).cpu().numpy()

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(visibility, cmap='inferno', origin='lower',
                   extent=[0, 360, 0, 90], aspect='auto')
    ax.set_title(f"SG Visibility Map (Vertex {i})")
    ax.set_xlabel("Azimuth φ (deg)")
    ax.set_ylabel("Polar θ (deg)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Approximated Visibility")

    fig.savefig(f"{output_dir}/sg_vis_{i:05d}.png")
    plt.close(fig)
