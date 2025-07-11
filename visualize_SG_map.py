
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def get_lambda_for_visibility(theta_deg, target_value=1e-3):
    theta = np.radians(theta_deg)
    return np.log(1 / target_value) / (1 - np.cos(theta))

def initialize_visSGs(J=5, theta_deg=45.0):
    theta_phi = [
        (0.0, 0.0),
        (np.pi/4, 0.0),
        (np.pi/4, np.pi/2),
        (np.pi/4, np.pi),
        (np.pi/4, 3*np.pi/2)
    ]
    axes = []
    for theta, phi in theta_phi:
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        axes.append([x, y, z])
    axes = np.array(axes)  # [J, 3]
    lambdas = np.full((len(axes), 1), get_lambda_for_visibility(theta_deg))
    return torch.tensor(axes, dtype=torch.float32), torch.tensor(lambdas, dtype=torch.float32)

def compute_vmap(mu, axes, lambdas, dirs):
    V_map = torch.zeros(dirs.shape[:2])
    for j in range(mu.shape[0]):
        a = axes[j]
        lam = lambdas[j]
        dot = torch.sum(dirs * a, dim=-1, keepdim=True)
        G = mu[j].mean() * torch.exp(lam * (dot - 1.0))
        V_map += G.squeeze(-1)
    return V_map

def visualize_all_vertices(mu_path="visibility/trained_mu_localSG.pt", theta_deg=45.0, res=100, output_dir="results/visibility_maps"):
    mu_all = torch.load(mu_path)  # [V, J, 3]
    V = mu_all.shape[0]

    axes, lambdas = initialize_visSGs(J=mu_all.shape[1], theta_deg=theta_deg)

    theta = torch.linspace(0, np.pi / 2, res)
    phi = torch.linspace(0, 2 * np.pi, res)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')

    x = torch.sin(theta_grid) * torch.cos(phi_grid)
    y = torch.sin(theta_grid) * torch.sin(phi_grid)
    z = torch.cos(theta_grid)
    dirs = torch.stack([x, y, z], dim=-1)  # [res, res, 3]

    os.makedirs(output_dir, exist_ok=True)

    # ===== 1. 全V_mapから共通vmin/vmaxを決定 =====
    print("グローバル最小/最大のスキャン中...")
    all_min, all_max = float("inf"), float("-inf")
    for mu in mu_all:
        V_map = compute_vmap(mu, axes, lambdas, dirs)
        all_min = min(all_min, V_map.min().item())
        all_max = max(all_max, V_map.max().item())
    print(f"全頂点共通スケール: vmin = {all_min:.6f}, vmax = {all_max:.6f}")

    # ===== 2. 可視化ループ =====
    for vertex_id in range(V):
        mu = mu_all[vertex_id]
        V_map = compute_vmap(mu, axes, lambdas, dirs)

        out_path = os.path.join(output_dir, f"visibility_map_vertex{vertex_id:04d}.png")
        fig = plt.figure(figsize=(6, 5))
        im = plt.imshow(V_map.cpu().numpy(), origin="lower", extent=[0, 360, 0, 90],
                        cmap='viridis', vmin=all_min, vmax=all_max)
        plt.xlabel("phi (°)")
        plt.ylabel("theta (°)")
        plt.title(f"可視関数の近似（頂点 {vertex_id}）")
        plt.colorbar(im, label="Visibility")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        print(f"保存しました: {out_path}")

if __name__ == "__main__":
    visualize_all_vertices()
