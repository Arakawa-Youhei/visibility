import torch
import numpy as np
import matplotlib.pyplot as plt
import os
# ====== SG初期化（Z軸基準） ======
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

# ====== 可視化関数 ======
def visualize_visibility(vertex_id=0, mu_path="visibility/trained_mu_localSG.pt", theta_deg=45.0, res=100, output_dir="visibility/visibility_maps"):
    mu_all = torch.load(mu_path)  # [V, J, 3]
    mu = mu_all[vertex_id]        # [J, 3]

    axes, lambdas = initialize_visSGs(J=mu.shape[0], theta_deg=theta_deg)
    
    theta = torch.linspace(0, np.pi / 2, res)
    phi = torch.linspace(0, 2 * np.pi, res)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')

    # 球面方向ベクトル [res, res, 3]
    x = torch.sin(theta_grid) * torch.cos(phi_grid)
    y = torch.sin(theta_grid) * torch.sin(phi_grid)
    z = torch.cos(theta_grid)
    dirs = torch.stack([x, y, z], dim=-1)  # [res, res, 3]

    # 可視関数の評価
    V = torch.zeros_like(theta_grid)
    for j in range(mu.shape[0]):
        a = axes[j]
        lam = lambdas[j]
        dot = torch.sum(dirs * a, dim=-1, keepdim=True)  # [res, res, 1]
        G = mu[j].mean() * torch.exp(lam * (dot - 1.0))  # RGB平均で評価
        V += G.squeeze(-1)

    # 可視化
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"visibility_map_vertex{vertex_id}.png")
    plt.figure(figsize=(6, 5))
    im = plt.imshow(V.cpu().numpy(), origin="lower", extent=[0, 360, 0, 90], cmap='viridis')
    plt.xlabel("phi (°)")
    plt.ylabel("theta (°)")
    plt.title(f"近似可視関数（頂点 {vertex_id}）")
    plt.colorbar(im, label="Visibility")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"保存しました: {out_path}")

# 実行例
if __name__ == "__main__":
    visualize_visibility(vertex_id=0, output_dir="visibility/visibility_maps")
