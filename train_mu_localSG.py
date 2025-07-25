import os
import numpy as np
import torch
from tqdm import tqdm

def get_lambda_for_visibility(theta_deg: float, target_value=1e-3):
    theta_rad = np.radians(theta_deg)
    cos_theta = np.cos(theta_rad)
    return np.log(1.0 / target_value) / (1 - cos_theta)

def initialize_visSGs(J=5, theta_deg=45.0, mu_init=1.0):
    theta_phi = [
        (0.0, 0.0),
        (np.pi/4, 0.0),
        (np.pi/4, np.pi/2),
        (np.pi/4, np.pi),
        (np.pi/4, 3*np.pi/2),
    ]
    axes = []
    for theta, phi in theta_phi:
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        axes.append([x, y, z])
    axes = torch.tensor(axes, dtype=torch.float32)
    lambda_val = get_lambda_for_visibility(theta_deg)
    lambdas = torch.full((len(axes), 1), lambda_val, dtype=torch.float32)
    mus = torch.full((len(axes), 3), mu_init, dtype=torch.float32)
    visSGs = torch.cat([axes, lambdas, mus], dim=1)
    return visSGs

def world_to_local(dirs, normals):
    up = torch.tensor([0.0, 1.0, 0.0], device=dirs.device).expand_as(normals)
    mask = torch.abs(normals[:, 1]) >= 0.99
    up[mask] = torch.tensor([1.0, 0.0, 0.0], device=dirs.device)
    z = normals
    x = torch.nn.functional.normalize(torch.cross(up, z), dim=1)
    y = torch.cross(z, x)
    R = torch.stack([x, y, z], dim=2)  # [B, 3, 3]
    return torch.bmm(dirs, R)          # [B, D, 3]

def visibility_loss(V_target, mus, dirs_local, axes, lambdas):
    V, D, _ = dirs_local.shape
    dirs_exp = dirs_local.unsqueeze(2)     # [V, D, 1, 3]
    axes_exp = axes.unsqueeze(0).unsqueeze(0)     # [1, 1, J, 3]
    lambdas_exp = lambdas.unsqueeze(0).unsqueeze(0)   # [1, 1, J, 1]
    mus_exp = mus.unsqueeze(1)     # [V, 1, J, 3]
    dot = torch.sum(dirs_exp * axes_exp, dim=-1, keepdim=True)  # [V, D, J, 1]
    G = mus_exp * torch.exp(lambdas_exp * (dot - 1.))  # [V, D, J, 3]
    G_sum = G.sum(dim=2).mean(dim=-1)  # [V, D, 3]
    loss = torch.nn.functional.mse_loss(G_sum.mean(dim=-1), V_target.squeeze(1))
    return loss

# ========== メイン処理 ==========

J = 5
theta_deg = 45.0
learning_rate = 5e-3
num_steps = 2000
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

# データ読み込み
npy_dir = "raytracing_results/202507111807/npy"
npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith(".npy") and f.startswith("vertex_")])
V_targets_np = [np.load(os.path.join(npy_dir, f)) for f in npy_files]
V_targets = torch.from_numpy(np.stack(V_targets_np)).float()  # [V, D]
dirs_all = torch.from_numpy(np.load(os.path.join(npy_dir, "directions.npy"))).float()  # [D, 3]
normals_all = torch.from_numpy(np.load(os.path.join(npy_dir, "normals.npy"))).float()  # [V, 3]

V, D = V_targets.shape
dirs_all = dirs_all.unsqueeze(0).expand(V, -1, -1)  # [V, D, 3]

# SG方向とλは固定
base_visSGs = initialize_visSGs(J=J, theta_deg=theta_deg)
axes_local = base_visSGs[:, :3].to(device)      # [J, 3]
lambdas = base_visSGs[:, 3:4].to(device)        # [J, 1]

# μを初期化
mu_all = torch.zeros(V, J, 3, device=device)

print("バッチごとのμ学習を開始...")
for i in tqdm(range(0, V, batch_size)):
    v_batch = slice(i, min(i + batch_size, V))
    batch_V = v_batch.stop - v_batch.start

    dirs_batch = dirs_all[v_batch].to(device)          # [B, D, 3]
    normals_batch = normals_all[v_batch].to(device)    # [B, 3]
    dirs_local = world_to_local(dirs_batch, normals_batch)  # [B, D, 3]
    V_target_batch = V_targets[v_batch].to(device)     # [B, D]

    mu = torch.nn.Parameter(torch.ones(batch_V, J, 3, device=device)*0.2 + 0.9)  # [B, J, 3]
    optimizer = torch.optim.Adam([mu], lr=learning_rate)

    for step in range(num_steps):
        optimizer.zero_grad()
        loss = visibility_loss(V_target_batch, mu, dirs_local, axes_local, lambdas)
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0 or step == num_steps - 1:
        print(f"バッチ {i} ～ {v_batch.stop - 1}, Step {step}: Loss = {loss.item():.6f}")
    mu_all[v_batch] = mu.detach()

# 保存
os.makedirs("visibility", exist_ok=True)
torch.save(mu_all.cpu(), "visibility/trained_mu_localSG.pt")
print("μ（ローカルSG）のバッチ学習完了。保存先: visibility/trained_mu_localSG.pt")
