
import os
import numpy as np
import torch
from tqdm import tqdm

# =========================
# vis_sg_init.py の内容（統合）
# =========================
def get_lambda_for_visibility(theta_deg: float, target_value=1e-3):
    theta_rad = np.radians(theta_deg)
    cos_theta = np.cos(theta_rad)
    return np.log(1.0 / target_value) / (1 - cos_theta)

def initialize_visSGs(J=5, theta_deg=45.0, mu_init=1.0):
    theta_phi = [
        (0.0, 0.0),               # 上方向
        (np.pi/4, 0.0),           # 前斜め
        (np.pi/4, np.pi/2),       # 右斜め
        (np.pi/4, np.pi),         # 後斜め
        (np.pi/4, 3*np.pi/2),     # 左斜め
    ]
    axes = []
    for theta, phi in theta_phi:
        x = np.sin(theta) * np.cos(phi)
        y = np.cos(theta)
        z = np.sin(theta) * np.sin(phi)
        axes.append([x, y, z])
    axes = torch.tensor(axes, dtype=torch.float32)  # [J, 3]
    lambda_val = get_lambda_for_visibility(theta_deg)
    lambdas = torch.full((len(axes), 1), lambda_val, dtype=torch.float32)
    if isinstance(mu_init, (int, float)):
        mus = torch.full((len(axes), 3), mu_init, dtype=torch.float32)
    else:
        mus = torch.tensor(mu_init, dtype=torch.float32).expand(len(axes), 3)
    visSGs = torch.cat([axes, lambdas, mus], dim=1)
    return visSGs

# =========================
# ローカルSGによるμの学習コード
# =========================

# パラメータ
J = 5
theta_deg = 45.0
learning_rate = 1e-2
num_steps = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"

# npyの読み込み
npy_dir = "visibility/npy_data"
npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith(".npy") and f.startswith("vertex_")])
V_targets_np = [np.load(os.path.join(npy_dir, f)) for f in npy_files]
V_targets = torch.from_numpy(np.stack(V_targets_np)).float().to(device)  # [V, D]
dirs = torch.from_numpy(np.load(os.path.join(npy_dir, "directions.npy"))).float().to(device)  # [D, 3]
normals = torch.from_numpy(np.load(os.path.join(npy_dir, "normals.npy"))).float().to(device)  # [V, 3]

V, D = V_targets.shape
dirs = dirs.unsqueeze(0).expand(V, -1, -1)  # [V, D, 3]
normals = normals / normals.norm(dim=1, keepdim=True)

# ローカル座標系への変換
def world_to_local(dirs, normals):
    up = torch.tensor([0.0, 1.0, 0.0], device=dirs.device).expand_as(normals)
    mask = torch.abs(normals[:, 1]) >= 0.99
    up[mask] = torch.tensor([1.0, 0.0, 0.0], device=dirs.device)

    z = normals
    x = torch.nn.functional.normalize(torch.cross(up, z), dim=1)
    y = torch.cross(z, x)
    R = torch.stack([x, y, z], dim=2)  # [V, 3, 3]
    dirs_local = torch.bmm(dirs, R)  # [V, D, 3]
    return dirs_local

dirs_local = world_to_local(dirs, normals)

# SG初期化（ローカル方向）
base_visSGs = initialize_visSGs(J=J, theta_deg=theta_deg)
axes_local = base_visSGs[:, :3].to(device)      # [J, 3]
lambdas = base_visSGs[:, 3:4].to(device)        # [J, 1]

# 学習パラメータ（μのみ学習）
mu_all = torch.nn.Parameter(torch.ones(V, J, 3, device=device))
optimizer = torch.optim.Adam([mu_all], lr=learning_rate)

# ロス関数
def visibility_loss(V_target, mus, dirs_local, axes, lambdas):
    V, D, _ = dirs_local.shape
    dirs_exp = dirs_local.unsqueeze(2)          # [V, D, 1, 3]
    axes_exp = axes.unsqueeze(0).unsqueeze(0)   # [1, 1, J, 3]
    lambdas_exp = lambdas.unsqueeze(0).unsqueeze(0)  # [1, 1, J, 1]
    mus_exp = mus.unsqueeze(1)                  # [V, 1, J, 3]

    dot = torch.sum(dirs_exp * axes_exp, dim=-1, keepdim=True)  # [V, D, J, 1]
    G = mus_exp * torch.exp(lambdas_exp * (dot - 1.))           # [V, D, J, 3]
    G_sum = G.sum(dim=2).mean(dim=-1)                           # [V, D]
    loss = torch.nn.functional.mse_loss(G_sum, V_target)
    return loss

# 学習ループ
for step in tqdm(range(num_steps)):
    loss = visibility_loss(V_targets, mu_all, dirs_local, axes_local, lambdas)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"Step {step}: Loss = {loss.item():.6f}")

# 保存
os.makedirs("visibility", exist_ok=True)
torch.save(mu_all.detach().cpu(), "visibility/trained_mu_localSG.pt")
print("μ（ローカルSG）の学習完了。保存先: visibility/trained_mu_localSG.pt")
