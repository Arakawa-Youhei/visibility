
import os
import numpy as np
import torch
from tqdm import tqdm

def get_lambda_for_visibility(theta_deg: float, target_value=1e-3):
    theta_rad = np.radians(theta_deg)
    cos_theta = np.cos(theta_rad)
    return np.log(1.0 / target_value) / (1 - cos_theta)

def initialize_visSGs(J=5, theta_deg=60.0, mu_init=1.0):
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
    dirs_exp = dirs_local.unsqueeze(2)
    axes_exp = axes.unsqueeze(0).unsqueeze(0)
    lambdas_exp = lambdas.unsqueeze(0).unsqueeze(0)
    mus_exp = mus.unsqueeze(1)
    dot = torch.sum(dirs_exp * axes_exp, dim=-1, keepdim=True)
    G = mus_exp * torch.exp(lambdas_exp * (dot - 1.))
    G_sum = G.sum(dim=2).mean(dim=-1)
    loss = torch.nn.functional.mse_loss(G_sum, V_target)
    return loss

# ========== メイン処理 ==========
J = 5
theta_deg = 60.0
learning_rate = 5e-3
num_steps = 2000
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

npy_dir = "raytracing_results/202507111807/npy"
npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith(".npy") and f.startswith("vertex_")])
V_targets_np = [np.load(os.path.join(npy_dir, f)) for f in npy_files]
V_targets = torch.from_numpy(np.stack(V_targets_np)).float()
dirs_all = torch.from_numpy(np.load(os.path.join(npy_dir, "directions.npy"))).float()
normals_all = torch.from_numpy(np.load(os.path.join(npy_dir, "normals.npy"))).float()

V, D = V_targets.shape
dirs_all = dirs_all.unsqueeze(0).expand(V, -1, -1)

base_visSGs = initialize_visSGs(J=J, theta_deg=theta_deg)
axes_local = base_visSGs[:, :3].to(device)
lambdas = base_visSGs[:, 3:4].to(device)

mu_all = torch.nn.Parameter(torch.ones(V, J, 3, device=device))
optimizer = torch.optim.Adam([mu_all], lr=learning_rate)

for step in tqdm(range(num_steps)):
    total_loss = 0.0
    for i in range(0, V, batch_size):
        V_batch = slice(i, min(i + batch_size, V))
        dirs_batch = dirs_all[V_batch].to(device)
        normals_batch = normals_all[V_batch].to(device)
        dirs_local_batch = world_to_local(dirs_batch, normals_batch)

        V_targets_batch = V_targets[V_batch].to(device)
        mu_batch = mu_all[V_batch]

        loss = visibility_loss(V_targets_batch, mu_batch, dirs_local_batch, axes_local, lambdas)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / (V // batch_size + 1)
    if step % 100 == 0:
        print(f"Step {step}: Loss = {avg_loss:.6f}")
        print(f"  μ 平均: {mu_all.mean().item():.6f}  最大: {mu_all.max().item():.6f}")

os.makedirs("visibility", exist_ok=True)
torch.save(mu_all.detach().cpu(), "visibility/trained_mu_localSG.pt")
print("μ（ローカルSG）の学習完了。保存先: visibility/trained_mu_localSG.pt")
