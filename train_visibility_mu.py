
import numpy as np
import torch
from tqdm import tqdm
from vis_sg_init import initialize_visSGs

# パラメータ
J = 5                     # SGの数
theta_deg = 45.0          # SGのローブ開き角
learning_rate = 1e-2
num_steps = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"

# 遮蔽マップと方向を読み込む
# ↓ 変更開始：頂点ごとの .npy を読み込むように変更
import os
npy_dir = "raytracing_results/202507111807/npy"  # 出力先に合わせて変更
npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy') and f.startswith('vertex_')])
all_data = [np.load(os.path.join(npy_dir, f)) for f in npy_files]
V_targets = torch.from_numpy(np.stack(all_data)).float().to(device)  # [V, D]
# ↑ 変更終了
dirs = torch.from_numpy(np.load(os.path.join(npy_dir, 'directions.npy'))).float().to(device)

V, D = V_targets.shape
mu_all = torch.nn.Parameter(torch.ones(V, J, 3, device=device))  # 学習対象

optimizer = torch.optim.Adam([mu_all], lr=learning_rate)

# 可視関数の近似損失
def visibility_loss(V_target, mus, dirs, vis_axes, vis_lambdas):
    dirs = dirs.unsqueeze(0).unsqueeze(-2)       # [1, D, 1, 3]
    axes = vis_axes.unsqueeze(0).unsqueeze(1)    # [1, 1, J, 3]
    lambdas = vis_lambdas.unsqueeze(0).unsqueeze(1)  # [1, 1, J, 1]
    mus = mus.unsqueeze(1)                       # [V, 1, J, 3]

    dot = torch.sum(dirs * axes, dim=-1, keepdim=True)  # [1, D, J, 1]
    G = mus * torch.exp(lambdas * (dot - 1.))           # [V, D, J, 3]
    G_sum = G.sum(dim=2).mean(dim=-1)                   # [V, D]
    loss = torch.nn.functional.mse_loss(G_sum, V_target)
    return loss

# SG構造（軸とλ）を固定して初期化
base_visSGs = initialize_visSGs(J=J, theta_deg=theta_deg)  # [J, 7]
vis_axes = base_visSGs[:, :3].to(device)       # [J, 3]
vis_lambdas = base_visSGs[:, 3:4].to(device)   # [J, 1]

# 学習ループ
for step in tqdm(range(num_steps)):
    loss = visibility_loss(V_targets, mu_all, dirs, vis_axes, vis_lambdas)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}: Loss = {loss.item():.6f}")

# 学習済みμを保存
torch.save(mu_all.detach().cpu(), "visibility/trained_mu.pt")
print("μ の学習完了。saved to: visibility/trained_mu.pt")
