
import torch
import os
import numpy as np
from vis_sg_init import initialize_visSGs

# パラメータ
J = 5
theta_deg = 45.0
mu_path = "visibility/trained_mu.pt"
output_path = "visibility/visSGs_all.pt"

# μ を読み込む
mu_all = torch.load(mu_path)  # [V, J, 3]
V = mu_all.shape[0]

# 各頂点の SG を構築
visSGs_all = []
for i in range(V):
    visSG = initialize_visSGs(J=J, theta_deg=theta_deg)  # [J, 7]
    visSG[:, 4:] = mu_all[i]  # μ を埋め込む
    visSGs_all.append(visSG)

visSGs_all = torch.stack(visSGs_all)  # [V, J, 7]

# 保存
os.makedirs("visibility", exist_ok=True)
torch.save(visSGs_all, output_path)
print(f"Saved: {output_path} (shape: {visSGs_all.shape})")
