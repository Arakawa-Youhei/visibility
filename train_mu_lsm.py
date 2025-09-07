
"""
可視性を球面ガウス(SG)の和で近似し、各頂点の μ を
最小二乗（リッジ最小二乗も可）で一発解するスクリプト。

数式：
  g_j(d_k) = exp(λ_j (a_j^T d_k - 1))
  G_{k j} = g_j(d_k)
  最小二乗:  minimize ||G μ - B||_F^2
  リッジ   :  minimize ||G μ - B||_F^2 + α ||μ||_F^2
  解       :  μ* = (G^T G + α I)^{-1} G^T B
  実装     :  torch.linalg.lstsq で (A x ≈ B) をバッチ解
              （リッジは A=[G; sqrt(α)I], B=[b; 0] としてダミー行付加）
"""

import os
import numpy as np
import torch
from tqdm import tqdm


# ===================== ユーティリティ関数 =====================

def get_lambda_for_visibility(theta_deg: float, target_value: float = 1e-3) -> float:
    """
    半値開口角 theta_deg（目安）から SG のシャープネス λ を推定。
    exp(λ (cos θ - 1)) ≈ target_value となる λ を返す。
    """
    theta_rad = np.radians(theta_deg)
    cos_theta = np.cos(theta_rad)
    return float(np.log(1.0 / target_value) / (1.0 - cos_theta))


def initialize_visSGs(J: int = 5, theta_deg: float = 45.0, mu_init: float = 1.0) -> torch.Tensor:
    """
    J=5 固定の軸配置（頂点1つ＋周囲4つ）で可視性 SG を初期化。
    返り値: [J, 7] = [axis(3), lambda(1), mu(3)] ただし mu はダミー。
    """
    # 固定の方向（1本は北極、4本は45度で等間隔）
    theta_phi = [
        (0.0, 0.0),
        (np.pi / 4, 0.0),
        (np.pi / 4, np.pi / 2),
        (np.pi / 4, np.pi),
        (np.pi / 4, 3 * np.pi / 2),
    ]
    axes = []
    for theta, phi in theta_phi:
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        axes.append([x, y, z])
    axes = torch.tensor(axes, dtype=torch.float32)           # [J, 3]
    axes = torch.nn.functional.normalize(axes, dim=-1)       # 念のため正規化

    lambda_val = get_lambda_for_visibility(theta_deg)
    lambdas = torch.full((axes.shape[0], 1), lambda_val, dtype=torch.float32)  # [J, 1]
    mus = torch.full((axes.shape[0], 3), mu_init, dtype=torch.float32)         # [J, 3]（未使用）

    visSGs = torch.cat([axes, lambdas, mus], dim=1)  # [J, 7]
    return visSGs


def world_to_local(dirs: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """
    世界→ローカル（Z=法線，上向きはYを優先、法線がYに近いときはXを代替Up）
    dirs:    [B, D, 3]
    normals: [B, 3]
    return:  [B, D, 3]
    """
    B = normals.shape[0]
    device = dirs.device

    # デフォルトUpはY
    up = torch.tensor([0.0, 1.0, 0.0], device=device).expand(B, -1).clone()  # [B, 3]
    # 法線がYとほぼ平行なら X をUpに切替
    mask = torch.abs(normals[:, 1]) >= 0.99
    up[mask] = torch.tensor([1.0, 0.0, 0.0], device=device)

    z = torch.nn.functional.normalize(normals, dim=-1)              # [B, 3]
    x = torch.nn.functional.normalize(torch.cross(up, z), dim=-1)   # [B, 3]
    y = torch.cross(z, x)                                           # [B, 3]

    # R: [B, 3, 3] （列が基底ベクトル x,y,z）
    R = torch.stack([x, y, z], dim=2)
    # dirs_local = dirs * R
    return torch.bmm(dirs, R)  # [B, D, 3]


# ===================== 最小二乗（バッチ）本体 =====================

def solve_mu_least_squares(
    dirs_local: torch.Tensor,   # [B, D, 3]
    targets: torch.Tensor,      # [B, D]  (0/1 可視性)
    axes_local: torch.Tensor,   # [J, 3]
    lambdas: torch.Tensor,      # [J, 1]
    ridge_alpha: float = 1e-4,  # 0.0 で純最小二乗
) -> (torch.Tensor, float):
    """
    各バッチ B について、最小二乗 (リッジ可) で μ を一発解。
    返り値:
      mu:  [B, J, 3]
      mse: バッチの再構成MSE（参考表示用）
    数式：
      G[b, k, j] = exp( λ_j * (a_j^T d_{b,k} - 1) )
      solve (G μ ≈ B), B = [b, b, b]  (3ch同時)
    """
    device = dirs_local.device
    Bsz, D, _ = dirs_local.shape
    J = axes_local.shape[0]

    # a_j^T d_{b,k} → [B, D, J]
    dots = torch.matmul(dirs_local, axes_local.T)  # [B, D, J]
    # G = exp(λ_j (dots - 1))
    lam = lambdas.view(1, 1, J)                    # [1,1,J]
    G = torch.exp(lam * (dots - 1.0))              # [B, D, J]

    # B = [b, b, b]
    Bmat = targets.unsqueeze(-1).expand(Bsz, D, 3)  # [B, D, 3]

    if ridge_alpha > 0.0:
        # A = [G; sqrt(α) I],  B = [b; 0]
        I = torch.eye(J, device=device).unsqueeze(0).expand(Bsz, J, J)                # [B, J, J]
        sqrt_alpha = torch.sqrt(torch.tensor(ridge_alpha, device=device))
        A_aug = torch.cat([G, sqrt_alpha * I], dim=1)                                  # [B, D+J, J]
        zeros_reg = torch.zeros(Bsz, J, 3, device=device)                              # [B, J, 3]
        B_aug = torch.cat([Bmat, zeros_reg], dim=1)                                     # [B, D+J, 3]
        X = torch.linalg.lstsq(A_aug, B_aug).solution                                   # [B, J, 3]
    else:
        X = torch.linalg.lstsq(G, Bmat).solution                                        # [B, J, 3]

    # 参考 MSE（3ch平均を可視性スカラーとみなす）
    pred = torch.matmul(G, X).mean(dim=-1)  # [B, D]
    mse = torch.mean((pred - targets) ** 2).item()
    return X, mse


# ===================== メイン =====================

def main():
    # --------- 設定 ----------
    J = 5
    theta_deg = 45.0
    batch_size = 64
    ridge_alpha = 1e-4      # 0.0 で純最小二乗。1e-6~1e-3 を推奨
    npy_dir = "frog3c_raytracing/20250722/npy"
    out_dir = "frog3c_visibility"
    out_path = os.path.join(out_dir, "trained_mu_localSG.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high") if device == "cuda" else None

    # --------- データ読み込み ----------
    npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith(".npy") and f.startswith("vertex_")])
    if len(npy_files) == 0:
        raise FileNotFoundError(f"No 'vertex_*.npy' found in: {npy_dir}")

    V_targets_np = [np.load(os.path.join(npy_dir, f)) for f in npy_files]  # 各頂点: [D]
    V_targets = 1.0 - torch.from_numpy(np.stack(V_targets_np)).float()     # [V, D]（非衝突=1に変換）
    dirs_all_np = np.load(os.path.join(npy_dir, "directions.npy"))         # [D, 3]
    normals_all_np = np.load(os.path.join(npy_dir, "normals.npy"))         # [V, 3]

    dirs_D3 = torch.from_numpy(dirs_all_np).float()        # [D, 3]
    normals_all = torch.from_numpy(normals_all_np).float() # [V, 3]
    V, D = V_targets.shape

    # 各頂点に同じ方向集合を持たせる
    dirs_all = dirs_D3.unsqueeze(0).expand(V, -1, -1)      # [V, D, 3]

    # --------- SG方向とλ（固定） ----------
    base_visSGs = initialize_visSGs(J=J, theta_deg=theta_deg)
    axes_local = base_visSGs[:, :3].to(device)     # [J, 3]
    lambdas = base_visSGs[:, 3:4].to(device)       # [J, 1]

    # --------- 出力バッファ ----------
    mu_all = torch.zeros(V, J, 3, device=device)

    # --------- バッチ最小二乗 ----------
    print("バッチごとの μ を最小二乗で一括解します...")
    for i in tqdm(range(0, V, batch_size)):
        v_batch = slice(i, min(i + batch_size, V))
        Bsz = v_batch.stop - v_batch.start

        dirs_batch = dirs_all[v_batch].to(device)         # [B, D, 3]
        normals_batch = normals_all[v_batch].to(device)   # [B, 3]
        dirs_local = world_to_local(dirs_batch, normals_batch)   # [B, D, 3]
        b = V_targets[v_batch].to(device)                 # [B, D]

        mu_batch, mse = solve_mu_least_squares(
            dirs_local=dirs_local,
            targets=b,
            axes_local=axes_local,
            lambdas=lambdas,
            ridge_alpha=ridge_alpha,
        )
        mu_all[v_batch] = mu_batch.detach()

        print(f"  バッチ {i:>6} ～ {v_batch.stop - 1:<6} | MSE = {mse:.6f}")

    # --------- 保存 ----------
    os.makedirs(out_dir, exist_ok=True)
    torch.save(mu_all.cpu(), out_path)
    print(f"μ（ローカルSG）の最小二乗解を保存: {out_path}")


if __name__ == "__main__":
    main()
