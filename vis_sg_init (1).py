
import numpy as np
import torch

def get_lambda_for_visibility(theta_deg: float, target_value=1e-3):
    """
    θ度離れた方向で球面ガウス関数の値がtarget_valueになるようなλを返す
    """
    theta_rad = np.radians(theta_deg)
    cos_theta = np.cos(theta_rad)
    return np.log(1.0 / target_value) / (1 - cos_theta)

def initialize_visSGs(J=5, theta_deg=45.0, mu_init=1.0):
    """
    可視関数用のSGを半球上に配置して初期化する
    - J: SGの数（5個推奨）
    - theta_deg: λの計算に使う開き角度（例: 45°で λ ≈ 26.6）
    - mu_init: μの初期値（floatまたは[3]のTensor）
    """
    # 固定方向（θ, φ）5方向
    theta_phi = [
        (0.0, 0.0),               # G1: 上方向
        (np.pi/4, 0.0),           # G2: 前斜め
        (np.pi/4, np.pi/2),       # G3: 右斜め
        (np.pi/4, np.pi),         # G4: 後斜め
        (np.pi/4, 3*np.pi/2),     # G5: 左斜め
    ]

    # 各方向の単位ベクトルを計算
    axes = []
    for theta, phi in theta_phi:
        x = np.sin(theta) * np.cos(phi)
        y = np.cos(theta)
        z = np.sin(theta) * np.sin(phi)
        axes.append([x, y, z])
    axes = torch.tensor(axes, dtype=torch.float32)  # [J, 3]

    # λをθから自動計算
    lambda_val = get_lambda_for_visibility(theta_deg)
    lambdas = torch.full((J, 1), lambda_val, dtype=torch.float32)  # [J, 1]

    # μの初期値を設定
    if isinstance(mu_init, (int, float)):
        mus = torch.full((J, 3), mu_init, dtype=torch.float32)
    else:
        mus = torch.tensor(mu_init, dtype=torch.float32).expand(J, 3)

    # SG形式 [J, 7]： [axis (3), lambda (1), mu (3)]
    visSGs = torch.cat([axes, lambdas, mus], dim=1)  # [J, 7]
    return visSGs
