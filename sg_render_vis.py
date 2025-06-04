import torch
import numpy as np

# 小さな定数（ゼロ除算や数値安定性対策）
TINY_NUMBER = 1e-6

# 光源SGと可視性SGを合成（Hadamard積に相当）
def apply_visibility_sg(lgtSGs, visSGs):
    # M: 光源SGの個数, J: 可視性SGの個数
    M = lgtSGs.shape[0]
    J = visSGs.shape[0]

    # 全組み合わせでブロードキャスト展開（M x J x 7）
    lgtSGs_exp = lgtSGs.unsqueeze(1).expand(M, J, 7)
    visSGs_exp = visSGs.unsqueeze(0).expand(M, J, 7)

    # 各SGの方向(xi)、スケール(mu)、シャープネス(lambda) を分解
    xi1, lam1, mu1 = lgtSGs_exp[..., :3], torch.abs(lgtSGs_exp[..., 3:4]), torch.abs(lgtSGs_exp[..., 4:])
    xi2, lam2, mu2 = visSGs_exp[..., :3], torch.abs(visSGs_exp[..., 3:4]), torch.abs(visSGs_exp[..., 4:])

    # 合成後のシャープネス
    lam3 = torch.norm(lam1 * xi1 + lam2 * xi2, dim=-1, keepdim=True) + TINY_NUMBER
    # 合成後の方向ベクトル
    xi3 = (lam1 * xi1 + lam2 * xi2) / lam3
    # 合成後のスケール（エネルギー）
    mu3 = mu1 * mu2 * torch.exp(lam3 - lam1 - lam2)

    # 合成されたSGの形に戻す（M*J x 7）
    return torch.cat([xi3, lam3, mu3], dim=-1).view(-1, 7)

# 可視性SGをfitする関数（回帰的に最適化）
def fit_visibility_sg(visibility_fn, viewdirs, J=5, fixed_lamb=32.0):
    N = viewdirs.shape[0]
    device = viewdirs.device

    # θ: 仰角, φ: 方位角 を使って均等な方向を生成
    theta = torch.linspace(0, np.pi / 2, J, device=device)
    phi = torch.linspace(-np.pi, np.pi, J, device=device)
    dirs = torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.cos(theta),
        torch.sin(theta) * torch.sin(phi)
    ], dim=-1)[:J]

    # 正規化
    dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + TINY_NUMBER)

    # 学習対象は mu のみ（エネルギー）
    mus = torch.ones((J, 3), device=device, requires_grad=True)
    lambdas = torch.ones((J, 1), device=device) * fixed_lamb
    optimizer = torch.optim.Adam([mus], lr=0.1)

    # 500回の最適化ステップ
    for _ in range(500):
        optimizer.zero_grad()
        dot = torch.sum(viewdirs.unsqueeze(1) * dirs.unsqueeze(0), dim=-1, keepdim=True)
        sg_val = mus.unsqueeze(0) * torch.exp(fixed_lamb * (dot - 1.0))
        pred = torch.sum(sg_val, dim=1)
        gt = visibility_fn(viewdirs)
        loss = torch.mean((pred - gt) ** 2)
        loss.backward()
        optimizer.step()

    return torch.cat([dirs, lambdas, torch.clamp(mus.detach(), min=0)], dim=-1)

# 可視性を含めたレンダリング計算
def render_with_sg_visibility(lgtSGs, diffuse_visSGs, specular_visSGs,
                               diffuse_albedo, specular_reflectance,
                               normal, viewdirs, roughness):
    # λトリック: 2つのSGの合成処理
    def lambda_trick(lobe1, lambda1, mu1, lobe2, lambda2, mu2):
        ratio = lambda1 / lambda2
        dot = torch.sum(lobe1 * lobe2, dim=-1, keepdim=True)
        tmp = torch.sqrt(ratio ** 2 + 1. + 2. * ratio * dot)
        tmp = torch.min(tmp, ratio + 1.)
        lambda3 = lambda2 * tmp
        lambda1_over_lambda3 = ratio / tmp
        lambda2_over_lambda3 = 1. / tmp
        diff = lambda2 * (tmp - ratio - 1.)
        final_lobes = lambda1_over_lambda3 * lobe1 + lambda2_over_lambda3 * lobe2
        final_lambdas = lambda3
        final_mus = mu1 * mu2 * torch.exp(diff)
        return final_lobes, final_lambdas, final_mus

    # 半球積分の近似
    def hemisphere_int(lambda_val, cos_beta):
        lambda_val = lambda_val + TINY_NUMBER
        inv_lambda = 1. / lambda_val
        t = torch.sqrt(lambda_val) * (1.6988 + 10.8438 * inv_lambda) / (1. + 6.2201 * inv_lambda + 10.2415 * inv_lambda ** 2)
        inv_a = torch.exp(-t)
        mask = (cos_beta >= 0).float()
        inv_b = torch.exp(-t * torch.clamp(cos_beta, min=0.))
        s1 = (1. - inv_a * inv_b) / (1. - inv_a + inv_b - inv_a * inv_b)
        b = torch.exp(t * torch.clamp(cos_beta, max=0.))
        s2 = (b - inv_a) / ((1. - inv_a) * (b + 1.))
        s = mask * s1 + (1. - mask) * s2
        A_b = 2. * np.pi / lambda_val * (torch.exp(-lambda_val) - torch.exp(-2. * lambda_val))
        A_u = 2. * np.pi / lambda_val * (1. - torch.exp(-lambda_val))
        return A_b * (1. - s) + A_u * s

    # SGをBRDFと統合し積分評価
def integrate_sg(lgtSGs, diffuse_albedo, normal, viewdirs, roughness, is_specular):
        dots_shape = list(normal.shape[:-1])
        M = lgtSGs.shape[0]
        normal = normal.unsqueeze(-2).expand(dots_shape + [M, 3])
        viewdirs = viewdirs.unsqueeze(-2).expand(dots_shape + [M, 3])
        lgtSGs = lgtSGs.view([1] * len(dots_shape) + [M, 7]).expand(dots_shape + [M, 7])
        lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
        lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
        lgtSGMus = torch.abs(lgtSGs[..., -3:])

        if is_specular:
            inv_rough4 = 1. / (roughness ** 4 + TINY_NUMBER)
            brdf_lambdas = (2. * inv_rough4).unsqueeze(-2).expand(dots_shape + [M, 1])
            brdf_mus = (inv_rough4 / np.pi).expand(dots_shape + [3]).unsqueeze(-2).expand(dots_shape + [M, 3])
            v_dot_lobe = torch.sum(normal * viewdirs, dim=-1, keepdim=True).clamp(min=0.)
            warp_lobes = 2 * v_dot_lobe * normal - viewdirs
            warp_lobes = warp_lobes / (torch.norm(warp_lobes, dim=-1, keepdim=True) + TINY_NUMBER)
            warp_lambdas = brdf_lambdas / (4 * v_dot_lobe + TINY_NUMBER)
            warp_mus = brdf_mus
            lobes, lambdas, mus = lambda_trick(lgtSGLobes, lgtSGLambdas, lgtSGMus, warp_lobes, warp_lambdas, warp_mus)
        else:
            diffuse = (diffuse_albedo / np.pi).unsqueeze(-2).expand(dots_shape + [M, 3])
            lobes = lgtSGLobes
            lambdas = lgtSGLambdas
            mus = lgtSGMus * diffuse

        mu_cos, lambda_cos, alpha_cos = 32.7080, 0.0315, 31.7003
        lobe_prime, lambda_prime, mu_prime = lambda_trick(normal, lambda_cos, mu_cos, lobes, lambdas, mus)
        dot1 = torch.sum(lobe_prime * normal, dim=-1, keepdim=True)
        dot2 = torch.sum(lobes * normal, dim=-1, keepdim=True)
        out = mu_prime * hemisphere_int(lambda_prime, dot1) - mus * alpha_cos * hemisphere_int(lambdas, dot2)
        return out.sum(dim=-2)

    # 可視性を合成（なければそのまま）
    lgtSGs_diffuse = apply_visibility_sg(lgtSGs, diffuse_visSGs) if diffuse_visSGs is not None else lgtSGs
    lgtSGs_specular = apply_visibility_sg(lgtSGs, specular_visSGs) if specular_visSGs is not None else lgtSGs

    # 拡散と鏡面成分のレンダリング結果
    Ld = integrate_sg(lgtSGs_diffuse, diffuse_albedo, normal, viewdirs, roughness, is_specular=False)
    Ls = integrate_sg(lgtSGs_specular, diffuse_albedo, normal, viewdirs, roughness, is_specular=True)
    rgb = Ld + Ls

    return {
        'rgb': rgb,
        'Ld': Ld,
        'Ls': Ls,
    }
