
import torch
import numpy as np

TINY_NUMBER = 1e-6

def prepend_dims(tensor, shape):
    orig_shape = list(tensor.shape)
    return tensor.view([1] * len(shape) + orig_shape).expand(shape + [-1] * len(orig_shape))

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
    final_mus = mu1 * mu2 * torch.exp(diff)
    return final_lobes, lambda3, final_mus

def hemisphere_int(lambda_val, cos_beta):
    lambda_val = lambda_val + TINY_NUMBER
    inv_lambda_val = 1. / lambda_val
    t = torch.sqrt(lambda_val) * (1.6988 + 10.8438 * inv_lambda_val) / (1. + 6.2201 * inv_lambda_val + 10.2415 * inv_lambda_val * inv_lambda_val)
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

def render_with_sg(lgtSGs, specular_reflectance, roughness, diffuse_albedo,
                   normal, viewdirs, visSGs=None, diffuse_rgb=None):
    roughness = roughness.view(-1, 1)
    dots_shape = list(normal.shape[:-1])
    K = 1

    # 可視性SGと畳み込み
    if visSGs is not None:
        visSGs = prepend_dims(visSGs, dots_shape)
        visSGs = visSGs.unsqueeze(-2).expand(dots_shape + [visSGs.shape[-2], 1, 7])
        lgtSGs = prepend_dims(lgtSGs, dots_shape)
        lgtSGs = lgtSGs.unsqueeze(-2).expand(dots_shape + [lgtSGs.shape[-2], 1, 7])
        lgtSGLobes, lgtSGLambdas, lgtSGMus = lambda_trick(
            lgtSGs[..., :3], lgtSGs[..., 3:4], lgtSGs[..., 4:],
            visSGs[..., :3], visSGs[..., 3:4], visSGs[..., 4:]
        )
        lgtSGs = torch.cat([lgtSGLobes, lgtSGLambdas, lgtSGMus], dim=-1)
    else:
        lgtSGs = prepend_dims(lgtSGs, dots_shape)
        lgtSGs = lgtSGs.unsqueeze(-2).expand(dots_shape + [lgtSGs.shape[-2], 1, 7])

    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:])

    normal = normal.unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [lgtSGs.shape[-2], K, 3])
    viewdirs = viewdirs.unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [lgtSGs.shape[-2], K, 3])

    inv_roughness_pow4 = 1. / (roughness ** 4 + TINY_NUMBER)
    brdfSGLambdas = (2. * inv_roughness_pow4).unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [lgtSGs.shape[-2], K, 1])
    mu_val = (inv_roughness_pow4 / np.pi).expand(dots_shape + [3])
    brdfSGMus = mu_val.unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [lgtSGs.shape[-2], K, 3])
    warpBrdfSGLobes = 2 * torch.clamp(torch.sum(normal * viewdirs, dim=-1, keepdim=True), min=0.) * normal - viewdirs
    warpBrdfSGLobes = warpBrdfSGLobes / (torch.norm(warpBrdfSGLobes, dim=-1, keepdim=True) + TINY_NUMBER)
    warpBrdfSGLambdas = brdfSGLambdas / (4 * torch.clamp(torch.sum(normal * viewdirs, dim=-1, keepdim=True), min=0.) + TINY_NUMBER)
    warpBrdfSGMus = brdfSGMus

    new_half = warpBrdfSGLobes + viewdirs
    new_half = new_half / (torch.norm(new_half, dim=-1, keepdim=True) + TINY_NUMBER)
    v_dot_h = torch.clamp(torch.sum(viewdirs * new_half, dim=-1, keepdim=True), min=0.)
    specular_reflectance = specular_reflectance.unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [lgtSGs.shape[-2], K, 3])
    F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)

    dot1 = torch.clamp(torch.sum(warpBrdfSGLobes * normal, dim=-1, keepdim=True), min=0.)
    dot2 = torch.clamp(torch.sum(viewdirs * normal, dim=-1, keepdim=True), min=0.)
    k = ((roughness + 1.) ** 2 / 8.).unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [lgtSGs.shape[-2], K, 1])
    G = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER) * dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
    Moi = F * G / (4 * dot1 * dot2 + TINY_NUMBER)
    warpBrdfSGMus = warpBrdfSGMus * Moi

    final_lobes, final_lambdas, final_mus = lambda_trick(
        lgtSGLobes, lgtSGLambdas, lgtSGMus,
        warpBrdfSGLobes, warpBrdfSGLambdas, warpBrdfSGMus
    )

    mu_cos = 32.7080
    lambda_cos = 0.0315
    alpha_cos = 31.7003
    lobe_prime, lambda_prime, mu_prime = lambda_trick(normal, lambda_cos, mu_cos, final_lobes, final_lambdas, final_mus)
    dot1 = torch.sum(lobe_prime * normal, dim=-1, keepdim=True)
    dot2 = torch.sum(final_lobes * normal, dim=-1, keepdim=True)
    specular_rgb = mu_prime * hemisphere_int(lambda_prime, dot1) - final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)
    specular_rgb = specular_rgb.sum(dim=-2).sum(dim=-2)
    specular_rgb = torch.clamp(specular_rgb, min=0.)

    if diffuse_rgb is None:
        diffuse = (diffuse_albedo / np.pi).unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [lgtSGs.shape[-2], 1, 3])
        final_lobes = lgtSGLobes.narrow(dim=-2, start=0, length=1)
        final_mus = lgtSGMus.narrow(dim=-2, start=0, length=1) * diffuse
        final_lambdas = lgtSGLambdas.narrow(dim=-2, start=0, length=1)
        lobe_prime, lambda_prime, mu_prime = lambda_trick(normal, lambda_cos, mu_cos, final_lobes, final_lambdas, final_mus)
        dot1 = torch.sum(lobe_prime * normal, dim=-1, keepdim=True)
        dot2 = torch.sum(final_lobes * normal, dim=-1, keepdim=True)
        diffuse_rgb = mu_prime * hemisphere_int(lambda_prime, dot1) - final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)
        diffuse_rgb = diffuse_rgb.sum(dim=-2).sum(dim=-2)
        diffuse_rgb = torch.clamp(diffuse_rgb, min=0.)

    rgb = specular_rgb + diffuse_rgb
    return {
        'sg_rgb': rgb,
        'sg_specular_rgb': specular_rgb,
        'sg_diffuse_rgb': diffuse_rgb,
        'sg_diffuse_albedo': diffuse_albedo,
        'sg_roughness': roughness
    }
