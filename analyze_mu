import torch

def analyze_mu(mu_path="visibility/trained_mu_localSG.pt"):
    mu_all = torch.load(mu_path)  # [V, J, 3]

    print("μの形状:", mu_all.shape)
    print("μの統計情報:")
    print(f"  最小値: {mu_all.min().item():.6f}")
    print(f"  最大値: {mu_all.max().item():.6f}")
    print(f"  平均値: {mu_all.mean().item():.6f}")
    print(f"  標準偏差: {mu_all.std().item():.6f}")

    # 各SGごとの平均μ（RGB平均）
    mu_rgb_mean = mu_all.mean(dim=2)  # [V, J]
    mu_mean_per_SG = mu_rgb_mean.mean(dim=0)  # [J]
    print("\n各SG（方向）ごとの平均μ（RGB平均）:")
    for j, val in enumerate(mu_mean_per_SG):
        print(f"  SG{j}: {val.item():.6f}")

if __name__ == "__main__":
    analyze_mu()
