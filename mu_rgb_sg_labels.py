import torch

# μを読み込み [V, 5, 3]
mu_all = torch.load("visibility/trained_mu_localSG.pt")

output_path = "visibility/mu_rgb_mean_with_sg_labels.txt"
with open(output_path, "w") as f:
    for v_id, mu_vertex in enumerate(mu_all):  # mu_vertex: [5, 3]
        # 各SGのRGB平均を計算してラベル付きで整形
        mu_line = " ".join([f"SG{j+1}={mu_rgb.mean().item():.4f}" for j, mu_rgb in enumerate(mu_vertex)])
        f.write(f"頂点 {v_id}: {mu_line}\n")

print(f"SG番号付きでRGB平均μを出力しました → {output_path}")
