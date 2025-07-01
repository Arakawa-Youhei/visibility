
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 可視SGの構造を読み込み
visSGs_all = torch.load("visibility/visSGs_all.pt")  # [V, J, 7]
visSGs = visSGs_all[0]  # 任意の1頂点分を可視化（[J, 7]）

axes = visSGs[:, :3].numpy()  # 軸ベクトル [J, 3]

# 単位半球を描画（Y軸が上方向）
phi, theta = np.mgrid[0:np.pi/2:50j, 0:2*np.pi:50j]
x = np.sin(phi) * np.cos(theta)
y = np.cos(phi)
z = np.sin(phi) * np.sin(theta)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='lightgray', alpha=0.3, linewidth=0)

# 軸ベクトルを矢印で表示
for i, a in enumerate(axes):
    ax.quiver(0, 0, 0, a[0], a[1], a[2], color='r', length=1.0, normalize=True)
    ax.text(a[0]*1.1, a[1]*1.1, a[2]*1.1, f"SG{i+1}", color='black')

ax.set_title("SG Axis Directions (Vertex 0)")
ax.set_xlim([-1, 1])
ax.set_ylim([0, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel("X")
ax.set_ylabel("Y (up)")
ax.set_zlabel("Z")
plt.tight_layout()

# 保存用ディレクトリとファイル
output_dir = "visibility/sg_axis_visualization"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/sg_axes_vertex0.png"
plt.savefig(output_path)
print(f"Saved SG axis visualization to: {output_path}")
plt.close()
