
import os
import numpy as np
import trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# ====== 入力 ======
mesh_path = "exp/strawberry_s2/mesh/mesh.obj"
os.makedirs("visibility", exist_ok=True)

# ====== メッシュ読み込み ======
mesh = trimesh.load(mesh_path, process=False)
if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
    mesh.recompute_vertex_normals()
vertices = np.array(mesh.vertices)
normals = np.array(mesh.vertex_normals)
V = vertices.shape[0]

# ====== ローカル半球方向の生成（Y軸が上） ======
def generate_local_directions(grid_size):
    theta = np.linspace(0, 0.5 * np.pi, grid_size)  # 天頂角: 0〜90°
    phi = np.linspace(0, 2 * np.pi, grid_size)      # 方位角: 0〜360°
    theta, phi = np.meshgrid(theta, phi)
    x = np.sin(theta) * np.cos(phi)
    y = np.cos(theta)
    z = np.sin(theta) * np.sin(phi)
    dirs = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

grid_size = 64
local_dirs = generate_local_directions(grid_size)  # [D, 3]
D = local_dirs.shape[0]
np.save("visibility/directions.npy", local_dirs)

# ====== 回転行列（ローカル → ワールド） ======
def get_rotation_to_world(normal):
    normal = normal / np.linalg.norm(normal)
    y_axis = np.array([0, 1, 0])
    axis = np.cross(y_axis, normal)
    if np.linalg.norm(axis) < 1e-6:
        return np.eye(3)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(y_axis, normal), -1.0, 1.0))
    return R.from_rotvec(axis * angle).as_matrix()

# ====== 各頂点ごとのレイ方向（ワールド空間で） ======
offset = 0.01
all_results = np.zeros((V, D), dtype=np.uint8)

for i in tqdm(range(V), desc="Raytracing"):
    origin = vertices[i] + offset * normals[i]
    R_world = get_rotation_to_world(normals[i])
    world_dirs = (R_world @ local_dirs.T).T

    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=np.tile(origin, (D, 1)),
        ray_directions=world_dirs,
        multiple_hits=False
    )
    hit_flags = np.zeros(D, dtype=np.uint8)
    hit_flags[index_ray] = 1
    all_results[i] = 1 - hit_flags  # 1: visible, 0: occluded

# ====== 結果保存 ======
np.save("visibility/visibility_dataset.npy", all_results)
print("Saved: visibility_dataset.npy")

# ====== 可視性マップの画像出力（ローカル半球） ======
output_dir = "vertex_raytracing_images_202507101800"
os.makedirs(output_dir, exist_ok=True)

for i in tqdm(range(V), desc="Saving images"):
    image = all_results[i].reshape(grid_size, grid_size)
    fig, ax = plt.subplots()
    im = ax.imshow(image, cmap='gray', origin='lower', extent=[0, 360, 0, 90])
    ax.set_title(f"Vertex {i}")
    ax.set_xlabel("Azimuth φ (deg)")    # 方位角 φ
    ax.set_ylabel("Elevation θ (deg)")  # 天頂角 θ
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Visibility")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Occluded", "Visible"])
    plt.savefig(f"{output_dir}/vertex_{i:05d}.png")
    plt.close()
