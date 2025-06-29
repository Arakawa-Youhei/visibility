import trimesh
import numpy as np
import cupy as cp
from PIL import Image
import os
from tqdm import tqdm

# ======== メッシュ読み込み ========
mesh_path = "mesh/model.obj"
mesh = trimesh.load(mesh_path)

if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
    mesh.recompute_vertex_normals()

vertices = np.array(mesh.vertices)  # [V, 3]
V = vertices.shape[0]

# ======== 球面方向の生成 ========
def generate_sphere_directions(grid_size):
    phi = np.linspace(0, np.pi, grid_size)
    theta = np.linspace(0, 2 * np.pi, grid_size)
    phi, theta = np.meshgrid(phi, theta)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    directions = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions  # [D, 3]

grid_size = 100
directions = generate_sphere_directions(grid_size)  # [D, 3]
D = directions.shape[0]
cp_directions = cp.array(directions, dtype=cp.float32)

# ======== レイトレーシング ========
def gpu_raytrace_batch(mesh, vertices, directions_gpu, offset=0.01):
    offsets = offset * mesh.vertex_normals
    origins = vertices + offsets
    origins_gpu = cp.array(origins, dtype=cp.float32)
    results = cp.zeros((V, D), dtype=cp.bool_)

    for i in tqdm(range(V), desc="Processing vertices"):
        ray_origins = cp.tile(origins_gpu[i], (D, 1))
        ray_directions = directions_gpu
        intersections, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=ray_origins.get(),
            ray_directions=ray_directions.get()
        )
        results[i, index_ray] = True
    return results.get()  # [V, D] numpy.bool_

print("Running ray tracing...")
results = gpu_raytrace_batch(mesh, vertices, cp_directions)

# ======== 出力保存 ========
os.makedirs("visibility", exist_ok=True)
np.save("visibility/directions.npy", directions)
np.save("visibility/visibility_dataset.npy", 1.0 - results.astype(np.float32))
print("Saved: visibility_dataset.npy and directions.npy")

# ======== 可視性マップの画像出力 ========
output_dir = "vertex_raytracing_images"
os.makedirs(output_dir, exist_ok=True)

for i, result in enumerate(tqdm(results, desc="Saving images")):
    image = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for k, value in enumerate(result):
        x = k % grid_size
        y = k // grid_size
        image[y, x] = 255 if not value else 0  # 白=可視, 黒=遮蔽
    Image.fromarray(image, mode='L').save(f"{output_dir}/vertex_{i:05d}.png")
print(f"Saved raytrace images to: {output_dir}")
