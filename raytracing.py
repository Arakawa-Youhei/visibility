import trimesh
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import os
from trimesh.ray.ray_pyembree import RayMeshIntersector
from scipy.spatial.transform import Rotation as R

# 半球方向ベクトルを生成（Z+方向基準）
def generate_hemisphere_directions(n_theta=100, n_phi=100):
    theta = np.linspace(0, np.pi / 2, n_theta)  # 天頂角
    phi = np.linspace(0, 2 * np.pi, n_phi)      # 方位角
    theta, phi = np.meshgrid(theta, phi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    directions = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # (n, 3)
    return directions, theta.shape

# 回転行列で方向ベクトル群を回転
def rotate_directions(directions, normal):
    if np.allclose(normal, [0, 0, 1]):
        return directions  # 回転不要
    rot, _ = R.align_vectors([[0, 0, 1]], [normal])
    return rot.apply(directions)

# レイトレーシング
def raytrace_local_hemisphere(mesh, directions_z, shape_2d):
    ray_intersector = RayMeshIntersector(mesh)
    results = []

    for i in tqdm(range(len(mesh.vertices)), desc="Raytracing (local hemisphere)"):
        origin = mesh.vertices[i]
        normal = mesh.vertex_normals[i]
        directions = rotate_directions(directions_z, normal)

        origins = np.tile(origin, (directions.shape[0], 1))
        hits = ray_intersector.intersects_any(origins, directions)
        results.append(hits.astype(np.uint8))

    return np.array(results), shape_2d

# 画像出力（軸・凡例あり）
def save_images(results, shape_2d, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, result in enumerate(results):
        img = (1 - result).reshape(shape_2d) * 255  # 1: white, 0: black
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray', extent=[0, 180, 0, 360], origin='lower')
        ax.set_title(f"Vertex {i}")
        ax.set_xlabel("Theta (deg)")
        ax.set_ylabel("Phi (deg)")
        ax.set_xticks(np.linspace(0, 180, 5))
        ax.set_yticks(np.linspace(0, 360, 5))
        fig.savefig(os.path.join(output_dir, f"raytrace_vertex_{i}.png"))
        plt.close(fig)

# npy保存
def save_npy(results, output_path):
    np.save(output_path, results)

# メイン処理
def main(obj_path, output_dir):
    mesh = trimesh.load(obj_path, force='mesh')
    directions_z, shape_2d = generate_hemisphere_directions(n_theta=100, n_phi=100)
    results, shape_2d = raytrace_local_hemisphere(mesh, directions_z, shape_2d)
    save_images(results, shape_2d, os.path.join(output_dir, "images"))
    save_npy(results, os.path.join(output_dir, "results.npy"))

# 実行例
if __name__ == "__main__":
    obj_path = "your_model.obj"  # ←ファイル名を適宜変更
    output_dir = "output_local_raytrace"
    main(obj_path, output_dir)
