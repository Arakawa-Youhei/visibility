import trimesh
import numpy as np
import cupy as cp
from PIL import Image
from tqdm import tqdm
import os

# メッシュの読み込み
mesh = trimesh.load('/home/arakawa/HyperDreamer/visibility/mesh.obj')

# メッシュの法線を確認し、必要なら再計算
if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
    mesh.recompute_vertex_normals()

# メッシュ内の頂点を取得
vertices = np.array(mesh.vertices)  # 頂点座標
num_vertices = len(vertices)

# 球面方向ベクトルの生成
def generate_sphere_directions(grid_size):
    """
    球面上の一様な方向ベクトルを生成
    """
    phi = np.linspace(0, np.pi, grid_size)  # 緯度
    theta = np.linspace(0, 2 * np.pi, grid_size)  # 経度
    phi, theta = np.meshgrid(phi, theta)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    directions = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)  # 正規化
    return directions

# グリッドサイズを設定
grid_size = 100  # グリッド解像度（例: 100x100）
directions = generate_sphere_directions(grid_size)

# CuPy に方向ベクトルを転送（キャッシュ）
directions_gpu = cp.array(directions, dtype=cp.float32)

# 全頂点をバッチで処理
def gpu_raytrace_batch(mesh, vertices, directions_gpu, grid_size, offset=0.01):
    """
    全頂点をバッチ処理で一括してレイトレーシングを行う
    """
    # 頂点をオフセット（表面法線に沿って移動）
    offsets = offset * mesh.vertex_normals
    origins = vertices + offsets

    # CuPy に転送
    origins_gpu = cp.array(origins, dtype=cp.float32)

    # 結果を記録する配列
    num_directions = len(directions_gpu)
    num_vertices = len(origins_gpu)
    results = cp.zeros((num_vertices, num_directions), dtype=cp.bool_)

    # 各頂点に対して方向ベクトルを処理
    for i in tqdm(range(num_vertices), desc="Processing vertices"):
        ray_origins = cp.tile(origins_gpu[i], (num_directions, 1))  # 各方向に同じ起点
        ray_directions = directions_gpu  # 方向ベクトル

        # trimesh の光線交差処理
        intersections, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=ray_origins.get(),  # GPU → CPU に変換
            ray_directions=ray_directions.get()
        )

        # 衝突結果を記録
        results[i, index_ray] = True  # 衝突した方向を記録

    return results.get()  # GPU → CPU に戻す

# すべての頂点を処理
results = gpu_raytrace_batch(mesh, vertices, directions_gpu, grid_size)

# 結果を画像として保存
output_dir = "vertex_raytracing_images"
os.makedirs(output_dir, exist_ok=True)

for i, result in enumerate(tqdm(results, desc="Saving images")):
    # 結果を画像化
    image = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for k, value in enumerate(result):
        x = k % grid_size
        y = k // grid_size
        image[y, x] = 255 if not value else 0  # 白 (255): 非衝突, 黒 (0): 衝突

    # 画像を保存
    Image.fromarray(image, mode='L').save(f"{output_dir}/vertex_{i:05d}.png")

