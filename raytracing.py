import trimesh
import numpy as np
from trimesh.ray.ray_pyembree import RayMeshIntersector
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def load_mesh(path):
    mesh = trimesh.load(path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("OBJファイルは単一メッシュである必要があります")
    return mesh

def generate_rays(origin, resolution=64):
    theta = np.linspace(0, np.pi / 2, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution)
    theta, phi = np.meshgrid(theta, phi)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.cos(theta)
    z = np.sin(theta) * np.sin(phi)
    
    directions = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    origins = np.tile(origin, (directions.shape[0], 1))
    return origins, directions

def raytrace(mesh, resolution=64, vertex_index=0):
    intersector = RayMeshIntersector(mesh)
    origin = mesh.vertices[vertex_index]
    origins, directions = generate_rays(origin, resolution=resolution)
    hit, _, _ = intersector.intersects_id(origins, directions, return_locations=False)
    hit_image = hit.reshape((resolution, resolution)).astype(np.uint8) * 255
    return hit_image

def save_image(array, file_path):
    img = Image.fromarray(array.astype(np.uint8), mode='L')
    img.save(file_path)

if __name__ == "__main__":
    obj_path = "exp/strawberry_s2/mesh/mesh.obj"  # 任意の.objファイルを指定
    mesh = load_mesh(obj_path)
    
    output_dir = "./raytracing/output_images_20250711"
    import os
    os.makedirs(output_dir, exist_ok=True)

    for idx in tqdm(range(len(mesh.vertices))):
        img = raytrace(mesh, resolution=64, vertex_index=idx)
        save_image(img, f"{output_dir}/raytrace_result_vertex_{idx}.png")
