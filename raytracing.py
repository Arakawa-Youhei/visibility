import numpy as np
import trimesh
import os
import matplotlib.pyplot as plt

def generate_hemisphere_rays(n_theta=100, n_phi=100):
    theta = np.linspace(0, np.pi / 2, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    directions = []

    for t in theta:
        for p in phi:
            x = np.sin(t) * np.cos(p)
            y = np.sin(t) * np.sin(p)
            z = np.cos(t)
            directions.append([x, y, z])

    return np.array(directions), theta, phi

def local_to_world(directions, normal):
    z = normal / np.linalg.norm(normal)
    up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.99 else np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return directions @ np.stack([x, y, z], axis=1)

def process_vertex_rays(vertex, normal, mesh, directions):
    world_dirs = local_to_world(directions, normal)
    origins = np.tile(vertex, (world_dirs.shape[0], 1)) + 1e-4 * normal
    ray_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    hits = ray_intersector.intersects_any(origins, world_dirs)
    return hits.astype(np.uint8)

def save_image(hits, theta_res, phi_res, save_path):
    img = hits.reshape((theta_res, phi_res))
    plt.imshow(img, cmap='gray_r', origin='lower', extent=[0, 360, 0, 90])
    plt.xlabel('phi (°)')
    plt.ylabel('theta (°)')
    plt.title('Ray Intersection')
    plt.colorbar(label='Hit')
    plt.xticks(np.linspace(0, 360, 5))
    plt.yticks(np.linspace(0, 90, 4))
    plt.savefig(save_path)
    plt.close()

def main(obj_path, output_dir, npy_dir, theta_res=100, phi_res=100):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    mesh = trimesh.load(obj_path, force='mesh')
    if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
        mesh = mesh.copy()
        mesh.rezero()
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.fix_normals()
        mesh.vertex_normals = mesh.vertex_normals

    directions, theta_list, phi_list = generate_hemisphere_rays(theta_res, phi_res)

    for i, (v, n) in enumerate(zip(mesh.vertices, mesh.vertex_normals)):
        hits = process_vertex_rays(v, n, mesh, directions)
        np.save(os.path.join(npy_dir, f"vertex_{i:04d}.npy"), hits)
        save_image(hits, theta_res, phi_res, os.path.join(output_dir, f"vertex_{i:04d}.png"))
        print(f"Processed vertex {i+1}/{len(mesh.vertices)}")
        
        directions_path = os.path.join(npy_dir, "directions.npy")
        if not os.path.exists(directions_path):
            np.save(directions_path,  directions)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python raytracing.py <input.obj> <output_dir> <npy_dir>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
