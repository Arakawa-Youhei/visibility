import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
import matplotlib.pyplot as plt
import os

def generate_hemisphere_directions(n_theta=100, n_phi=100):
    theta = np.linspace(0, np.pi / 2, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    directions = np.stack([x, y, z], axis=-1)
    return directions.reshape(-1, 3), theta_grid, phi_grid

def local_to_world(local_dir, normal):
    up = np.array([0, 0, 1])
    v = np.cross(up, normal)
    c = np.dot(up, normal)
    if np.linalg.norm(v) < 1e-6:
        return local_dir
    kmat = np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]])
    rotation = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (np.linalg.norm(v)**2))
    return (rotation @ local_dir.T).T

def raytrace_vertex(mesh, intersector, vertex, normal, directions):
    world_dirs = local_to_world(directions, normal)
    origins = np.tile(vertex, (len(world_dirs), 1))
    locations, index_ray, _ = intersector.intersects_location(origins, world_dirs, multiple_hits=False)
    hit_mask = np.zeros(len(world_dirs), dtype=bool)
    hit_mask[index_ray] = True
    return hit_mask

def save_results_image(hit_mask, theta_grid, phi_grid, output_path):
    image = hit_mask.reshape(theta_grid.shape)
    plt.imshow(image, cmap='gray', origin='lower', extent=[0, 360, 0, 90])
    plt.xlabel('phi [deg]')
    plt.ylabel('theta [deg]')
    plt.xticks(np.linspace(0, 360, 9))
    plt.yticks(np.linspace(0, 90, 10))
    plt.title("Ray Hit Map")
    plt.colorbar(label="Hit (Black) / No Hit (White)")
    plt.savefig(output_path)
    plt.close()

def main(obj_path, output_dir):
    mesh = trimesh.load(obj_path, force='mesh')
    mesh.compute_vertex_normals()
    intersector = RayMeshIntersector(mesh)
    directions, theta_grid, phi_grid = generate_hemisphere_directions()

    os.makedirs(output_dir, exist_ok=True)
    for idx, (v, n) in enumerate(zip(mesh.vertices, mesh.vertex_normals)):
        print(f"Processing vertex {idx+1}/{len(mesh.vertices)}...")
        hit_mask = raytrace_vertex(mesh, intersector, v, n, directions)
        np.save(os.path.join(output_dir, f"vertex_{idx}_hit.npy"), hit_mask)
        save_results_image(hit_mask, theta_grid, phi_grid, os.path.join(output_dir, f"vertex_{idx}_result.png"))

if __name__ == "__main__":
    main("exp/strawberry_s2/mesh/mesh.obj", "output_results_202507110058")
