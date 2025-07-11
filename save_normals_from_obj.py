
# save_normals_from_obj.py
import trimesh
import numpy as np
import os

def save_vertex_normals(obj_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mesh = trimesh.load(obj_path, force='mesh')

    if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
        mesh.rezero()
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.fix_normals()
        mesh.vertex_normals = mesh.vertex_normals  # 再計算

    normals = mesh.vertex_normals  # shape: [V, 3]
    save_path = os.path.join(output_dir, "normals.npy")
    np.save(save_path, normals)
    print(f"法線ベクトルを保存しました: {save_path}")

# 使用例
if __name__ == "__main__":
    obj_path = "exp/strawberry_s2/mesh/mesh.obj"  # <- 読み込む .obj ファイルのパス
    output_dir = "raytracing_results/202507111807/npy"  # <- normals.npy を保存するディレクトリ
    save_vertex_normals(obj_path, output_dir)
