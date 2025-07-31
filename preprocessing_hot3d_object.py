import os
import os.path as osp

import numpy as np
import pymeshlab as ml
import trimesh
import glob
import tqdm


def main():
    import json
    with open("../hot3d/dataset/assets/instance.json", 'r') as f:
        mapping = json.load(f)
    original_folder = '../hot3d/dataset/assets'
    save_folder = 'data/hot3d/processed_object_meshes'
    os.makedirs(save_folder, exist_ok=True)

    n_target_vertices = 4000

    original_meshes = glob.glob(os.path.join(original_folder, "*.glb"))
    for mesh_path in tqdm.tqdm(original_meshes, desc="Processing object meshes"):
        uid = os.path.basename(mesh_path).split('.')[0]
        basename = mapping[uid]['instance_name'] + '.ply'
        ms = ml.MeshSet(verbose=0)

        mesh = trimesh.load_mesh(mesh_path)
        ##### it is for alignment with GRAB object pose, not used now #####
        # # rotate mesh for 90 degree clockwwise along x-axis
        # # then rotate for 180 degree along z-axis
        # vert = mesh.vertices.copy() @ np.array([
        #         [1,  0,  0], 
        #         [0,  0,  1], 
        #         [0, -1,  0] 
        #     ]).astype(np.float64) @ np.array([
        #         [-1,  0,  0], 
        #         [ 0, -1,  0], 
        #         [ 0,  0,  1] 
        #     ]).astype(np.float64)
        # # vert = mesh.vertices
        # # vert[:,1], vert[:, 2] = -vert[:,2], vert[:, 1]
        # face = mesh.faces
        # ms.add_mesh(ml.Mesh(vert, face))
        ##### it is for alignment with GRAB object pose, not used now #####
        ms.add_mesh(ml.Mesh(mesh.vertices.copy(), mesh.faces))
        m = ms.current_mesh()
        TARGET = n_target_vertices
        numFaces = 100 + 2 * TARGET
        while (ms.current_mesh().vertex_number() > TARGET):
            # ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=numFaces, preservenormal=True)
            ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=numFaces, preservenormal=True)
            numFaces = numFaces - (ms.current_mesh().vertex_number() - TARGET)
        m = ms.current_mesh()
        print('output mesh has', m.vertex_number(), 'vertex and', m.face_number(), 'faces')

        ms.save_current_mesh(osp.join(save_folder, basename))

if __name__ == "__main__":
    main()
    from preprocess.preprocessing_hot3d import preprocessing_object
    preprocessing_object()