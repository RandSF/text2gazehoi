import json
import pickle

import trimesh
import smplx
import torch
import numpy as np

import glob
import os
import os.path as osp
from collections import Counter

import tqdm

from lib.utils.rot import (
    quaternion_to_angle_axis, 
    quaternion_to_rotation_matrix,
    rotmat_to_rot6d, 
    axis_angle_to_rot6d
)
from lib.utils.file import load_config
from constants.hot3d_constants import hot3d_obj_name

def hoi_to_str(annt):
    # convert the hoi info to string in the format: 
    # "left_actions+left_objects|r_as+r_os|both_as+both_os" 
    # ("|" is the hand separator, "+" is the action-object separator, "=" is the inner separator inside actions/objects)
    if annt['left']['valid']:
        l_a = "=".join(annt['left']['actions'])
        l_o = "=".join(annt['left']['objects'])
        left = f"{l_a}+{l_o}"
    else:
        left = ""
    if annt['right']['valid']:
        r_a = "=".join(annt['right']['actions'])
        r_o = "=".join(annt['right']['objects'])
        right = f"{r_a}+{r_o}"
    else:
        right = ""
    if annt['both']['valid']:
        b_a = "=".join(annt['both']['actions'])
        b_o = "=".join(annt['both']['objects'])
        both = f"{b_a}+{b_o}"
    else:
        both = ""
    return f"{left}|{right}|{both}"

def batched_indexing(idx_batch: torch.Tensor, mapping: dict):
    res = []
    assert torch.is_tensor(idx_batch)
    assert idx_batch.ndim == 2
    for idx in idx_batch:
        res.append(mapping[idx.item()])
    return torch.stack(res, dim=0)

### a modified version, all data is available
def get_contact_info(
        left_joint, right_joint, left_obj_verts, right_obj_verts
    ):
    contact_threshold = 0.02

    ldist = torch.cdist(left_obj_verts, left_joint)
    lcf_idx, lcov_idx, lchj_idx =torch.where(ldist < contact_threshold) # c: contact, ov: object vertex, hj: hand joint
    ldist_value = ldist[lcf_idx, lcov_idx, lchj_idx]
    rdist = torch.cdist(right_obj_verts, right_joint)
    rcf_idx, rcov_idx, rchj_idx = torch.where(rdist < contact_threshold)
    rdist_value = rdist[rcf_idx, rcov_idx, rchj_idx]

    is_lhand, is_rhand = 1, 1
    
    return (lcf_idx, lcov_idx, lchj_idx, ldist_value, 
            rcf_idx, rcov_idx, rchj_idx, rdist_value, 
            is_lhand, is_rhand)

def farthest_point_sample(xyz, npoint, random=False):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if random:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    else:
        farthest = 0
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids
### a modified version, all data is available

def preprocessing_object():
    _config = load_config("configs/dataset/hot3d.yaml")
    objects_folder = glob.glob(osp.join(_config.obj_root, "*"))
    with open(_config.instance_path, 'r') as f:
        ins_json = json.loads(f.read())
    uid_obj_mapping = {k: v['instance_name'] for k, v in ins_json.items()}
    obj_pcs = {}
    obj_pc_normals = {}
    point_sets = {}
    obj_path = {}
    for object_path in tqdm.tqdm(objects_folder):
        # object_paths = glob.glob(osp.join(object_folder, "*.obj"))
        # for object_path in tqdm.tqdm(object_paths):
        mesh = trimesh.load(object_path, maintain_order=True)
        verts = torch.tensor(mesh.vertices).float().unsqueeze(0).cuda()
        normal = torch.tensor(mesh.vertex_normals).float().unsqueeze(0).cuda()
        normal = normal / torch.norm(normal, dim=2, keepdim=True)
        point_set = farthest_point_sample(verts, 1024)
        sampled_pc = verts[0, point_set[0]].cpu().numpy()
        sampled_normal = normal[0, point_set[0]].cpu().numpy()
        object_name = object_path.split("/")[-1]
        key = f"{object_name}"
        obj_pcs[key] = sampled_pc
        obj_pc_normals[key] = sampled_normal
        point_sets[key] = point_set[0].cpu().numpy()
        obj_path[key] = "/".join(object_path.split("/")[-2:])
    
    os.makedirs("data/hot3d", exist_ok=True)
    with open("data/hot3d/obj.pkl", "wb") as f:
        pickle.dump(
            {
                "object_name": hot3d_obj_name, 
                "obj_pcs": obj_pcs, 
                "obj_pc_normals": obj_pc_normals, 
                "point_sets": point_sets, 
                "obj_path": obj_path, 
            }, f)
        
def preprocessing_data():
    _config = load_config("configs/dataset/hot3d.yaml")
    data_save_path = _config.data_path

    mano_dir = osp.abspath('../mano_v1_2/models/')
    left_mano = smplx.create(
        osp.join(mano_dir, "MANO_LEFT.pkl"), "mano", num_pca_comps=15, 
    )
    right_mano = smplx.create(
        osp.join(mano_dir, "MANO_RIGHT.pkl"), "mano", num_pca_comps=15, 
    )
    hand_components = smplx.create(
        osp.join(mano_dir, "MANO_RIGHT.pkl"), "mano", num_pca_comps=15, 
    ).hand_components.clone()    # left and right components are the same

    with open(_config.instance_path, 'r') as f:
        ins_json = json.loads(f.read())
    uid_obj_mapping = {v['instance_name']: int(k)  for k, v in ins_json.items()} # {name: uid}
    with open(_config.data_obj_pc_path, 'rb') as f:
        obj_data = pickle.load(f)
    uid_pcs_mapping = {uid_obj_mapping[name.split('.')[0]]: torch.from_numpy(pcs) for name, pcs in obj_data['obj_pcs'].items()}

    data_root = _config.root
    sub_folders = [osp.join(data_root, subf) for subf in os.listdir(data_root) if osp.isdir(osp.join(data_root, subf))]

    x_lhand_total = []
    x_rhand_total = []
    j_lhand_total = []
    j_rhand_total = []
    x_obj_total = []
    lhand_beta_total = []
    rhand_beta_total = []
    x_lhand_org_total = []
    x_rhand_org_total = []
    lcf_idx_total = []
    lcov_idx_total = []
    lchj_idx_total = []
    ldist_value_total = []
    rcf_idx_total = []
    rcov_idx_total = []
    rchj_idx_total = []
    rdist_value_total = []
    is_lhand_total = []
    is_rhand_total = []
    object_name_total = []
    action_name_total = []
    nframes_total = []
    
    for sf in tqdm.tqdm(sub_folders):
        res = sorted(glob.glob(osp.join(sf, "*")), key=lambda s: s.split('/')[-1].split('.')[0]) 
        # annt, (clip,) contact, gaze, hand, handjoints, handmeshs, head, masks, objects, timestamps
        # print(res)  
        ### only use annt(text), clip, hand, masks, objects ###
        for fp in res:
            if fp.find('annotations.json')!=-1:
                annt = json.load(open(res[0], 'r'))
                break
        clip, contact, gaze, hand, _, _, _, masks, objects, _ = [np.load(fp) for fp in res if fp.find('.npy')!=-1]

        nframe = hand.shape[0]

        left_hand, right_hand = torch.from_numpy(hand[:,:22]).float(), torch.from_numpy(hand[:,22:]).float()
        left_transl, left_wrist_quat, left_pca_pose = left_hand[:,:3], left_hand[:,3:7], left_hand[:,7:]
        right_transl, right_wrist_quat, right_pca_pose = right_hand[:,:3], right_hand[:,3:7], right_hand[:,7:]
        # left_pose, right_pose = left_pca_pose @ hand_components, right_pca_pose @ hand_components
        left_orient, right_orient = quaternion_to_angle_axis(torch.roll(left_wrist_quat, shifts=1, dims=1)), quaternion_to_angle_axis(torch.roll(right_wrist_quat, shifts=1, dims=1))

        beta = torch.zeros(nframe, 10)

        left_output = left_mano(global_orient=left_orient, hand_pose=left_pca_pose, transl=left_transl, betas=beta, return_full_pose=True)
        left_joint = left_output.joints
        left_pose = left_output.full_pose

        right_output = right_mano(global_orient=right_orient, hand_pose=right_pca_pose, transl=right_transl, betas=beta, return_full_pose=True)
        right_joint = right_output.joints
        right_pose = right_output.full_pose

        left_rot6 = axis_angle_to_rot6d(left_output.full_pose.reshape(-1, 3)).reshape(nframe, 16*6)
        right_rot6 = axis_angle_to_rot6d(right_output.full_pose.reshape(-1, 3)).reshape(nframe, 16*6)
        
        x_left = torch.cat([left_transl, left_rot6], dim=-1)    # [N, 99]
        x_right = torch.cat([right_transl, right_rot6], dim=-1)    # [N, 99]

        ### objects
        nobj = 6
        objects = torch.from_numpy(objects).reshape(-1, nobj, 8)
        obj_uids, obj_transl, obj_quat = objects[:,:,0], objects[:,:,1:4].float(), objects[:,:,4:].float()
        # adhoc process to make uids available all frames
        obj_uids = obj_uids[:1].expand(nframe, -1)
        obj_quat = torch.roll(obj_quat, shifts=1, dims=-1) # xyzw -> wxyz
        obj_rot = quaternion_to_rotation_matrix(obj_quat.reshape(-1, 4)).reshape(nframe, nobj, 3, 3)
        obj_rot6d = rotmat_to_rot6d(obj_rot.reshape(-1, 3, 3)).reshape(nframe, nobj, 6)
        x_obj = torch.cat([obj_transl, obj_rot6d], dim=-1)  # [N, o, 9]
        ### contect[:,[0, 2]] are the uids of the (approx) closest object to the left/right hand
        contact = torch.from_numpy(contact).to(int)
        obj_uids = obj_uids.to(int)
        left_nearest_uid, right_nearest_uid = contact[:,0].unsqueeze(-1), contact[:, 2].unsqueeze(-1)
        
        left_idxs = torch.where(left_nearest_uid==obj_uids)
        right_idxs = torch.where(right_nearest_uid==obj_uids)
        left_nearest_obj_rot, right_nearest_obj_rot = obj_rot[left_idxs], obj_rot[right_idxs]
        left_nearest_obj_transl, right_nearest_obj_transl = obj_transl[left_idxs], obj_transl[right_idxs]

        left_nearest_obj_pc, right_nearest_obj_pc = batched_indexing(left_nearest_uid, uid_pcs_mapping), batched_indexing(right_nearest_uid, uid_pcs_mapping)
        left_nearest_obj_pc = torch.matmul(left_nearest_obj_rot, left_nearest_obj_pc.transpose(1, 2)).transpose(1, 2) + left_nearest_obj_transl.unsqueeze(1)
        right_nearest_obj_pc = torch.matmul(right_nearest_obj_rot, right_nearest_obj_pc.transpose(1, 2)).transpose(1, 2) + right_nearest_obj_transl.unsqueeze(1)

        ### process short clips for training
        #### availability was processed in `clip`
        for i in range(clip.shape[0]):
            sf, ef = clip[i] # start_frame, end_frame

            actions = annt['annotations'][str(i)]['left']['actions']+annt['annotations'][str(i)]['right']['actions']+annt['annotations'][str(i)]['both']['actions']
            objects = annt['annotations'][str(i)]['left']['objects']+annt['annotations'][str(i)]['right']['objects']+annt['annotations'][str(i)]['both']['objects']
            ## process the contact info
            lcf_idx, lcov_idx, lchj_idx, ldist_value, \
            rcf_idx, rcov_idx, rchj_idx, rdist_value, \
            is_lhand, is_rhand = get_contact_info(
                left_joint[sf: ef], right_joint[sf: ef], 
                left_nearest_obj_pc[sf: ef], right_nearest_obj_pc[sf: ef]
            )
            x_lhand_total.append(x_left[sf: ef].numpy())
            x_rhand_total.append(x_right[sf: ef].numpy())
            j_lhand_total.append(left_joint[sf: ef].numpy())
            j_rhand_total.append(right_hand[sf: ef].numpy())
            x_obj_total.append(x_obj[sf: ef].numpy())
            lhand_beta_total.append(beta[sf: ef].numpy())
            rhand_beta_total.append(beta[sf: ef].numpy())
            x_lhand_org_total.append(left_pose[sf: ef].numpy())
            x_rhand_org_total.append(right_pose[sf: ef].numpy())
            lcf_idx_total.append(lcf_idx.numpy())
            lcov_idx_total.append(lcov_idx.numpy())
            lchj_idx_total.append(lchj_idx.numpy())
            ldist_value_total.append(ldist_value.numpy())
            rcf_idx_total.append(rcf_idx.numpy())
            rcov_idx_total.append(rcov_idx.numpy())
            rchj_idx_total.append(rchj_idx.numpy())
            rdist_value_total.append(rdist_value.numpy())
            is_lhand_total.append(is_lhand)
            is_rhand_total.append(is_rhand)
            object_name_total.append(objects)
            action_name_total.append(actions)
            nframes_total.append(ef-sf+1)
            
    np.savez(
        file = data_save_path, 
        x_land = np.array(x_lhand_total, dtype = object), 
        x_rhand = np.array(x_rhand_total, dtype = object), 
        j_lhand = np.array(j_lhand_total, dtype = object), 
        j_rhand = np.array(j_rhand_total, dtype = object), 
        x_obj = np.array(x_obj_total, dtype = object), 
        lhand_beta = np.array(lhand_beta_total, dtype = object), 
        rhand_beta = np.array(rhand_beta_total, dtype = object), 
        x_lhand_org = np.array(x_lhand_org_total, dtype = object), 
        x_rhand_org = np.array(x_rhand_org_total, dtype = object), 
        lcf_idx = np.array(lcf_idx_total, dtype = object), 
        lcov_idx = np.array(lcov_idx_total, dtype = object), 
        lchj_idx = np.array(lchj_idx_total, dtype = object), 
        ldist_value = np.array(ldist_value_total, dtype = object), 
        rcf_idx = np.array(rcf_idx_total, dtype = object), 
        rcov_idx = np.array(rcov_idx_total, dtype = object), 
        rchj_idx = np.array(rchj_idx_total, dtype = object), 
        rdist_value = np.array(rdist_value_total, dtype = object), 
        is_lhand = np.array(is_lhand_total, dtype = object), 
        is_rhand = np.array(is_rhand_total, dtype = object), 
        object_name = np.array(object_name_total, dtype = object), 
        action_name = np.array(action_name_total, dtype = object), 
        nframes = np.array(nframes_total, dtype = object), 
        
    )

def processing_text():
    _config = load_config("configs/dataset/hot3d.yaml")
    data_root = _config.root
    sub_folders = [osp.join(data_root, subf) for subf in os.listdir(data_root) if osp.isdir(osp.join(data_root, subf))]
    text_description = {}
    for sf in sub_folders:
        annt_path = osp.join(sf, 'annotations.json')
        with open(annt_path) as f:
            annt = json.load(f)
        for idx in annt['annotations'].keys():
            hoi = hoi_to_str(annt['annotations'][idx])
            if not text_description.get(hoi, False):
                text_description[hoi] = []
            if annt['annotations'][idx]['text'] not in text_description[hoi]:
                text_description[hoi].append(annt['annotations'][idx]['text'])

    with open("data/hot3d/text.json", "w") as f:
        json.dump(text_description, f,indent=0)

def preprocessing_balance_weights(is_action=False):
    _config = load_config("configs/dataset/hot3d.yaml")
    if is_action:
        data_path = _config.action_train_data_path
        balance_weights_path = _config.action_balance_weights_path
    else:
        data_path = _config.data_path
        balance_weights_path = _config.balance_weights_path
    t2c_json_path = _config.t2c_json
    data_root = _config.root
    
    hoi_all = []
    sub_folders = [osp.join(data_root, subf) for subf in os.listdir(data_root) if osp.isdir(osp.join(data_root, subf))]
    for sf in sub_folders:
        annt_path = osp.join(sf, 'annotations.json')
        with open(annt_path) as f:
            annt = json.load(f)
        for annotation in annt['annotations'].values():
            hoi_all.append(hoi_to_str(annotation))

    
    text_counter = Counter(hoi_all)
    text_dict = dict(text_counter)
    text_prob = {k:1/v for k, v in text_dict.items()}
    balance_weights = [text_prob[text] for text in hoi_all]
    with open(balance_weights_path, "wb") as f:
        pickle.dump(balance_weights, f)
    with open(t2c_json_path, "w") as f:
        json.dump(text_dict, f)
        
def preprocessing_text2length():
    _config = load_config("configs/dataset/hot3d.yaml")
    data_root = _config.root
    
    text_dict = {}
    sub_folders = [osp.join(data_root, subf) for subf in os.listdir(data_root) if osp.isdir(osp.join(data_root, subf))]
    for sf in sub_folders:
        clip_path = osp.join(sf, 'clip.npy')
        annt_path = osp.join(sf, 'annotations.json')
        with open(annt_path) as f:
            annt = json.load(f)
        with open(clip_path) as f:
            clip = np.load(clip_path)
        for idx in range(clip.shape[0]):
            num_frames = (clip[idx, 1] - clip[idx, 0]).item()
            text_key = hoi_to_str(annt['annotations'][str(idx)])
            if text_key not in text_dict:
                text_dict[text_key] = [num_frames]
            else:
                text_dict[text_key].append(num_frames)

    with open("data/hot3d/text_length.json", "w") as f:
        json.dump(text_dict, f)

def print_text_data_num():
    arctic_config = load_config("configs/dataset/arctic.yaml")
    data_path = arctic_config.data_path
    t2l_json_path = arctic_config.t2l_json
    
    with np.load(data_path, allow_pickle=True) as data:
        action_name = data["action_name"]
    print(f"data num: {len(action_name)}")
    
    with open(t2l_json_path, "r") as f:
        text = json.load(f)
    print(f"text num: {len(text)}")

if __name__ == '__main__':
    # preprocessing_object()
    # preprocessing_data()
    # processing_text()
    # preprocessing_balance_weights()
    preprocessing_text2length()