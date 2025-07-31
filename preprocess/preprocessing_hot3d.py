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
from lib.utils.file import load_config
from lib.utils.frame import align_frame
from lib.utils.rot import (
    quaternion_to_angle_axis, 
    quaternion_to_rotation_matrix,
    rotmat_to_rot6d, 
    axis_angle_to_rot6d
)
from lib.utils.proc import (
    farthest_point_sample, 
    get_contact_info,
)
from lib.utils.hot3d_utils import (
    hoi_to_str, 
    hot3d_align_frame
)
from lib.models.mano import build_mano_aa

from constants.hot3d_constants import hot3d_obj_name, hot3d_obj_name_mapping



def preprocessing_object():
    _config = load_config("configs/dataset/hot3d.yaml")
    objects_folder = glob.glob(osp.join(_config.obj_root, "*"))
    with open(_config.instance_path, 'r') as f:
        ins_json = json.loads(f.read())
    uid_obj_mapping = {int(k): v['instance_name'] for k, v in ins_json.items()}
    obj_uid_mapping = {v['instance_name']: int(k) for k, v in ins_json.items()}
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
        # key = f"{object_name}"
        key = int(obj_uid_mapping[object_name.split(".")[0]])
        obj_pcs[key] = sampled_pc
        obj_pc_normals[key] = sampled_normal
        point_sets[key] = point_set[0].cpu().numpy()
        obj_path[key] = "/".join(object_path.split("/")[-2:])
    ### special case for null object
    key = int(0)
    obj_pcs[key] = np.ones_like(sampled_pc)
    obj_pc_normals[key] = np.ones_like(sampled_normal)
    point_sets[key] = np.ones_like(point_set[0].cpu().numpy())
    obj_path[key] = ""
    ### special case for null object

    os.makedirs("data/hot3d", exist_ok=True)
    with open("data/hot3d/obj.pkl", "wb") as f:
        pickle.dump(
            {
                "object_name": obj_uid_mapping, 
                "obj_pcs": obj_pcs, 
                "obj_pc_normals": obj_pc_normals, 
                "point_sets": point_sets, 
                "obj_path": obj_path, 
            }, f)
        
def preprocessing_data():
    _config = load_config("configs/dataset/hot3d.yaml")
    data_save_path = _config.data_path

    mano_dir = osp.abspath('../mano_v1_2/models/')
    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=False, num_pca=15)
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=False, num_pca=15)
    # full_joint_mapper = FullJointMapper(right_mano.J_regressor)

    with open(_config.instance_path, 'r') as f:
        ins_json = json.loads(f.read())
    obj_uid_mapping = {v['instance_name']: int(k)  for k, v in ins_json.items()} # {name: uid}
    uid_obj_mapping = {k: v['instance_name']   for k, v in ins_json.items()} # {uid: name}
    with open(_config.data_obj_pc_path, 'rb') as f:
        obj_data = pickle.load(f)
    uid_pcs_mapping = {int(uid): torch.from_numpy(pcs) for uid, pcs in obj_data['obj_pcs'].items()}

    data_root = _config.root
    sub_folders = [osp.join(data_root, subf) for subf in os.listdir(data_root) if osp.isdir(osp.join(data_root, subf))]

    # basic data
    nframes_total = []  # [Nclip, ], int, length of each clip
    all_objs_total = [] # [Nclip, Nobject], int, uid of all objects (may contain zero)
    obj_exist_total = []    # [Nclip, Nobject], bool, 
    active_flag_total = []  # [Nclip, Nframe, Nobject], bool, if the object is active (close enogh to the hands)
    
    # pose & position data
    x_lhand_total = []
    x_rhand_total = []
    j_lhand_total = []
    j_rhand_total = []
    x_obj_total = []
    lhand_beta_total = []
    rhand_beta_total = []
    x_lhand_org_total = []
    x_rhand_org_total = []

    # (maybe not used) active data
    luid_nearest_total = [] # [Nclip, Nobject], int, uid of the nearest object
    ruid_nearest_total = []

    # contact data
    lcf_idx_total = []      # [Nclip, _dynamic], int, indices of contact map along the dimension of frame
    lcon_idx_total = []     # [Nclip, _dynamic], int, indices of contact map along the dimension of object-number
    lcov_idx_total = []     # [Nclip, _dynamic], int, indices of contact map along the dimension of object-vertices
    lchj_idx_total = []     # [Nclip, _dynamic], int, indices of contact map along the dimension of hand joints
    ldist_value_total = []  # [Nclip, _dynamic], float, value of contact map
    rcf_idx_total = []
    rcon_idx_total = []
    rcov_idx_total = []
    rchj_idx_total = []
    rdist_value_total = []

    is_lhand_total = []     # [Nclip, ], int, if the hand is active (always True now)
    is_rhand_total = []

    object_name_total = []  # [Nclip, Nobject], str, uid of all objects, can be zero for not existing objects
    action_name_total = []  # [Nclip, _dynamic], str, name of the actions, can be multiple
    hoi_name_total = []     # [Nclip, ], str, a string that encodes the hoi type

    mask_hand_total = []
    mask_object_total = []
    mask_global_total = []
    
    
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

        masks = torch.from_numpy(masks).bool()
        masks_hand = masks[:, 1]
        masks_object = masks[:, 0]
        masks_global = masks[:, 3]

        left_hand, right_hand = torch.from_numpy(hand[:,:22]).float(), torch.from_numpy(hand[:,22:]).float()
        left_transl, left_wrist_quat, left_pca_pose = left_hand[:,:3], left_hand[:,3:7], left_hand[:,7:]
        right_transl, right_wrist_quat, right_pca_pose = right_hand[:,:3], right_hand[:,3:7], right_hand[:,7:]
        # left_pose, right_pose = left_pca_pose @ hand_components, right_pca_pose @ hand_components
        left_orient, right_orient = quaternion_to_angle_axis(torch.roll(left_wrist_quat, shifts=1, dims=1)), quaternion_to_angle_axis(torch.roll(right_wrist_quat, shifts=1, dims=1))

        beta = torch.zeros(nframe, 10)

        left_output = lhand_layer(global_orient=left_orient, hand_pose=left_pca_pose, transl=left_transl, betas=beta, return_full_pose=True)
        # left_joint = left_output.joints   # 16
        left_joint = left_output.joints_w_tip    # 21
        left_pose = left_output.full_pose

        right_output = rhand_layer(global_orient=right_orient, hand_pose=right_pca_pose, transl=right_transl, betas=beta, return_full_pose=True)
        # right_joint = right_output.joints
        right_joint = left_output.joints_w_tip
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
        obj_uids = obj_uids.to(int)

        # sort objects pose
        obj_uids, sort_idx = torch.sort(obj_uids, dim=1, descending=True)    # [nf, no], descending for 0 at tail
        obj_quat = torch.gather(obj_quat, index=sort_idx[...,None].expand_as(obj_quat), dim=1)
        obj_transl = torch.gather(obj_transl, index=sort_idx[...,None].expand_as(obj_transl), dim=1)

        obj_quat = torch.roll(obj_quat, shifts=1, dims=-1) # xyzw -> wxyz
        obj_rot = quaternion_to_rotation_matrix(obj_quat.reshape(-1, 4)).reshape(nframe, nobj, 3, 3)
        obj_rot6d = rotmat_to_rot6d(obj_rot.reshape(-1, 3, 3)).reshape(nframe, nobj, 6)
        x_obj = torch.cat([obj_transl, obj_rot6d], dim=-1)  # [nf, no, 9]
        ### contect[:,[0, 2]] are the uids of the (approx) closest object to the left/right hand
        contact = torch.from_numpy(contact).to(int)
        
        luid_nearest, ruid_nearest = contact[:,0].unsqueeze(-1), contact[:, 2].unsqueeze(-1)
        
        # luid_nearest_idx = torch.where(luid_nearest==obj_uids)
        # ruid_nearest_idx = torch.where(ruid_nearest==obj_uids)
        # left_nearest_obj_rot, right_nearest_obj_rot = obj_rot[luid_nearest_idx], obj_rot[ruid_nearest_idx]
        # left_nearest_obj_transl, right_nearest_obj_transl = obj_transl[luid_nearest_idx], obj_transl[ruid_nearest_idx]

        # left_nearest_obj_pc, right_nearest_obj_pc = batched_indexing(luid_nearest, uid_pcs_mapping), batched_indexing(ruid_nearest, uid_pcs_mapping)
        # left_nearest_obj_pc = torch.matmul(left_nearest_obj_rot, left_nearest_obj_pc.transpose(1, 2)).transpose(1, 2) + left_nearest_obj_transl.unsqueeze(1)
        # right_nearest_obj_pc = torch.matmul(right_nearest_obj_rot, right_nearest_obj_pc.transpose(1, 2)).transpose(1, 2) + right_nearest_obj_transl.unsqueeze(1)

        objs_pc = torch.stack([uid_pcs_mapping[uid.item()] for uid in obj_uids[0]], dim=0)[None].expand(nframe, -1, -1, -1)
        objs_pc = torch.matmul(obj_rot, objs_pc.transpose(-1, -2)).transpose(-1, -2) + obj_transl.unsqueeze(-2)

        obj_exist = obj_uids!=0

        ### process short clips for training
        #### availability was processed in `clip`
        for i in range(clip.shape[0]):
            if not annt['annotations'][str(i)]['valid']:
                continue
            sf, ef = clip[i] # start_frame, end_frame
            overall_mask = obj_exist[sf:ef] & masks_hand[sf:ef, None] & masks_object[sf:ef, None] & masks_global[sf:ef, None]
            if not overall_mask.any():  # if all data are masked
                continue

            hoistr = hoi_to_str(annt['annotations'][str(i)])
            actions, objects = set(), set()
            for hoi in annt['annotations'][str(i)]['hois']:
                actions.add(hoi['action'])
                objects.update(hoi['objects'])

            ## process the contact info
            lcf_idx, lcon_idx, lcov_idx, lchj_idx, ldist_value, \
            rcf_idx, rcon_idx, rcov_idx, rchj_idx, rdist_value, \
            is_lhand, is_rhand, active_flag = get_contact_info(
                left_joint[sf: ef], right_joint[sf: ef], objs_pc[sf: ef], overall_mask, 
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

            luid_nearest_total.append(luid_nearest[sf: ef, 0].numpy())
            ruid_nearest_total.append(ruid_nearest[sf: ef, 0].numpy())

            lcf_idx_total.append(lcf_idx.numpy())
            lcon_idx_total.append(lcon_idx.numpy())
            lcov_idx_total.append(lcov_idx.numpy())
            lchj_idx_total.append(lchj_idx.numpy())
            ldist_value_total.append(ldist_value.numpy())

            rcf_idx_total.append(rcf_idx.numpy())
            rcon_idx_total.append(rcon_idx.numpy())
            rcov_idx_total.append(rcov_idx.numpy())
            rchj_idx_total.append(rchj_idx.numpy())
            rdist_value_total.append(rdist_value.numpy())

            is_lhand_total.append(is_lhand)
            is_rhand_total.append(is_rhand)

            object_name_total.append(list(objects))
            action_name_total.append(list(actions))
            hoi_name_total.append(hoistr)

            nframes_total.append(ef-sf)

            all_objs_total.append(obj_uids[0].numpy())
            obj_exist_total.append(obj_exist[sf:ef].numpy())
            active_flag_total.append(active_flag.numpy())

            mask_hand_total.append(masks_hand[sf:ef].numpy())
            mask_object_total.append(masks_object[sf:ef].numpy())
            mask_global_total.append(masks_global[sf:ef].numpy())

    data_dict = dict(
        active_flag = active_flag_total,
        obj_exist = obj_exist_total, 
        luid_nearest = luid_nearest_total, 
        ruid_nearest = ruid_nearest_total, 
        x_lhand = x_lhand_total,
        x_rhand = x_rhand_total,
        x_obj = x_obj_total,
        j_lhand = j_lhand_total, 
        j_rhand = j_rhand_total, 
        lhand_beta = lhand_beta_total,
        rhand_beta = rhand_beta_total,
        lhand_org = x_lhand_org_total, 
        rhand_org = x_rhand_org_total,
        
        mask_hand = mask_hand_total, 
        mask_object = mask_object_total, 
        mask_global = mask_global_total, 
    )
            
    np.savez(
        file = data_save_path, 

        **hot3d_align_frame(data_dict, preset_max_nframes=_config.max_frames),
        nframes = np.array(nframes_total), 
        all_objs = np.array(all_objs_total), 
        is_lhand = np.array(is_lhand_total), 
        is_rhand = np.array(is_rhand_total), 
        
        lcf_idx = np.array(lcf_idx_total, dtype = object), 
        lcon_idx = np.array(lcon_idx_total, dtype = object), 
        lcov_idx = np.array(lcov_idx_total, dtype = object), 
        lchj_idx = np.array(lchj_idx_total, dtype = object), 
        ldist_value = np.array(ldist_value_total, dtype = object), 
        rcf_idx = np.array(rcf_idx_total, dtype = object), 
        rcon_idx = np.array(rcon_idx_total, dtype = object), 
        rcov_idx = np.array(rcov_idx_total, dtype = object), 
        rchj_idx = np.array(rchj_idx_total, dtype = object), 
        rdist_value = np.array(rdist_value_total, dtype = object), 

        object_name = np.array(object_name_total, dtype = object), 
        action_name = np.array(action_name_total, dtype = object), 
        hoi_name = np.array(hoi_name_total, dtype = object), 
        
        
    )

def processing_text():
    _hand_type = {'left': 'the left hand', 'right': 'the right hand', 'both': 'both hands', }
    def process_text(hois):
        all_text = []
        for hoi in hois:
            all_text.append(make_text(hand=hoi['hand'], act= hoi['action'], objs=hoi['objects']))   # list[list[str]]
        final_text = []
        cur_text = []
        def _dfs(depth):
            if depth == len(all_text):
                final_text.append(', '.join(cur_text))
                return
            for text in all_text[depth]:
                cur_text.append(text)
                _dfs(depth + 1)
                cur_text.pop()
        _dfs(0)
        return final_text

    def make_text(act, objs, hand,):
        hand = _hand_type[hand]
        text_list = []
        if len(objs)==1:
            obj0 = objs[0]
            for obj0_name in hot3d_obj_name_mapping[obj0]:
                text_list.append(make_action(act, hand, obj0_name))
        elif len(objs)==2:
            for obj0_name in hot3d_obj_name_mapping[objs[0]]:
                for obj1_name in hot3d_obj_name_mapping[objs[1]]:
                    text_list.append(make_action(act, hand, obj0_name, obj1_name))
        else:
            raise NotImplementedError
        return text_list
        
    def make_action(act, hand, obj0, obj1=None):
        if act in ['clean the whiteboard', 'cook', 'do exercise', 'drink coffee', 'write on the whiteboard', 'make a call', 'take photo', ]:
            if obj1 != None:
                return f"{act} using the {obj0} and the {obj1} with {hand}"
            else:
                return f"{act} using the {obj0} with {hand}"
        if act in ['hold', 'move', 'pick up', 'play', 'put down', 'reach for', 'rotate', 'type', 'use', ]:
            if obj1 != None:
                return f"{act} the {obj0} and the {obj1} with {hand}"
            else:
                return f"{act} the {obj0} with {hand}"
        if act == 'pour coffee':
            if obj1 != None:
                return f"{act} into the {obj0} from the {obj1} with {hand}"
            else:
                return f"{act} using the {obj0} with {hand}"
        if act == 'pour the ingredient':
            if obj1 != None:
                return f"{act} of the {obj0} into the {obj1} with {hand}"
            else:
                return f"{act} from the {obj0} with {hand}"
        
        raise NotImplementedError
    
    _config = load_config("configs/dataset/hot3d.yaml")
    data_root = _config.root
    sub_folders = [osp.join(data_root, subf) for subf in os.listdir(data_root) if osp.isdir(osp.join(data_root, subf))]
    text_description = {}
    for sf in sub_folders:
        annt_path = osp.join(sf, 'annotations.json')
        with open(annt_path) as f:
            annt = json.load(f)
        res = sorted(glob.glob(osp.join(sf, "*")), key=lambda s: s.split('/')[-1].split('.')[0]) 
        clip, contact, gaze, hand, _, _, _, masks, objects, _ = [np.load(fp) for fp in res if fp.find('.npy')!=-1]
        masks = torch.from_numpy(masks).bool()
        masks_hand = masks[:, 1]
        masks_object = masks[:, 0]
        masks_global = masks[:, 3]
        obj_uids = torch.from_numpy(objects[:1,::8]).to(int)
        # sort objects pose
        obj_uids, sort_idx = torch.sort(obj_uids, dim=1, descending=True)    # [nf, no], descending for 0 at tail
        obj_exist = obj_uids!=0
        for i in range(clip.shape[0]):
            if not annt['annotations'][str(i)]['valid']:
                continue
            sf, ef = clip[i] # start_frame, end_frame
            overall_mask = obj_exist & masks_hand[sf:ef, None] & masks_object[sf:ef, None] & masks_global[sf:ef, None]
            if not overall_mask.any():  # if all data are masked
                continue

        
        for idx in annt['annotations'].keys():
            hoi_name = hoi_to_str(annt['annotations'][idx])
            if not text_description.get(hoi_name, False):
                text_description[hoi_name] = []
            text_description[hoi_name].extend(
                process_text(annt['annotations'][idx]['hois'])
            )


    
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
        res = sorted(glob.glob(osp.join(sf, "*")), key=lambda s: s.split('/')[-1].split('.')[0]) 
        clip, contact, gaze, hand, _, _, _, masks, objects, _ = [np.load(fp) for fp in res if fp.find('.npy')!=-1]
        masks = torch.from_numpy(masks).bool()
        masks_hand = masks[:, 1]
        masks_object = masks[:, 0]
        masks_global = masks[:, 3]
        obj_uids = torch.from_numpy(objects[:1,::8]).to(int)
        # sort objects pose
        obj_uids, sort_idx = torch.sort(obj_uids, dim=1, descending=True)    # [nf, no], descending for 0 at tail
        obj_exist = obj_uids!=0
        for i in range(clip.shape[0]):
            if not annt['annotations'][str(i)]['valid']:
                continue
            sf, ef = clip[i] # start_frame, end_frame
            overall_mask = obj_exist & masks_hand[sf:ef, None] & masks_object[sf:ef, None] & masks_global[sf:ef, None]
            if not overall_mask.any():  # if all data are masked
                continue
            hoi_all.append(hoi_to_str(annt['annotations'][str(i)]))

    
    text_counter = Counter(hoi_all)
    text_dict = dict(text_counter)
    text_prob = {k:1/v for k, v in text_dict.items()}
    balance_weights = [text_prob[text] for text in hoi_all]
    with open(balance_weights_path, "wb") as f:
        pickle.dump(balance_weights, f)
    with open(t2c_json_path, "w") as f:
        json.dump(text_dict, f, indent=0)
        
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
            if not annt['annotations'][str(idx)]['valid']: continue
            num_frames = (clip[idx, 1] - clip[idx, 0]).item()
            text_key = hoi_to_str(annt['annotations'][str(idx)])
            if text_key not in text_dict:
                text_dict[text_key] = [num_frames]
            else:
                text_dict[text_key].append(num_frames)

    with open("data/hot3d/text_length.json", "w") as f:
        json.dump(text_dict, f, indent=0)

def print_text_data_num():
    arctic_config = load_config("configs/dataset/hot3d.yaml")
    data_path = arctic_config.data_path
    t2l_json_path = arctic_config.t2l_json
    
    with np.load(data_path, allow_pickle=True) as data:
        action_name = data["action_name"]
    print(f"data num: {len(action_name)}")
    
    with open(t2l_json_path, "r") as f:
        text = json.load(f)
    print(f"text num: {len(text)}")