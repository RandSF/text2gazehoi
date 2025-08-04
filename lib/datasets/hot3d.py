import time
import random
import numpy as np
import json

from torch.utils.data import Dataset

from lib.models.object import build_object_model
from lib.utils.frame import get_valid_mask
from lib.utils.augm import (
    augmentation, 
    augmentation_joints, 
    get_augm_rot, 
    get_augm_scale, 
)

from lib.utils.proc import (
    pc_normalize, 
    process_dist_map, 
    select_from_groups, 
)
from lib.utils.hot3d_utils import (
    hot3d_get_contact_map, 
    hot3d_process_dist_map, 
    hot3d_get_valid_mask
)
import os
import os.path as osp
import json
from constants.hot3d_constants import hot3d_obj_name

NUM_OBJ = 6

# rotation: (xyzw) quaternion
def load_data(base_path):
    timestamp = np.load(osp.join(base_path, 'timestamps.npy'))
    gaze = np.load(osp.join(base_path, 'gaze.npy'))    # gaze_direction (3) + gaze_center_in_world (3)
    hand = np.load(osp.join(base_path, 'hand.npy'))    # left_hand (22) + right_hand (22), hand = wrist_pose (7, translation (3) + rotation (4)) + joint_angles (15)
    handjoints = np.load(osp.join(base_path, 'handjoints.npy'))    # left_hand (20*3) + right_hand (20*3)-
    head =np.load(osp.join(base_path, 'head.npy'))     # head_direction (3) + head_translation (3) + head_rotation (4, quat_xyzw)
    objects = np.load(osp.join(base_path, 'objects.npy'))  # object_data (8) * 6 objects (at most 6 objects), object_data = object_uid (1) + object_pose (7, translation (3) + rotation (4))         
    masks = np.load(osp.join(base_path, 'masks.npy'))  # 4: obj, hand, head, qa
    contact = np.load(osp.join(base_path, 'contact.npy'))   # 4: left_hand_object_info(2, nearest_uid + distance) + left_hand_object_info(2)
    return timestamp, gaze, hand, handjoints, head, objects, masks, contact
        
class BaseHOT3D(Dataset):
    def __init__(
        self, 
        data_path, 
        data_obj_pc_path, 
        text_json, 
        max_frames, 
        data_ratio=1.0, 
        augm=False, 
        instance_path = 'data/hot3dgaze/instance.json', 
        **kwargs
    ):
        super().__init__()

        self.data_path = data_path
        self.data_obj_pc_path = data_obj_pc_path
        self.max_frames = max_frames
        self.data_ratio = data_ratio
        self.augm = augm

        with np.load(data_path, allow_pickle=True) as data:
            self.hoi_name = data["hoi_name"]
            self.nframes = data["nframes"]
        with open(text_json, "r") as f:
            self.text_description = json.load(f)

        with open(instance_path, 'r') as f:
            ins_json = json.loads(f.read())
        self.uid_obj_mapping = {k: v['instance_name'] for k, v in ins_json.items()}
        self.obj_uid_mapping = {v['instance_name']: k for k, v in ins_json.items()}

        self.object_model = build_object_model(data_obj_pc_path)
        
    def __len__(self):
        return int(len(self.hoi_name)*self.data_ratio)
    
    def _tic(self):
        self.start_time = time.time()
        print("Start to read data hot3d!!!")
    def _toc(self):
        print(f"Finish to read data hot3d!! {time.time()-self.start_time:.2f}s")
        print(f"length of data: {self.__len__()}")

class SequenceHOT3D(BaseHOT3D): # point encoder
    def __init__(
        self, 
        data_path, 
        data_obj_pc_path, 
        text_json, 
        max_frames, 
        data_ratio=1.0, 
        augm=False, 
        instance_path = 'data/hot3dgaze/instance.json', 
        **kwargs
    ):
        self._tic()
        super().__init__(data_path, data_obj_pc_path, text_json, max_frames, data_ratio, augm, instance_path)
        self._toc()
        
    def __getitem__(self, index):
        item = {}
        
        nframes = self.nframes[index]
        if nframes > self.max_frames:
            nframes = self.max_frames
        seq_time = np.array([nframes/self.max_frames], dtype=np.float32)
        if self.augm:
            augm_scale = 1-(2*np.random.rand()*0.05-0.05)
            seq_time *= augm_scale
            if seq_time > 1:
                seq_time = np.array([1.0], dtype=np.float32)
        item["seq_time"] = seq_time
            
        hoi_name = self.hoi_name[index]
        text_candidates = self.text_description[hoi_name]
        item["text"] = random.choice(text_candidates)
        return item
    
    
class ContactHOT3D(BaseHOT3D): # point encoder
    def __init__(
        self, 
        data_path, 
        data_obj_pc_path, 
        text_json, 
        max_frames,
        data_ratio=1.0, 
        augm=False, 
        instance_path = 'data/hot3dgaze/instance.json', 
        **kwargs
    ):
        self._tic()
        super().__init__(data_path, data_obj_pc_path, text_json, max_frames, data_ratio, augm, instance_path)

        with np.load(data_path, allow_pickle=True) as data:
            self.all_obj_uids = data["all_objs"]
            self.obj_name = data["object_name"]
            # self.luid_nearest = data["luid_nearest"]
            # self.ruid_nearest = data["ruid_nearest"]
            self.lcon_idx = data['lcon_idx'] # left contact object number idx
            self.lcov_idx = data["lcov_idx"] # left contact object verts idx
            self.rcon_idx = data['rcon_idx'] # right contact object number idx
            self.rcov_idx = data["rcov_idx"] # right contact object verts idx

            self.obj_exist = data['obj_exist']

            self.is_lhand = data["is_lhand"]
            self.is_rhand = data["is_rhand"]
        self._toc()
    
    def __getitem__(self, index):
        item = {}
        
        is_lhand = self.is_lhand[index]
        is_rhand = self.is_rhand[index]
        item["is_lhand"] = is_lhand
        item["is_rhand"] = is_rhand
        
        hoi_name = self.hoi_name[index]
        
        text_candidates = self.text_description[hoi_name]
        text = random.choice(text_candidates) ##!##!#!
        item["text"] = text
        
        normalized_obj_pc, obj_norm_scale = [], []
        for uid in self.all_obj_uids[index]:
            # _, obj_pc, _, _  = self.object_model(str(uid))
            _, obj_pc, _, _  = self.object_model(uid)
            if self.augm:
                raise NotImplementedError
                aug_scale = get_augm_scale(0.2).numpy()
                obj_pc = obj_pc*aug_scale
                aug_rotmat = get_augm_rot(15, 15, 15).numpy()
                obj_pc = np.einsum("ij,kj->ki", aug_rotmat, obj_pc)
            _normalized_obj_pc, _, _obj_norm_scale = pc_normalize(obj_pc, return_params=True)
            normalized_obj_pc.append(_normalized_obj_pc)
            obj_norm_scale.append(_obj_norm_scale)
        normalized_obj_pc = np.stack(normalized_obj_pc, axis=0)
        obj_norm_scale = np.array(obj_norm_scale)

        item['normalized_obj_pc'] = normalized_obj_pc
        item['obj_scale'] = obj_norm_scale

        lcon_idx = self.lcon_idx[index]
        lcov_idx = self.lcov_idx[index]
        rcon_idx = self.rcon_idx[index]
        rcov_idx = self.rcov_idx[index]
        lcov_map = hot3d_get_contact_map(lcon_idx, lcov_idx, 6, 1024, is_lhand).reshape(6, 1024, -1)
        rcov_map = hot3d_get_contact_map(rcon_idx, rcov_idx, 6, 1024, is_rhand).reshape(6, 1024, -1)
        cov_map = (lcov_map+rcov_map)>0
        item["cov_map"] = cov_map.astype(np.float32)
        # item["lcov_map"] = lcov_map.astype(np.float32)
        # item["rcov_map"] = rcov_map.astype(np.float32)
        
        no, v, j = 6, 1024, 21
        # obj_vert_flag = self.obj_exist[index][...,None,None]
        # # obj_vert_flag = obj_vert_flag.repeat(v, axis=-2).repeat(j, axis=-1).reshape(-1, no*v, j)
        # item['map_mask'] = obj_vert_flag[0] # contact map is static for the whole clip

        active_flag = cov_map.sum(-1).sum(-1) > 0
        item['map_mask'] = active_flag[...,None,None].astype(np.float32)

        return item
    
    
class MotionHOT3D(BaseHOT3D):
    def __init__(
        self, 
        data_path, 
        data_obj_pc_path, 
        text_json, 
        max_frames,
        data_ratio=1.0, 
        augm=False, 
        instance_path = 'data/hot3dgaze/instance.json', 
        **kwargs
    ):
        self._tic()
        super().__init__(data_path, data_obj_pc_path, text_json, max_frames, data_ratio, augm, instance_path)
        with np.load(data_path, allow_pickle=True) as data:
            self.all_obj_uids = data["all_objs"]
            self.luid_nearest = data["luid_nearest"]
            self.ruid_nearest = data["ruid_nearest"]
            self.x_lhand = data["x_lhand"]
            self.x_rhand = data["x_rhand"]
            self.x_obj = data["x_obj"]
            self.lhand_org = data["lhand_org"]
            self.rhand_org = data["rhand_org"]
            self.lcf_idx = data["lcf_idx"] # left hand contact frame idx
            self.lcon_idx = data["lcon_idx"] # left contact object number idx
            self.lcov_idx = data["lcov_idx"] # left contact object verts idx
            self.lchj_idx = data["lchj_idx"] # left contact hand joints idx
            self.ldist_value = data["ldist_value"]
            self.rcf_idx = data["rcf_idx"] # right hand contact frame idx
            self.rcon_idx = data["rcon_idx"] # right contact object number idx
            self.rcov_idx = data["rcov_idx"] # right contact object verts idx
            self.rchj_idx = data["rchj_idx"] # right contact hand joints idx
            self.rdist_value = data["rdist_value"]

            self.active_flag = data['active_flag']
            self.obj_exist = data['obj_exist']
            self.mask_hand = data['mask_hand']
            self.mask_object = data['mask_object']
            self.mask_global = data['mask_global']

            self.is_lhand = data["is_lhand"]
            self.is_rhand = data["is_rhand"]


        self._toc()
    
    def __getitem__(self, index):
        item = {}

        nframes = self.nframes[index]
        is_lhand = self.is_lhand[index]
        is_rhand = self.is_rhand[index]
        
        item["is_lhand"] = is_lhand
        item["is_rhand"] = is_rhand

        if nframes > self.max_frames:
            init_frame = np.random.randint(0, nframes-self.max_frames)
            nframes = self.max_frames
        else:
            init_frame = 0
        
        item['init_frame'] = init_frame
        item['nframes'] = nframes

        x_obj = self.x_obj[index][init_frame:init_frame+self.max_frames]
        if self.augm:
            raise NotImplementedError("augmentation not valid")
            x_obj[:nframes], aug_rotmat, aug_trans = augmentation(x_obj[:nframes])
        item["x_obj"] = x_obj

        if is_lhand:
            x_lhand = self.x_lhand[index][init_frame:init_frame+self.max_frames]
            if self.augm:
                raise NotImplementedError("augmentation not valid")
                lhand_org = self.lhand_org[index][init_frame:init_frame+nframes]
                x_lhand[:nframes], _, _ \
                    = augmentation(
                        x_lhand[:nframes], 
                        hand_org=lhand_org, 
                        aug_rotmat=aug_rotmat, 
                        aug_trans=aug_trans
                    )
        else:
            x_lhand = np.zeros((self.max_frames, 99), dtype=np.float32)
        
        item["x_lhand"] = x_lhand

        if is_rhand:
            x_rhand = self.x_rhand[index][init_frame:init_frame+self.max_frames]
            if self.augm:
                raise NotImplementedError("augmentation not valid")
                rhand_org = self.rhand_org[index][init_frame:init_frame+nframes]
                x_rhand[:nframes], _, _ \
                    = augmentation(
                        x_rhand[:nframes], 
                        hand_org=rhand_org, 
                        aug_rotmat=aug_rotmat, 
                        aug_trans=aug_trans
                    )
        else:
            x_rhand = np.zeros((self.max_frames, 99), dtype=np.float32)
        item["x_rhand"] = x_rhand

        # action_name = self.action_name[index]
        hoi_name = self.hoi_name[index]
        max_frames = self.max_frames
        
        mask_hand = self.mask_hand[index][init_frame:init_frame+self.max_frames]
        mask_object = self.mask_object[index][init_frame:init_frame+self.max_frames]
        mask_global = self.mask_global[index][init_frame:init_frame+self.max_frames]
        active_flag = self.active_flag[index][init_frame:init_frame+self.max_frames]
        exist_flag = self.obj_exist[index][init_frame:init_frame+self.max_frames]
        item['_mask_hand'] = mask_hand
        item['_mask_object'] = mask_object
        item['_mask_global'] = mask_global
        valid_mask_lhand, valid_mask_rhand, valid_mask_obj = get_valid_mask(
            is_lhand, is_rhand, max_frames, 
            NUM_OBJ, nframes, 
            exist_flag, mask_hand, mask_object, mask_global, ) # max_frames: 2x frames
        item["valid_mask_lhand"] = valid_mask_lhand
        item["valid_mask_rhand"] = valid_mask_rhand
        item["valid_mask_obj"] = valid_mask_obj

        

        text_candidates = self.text_description[hoi_name]
        text = random.choice(text_candidates)
        item["text"] = text
        item["text_orig"] = hoi_name
        
        ### get point clouds of all objects
        obj_pc, normalized_obj_pc, obj_pc_normal, obj_norm_cent, obj_norm_scale = [], [], [], [], []
        for obj in self.all_obj_uids[index]:
            # _, _obj_pc, _obj_pc_normal, _ = self.object_model(str(obj))
            _, _obj_pc, _obj_pc_normal, _ = self.object_model(obj)
            _normalized_obj_pc, _obj_norm_cent, _obj_norm_scale = pc_normalize(_obj_pc, return_params=True)
            obj_pc.append(_obj_pc)
            obj_pc_normal.append(_obj_pc_normal)
            normalized_obj_pc.append(_normalized_obj_pc)
            obj_norm_cent.append(_obj_norm_cent)
            obj_norm_scale.append(_obj_norm_scale)
        item["obj_pc"] = np.stack(obj_pc, axis=0)
        item["normalized_obj_pc"] = np.stack(normalized_obj_pc, axis=0)
        item["obj_pc_normal"] = np.stack(obj_pc_normal, axis=0)
        item["obj_cent"] = np.stack(obj_norm_cent, axis=0)
        item["obj_scale"] = np.stack(obj_norm_scale, axis=0)

        lcf_idx = self.lcf_idx[index] 
        lcon_idx = self.lcon_idx[index]
        lcov_idx = self.lcov_idx[index]
        lchj_idx = self.lchj_idx[index]
        ldist_value = self.ldist_value[index]

        ldist_map = hot3d_process_dist_map(
            self.max_frames, init_frame, NUM_OBJ, 
            lcf_idx, lcon_idx, lcov_idx, lchj_idx, 
            ldist_value, is_lhand)
        item["ldist_map"] = ldist_map

        rcf_idx = self.rcf_idx[index]
        rcon_idx = self.rcon_idx[index]
        rcov_idx = self.rcov_idx[index]
        rchj_idx = self.rchj_idx[index]
        rdist_value = self.rdist_value[index]
        
        rdist_map = hot3d_process_dist_map(
            self.max_frames, init_frame, NUM_OBJ, 
            rcf_idx, rcon_idx, rcov_idx, rchj_idx, 
            rdist_value, is_rhand)
        item["rdist_map"] = rdist_map
        
        lcov_map = hot3d_get_contact_map(lcon_idx, lcov_idx, 6, 1024, is_lhand).reshape(6, 1024)
        rcov_map = hot3d_get_contact_map(rcon_idx, rcov_idx, 6, 1024, is_rhand).reshape(6, 1024)
        cov_map = (lcov_map+rcov_map)>0
        item["cov_map"] = cov_map.astype(np.float32)

        active_flag = np.logical_or(rdist_map>0, ldist_map>0).max(axis=-1).max(axis=-1)
        item['active_flag'] = active_flag.astype(np.float32)
        return item