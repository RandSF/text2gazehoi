import torch
import numpy as np

class FullJointMapper:
    def __init__(self, orig_joint_regressor, ):
        self.orig_joint_regressor = orig_joint_regressor # same for the right and left hands
        self.joint_regressor = torch.zeros(21, 778)

        # changed MANO joint set
        self.joint_num = 21 # manually added fingertips
        self.orig_joints_name = ('Wrist', 'Index_1', 'Index_2', 'Index_3', 'Middle_1', 'Middle_2', 'Middle_3', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Ring_1', 'Ring_2', 'Ring_3', 'Thumb_1', 'Thumb_2', 'Thumb_3')
        self.joints_name = ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4')

        for i, jname in enumerate(self.orig_joints_name):
            self.joint_regressor[self.joints_name.index(jname)] = self.orig_joint_regressor[i]

        self.joint_regressor[self.joints_name.index('Thumb_4')] = torch.tensor([1 if i == 745 else 0 for i in range(778)], dtype=torch.float32).reshape(1,-1)
        self.joint_regressor[self.joints_name.index('Index_4')] = torch.tensor([1 if i == 317 else 0 for i in range(778)], dtype=torch.float32).reshape(1,-1)
        self.joint_regressor[self.joints_name.index('Middle_4')] = torch.tensor([1 if i == 445 else 0 for i in range(778)], dtype=torch.float32).reshape(1,-1)
        self.joint_regressor[self.joints_name.index('Ring_4')] = torch.tensor([1 if i == 556 else 0 for i in range(778)], dtype=torch.float32).reshape(1,-1)
        self.joint_regressor[self.joints_name.index('Pinky_4')] = torch.tensor([1 if i == 673 else 0 for i in range(778)], dtype=torch.float32).reshape(1,-1)
    def get_full_joint(self, vertices, orig_joints=None):
        joints = torch.einsum("bvd,jv->bjd", vertices, self.joint_regressor)
        if orig_joints != None:
            for i, jname in enumerate(self.orig_joints_name):
                assert torch.norm(joints[:,self.joints_name.index(jname)]-orig_joints[:,i])<0.5
        return joints


def hoi_to_str(annt, include_action=True, include_object=True):
    # convert the hoi info to string in the format: 
    # "left_actions+left_objects|r_as+r_os|both_as+both_os" 
    # ("|" is the hand separator, "+" is the action-object separator, "=" is the inner separator inside actions/objects)
    hoi_list = []
    for hoi in annt['hois']:
        hand = hoi['hand']
        act = hoi['action'] if include_action else ""
        objs = hoi['objects'] if include_object else []
        hoi_list.append(hoi_to_str_single(hand,act,objs))
    
    hoi_total = "|".join(hoi_list)
    return hoi_total

def hoi_to_str_single(h, a, o):
    o = "=".join(o)
    return f"{h}+{a}+{o}"

def batched_indexing(idx_batch: torch.Tensor, mapping: dict):
    res = []
    assert torch.is_tensor(idx_batch)
    assert idx_batch.ndim == 2
    for idx in idx_batch:
        res.append(mapping[idx.item()])
    return torch.stack(res, dim=0)

### a modified version, all data is available
def get_contact_info(
        left_joint, right_joint, obj_verts, obj_exist
    ):
    # joints: [Nf, J, 3]
    # obj: [Nf, No, V, 3], [Nf, No, ]
    # exist: [Nf, No]
    contact_threshold = 0.02
    LARGE = 100 + contact_threshold

    nf, j, d = left_joint.shape
    nf, no, v, d = obj_verts.shape
    obj_verts = obj_verts.flatten(1, 2) # [Nf, No*V, 3]

    ldist = torch.cdist(obj_verts, left_joint).reshape(nf, no, v, j)
    ldist = ldist + LARGE * torch.ones_like(ldist) * (~obj_exist[...,None,None])   # where objects do not exist are added with a LARGE value
    lcf_idx, lcon_idx, lcov_idx, lchj_idx =torch.where(ldist < contact_threshold) # c: contact, ov: object vertex, hj: hand joint
    ldist_value = ldist[lcf_idx, lcon_idx, lcov_idx, lchj_idx]

    rdist = torch.cdist(obj_verts, right_joint).reshape(nf, no, v, j)
    rdist = rdist + LARGE * torch.ones_like(rdist) * (~obj_exist[...,None,None])   # where objects do not exist are added with a LARGE value
    rcf_idx, rcon_idx, rcov_idx, rchj_idx = torch.where(rdist < contact_threshold)
    rdist_value = rdist[rcf_idx, rcon_idx, rcov_idx, rchj_idx]

    is_lhand, is_rhand = 1, 1
    
    active_flag = torch.min(torch.min(torch.minimum(ldist, rdist), dim=-1)[0], dim=-1)[0] < contact_threshold

    return (lcf_idx, lcon_idx, lcov_idx, lchj_idx, ldist_value, 
            rcf_idx, rcon_idx, rcov_idx, rchj_idx, rdist_value, 
            is_lhand, is_rhand, active_flag)

# def farthest_point_sample(xyz, npoint, random=False):
#     """
#     Input:
#         xyz: pointcloud data, [B, N, 3]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [B, npoint]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
#     distance = torch.ones(B, N).to(device) * 1e10
#     if random:
#         farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
#     else:
#         farthest = 0
#     batch_indices = torch.arange(B, dtype=torch.long).to(device)
#     for i in range(npoint):
#         centroids[:, i] = farthest
#         centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
#         dist = torch.sum((xyz - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = torch.max(distance, -1)[1]
#     return centroids
# ### a modified version, all data is available

import warnings
def hot3d_align_frame(total_dict: dict, preset_max_nframes = 300):
    # fill the data with shape of `[nframes, ...]` to be with shape of '[max_nframes, ...]', which is filled by zero
    max_nframes = preset_max_nframes
    value = next(iter(total_dict.values())) # ndata, frames, _, 
    # ndata: the number of data in `total_dict.values()`
    for _value in value:
        nframes = len(_value)
        if nframes > max_nframes:
            max_nframes = nframes
    ndata = len(value)

    final_dict = {}
    for key, value in total_dict.items():
        # value: list of clips
        final_value = []
        arr_flag = isinstance(value[0], np.ndarray)
        for i, data in enumerate(value):
            nframes = data.shape[0] if arr_flag else len(data)
            if nframes == 0:
                warnings.warn(f"find zero-length data in {key}, it is the {i}-th clip, check it if it is unexpected.")
                continue
            if arr_flag:
                padding = np.zeros([max_nframes-nframes, *data.shape[1:]], dtype=data.dtype)
                final_data = np.concatenate([data, padding], axis=0)
            elif isinstance(data, list):   # list
                final_data = data + [0 for _ in range(max_nframes-nframes)]
            else:raise ValueError
            final_value.append(final_data)
        final_dict[key] = np.array(final_value)
    return final_dict

def hot3d_get_contact_map(con_idx, cov_idx, n_obj, n_vert, is_hand):
    contact_map = np.zeros([n_obj, n_vert])
    if is_hand:
        contact_map[con_idx, cov_idx] = 1
    return contact_map

def hot3d_process_dist_map(
    max_nframes, init_frame, num_object, 
    cf_idx, con_idx, cov_idx, chj_idx, 
    dist_value, is_hand
):
    dist_map = np.zeros((max_nframes, num_object, 1024, 21), dtype=np.float32)
    if is_hand:
        f_idx_filtered = np.where((init_frame<=cf_idx) & (cf_idx<init_frame+max_nframes))[0]
        cf_idx_selected = cf_idx[f_idx_filtered]
        cf_idx_moved = cf_idx_selected-init_frame
        con_idx_selected = con_idx[f_idx_filtered]
        cov_idx_selected = cov_idx[f_idx_filtered]
        chj_idx_selected = chj_idx[f_idx_filtered]
        dist_value_selected = dist_value[f_idx_filtered]
        dist_map[cf_idx_moved, con_idx_selected, cov_idx_selected, chj_idx_selected] = dist_value_selected
    return dist_map

def hot3d_get_valid_mask(is_lhand, is_rhand, exist_flag, 
                         mask_hand, mask_object, mask_global, 
                         init_frame, nframes, 
                         num_object, valid_nframes):
    valid_mask_lhand = np.zeros((nframes))
    valid_mask_rhand = np.zeros((nframes))
    valid_mask_obj = np.zeros((nframes, num_object))
    valid_mask_lhand[:valid_nframes] = 1
    valid_mask_rhand[:valid_nframes] = 1
    valid_mask_obj[np.where(exist_flag)] = 1
    if not is_lhand:
        valid_mask_lhand[:] = 0
    if not is_rhand:
        valid_mask_rhand[:] = 0
    return (
        (valid_mask_lhand * mask_hand * mask_global).astype(np.bool), 
        (valid_mask_rhand * mask_hand * mask_global).astype(np.bool), 
        (valid_mask_obj * mask_object[...,None] * mask_global[...,None]).astype(np.bool)
    )


def hot3d_proc_cond_contact_estimator(obj_scale, obj_feat, enc_text, nobjs, npts, use_scale):
    if use_scale:
        enc_text_expand = enc_text.unsqueeze(1).unsqueeze(1)
        enc_text_expand = enc_text_expand.expand(-1, nobjs, npts, -1)
        obj_scale_expand2 = obj_scale.unsqueeze(-1).unsqueeze(-1)
        obj_scale_expand2 = obj_scale_expand2.expand(-1, nobjs, npts, -1)
        condition = torch.cat([obj_scale_expand2, obj_feat, enc_text_expand], dim=-1)
    else:
        raise NotImplementedError
        condition = torch.cat([obj_feat, enc_text_expand], dim=2)
    return condition

def hot3d_proc_obj_feat_final(
    contact_estimator, 
    obj_scale, obj_cent, 
    obj_feat, enc_text, npts, 
    use_obj_scale_centroid, 
    use_scale, use_contact_feat, 
    return_plot=False, 
):
    obj_feat_cov, est_contact_map, est_contact_map_plot = proc_obj_feat_cov(
        contact_estimator, 
        obj_scale, obj_feat, enc_text, npts, 
        use_scale, use_contact_feat, 
    )
    if use_obj_scale_centroid:
        obj_scale_expand1 = obj_scale.unsqueeze(1)
        obj_feat_final = torch.cat([obj_feat_cov, obj_scale_expand1, obj_cent], dim=1)
    else:
        obj_feat_final = obj_feat_cov
    if return_plot:
        return obj_feat_final, est_contact_map, est_contact_map_plot
    else:
        return obj_feat_final, est_contact_map
    
# def hot3d_proc_obj_feat_final_train(cov_map, obj_scale, obj_cent, obj_feat, use_obj_scale_centroid, use_contact_feat):
#     obj_feat_global = obj_feat[:, 0, :1024]
#     if use_contact_feat:
#         obj_feat_cov = torch.cat([obj_feat_global, cov_map], dim=1)
#     else:
#         obj_feat_cov = obj_feat_global

#     if use_obj_scale_centroid:
#         obj_scale_expand1 = obj_scale.unsqueeze(1)
#         obj_feat_final = torch.cat([obj_feat_cov, obj_scale_expand1, obj_cent], dim=1)
#     else:
#         obj_feat_final = obj_feat_cov
#     return obj_feat_final

def hot3d_proc_obj_feat_final_train(cov_map, obj_scale, obj_cent, obj_feat, use_obj_scale_centroid, use_contact_feat):
    obj_feat_global = obj_feat[:, :, 0, :1024]
    if use_contact_feat:
        obj_feat_cov = torch.cat([obj_feat_global, cov_map], dim=-1)
    else:
        obj_feat_cov = obj_feat_global

    if use_obj_scale_centroid:
        obj_scale_expand1 = obj_scale.unsqueeze(-1)
        obj_feat_final = torch.cat([obj_feat_cov, obj_scale_expand1, obj_cent], dim=-1)
    else:
        obj_feat_final = obj_feat_cov
    return obj_feat_final