import numpy as np

import torch


def get_valid_mask(is_lhand, is_rhand, frames, 
                    num_object, valid_nframes,
                    exist_flag=None, mask_hand=None, mask_object=None, mask_global=None):
    valid_mask_lhand = np.zeros((frames))
    valid_mask_rhand = np.zeros((frames))
    valid_mask_obj = np.zeros((frames, num_object))
    valid_mask_lhand[:valid_nframes] = 1
    valid_mask_rhand[:valid_nframes] = 1
    if exist_flag is not None:
        valid_mask_obj[np.where(exist_flag)] = 1
    else:
        valid_mask_obj[:valid_nframes] = 1
    if not is_lhand:
        valid_mask_lhand[:] = 0
    if not is_rhand:
        valid_mask_rhand[:] = 0
    if mask_hand is not None:
        valid_mask_lhand *= mask_hand
        valid_mask_rhand *= mask_hand
    if mask_object is not None:
        valid_mask_obj *= mask_object[...,None]
    if mask_global is not None:
        valid_mask_lhand *= mask_global
        valid_mask_rhand *= mask_global
        valid_mask_obj *= mask_global[...,None]
    return (
        (valid_mask_lhand).astype(np.bool), 
        (valid_mask_rhand).astype(np.bool), 
        (valid_mask_obj).astype(np.bool)
    )

def get_frame_align_data_format(key, ndata, max_nframes):
    if "beta" in key:
        final_data = np.zeros((ndata, max_nframes, 10), dtype=np.float32)
    elif key == "x_lhand" or key == "x_rhand":
        final_data = np.zeros((ndata, max_nframes, 3+16*6), dtype=np.float32)
    elif key == "j_lhand" or key == "j_rhand":
        final_data = np.zeros((ndata, max_nframes, 21, 3), dtype=np.float32)
    elif key == "obj_alpha":
        final_data = np.zeros((ndata, max_nframes), dtype=np.float32)
    elif key == "x_obj":
        final_data = np.zeros((ndata, max_nframes, 3+6), dtype=np.float32)
    elif key == "x_obj_angle":
        final_data = np.zeros((ndata, max_nframes, 1), dtype=np.float32)
    elif "org" in key:
        final_data = np.zeros((ndata, max_nframes, 3), dtype=np.float32)
    elif "idx" in key or "dist" in key:
        final_data = np.zeros((ndata, max_nframes), dtype=np.float32)
    return final_data

def align_frame(total_dict: dict, preset_max_nframes = 300):
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

def sample_with_window_size(pred, targ, valid_mask, window_size=20, window_step=1):
    ### data: B, T, C
    valid_mask_max_inx = get_mask_max_idx(valid_mask)
    valid_mask_max_inx_w = valid_mask_max_inx-(window_size*window_step)
    valid_mask_max_inx_w[valid_mask_max_inx_w<0] = 0

    sampled_f_idx = sample_frame_index(valid_mask_max_inx_w, window_size, window_step)
    sampled_pred = torch.gather(pred, 1, sampled_f_idx.unsqueeze(2).expand(-1, -1, pred.shape[2]))
    sampled_targ = torch.gather(targ, 1, sampled_f_idx.unsqueeze(2).expand(-1, -1, targ.shape[2]))
    return sampled_pred, sampled_targ
    
def sample_frame_index(frame_indices, window_size, window_step):
    sampled_indices = torch.rand(len(frame_indices), device=frame_indices.device)*frame_indices
    sampled_indices = sampled_indices.int()
    sampled_indices = sampled_indices.reshape(-1, 1)
    sampled_indices = torch.arange(
            start=0,
            end=window_step*window_size, 
            step=window_step,
                device=sampled_indices.device
        ).unsqueeze(0).expand(sampled_indices.shape[0], -1) \
        + sampled_indices
    return sampled_indices

def get_mask_max_idx(mask):
    '''
    Find the indices of the last occurrences of True in a boolean mask.

    Parameters:
        mask (torch.Tensor): A boolean tensor representing a mask.

    Returns:
        torch.Tensor: A 1-dimensional tensor containing the indices of the last occurrences of True in the input mask.
    '''
    
    # Compute the transitions from True to False in the mask
    true_to_false_transitions = mask.int().diff(dim=1)
    
    # Find the indices of the last True occurrences in each row
    first_false_indices = true_to_false_transitions.argmin(dim=1)

    # Set indices to the maximum frame idx 
    # if there are no False occurrences in a row
    first_false_indices[
        ~true_to_false_transitions.any(dim=1)
        ] = mask.size(1)-1
    return first_false_indices
