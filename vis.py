import os
import os.path as osp
from glob import glob
import sys
# sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))

import tqdm
import numpy as np
import hydra
import yaml
import json
from omegaconf import OmegaConf
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch3d.structures import Meshes

from lib.models.mano import build_mano_aa
from lib.networks.clip import load_and_freeze_clip, encoded_text
from lib.datasets.datasets import get_dataloader
from lib.utils.model_utils import (
    build_refiner, 
    build_model_and_diffusion, 
    build_seq_cvae, 
    build_pointnetfeat, 
    build_mpnet, 
    build_contact_estimator, 
)
from lib.models.object import build_object_model
from lib.utils.renderer import Renderer
from lib.utils.metric import AverageMeter
from lib.utils.file import (
    make_model_result_folder, 
    wandb_login, 
    save_video, 
)
from lib.utils.eval import (
    get_object_hand_info, 
    get_valid_mask_bunch, 
    proc_results, 
)
from lib.utils.proc import (
    proc_obj_feat_final, 
    proc_obj_feat_final_train, 
    proc_refiner_input, 
)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    data_config = config.dataset
    dataset_name = data_config.name

    model_name = config.classifier.model_name
    best_model_name = model_name+"_best"
    save_root = config.classifier.save_root
    model_folder, result_folder = make_model_result_folder(save_root, train_type="train")

    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=data_config.flat_hand).cuda()
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=data_config.flat_hand).cuda()
    dataloader = get_dataloader("Motion"+dataset_name, config, data_config)
    refiner = build_refiner(config, test=True)
    texthom, diffusion \
        = build_model_and_diffusion(config, lhand_layer, rhand_layer, test=True)
    seq_cvae = build_seq_cvae(config, test=True)
    pointnet = build_pointnetfeat(config, test=True)
    object_model = build_object_model(data_config.data_obj_pc_path)
    contact_estimator = build_contact_estimator(config, test=True)

    clip_model = load_and_freeze_clip(config.clip.clip_version)
    clip_model = clip_model.cuda()

    dump_data = torch.randn([1, 1024, 3]).cuda()
    pointnet(dump_data)

    hand_nfeats = config.texthom.hand_nfeats
    obj_nfeats = config.texthom.obj_nfeats
    with torch.no_grad():
        max_frames = data_config.max_frames
        text = config.text
        is_lhand, is_rhand, \
        obj_pc_org, obj_pc_normal_org, \
        normalized_obj_pc, point_sets, \
        obj_cent, obj_scale, \
        obj_verts, obj_faces, \
        obj_top_idx, obj_pc_top_idx \
            = get_object_hand_info(
                object_model, clip_model, text, 
                data_config.obj_root, data_config
            )
        bs, nobjs, npts = normalized_obj_pc.shape[:3]
        
        enc_text = encoded_text(clip_model, text)
        
        obj_feat = pointnet(normalized_obj_pc.flatten(0, 1)).reshape(bs, nobjs, npts, -1)
        obj_feat_final, est_contact_map = proc_obj_feat_final(
            contact_estimator, obj_scale, obj_cent, 
            obj_feat, enc_text, nobjs, npts, 
            config.texthom.use_obj_scale_centroid, config.contact.use_scale, config.texthom.use_contact_feat
        )
        duration = seq_cvae.decode(enc_text)
        duration *= config.max_frames
        duration = duration.long()
        valid_mask_lhand, valid_mask_rhand, valid_mask_obj \
                = get_valid_mask_bunch(is_lhand, is_rhand, max_frames, nobjs, duration)
        coarse_x_lhand, coarse_x_rhand, coarse_x_obj \
            = diffusion.sampling(
                texthom, obj_feat_final, 
                enc_text, max_frames, 
                hand_nfeats, obj_nfeats, 
                valid_mask_lhand, 
                valid_mask_rhand, 
                valid_mask_obj, 
                device=torch.device("cuda")
            )
            
        input_lhand, input_rhand, refined_x_obj \
            = proc_refiner_input(
                coarse_x_lhand, coarse_x_rhand, coarse_x_obj, 
                lhand_layer, rhand_layer, obj_pc_org, obj_pc_normal_org, 
                valid_mask_lhand, valid_mask_rhand, valid_mask_obj, 
                est_contact_map, dataset_name, obj_pc_top_idx=obj_pc_top_idx, 
            )

        refined_x_lhand, refined_x_rhand \
            = refiner(
                input_lhand, input_rhand,  
                valid_mask_lhand=valid_mask_lhand, 
                valid_mask_rhand=valid_mask_rhand, 
            )
        for text_idx in range(len(text)):
            text_duration = duration[text_idx].item()
            obj_verts_text = obj_verts[text_idx]
            obj_faces_text = obj_faces[text_idx]
            point_set_text = point_sets[text_idx]
            est_contact_map_text = est_contact_map[text_idx]
            is_lhand_text = is_lhand[text_idx]
            is_rhand_text = is_rhand[text_idx]
            if dataset_name == "arctic":
                obj_top_idx_text = obj_top_idx[text_idx]
            else:
                obj_top_idx_text = None
                
            coarse_x_lhand_sampled = coarse_x_lhand[text_idx][:text_duration]
            coarse_x_rhand_sampled = coarse_x_rhand[text_idx][:text_duration]
            coarse_x_obj_sampled = coarse_x_obj[text_idx][:text_duration]
            
            refined_x_lhand_sampled = refined_x_lhand[text_idx][:text_duration]
            refined_x_rhand_sampled = refined_x_rhand[text_idx][:text_duration]
            refined_x_obj_sampled = refined_x_obj[text_idx][:text_duration]

            refined_obj_verts_tf, refined_lhand_verts, lhand_faces, \
            refined_rhand_verts, rhand_faces = \
                proc_results(
                    refined_x_lhand_sampled, refined_x_rhand_sampled, refined_x_obj_sampled, 
                    obj_verts_text, lhand_layer, rhand_layer, 
                    is_lhand_text, is_rhand_text, 
                    dataset_name, obj_top_idx_text
                )
            
            print(text)
            print(refined_obj_verts_tf.shape)
            print(refined_lhand_verts.shape)
            print(refined_rhand_verts.shape)
            print(lhand_faces.shape)
            print(rhand_faces.shape)
            print([face.shape for face in obj_faces_text])

            np.savez(
                'eval_data.npz',
                obj_vert = refined_obj_verts_tf.cpu().numpy(),
                obj_face = np.array([face.cpu().numpy() for face in obj_faces_text], dtype=object), 
                lhand_vert = refined_lhand_verts.cpu().numpy(),
                lhand_face = lhand_faces.cpu().numpy(),
                rhand_vert = refined_rhand_verts.cpu().numpy(),
                rhand_face = rhand_faces.cpu().numpy(),
            )

if __name__ == '__main__':
    main()