import os
import os.path as osp
from glob import glob
import sys
sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))

import tqdm
import numpy as np
import hydra
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
from lib.utils.visualize import render_videos

from lib.networks.action_rec import LSTM_Action_Classifier
from train.evaluate_statistical_metrics import calculate_frechet_distance, calculate_diversity_, calculate_multimodality_
import umap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_object(config)
    config = edict(config)
    data_config = config.dataset
    dataset_name = data_config.name
    
    wandb = wandb_login(config, config.classifier, relogin=True)

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
    dump_data = torch.randn([1, 1024, 3]).cuda()
    pointnet(dump_data)
    contact_estimator = build_contact_estimator(config, test=True)

    classifier = LSTM_Action_Classifier(batch_size=config.batch_size).cuda()
    
    clip_model = load_and_freeze_clip(config.clip.clip_version)
    clip_model = clip_model.cuda()

    # mpnet = build_mpnet(config)

    renderer = Renderer(device="cuda", camera=f"{dataset_name}_front")
    
    hand_nfeats = config.texthom.hand_nfeats
    obj_nfeats = config.texthom.obj_nfeats
    
    lambda_ce = config.classifier.lambda_ce
    lambda_dict = {
        "lambda_ce": lambda_ce, 
    }

    def generate_res(item):
        ### for gen
        #### generation
        with torch.no_grad():
            text = item["text"]
            is_lhand, is_rhand, \
            obj_pc_org, obj_pc_normal_org, \
            normalized_obj_pc, point_sets, \
            obj_cent, obj_scale, \
            obj_verts, obj_faces, \
            obj_top_idx, obj_pc_top_idx \
                = get_object_hand_info(
                    object_model, 
                    clip_model, 
                    text, 
                    data_config.obj_root, 
                    data_config,
                    # mpnet,  
                )
            
            text = item["text"]
            enc_text = encoded_text(clip_model, text)

            duration = seq_cvae.decode(enc_text)
            duration *= 150
            duration = duration.long()
            # duration_gt = torch.sum(valid_mask_obj, dim=-1).long()
            # duration = duration_gt
            # is_lhand, is_rhand = (valid_mask_lhand.sum(dim=-1)>0), (valid_mask_rhand.sum(dim=-1)>0)

            valid_mask_lhand, valid_mask_rhand, valid_mask_obj \
                = get_valid_mask_bunch(
                    is_lhand, is_rhand, 
                    data_config.max_nframes, duration
                )

            bs, npts = normalized_obj_pc.shape[:2]

            obj_feat = pointnet(normalized_obj_pc)
            obj_feat_final, est_contact_map = proc_obj_feat_final(
                contact_estimator, obj_scale, obj_cent, 
                obj_feat, enc_text, npts, 
                config.texthom.use_obj_scale_centroid, config.contact.use_scale, config.texthom.use_contact_feat
            )
            
            # coarse_x_lhand, coarse_x_rhand, coarse_x_obj \
            #     = diffusion(
            #         texthom, x_lhand, x_rhand, 
            #         x_obj, obj_feat_final, 
            #         enc_text=enc_text, get_losses=False, 
            #         valid_mask_lhand=valid_mask_lhand_pred, 
            #         valid_mask_rhand=valid_mask_rhand_pred, 
            #         valid_mask_obj=valid_mask_obj_pred, 
            #         ldist_map=ldist_map, 
            #         rdist_map=rdist_map, 
            #         obj_verts_org=obj_pc_org, 
            #         obj_pc_top_idx=obj_pc_top_idx, 
            #     )
            coarse_x_lhand, coarse_x_rhand, coarse_x_obj \
                = diffusion.sampling(
                    texthom, obj_feat_final, 
                    enc_text, data_config.max_nframes, 
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
        return refined_x_lhand, refined_x_rhand, refined_x_obj, duration, is_lhand, is_rhand

    with torch.no_grad():
        classifier.eval()
        if osp.isfile(config.ckpt_path): 
            ckpt_paths = [config.ckpt_path]
        else:
            def _get_epoch(fp, min_ep=-1):
                fname = osp.basename(fp).split('.')[0]
                split_res = fname.split('_')[-1]
                try: ep = int(split_res)
                except: return False
                if ep>min_ep: return ep
                else: return False
            ckpt_paths = glob(os.path.join(config.ckpt_path, "*.pth"))
            ckpt_paths = sorted([cp for cp in ckpt_paths if _get_epoch(cp, 90)], key=_get_epoch, reverse=False)
        for ckpt in ckpt_paths:
            classifier.load_ckpt(ckpt)
            y_pred_all, y_gt_all = [], []
            feat_gt_all, feat_pred_all = [], []
            top1_all, top3_all = [], []
            top1_pred_all, top3_pred_all = [], []
            for item in tqdm.tqdm(dataloader):
                valid_mask_lhand = item["valid_mask_lhand"].cuda()
                valid_mask_rhand = item["valid_mask_rhand"].cuda()
                valid_mask_obj = item["valid_mask_obj"].cuda()

                x_lhand = item["x_lhand"].cuda()
                x_rhand = item["x_rhand"].cuda()
                x_obj = item["x_obj"].cuda()
                action_idx = item['action_index'].cuda()
                refined_x_lhand, refined_x_rhand, refined_x_obj, duration, is_lhand_pred, is_rhand_pred = generate_res(item)
                duration_gt = torch.sum(valid_mask_obj, dim=-1).long()

                is_lhand_gt, is_rhand_gt = (valid_mask_lhand.sum(dim=-1)>0), (valid_mask_rhand.sum(dim=-1)>0)

                info_gt = classifier(
                    x_lhand, x_rhand, x_obj, action_idx, 
                    is_lhand_gt, is_rhand_gt, duration_gt
                )
                info_pred = classifier(
                    refined_x_lhand, refined_x_rhand, refined_x_obj, action_idx, 
                    is_lhand_pred, is_rhand_pred, duration
                )
                feat_gt_all.append(info_gt['activation'].cpu().numpy())
                feat_pred_all.append(info_pred['activation'].cpu().numpy())
                y_pred_all.append(info_pred['y_pred'].cpu().numpy())
                y_gt_all.append(action_idx.cpu().numpy())
                top1_all.append(info_gt['top1'])
                top3_all.append(info_gt['top3'])
                top1_pred_all.append(info_pred['top1'])
                top3_pred_all.append(info_pred['top3'])
                break
            ### calcu fid on full feat
            y_pred_all = np.concatenate(y_pred_all, axis=0)
            feat_gt_all, feat_pred_all = np.concatenate(feat_gt_all, axis=0), np.concatenate(feat_pred_all, axis=0)
            mu_gt, sigma_gt = np.mean(feat_gt_all, axis=0), np.cov(feat_gt_all, rowvar=False)
            mu_pred, sigma_pred = np.mean(feat_pred_all, axis=0), np.cov(feat_pred_all, rowvar=False)
            fid_full, _ = calculate_frechet_distance(mu_gt, sigma_gt, mu_pred, sigma_pred)
            divers = calculate_diversity_(torch.from_numpy(feat_pred_all), torch.from_numpy(y_pred_all), 37).item()
            multimodality = calculate_multimodality_(torch.from_numpy(feat_pred_all), torch.from_numpy(y_pred_all), 37).item()
                
            step = int(ckpt.split('_')[1].split('.')[0]) // 10 - 1
            wandb.log(
                {
                    "eval/fid": fid_full,
                    "eval/top1_pred": torch.mean(torch.cat(top1_pred_all)).item(), 
                    "eval/top3_pred": torch.mean(torch.cat(top3_pred_all)).item(), 
                    'eval/diversity': divers, 
                    'eval/multimodality': multimodality, 
                }, 
                step=step
            )
            acc_top3 = torch.mean(torch.cat(top3_pred_all)).item()

            ckpt_name = osp.basename(ckpt).split('.')[0]
            print(f"res of {ckpt_name}:\n\tACC(top3): {acc_top3:.2f}\tFID: {fid_full:.2f}\tDiv: {divers:.2f}\tMM: {multimodality:.2f}")

           


if __name__ == "__main__":
    main()