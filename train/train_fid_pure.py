import os
import os.path as osp
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
from train.evaluate_statistical_metrics import calculate_frechet_distance, calculate_multimodality_, calculate_diversity_
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

    classifier = LSTM_Action_Classifier(batch_size=config.batch_size).cuda()
    optimizer = optim.AdamW(classifier.parameters(), lr=config.classifier.lr)
    
    clip_model = load_and_freeze_clip(config.clip.clip_version)
    clip_model = clip_model.cuda()


    cur_loss = 9999
    best_loss = 9999
    best_epoch = 0
    
    lambda_ce = config.classifier.lambda_ce
    lambda_dict = {
        "lambda_ce": lambda_ce, 
    }

    nepoch = config.classifier.iteration / (data_config.data_num/data_config.text_num)
    nepoch = int(np.ceil(nepoch / 50.0) * 50)
    print(f"{config.classifier.iteration}/({data_config.data_num}/{data_config.text_num}) = {config.classifier.iteration / (data_config.data_num/data_config.text_num)} = rounded {(int(np.ceil(nepoch / 50.0) * 50))}")
    with tqdm.tqdm(range(nepoch)) as pbar:
        for epoch in pbar:
            save_flag = (epoch+1)%config.save_pth_freq==0 or epoch+1==nepoch

            classifier.train()
            loss_meter = AverageMeter()
            
            for item in dataloader:
                x_lhand = item["x_lhand"].cuda()
                x_rhand = item["x_rhand"].cuda()
                x_obj = item["x_obj"].cuda()
                action_idx = item['action_index'].cuda()
                
                valid_mask_lhand = item["valid_mask_lhand"].cuda()
                valid_mask_rhand = item["valid_mask_rhand"].cuda()
                valid_mask_obj = item["valid_mask_obj"].cuda()

                ### classification
                bs = x_obj.shape[0]

                duration_gt = torch.sum(valid_mask_obj, dim=-1).long()
                duration = duration_gt
                is_lhand_gt, is_rhand_gt = (valid_mask_lhand.sum(dim=-1)>0), (valid_mask_rhand.sum(dim=-1)>0)

                loss_info = classifier(
                    x_lhand, x_rhand, x_obj, action_idx, 
                    is_lhand_gt, is_rhand_gt, duration_gt
                )
                
                lambda_ce = lambda_dict['lambda_ce']

                losses = lambda_ce*loss_info['loss_ce']
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                loss_meter.update(losses.item(), bs)
            
            ### calculate metrics
            classifier.eval()
            with torch.no_grad():
                y_pred_all ,y_gt_all = [], []
                feat_gt_all, feat_pred_all = [], []
                top1_all, top3_all = [], []
                top1_pred_all, top3_pred_all = [], []
                
                for item in dataloader:
                    x_lhand = item["x_lhand"].cuda()
                    x_rhand = item["x_rhand"].cuda()
                    x_obj = item["x_obj"].cuda()
                    action_idx = item['action_index'].cuda()
                    
                    duration_gt = torch.sum(valid_mask_obj, dim=-1).long()
                    is_lhand_gt, is_rhand_gt = (valid_mask_lhand.sum(dim=-1)>0), (valid_mask_rhand.sum(dim=-1)>0)

                    info_gt = classifier(
                        x_lhand, x_rhand, x_obj, action_idx, 
                        is_lhand_gt, is_rhand_gt, duration_gt
                    )
                    feat_gt_all.append(info_gt['activation'].cpu())
                    top1_all.append(loss_info['top1'].cpu())
                    top3_all.append(loss_info['top3'].cpu())
                    y_gt_all.append(action_idx.cpu())

                ### one epoch ends
                feat_gt_all = torch.cat(feat_gt_all, dim=0)
                y_gt_all = torch.cat(y_gt_all, dim=0)
                divers_gt = calculate_diversity_(feat_gt_all, y_gt_all, 37).item()
                multimodality_gt = calculate_multimodality_(feat_gt_all, y_gt_all, 37).item()
                wandb.log(
                    {
                        "loss": cur_loss,
                        "top1": torch.mean(torch.cat(top1_all)).item(), 
                        "top3": torch.mean(torch.cat(top3_all)).item(), 
                        "diversity": divers_gt, 
                        "multimodality": multimodality_gt, 
                    }
                )

            ### calculate metrics
            cur_loss = loss_meter.avg
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_epoch = epoch+1
                torch.save(
                    {
                        "model": classifier.state_dict(), 
                        "epoch": epoch, 
                        "loss": cur_loss, 
                    },
                    osp.join(model_folder, best_model_name+".pth"), 
                )
            pbar.set_description(f"{model_name} | Best loss: {best_loss:.4f} ({best_epoch}), Cur loss: {cur_loss:.4f}")
            
            if save_flag:
                torch.save(
                    {
                        "model": classifier.state_dict(), 
                        "epoch": epoch, 
                        "loss": cur_loss, 
                    },
                    osp.join(model_folder, model_name+f"_{epoch+1}.pth"), 
                )
    wandb.finish()


if __name__ == "__main__":
    main()