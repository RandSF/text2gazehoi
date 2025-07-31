import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))

import tqdm
import numpy as np
import hydra
from easydict import EasyDict as edict
from omegaconf import OmegaConf
from copy import deepcopy 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lib.networks.clip import load_and_freeze_clip, encoded_text
from lib.datasets.datasets import get_dataloader
from lib.utils.model_utils import (
    build_pointnetfeat, 
    build_contact_estimator, 
)
from lib.utils.metric import AverageMeter
from lib.utils.file import (
    make_save_folder, 
    wandb_login,
)
from lib.utils.hot3d_utils import (
    hot3d_proc_cond_contact_estimator
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_object(config)
    config = edict(config)
    data_config = config.dataset
    dataset_name = data_config.name
    
    wandb = wandb_login(
        config, 
        config.contact, 
        relogin=False
    )

    feat_model_save_path = config.pointfeat.weight_path
    best_feat_model_save_path = feat_model_save_path.replace(".pth", "_best.pth")
    save_root = config.pointfeat.save_root
    make_save_folder(save_root)
    
    contact_model_name = config.contact.model_name
    contact_model_save_path = config.contact.weight_path
    best_contact_model_save_path = contact_model_save_path.replace(".pth", "_best.pth")
    save_root = config.contact.save_root
    make_save_folder(save_root)
    
    dataloader = get_dataloader("Contact"+dataset_name, config, data_config)
    valid_config = deepcopy(config)
    valid_config.shuffle = False
    valid_config.drop_last = False
    valid_data_config = deepcopy(data_config)
    valid_data_config.augm = False
    valid_dataloader = get_dataloader("Contact"+dataset_name, config, valid_data_config)
    pointnet = build_pointnetfeat(config)
    contact_estimator = build_contact_estimator(config)
    
    clip_model = load_and_freeze_clip(config.clip.clip_version)
    clip_model = clip_model.cuda()
    
    optimizer = optim.AdamW(
        list(contact_estimator.parameters()) \
        + list(pointnet.parameters()), 
        lr=config.contact.lr,
    )
    
    cur_loss = 99999
    best_loss = 99999
    best_epoch = -1
    
    nepoch = config.contact.iteration / (data_config.data_num/data_config.text_num)
    nepoch = int(np.ceil(nepoch / 50.0) * 50)
    with tqdm.tqdm(range(nepoch)) as pbar:
        for epoch in pbar:
            pointnet.train()
            contact_estimator.train()
            loss_meter = AverageMeter()
            recon_loss_meter = AverageMeter()
            kl_loss_meter = AverageMeter()
            dice_loss_meter = AverageMeter()
            for item in dataloader:
                normalized_obj_pc = item["normalized_obj_pc"].cuda()
                obj_scale = item["obj_scale"].cuda()
                text = item["text"]
                cov_map = item["cov_map"].cuda()
                cov_map_mask = item['map_mask'].unsqueeze(-1).unsqueeze(-1).cuda()
                bs, nobjs, npts, _ = normalized_obj_pc.shape
                
                obj_feat = pointnet(normalized_obj_pc.flatten(0, 1)).reshape(bs, nobjs, npts, -1)
                enc_text = encoded_text(clip_model, text)
                
                encoder_input = torch.cat([normalized_obj_pc, cov_map], dim=-1)
                condition = hot3d_proc_cond_contact_estimator(
                    obj_scale, obj_feat, enc_text, 
                    nobjs, npts, config.contact.use_scale
                )   # [B, O, P, D]
                contact_map, mu, logvar \
                    = contact_estimator(
                            encoder_input.flatten(0, 1), 
                            condition.flatten(0, 1), 
                        )
                contact_map, mu, logvar = contact_map.reshape(bs, nobjs, npts, -1), mu.reshape(bs, nobjs, -1), logvar.reshape(bs, nobjs, -1)
                recon_loss = F.binary_cross_entropy(contact_map, cov_map, reduction='none')
                recon_loss = torch.sum(recon_loss*cov_map_mask) / torch.sum(cov_map_mask)
                kl_div = -0.5 * torch.sum((1 + logvar - mu.pow(2) - logvar.exp())*cov_map_mask[...,0]) / torch.sum(cov_map_mask)
                
                if config.contact.use_dice_loss:
                    dice_loss = 1-2*(torch.sum(contact_map*cov_map*cov_map_mask)/(torch.sum(contact_map*cov_map_mask)+torch.sum(cov_map*cov_map_mask)))
                    losses = recon_loss+kl_div+dice_loss
                else:
                    losses = recon_loss+kl_div

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                loss_meter.update(losses.item(), bs)
                recon_loss_meter.update(recon_loss.item(), bs)
                kl_loss_meter.update(kl_div.item(), bs)
                dice_loss_meter.update(dice_loss.item(), bs)

            cur_loss = loss_meter.avg
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_epoch = epoch+1
                
                torch.save(
                    {
                        "model": contact_estimator.state_dict(), 
                        "epoch": epoch, 
                        "loss": cur_loss, 
                    },
                    best_contact_model_save_path
                )
                torch.save(
                    {
                        "model": pointnet.state_dict(), 
                        "epoch": epoch, 
                        "loss": cur_loss, 
                    },
                    best_feat_model_save_path
                )
            wandb.log(
                {
                    "loss": cur_loss,
                    "recon_loss": recon_loss_meter.avg,
                    "kl_div": kl_loss_meter.avg,
                    "dice_loss": dice_loss_meter.avg,
                }
            )
            pbar.set_description(f"{contact_model_name} | Best loss: {best_loss:.0f} ({best_epoch}), Cur loss: {cur_loss:.0f}")

    torch.save(
        {
            "model": contact_estimator.state_dict(), 
            "epoch": nepoch, 
            "loss": cur_loss, 
        },
        contact_model_save_path
    )
    torch.save(
        {
            "model": pointnet.state_dict(), 
            "epoch": nepoch, 
            "loss": cur_loss, 
        },
        feat_model_save_path
    )
    wandb.finish()

if __name__ == "__main__":
    main()