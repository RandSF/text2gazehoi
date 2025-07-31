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
    optimizer = optim.AdamW(classifier.parameters(), lr=config.classifier.lr)
    
    clip_model = load_and_freeze_clip(config.clip.clip_version)
    clip_model = clip_model.cuda()

    renderer = Renderer(device="cuda", camera=f"{dataset_name}_front")

    cur_loss = 9999
    best_loss = 9999
    best_epoch = 0
    
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

    nepoch = config.classifier.iteration / (data_config.data_num/data_config.text_num)
    nepoch = int(np.ceil(nepoch / 50.0) * 50)
    print(f"{config.classifier.iteration}/({data_config.data_num}/{data_config.text_num}) = {config.classifier.iteration / (data_config.data_num/data_config.text_num)} = rounded (int(np.ceil(nepoch / 50.0) * 50))")
    with tqdm.tqdm(range(nepoch)) as pbar:
        for epoch in pbar:
            save_flag = (epoch+1)%config.save_pth_freq==0 or epoch+1==nepoch
            eval_flag = False #(epoch+1)%(config.save_pth_freq*2)==0 or epoch+1==nepoch

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

                    if eval_flag:
                        refined_x_lhand, refined_x_rhand, refined_x_obj, duration, is_lhand_pred, is_rhand_pred = generate_res(item)
                        info_pred = classifier(
                            refined_x_lhand, refined_x_rhand, refined_x_obj, action_idx, 
                            is_lhand_gt, is_rhand_gt, duration
                        )
                        feat_pred_all.append(info_pred['activation'].cpu())
                        top1_pred_all.append(info_pred['top1'].cpu())
                        top3_pred_all.append(info_pred['top3'].cpu())
                        y_pred_all.append(info_pred['y_pred'].cpu())

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
                if eval_flag:
                    ### calcu fid on full feat
                    feat_pred_all = torch.cat(feat_pred_all, dim=0)
                    mu_gt, sigma_gt = np.mean(feat_gt_all.numpy(), axis=0), np.cov(feat_gt_all.numpy(), rowvar=False)
                    mu_pred, sigma_pred = np.mean(feat_pred_all.numpy(), axis=0), np.cov(feat_pred_all.numpy(), rowvar=False)
                    fid_full, tr_covmean = calculate_frechet_distance(mu_gt,sigma_gt, mu_pred, sigma_pred)
                    divers_pred = calculate_diversity_(feat_pred_all, y_pred_all, 37).item()
                    multimodality_pred = calculate_multimodality_(feat_pred_all, y_pred_all, 37).item()
                    wandb.log({
                        "top1_pred": torch.mean(torch.cat(top1_pred_all)).item(), 
                        "top3_pred": torch.mean(torch.cat(top3_pred_all)).item(), 

                        'data/mu_diff': np.dot(mu_gt-mu_pred, mu_gt-mu_pred), 
                        'data/tr_sig_gt': np.trace(sigma_gt), 
                        'data/tr_sig_pred': np.trace(sigma_pred), 
                        'data/tr_covmean': tr_covmean, 

                        "fid": fid_full,
                        "diversity_pred": divers_pred, 
                        "multimodality_pred": multimodality_pred, 
                    })
                    ### calcu fid on full feat

            # ### vis
            # reducer = umap.UMAP(
            #     n_components=2,      # 降维后的维度
            #     n_neighbors=15,      # 局部邻域大小
            #     min_dist=0.1,        # 点之间的最小距离
            #     metric='euclidean',  # 距离度量
            #     random_state=42,     # 随机种子
            #     n_jobs=1            # 使用所有CPU核心
            # )

            # # 执行降维
            # embedding = reducer.fit_transform(np.concatenate([feat_gt_all, feat_pred_all], axis=0))
            # N = feat_gt_all.shape[0]
            # fig = plt.figure(figsize=(10, 8))
            # canvas=FigureCanvasAgg(fig)
            # ax = fig.add_subplot()
            # ax.scatter(
            #     embedding[:N, 0], 
            #     embedding[:N, 1], 
            #     c='g',
            #     label='gt', 
            #     alpha=0.7,      
            #     s=15            
            # )
            # ax.scatter(
            #     embedding[N:, 0], 
            #     embedding[N:, 1], 
            #     c='r',
            #     label='pred', 
            #     alpha=0.7,      
            #     s=15            
            # )

            # ax.legend()
            # ax.grid(alpha=0.3)
            # fig.tight_layout()
            # canvas.draw()
            # image_array = np.asarray(canvas.buffer_rgba())
            # images = wandb.Image(image_array, caption=f"umap of epoch {epoch}")
            # wandb.log({"visualization": images})
            # ### vis

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
                        # "fid_full": fid_full,
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