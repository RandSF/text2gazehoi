import numpy as np

import torch
import torch.nn as nn


class TextHOM(nn.Module):
    def __init__(self, hand_nfeats=99, obj_nfeats=9, latent_dim=512, 
                ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                activation="gelu", clip_dim=512, obj_dim=1024,
                cond_mask_prob=0.1, use_cond_fc=True, 
                use_obj_scale_centroid=True, 
                use_contact_feat=True, 
                use_frame_pos=True, 
                use_inst_pos=True, 
                **kwargs):
        super().__init__()
        ### Variable

        self.cond_mask_prob = cond_mask_prob
        self.nfeats = hand_nfeats

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.clip_dim = clip_dim
        self.obj_dim = obj_dim
        if use_contact_feat:
            self.obj_dim = 2048

        self.input_feats_hand = hand_nfeats
        self.input_feats_obj = obj_nfeats

        self.use_cond_fc = use_cond_fc
        self.use_obj_scale_centroid = use_obj_scale_centroid
        self.use_frame_pos = use_frame_pos
        self.use_inst_pos = use_inst_pos
        
        ### Architecture
        self.init_fc_lhand = InitFC(self.input_feats_hand, self.latent_dim)
        self.init_fc_rhand = InitFC(self.input_feats_hand, self.latent_dim)
        self.init_fc_obj = InitFC(self.input_feats_obj, self.latent_dim)
        if not self.use_frame_pos:
            self.sequence_pos_encoder = OrgPositionalEncoding(self.latent_dim, self.dropout)
        else:
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.sequence_hand_encoder = HandObjectEncoding(self.latent_dim, self.dropout)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer,
            num_layers=self.num_layers
        )
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
        if use_obj_scale_centroid:
            self.embed_obj = nn.Linear(self.obj_dim+4, self.latent_dim)
            self.embed_obj_gate = nn.Linear(self.obj_dim+4, 1)
        else:
            self.embed_obj = nn.Linear(self.obj_dim, self.latent_dim)
            self.embed_obj_gate = nn.Linear(self.obj_dim, 1)

        if use_cond_fc:
            self.out_fc_lhand = CondOutFC(self.input_feats_hand, self.latent_dim)
            self.out_fc_rhand = CondOutFC(self.input_feats_hand, self.latent_dim)
        else:
            self.out_fc_lhand = OutFC(self.input_feats_hand, self.latent_dim)
            self.out_fc_rhand = OutFC(self.input_feats_hand, self.latent_dim)
        self.out_fc_obj = OutFC(self.input_feats_obj, self.latent_dim)
        
    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
    
    def forward(
        self, x_lhand, x_rhand, 
        x_obj, obj_feat, 
        timesteps, enc_text, 
        valid_mask_lhand=None, 
        valid_mask_rhand=None, 
        valid_mask_obj=None,
        active_flag=None,
    ):
        bs = timesteps.shape[0]
        emb = self.embed_timestep(timesteps)
        emb += self.embed_text(self.mask_cond(enc_text, force_mask=False))
        obj_feat = torch.sum(nn.functional.softmax(self.embed_obj_gate(obj_feat), dim=1) * obj_feat, dim=1)
        # embed_obj = self.embed_obj(self.mask_cond(obj_feat, force_mask=False))  # [Bs, No, D]
        emb += self.embed_obj(self.mask_cond(obj_feat, force_mask=False))
        x_init_lhand = self.init_fc_lhand(x_lhand)
        x_init_rhand = self.init_fc_rhand(x_rhand)
        nframe, n_obj = x_obj.shape[1:3]
        x_init_obj = self.init_fc_obj(x_obj.flatten(1, 2)).reshape(nframe, n_obj, bs, -1)

        # import pdb
        # pdb.set_trace()
        x_init = torch.cat((x_init_lhand.unsqueeze(1), x_init_rhand.unsqueeze(1), x_init_obj), dim=1)   # [Nf, 1, Bs, D], [Nf, No, Ds, D] -> [Nf, 2+No, Bs, D]
        x_orig = x_init.flatten(0, 1)
        x_init, emb = self.sequence_pos_encoder(x_init, emb)
        if self.use_inst_pos:
            x_init = self.sequence_hand_encoder(x_init)

        if active_flag is not None:
            ### only feed forward the active objects
            active_flag_extend = torch.nn.functional.pad(active_flag, [2,0], 'constant', 1).unsqueeze(1)   # [Bs, 1, 2+No]
            active_flag_extend = active_flag_extend.permute(1, 2, 0).unsqueeze(-1).expand_as(x_init)  # [Nf, 2+No, Bs, D]
            active_flag_flatten = active_flag_extend.flatten(0, 1)
            active_idx = torch.where(active_flag_flatten)[0]
            x_init = x_init.flatten(0, 1)
            xseq = x_init[active_idx]
        else:
            xseq = x_init.reshape(-1, bs, self.latent_dim)

        xseq = torch.cat((emb, xseq), dim=0)    # [1+Nf*(2+No), Bs, D]

        if valid_mask_obj is not None:
            bs = x_obj.shape[0]
            emb_mask = torch.ones((bs, 1), device=x_obj.device, dtype=bool)
            aug_mask_no_emb = torch.cat(
                [valid_mask_lhand.unsqueeze(-1), valid_mask_rhand.unsqueeze(-1), valid_mask_obj], dim=-1
            ).reshape(bs, -1)
            aug_mask_no_emb
            if active_flag is not None:
                aug_mask = torch.cat([emb_mask, aug_mask_no_emb[:, active_idx]], dim=1)
            else:
                aug_mask = torch.cat([emb_mask, aug_mask_no_emb], dim=1)
            x_out = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)[1:]
        else:
            x_out = self.seqTransEncoder(xseq, src_key_padding_mask=None)[1:]

        x_enc = torch.where(active_flag_flatten, x_out, x_orig) # fill the feature with original data

        x_enc = x_enc.reshape(nframe, 2+n_obj, bs, -1)
        x_enc_lhand = x_enc[:, 0]
        x_enc_rhand = x_enc[:, 1]
        x_enc_obj = x_enc[:, 2:]

        if self.use_cond_fc:
            pred_lhand = self.out_fc_lhand(x_enc_lhand, x_enc_obj)
            pred_rhand = self.out_fc_rhand(x_enc_rhand, x_enc_obj)
        else:
            pred_lhand = self.out_fc_lhand(x_enc_lhand)
            pred_rhand = self.out_fc_rhand(x_enc_rhand)
        pred_obj = self.out_fc_obj(x_enc_obj.flatten(0, 1)).reshape(bs, nframe, n_obj, -1)
        return pred_lhand, pred_rhand, pred_obj


class InitFC(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.fc(x)
        return x


class OutFC(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.latent_dim, self.input_feats)
        
    def forward(self, x):
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        return x
    

class CondOutFC(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.gate_layer = nn.Linear(self.latent_dim, 1)
        self.fc = nn.Linear(self.latent_dim*2, self.input_feats)
        
    def forward(self, x, cond):
        cond = torch.sum(nn.functional.softmax(self.gate_layer(cond), dim=1) * cond, dim=1)
        x = self.fc(torch.cat([x, cond], dim=2))
        x = x.permute(1, 0, 2)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class OrgPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(OrgPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)  # [L, 1, D]

    def forward(self, x, emb):  # x: [Nf, 2+No, Bs, D]
        emb = emb + self.pe[:1]
        Nf = x.shape[0] # num frames, aka length
        x = x + self.pe[1:Nf+1].unsqueeze(1)
        return self.dropout(x), self.dropout(emb)


class HandObjectEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=int(300*8)):
        super(HandObjectEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)  # [L, 1, D]
        stride = max(max_len//8, 1)
        self.lhand_idx = 0
        self.rhand_idx = stride*1
        self.obj_idx = [stride*(n+2) for n in range(6)]

    def forward(self, x):   # x: [Nf, 2+No, Bs, D]
        x[:, 0] = x[:, 0] + self.pe[self.lhand_idx].unsqueeze(0)    # left hand
        x[:, 1] = x[:, 1] + self.pe[self.rhand_idx].unsqueeze(0)    # right hand
        for n in range(6):                                          # obj 1 to 6
            x[:, 2+n] = x[:, 2+n] + self.pe[self.obj_idx[n]].unsqueeze(0)
        return self.dropout(x)