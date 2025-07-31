import torch
import numpy as np
import torch.nn as nn
import os
import sys
sys.path.append('.')
sys.path.append('..')

import torch.nn.functional as F
import torch.autograd as autograd

class LSTM_Action_Classifier(nn.Module):
	def __init__(self, feat_dim=108, hidden_dim=128, label_size=37, batch_size=64, num_layers=2, kernel_size=3, ckpt_path=False):   #LSTMClassifier(48, 128, 8, 1, 2, 3)
		super(LSTM_Action_Classifier, self).__init__()
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.num_layers = num_layers
		self.label_size = label_size
		joints_dim2d = feat_dim 
		
		self.lstm_left = nn.LSTM(joints_dim2d, hidden_dim, num_layers=self.num_layers)
		self.conv_left = nn.Conv1d(1, 1, kernel_size, stride=1, padding=1)
		self.lstm_right = nn.LSTM(joints_dim2d, hidden_dim, num_layers=self.num_layers)
		self.conv_right = nn.Conv1d(1, 1, kernel_size, stride=1, padding=1)
		# self.sig = nn.Sigmoid()
		self.smax = nn.Softmax(dim=-1)
		self.hidden2_2 = self.init_hidden2_2()
		self.hidden2_3 = self.init_hidden2_3()
		
		self.hidden2label = nn.Linear(hidden_dim, label_size)

		if ckpt_path:
			self.load_ckpt(ckpt_path)
	
	def load_ckpt(self, ckpt_path):
		ckpt = torch.load(ckpt_path)
		self.load_state_dict(ckpt['model'])
		print(f"classifier loaded from {ckpt_path}")
	
	def init_hidden2_1(self):
		# the first is the hidden h
		# the second is the cell  c
		return (autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim).cuda()),
				autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim).cuda()))
	def init_hidden2_2(self):
		# the first is the hidden h
		# the second is the cell  c
		return (autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim).cuda()),
				autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim).cuda()))
	def init_hidden2_3(self):
		# the first is the hidden h
		# the second is the cell  c
		return (autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim).cuda()),
				autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim).cuda()))
	
	def predict(self, x_lhand, x_rhand, x_obj, 
			 is_lhand, is_rhand, duration):
		assert self.batch_size == duration.shape[0]
		feat_all = []
		y_pred_all = []
		for idx in range(self.batch_size):
			x_o = x_obj[idx].view(-1, 1, 9)[:duration[idx]]
			x_l = x_lhand[idx].view(-1, 1, 99)[:duration[idx]] if is_lhand[idx] else torch.zeros([duration[idx], 1, 99]).to(x_o.device)
			x_r = x_rhand[idx].view(-1, 1, 99)[:duration[idx]] if is_rhand[idx] else torch.zeros([duration[idx], 1, 99]).to(x_o.device)

			lstm_out_l, _ = self.lstm_left(torch.cat([x_l, x_o], dim=-1), self.hidden2_2)
			lstm_out_r, _ = self.lstm_right(torch.cat([x_r, x_o], dim=-1), self.hidden2_3)
			t2_2 = lstm_out_l[-1].view(1,1,-1)
			t2_3 = lstm_out_r[-1].view(1,1,-1)
			y2_2 = self.conv_left(t2_2)
			y2_3 = self.conv_right(t2_3)
			y3 = y2_2 + y2_3
			y3 = y3.contiguous().view(-1, self.hidden_dim)        
			y4  = self.hidden2label(y3)
			y_pred = self.smax(y4)
			feat_all.append(y3.squeeze())
			y_pred_all.append(y_pred.squeeze())
		return  torch.stack(y_pred_all), torch.tanh(torch.stack(feat_all))*0.1


	def forward(self, x_lhand, x_rhand, x_obj, y, 
			 is_lhand, is_rhand, duration):
		
		y_pred, activation = self.predict(x_lhand, x_rhand, x_obj, is_lhand, is_rhand, duration)

		gt_one_hot = nn.functional.one_hot(y, num_classes=self.label_size).float()
		loss = F.binary_cross_entropy(y_pred, gt_one_hot)

		sorted_idx = torch.argsort(y_pred.detach(), dim=-1, descending=True)	
		acc_top1 = (
			(y[:,None].expand(-1, 1) == sorted_idx[:,:1]).sum(dim=-1) > 0
		).float()
		acc_top3 = (
			(y[:,None].expand(-1, 3) == sorted_idx[:,:3]).sum(dim=-1) > 0
		).float()
		return {'loss_ce': loss, 'top1': acc_top1, 'top3': acc_top3, 'activation': activation, 'y_pred': y_pred}
