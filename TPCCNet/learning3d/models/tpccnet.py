""" tpccNet
    References.
        .. Sarode V, Dhagat A, Srivatsan R A, et al. 
        "tpccNet: A Fully-Convolutional Network to Estimate Inlier Points"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('E:\Mtmpe-Net\TPCCNet\learning3d\data_utils')
# import dataloaders
sys.path.append('E:\Mtmpe-Net\TPCCNet\learning3d\models')
# from models.pointnet 
import pointnet
import pooling
from pointnet import PointNet
from pooling import Pooling

class PointNettpcc(nn.Module):
	def __init__(self, template_feature_size=1024, source_feature_size=1024, feature_model=PointNet()):
		super().__init__()
		self.feature_model = feature_model
		self.pooling = Pooling()

		input_size = template_feature_size + source_feature_size
		self.h3 = nn.Sequential(nn.Conv1d(input_size, 1024, 1), nn.ReLU(),#ReLU is greater than 0 reserved, and less than 0 becomes zero
								nn.Conv1d(1024, 512, 1), nn.ReLU(),
								nn.Conv1d(512, 256, 1), nn.ReLU(),
								nn.Conv1d(256, 128, 1), nn.ReLU(),
								nn.Conv1d(128,   1, 1), nn.Sigmoid())  #Group to C in the range of 0,1

	def find_tpcc(self, x, t_out_h1):
		batch_size, _ , num_points = t_out_h1.size()
		x = x.unsqueeze(2)  #Add a dimension in the third dimension
		x = x.repeat(1,1,num_points)#Replication forms the same number of rows C as the template point cloud for easy stitching
		x = torch.cat([t_out_h1, x], dim=1)#Stitch sources and templates
		x = self.h3(x)#The convolution operation described above
		return x.view(batch_size, -1)#Line up bath_size rows

	def forward(self, template, source):
		source_features = self.feature_model(source)				# [B x C x N]=B x Nx x 1024 ,pointnetOperation, C represents the number of template point cloud rows (Nx) in the paper, but also the number of rows in the paper C (C- Nx ×1) and the number of template point clouds, N for POINTNET generated dimensions (emb_dims s 1024)
		template_features = self.feature_model(template)			# [B x C x N]

		source_features = self.pooling(source_features)   #Pool the source point cloud 1×1024
		tpcc = self.find_tpcc(source_features, template_features) #Calculating musk (C in the paper)
		return tpcc 


class tpccNet(nn.Module):
	def __init__(self, feature_model=PointNet(), is_training=True):
		super().__init__()
		self.tpccNet = PointNettpcc(feature_model=feature_model)
		self.is_training = is_training

	@staticmethod
	def index_points(points, idx):#76 lines template = self.index_points(template, self.tpcc_idx)，index_points Is the address of the non-zero element of musk
		"""
		Input:
			points: input points data, [B, N, C]
			idx: sample index data, [B, S]
		Return:
			new_points:, indexed points data, [B, S, C]
		"""
		device = points.device
		B = points.shape[0] #Bath_size The value above is green B
		view_shape = list(idx.shape)#Convert to a list[B，S]
		view_shape[1:] = [1] * (len(view_shape) - 1)
		repeat_shape = list(idx.shape)
		repeat_shape[0] = 1
		batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)# torch.arange Within the () range
		new_points = points[batch_indices, idx, :]
		return new_points

	# This function is only useful for testing with a single pair of point clouds.
	@staticmethod
	def find_index(tpcc_val):
		tpcc_idx = torch.nonzero((tpcc_val[0]>0.5)*1.0)#Find the address (index) of an address that is not zero for an address that has a median value of more than 0.5 in the tpcc_val
		return tpcc_idx.view(1, -1)#Line up

	def forward(self, template, source, point_selection='threshold'):
		tpcc = self.tpccNet(template, source)#batch_size × m

		if point_selection == 'topk' or self.is_training:
			_, self.tpcc_idx = torch.topk(tpcc, source.shape[1], dim=1, sorted=False)
		elif point_selection == 'threshold':
			self.tpcc_idx = self.find_index(tpcc)

		template = self.index_points(template, self.tpcc_idx)
		return template, tpcc  #tpcced_template, predicted_tpcc


if __name__ == '__main__':
	template, source = torch.rand(10,1024,3), torch.rand(10,1024,3)
	net = tpccNet()
	result = net(template, source)
	import ipdb; ipdb.set_trace()