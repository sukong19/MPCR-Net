import torch
import torch.nn as nn
import torch.nn.functional as F


class Pooling(torch.nn.Module):
	def __init__(self, pool_type='max'):
		self.pool_type = pool_type
		super(Pooling, self).__init__()

	def forward(self, input):
		if self.pool_type == 'max':
			return torch.max(input, 2)[0].contiguous()#Enter the b×n×1024 shape, take the maximum value of each row of the second dimension, a total of 1024 rows, and become
		elif self.pool_type == 'avg' or self.pool_type == 'average':
			return torch.mean(input, 2).contiguous()