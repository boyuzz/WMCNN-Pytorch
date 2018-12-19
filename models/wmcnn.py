from bases.model_base import ModelBase
from models import srcnn
import torch.nn as nn


class Net(ModelBase):
	def __init__(self, config):
		super(Net, self).__init__(config)

		self.nets = nn.ModuleList([srcnn.Net(config) for i in range(4)])

	def forward(self, x):
		ca = self.nets[0](x)
		ch = self.nets[1](x)
		cv = self.nets[2](x)
		cd = self.nets[3](x)
		return ca, ch, cv, cd

	def load_from_mat(self, paths):
		assert len(self.nets) == len(paths)

		for i, path in enumerate(paths):
			self.nets[i].load_from_mat(path)
