import torch
import scipy.io as sio
import numpy as np
from models.base_module import ConvBlock
import utils.utils as utils
from bases.model_base import ModelBase


class Net(ModelBase):
    def __init__(self, config):
        super(Net, self).__init__(config)

        # self.layers = torch.nn.Sequential(
        #     ConvBlock(self.config['in_channels'], self.config['num_filter'], 9, 1, 4, activation='relu', bias=True),
        #     ConvBlock(self.config['num_filter'], self.config['num_filter'] // 2, 5, 1, 2, activation='relu', bias=True),
        #     ConvBlock(self.config['num_filter'] // 2, self.config['in_channels'], 5, 1, 2)
        # )
        self.w_input = ConvBlock(self.config['in_channels'], self.config['num_filter'], 9, 1, 4, activation='relu', bias=True)
        self.w_inter = ConvBlock(self.config['num_filter'], self.config['num_filter'] // 2, 5, 1, 2, activation='relu', bias=True)
        self.w_output = ConvBlock(self.config['num_filter'] // 2, self.config['in_channels'], 5, 1, 2, bias=True)
        # self.layers = torch.nn.Sequential(
        #     ConvBlock(self.config['in_channels'], self.config['num_filter'], 9, 1, 4),
        #     ConvBlock(self.config['num_filter'], self.config['num_filter'] // 2, 1, 1, 0),
        #     ConvBlock(self.config['num_filter'] // 2, self.config['in_channels'], 5, 1, 2, activation=None, norm=None)
        # )
        self.weight_init()

    def forward(self, x):
        # out = self.layers(x)
        out = self.w_input(x)
        out = self.w_inter(out)
        out = self.w_output(out)
        return out

    def weight_init(self, mean=0.0, std=0.001):
        for m in self.modules():
            utils.weights_init_normal(m, mean=mean, std=std)

    def load_from_mat(self, path):

        mat_model = sio.loadmat(path)

        weights_1 = np.reshape(mat_model['weights_conv1'], (9, 9, -1))
        weights_1 = np.transpose(weights_1, axes=[2,1,0])
        weights_1 = np.expand_dims(weights_1, 1)
        self.w_input.conv.weight.data = torch.from_numpy(weights_1)
        self.w_input.conv.bias.data = torch.from_numpy(mat_model['biases_conv1'].squeeze())

        weights_2 = np.reshape(mat_model['weights_conv2'], (64, 5, 5, -1))
        weights_2 = np.transpose(weights_2, axes=[3,0,2,1])
        self.w_inter.conv.weight.data = torch.from_numpy(weights_2)
        self.w_inter.conv.bias.data = torch.from_numpy(mat_model['biases_conv2'].squeeze())

        weights_3 = np.reshape(mat_model['weights_conv3'], (-1, 5, 5))
        weights_3 = np.transpose(weights_3, axes=[0,2,1])
        weights_3 = np.expand_dims(weights_3, 0)
        self.w_output.conv.weight.data = torch.from_numpy(weights_3)
        self.w_output.conv.bias.data = torch.from_numpy(mat_model['biases_conv3'].squeeze(1))
