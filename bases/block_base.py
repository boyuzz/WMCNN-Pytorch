import torch


class BlockBase(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation=None, norm=None):
        super(BlockBase, self).__init__()

        if norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(out_channels, eps=0.001)
        else:
            self.bn = None

        if activation == 'relu':
            self.act = torch.nn.ReLU(False)
        elif activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, False)
        elif activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        raise NotImplementedError
