import torch.nn as nn
import torch


def get_loss_function(config):
    try:
        if config['type'] == "Charbonnier":
            loss = L1_Charbonnier_loss(size_average=config['size_average'],
                                       batch_average=config['batch_average'])
        elif config['type'] == "Euclidean":
            loss = Euclidean_loss(size_average=config['size_average'],
                                       batch_average=config['batch_average'])
        else:
            loss = getattr(nn, config['type'])(reduction=config['reduction'])
        return loss
    except AttributeError:
        RuntimeError('The name of loss function is incorrect or does not exist!')


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, size_average=True, batch_average=True):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6
        self.size_average = size_average
        self.batch_average = batch_average

    def forward(self, x, y):
        '''
        Calculate loss and the ways of averaging.
        :param x: 4-D tensor, (N, C, H, W)
        :param y: 4-D tensor, (N, C, H, W)
        :return:
        '''
        diff = torch.add(x, -y)
        error = torch.sqrt(diff**2 + self.eps)
        if self.size_average:
            error = error.mean(-1).mean(-1).mean(-1)

        if self.batch_average:
            error = error.mean(0)

        loss = error.sum()
        return loss


class Euclidean_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, size_average=True, batch_average=True):
        super(Euclidean_loss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average

    def forward(self, x, y):
        '''
        Calculate loss and the ways of averaging.
        :param x: 4-D tensor, (N, C, H, W)
        :param y: 4-D tensor, (N, C, H, W)
        :return:
        '''
        diff = torch.add(x, -y)
        error = diff**2
        if self.size_average:
            error = error.mean(-1).mean(-1).mean(-1)

        if self.batch_average:
            error = error.mean(0)

        loss = error.sum()
        return loss
