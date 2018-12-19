# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
# @Time    : 11/09/2018 11:13 PM
# @Author  : Boyu Zhang
# @Site    : 
# @File    : sr_trainer.py
# @Software: PyCharm
"""

from bases.trainer_base import TrainerBase
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from utils import np_utils, utils
from trainers import get_optimizer
from trainers import get_loss_function
from skimage import measure
import pywt
import matlab
import matlab.engine as meng
from utils.imresize import imresize

# import random
import torch.nn as nn
import torch
import tqdm
import os
import numpy as np
import torch.nn.functional as F
import contextlib
import sys
# from torchviz import make_dot


class DummyFile(object):
    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.tqdm.write(x, file=self.file)

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile(sys.stdout)
    yield
    sys.stdout = save_stdout


class SRTrainer(TrainerBase):
    def __init__(self, model, data, config):
        super(SRTrainer, self).__init__(model, data, config)
        if self.config['cuda'] and not torch.cuda.is_available():
            print("GPU is not available on this device! Running in CPU!")
            self.config['cuda'] = False

        if self.config['resume']:
            model_path = os.path.join(self.config['checkpoint'])
            pretrained = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(pretrained)

        self.optim, self.scheduler = get_optimizer.get_optimizer(model, config['optimizer'])
        self.loss_func = get_loss_function.get_loss_function(config['loss'])

        if self.config['cuda']:
            self.model = self.model.cuda()
            self.loss_func = self.loss_func.cuda()

        self.highest_score = 0
        self.best_model = None
        self.draw_graph = False
        self.writer = SummaryWriter(self.config['tb_dir'])

    @staticmethod
    def tocuda(tensors):
        if isinstance(tensors, list):
            tensors = [img.cuda() for img in tensors]
        else:
            tensors = tensors.cuda()
        return tensors

    def feedforward(self, x, y):
        # TODO 修改为False则可以使用预训练的
        preds = self.model(x)
        loss = sum([self.loss_func(preds[i], y[i]) for i in range(len(preds))])

        return loss, preds

    def train(self):
        eng = meng.start_matlab()

        for epoch in range(1, self.config['num_epochs'] + 1):
            print('running epoch {}'.format(epoch))
            train_loss = 0.0
            self.model.train()

            for iteration, batch in enumerate(tqdm.tqdm(self.data['train'], file=sys.stdout)):
                # psnr = self.validate(self.data['test'], combine)
                # if iteration < len(self.data['train'])-5:
                #     continue
                with nostdout():
                    step = len(self.data['train']) * (epoch - 1) + iteration
                    x, y = batch[0], batch[1:]
                    if self.config['cuda']:
                        x = self.tocuda(x)
                        y = self.tocuda(y)

                    self.optim.zero_grad()

                    loss, preds = self.feedforward(x, y)

                    # for name, param in self.model.named_parameters():
                    #     a = param.clone().cpu().data.numpy()
                    #     print('weight before', name, a.max(), a.min())
                    # print('\n')

                    loss.backward()
                    if 'clip' in self.config.keys():
                        nn.utils.clip_grad_norm(self.model.parameters(), self.config['clip'])
                    # for name, param in self.model.named_parameters():
                    #     if param.grad is not None
                    #         a = param.grad.clone().cpu().data.numpy()
                    #         print('grad', name, a.max(), a.min())
                    # print('\n')
                    self.optim.step()

                    # for name, param in self.model.named_parameters():
                    #     a = param.clone().cpu().data.numpy()
                    #     print('weight after', name, a.max(), a.min())
                    # print('\n')
                    train_loss += loss.item()
                    tqdm.tqdm.write('Epoch {} bloss is {}'.format(epoch, loss.item()))

                    # add log for visualization
                    if iteration % self.config['log_freq'] == 0:
                        # print(self.validate(self.data['test']))
                        self.writer.add_scalar('train_loss', loss.item(), step)

                        for idx, channel in enumerate(preds):
                            dummy_sub = vutils.make_grid(channel[:9], normalize=False, scale_each=True)
                            self.writer.add_image('wm_channel_{}'.format(idx), dummy_sub, step)

                        weight_lr = self.optim.param_groups[0]['lr']
                        self.writer.add_scalar('weight_lr', weight_lr, step)

                        bias_lr = self.optim.param_groups[1]['lr']
                        self.writer.add_scalar('bias_lr', bias_lr, step)

                        for name, param in self.model.named_parameters():
                            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), step)
                            if param.grad is not None:
                                self.writer.add_histogram('{}_gradient'.format(name), param.grad.clone().cpu().data.numpy(), step)
                # break

            epoch_loss = train_loss / len(self.data['train'])

            # adjust learning rate according to training loss
            if self.scheduler is not None:
                if self.config["optimizer"]["lr_scheduler"] == "ReduceLROnPlateau":
                    self.scheduler.step(metrics=epoch_loss)
                else:
                    self.scheduler.step()

            # validation
            average_psnr = self.validate(self.data['test'], eng)
            for i, ap in enumerate(average_psnr):
                self.writer.add_scalar('{}x'.format(i+2), ap, epoch)

            if len(average_psnr) == 1:
                msg = 'Net: {}, Epoch: {}, Training Loss: {:.4f}, ' \
                      'Validation PSNR: {}x {:.4f}'.format(self.config['net'], epoch, epoch_loss, self.config['upscale'], average_psnr[0])
            else:
                msg = 'Net: {}, Epoch: {}, Training Loss: {:.4f}, Validation PSNR {}x:'.format(self.config['net'], epoch, epoch_loss, self.config['upscale'])
                for idx, psnr in enumerate(average_psnr):
                    msg += '{:.4f},'.format(psnr)

            print(msg)
            # save checkpoint
            if epoch % self.config['save_freq'] == 0 or np.max(average_psnr) > self.highest_score:
                self.checkpoint(epoch, average_psnr)
                print("saving checkpoint in {}".format(self.config['cp_dir']))

        self.writer.close()
        return self.highest_score, self.best_model

    def validate(self, data, eng):
        set_psnr = []
        self.model.eval()
        for x_list, y in data:
            x_list = self.tocuda(x_list)
            y = y.data.cpu().numpy().squeeze()
            y = utils.modcrop(y, self.config['upscale'])
            psnr_multi_scale = []
            for i, img in enumerate(x_list):
                if i != self.config['upscale']-2:
                    continue

                preds = self.model(img)
                # preds = [matlab.double(p.data.cpu().numpy().squeeze().tolist()) for p in preds]
                preds = [p.data.cpu().numpy().squeeze() for p in preds]

                pred_img = pywt.idwt2((preds[0], (preds[1:])), 'bior1.1')
                # pred_img = eng.idwt2(preds[0], preds[1], preds[2], preds[3], 'bior1.1')
                # pred_img = eng.idwt2(*preds, 'bior1.1')

                pred_img = np.clip(pred_img, 16 / 255, 235 / 255)
                pred_img = pred_img.squeeze()
                pred_img = utils.modcrop(pred_img, self.config['upscale'])
                psnr_multi_scale.append(measure.compare_psnr(pred_img, y))

            set_psnr.append(psnr_multi_scale)

        set_psnr = np.array(set_psnr)

        average_psnr = np.mean(set_psnr, 0)
        return average_psnr

    def checkpoint(self, epoch, val_psnr):
        max_score = np.max(val_psnr)
        model_name = 'weights.epoch_{}_mean_val_psnr_{:0.3f}.hdf5'.format(epoch, max_score)
        filepath = os.path.join(self.config['cp_dir'], model_name)
        if max_score > self.highest_score:
            self.highest_score = max_score
            self.best_model = model_name
        torch.save(self.model.state_dict(), filepath)

