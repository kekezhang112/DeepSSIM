# DeepSSIM-Lite loss
# Copyright(c) 2025 Keke Zhang, Weiling Chen, Tiesong Zhao and Zhou Wang.
# All Rights Reserved.
# Please refer to/cite the following paper: "Structural Similarity in Deep ...
# Features: Unified Image Quality Assessment Robust to Geometrically Disparate Reference"
# Usage: from DeepSSIM_lite_vgg16_loss import DeepSSIM
import numpy as np
import os, sys
import torch
from torchvision import models, transforms
import torch.nn as nn
import h5py
import vgg_v16s5c1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gram_matrix(input):
    batchsize, channel, h, w = input.size()
    features = input.view(batchsize,channel,h*w)
    features_t = features.permute(0,2,1)
    G = torch.bmm(features, features_t)
    G = G.div(channel * h * w)
    return G

class DeepSSIM(torch.nn.Module):
    def __init__(self):
        super(DeepSSIM, self).__init__()
        self.vgg_pretrained_features = vgg_v16s5c1.vgg16(pretrained=True).to(device)

        for param in self.parameters():
             param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def forward_once(self, x):
        x = x.to(device)
        h = (x - self.mean) / self.std
        h = self.vgg_pretrained_features(h)
        return h

    def forward(self, x, y):
        featsx = self.forward_once(x)
        featsy = self.forward_once(y)
        featsx = gram_matrix(featsx) # [bs , 512,512]
        featsy = gram_matrix(featsy)
        c = 1e-6
        x_mean = featsx.mean(dim=(1,2),keepdim=True)
        x_mean = x_mean.float()
        y_mean = featsy.mean(dim=(1,2),keepdim=True)
        y_mean = y_mean.float()
        x_var = ((featsx - x_mean) ** 2).mean(dim=(1,2),keepdim=True)
        y_var = ((featsy - y_mean) ** 2).mean(dim=(1,2),keepdim=True)
        xy_cov = (featsx * featsy).mean(dim=(1,2),keepdim=True) - x_mean * y_mean
        S2 = (2 * xy_cov + c) / (x_var + y_var + c)
        score = 1 - S2
        return score.mean()

def prepare_image(image):
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)

