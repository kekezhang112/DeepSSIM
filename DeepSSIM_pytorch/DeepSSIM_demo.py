# DeepSSIM Metric -- Pytorch version
# Copyright(c) 2025 Keke Zhang, Weiling Chen, Tiesong Zhao and Zhou Wang.
# All Rights Reserved.
# Please refer to/cite the following paper: "Structural Similarity in Deep ...
# Features: Unified Image Quality Assessment Robust to Geometrically Disparate Reference"

import numpy as np
import torch
from torchvision import transforms
import vgg_v16s5c1, vgg_v16s5c1sal
from PIL import Image

def gram_matrix(input):
    batchsize, channel, h, w = input.size()
    features = input.view(batchsize,channel,h*w)
    features_t = features.permute(0,2,1)
    G = torch.bmm(features, features_t)  # compute the gram product
    return G

class DeepSSIM(torch.nn.Module):
    def __init__(self):
        super(DeepSSIM, self).__init__()
        #### NOTE: If your application is not intended for image retargeting quality assessment...
        # or other clarity lossless quality assessment tasks, we recommend using the version...
        # noted "DeepSSIM without saliency calibration", which offers a simpler computation...
        # while delivering comparable performance.
        self.vgg_pretrained_features = vgg_v16s5c1.vgg16(pretrained=True) # DeepSSIM without saliency calibration
        # self.vgg_pretrained_features = vgg_v16s5c1sal.vgg16(pretrained=True) # DeepSSIM with saliency calibration

        self.vgg_pretrained_features.eval()

        for param in self.parameters():
             param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward_once(self, x): # DeepSSIM without saliency calibration
        h = (x - self.mean) / self.std
        h = self.vgg_pretrained_features(h)
        return h

    # def forward_once(self, x, map):  # DeepSSIM with saliency calibration
    #     h = (x - self.mean) / self.std
    #     h = self.vgg_pretrained_features(h, map)
    #     h_s5c1 = h
    #     return h_s5c1

    def forward(self, x, y): # DeepSSIM without saliency calibration

        featsx = self.forward_once(x)
        featsy = self.forward_once(y)
        featsx = gram_matrix(featsx)
        featsy = gram_matrix(featsy)

        eps = 1e-6
        x_grid = featsx.unfold(1, 4, 4).unfold(2, 4, 4)
        y_grid = featsy.unfold(1, 4, 4).unfold(2, 4, 4)

        x_grid_mean = x_grid.mean(dim=(3, 4), keepdim=True)
        y_grid_mean = y_grid.mean(dim=(3, 4), keepdim=True)

        x_var = ((x_grid - x_grid_mean) ** 2).mean(dim=(3, 4), keepdim=True)
        y_var = ((y_grid - y_grid_mean) ** 2).mean(dim=(3, 4), keepdim=True)

        xy_cov = (x_grid * y_grid).mean(dim=(3, 4), keepdim=True) - x_grid_mean * y_grid_mean

        S2 = (2 * xy_cov + eps) / (x_var + y_var + eps)
        S2 = S2.squeeze(-1).squeeze(-1)

        return S2.mean()


    # def forward(self, x, y, salx, saly): # DeepSSIM with saliency calibration
    #     featsx = self.forward_once(x,salx)
    #     featsy = self.forward_once(y,saly)
    #
    #     featsx = gram_matrix(featsx)
    #     featsy = gram_matrix(featsy)
    #     eps = 1e-6
    #     x_grid = featsx.unfold(1, 4, 4).unfold(2, 4, 4)
    #     y_grid = featsy.unfold(1, 4, 4).unfold(2, 4, 4)
    #     x_grid_mean = x_grid.mean(dim=(3, 4), keepdim=True)
    #     y_grid_mean = y_grid.mean(dim=(3, 4), keepdim=True)
    #     x_var = ((x_grid - x_grid_mean) ** 2).mean(dim=(3, 4), keepdim=True)
    #     y_var = ((y_grid - y_grid_mean) ** 2).mean(dim=(3, 4), keepdim=True)
    #     xy_cov = (x_grid * y_grid).mean(dim=(3, 4), keepdim=True) - x_grid_mean * y_grid_mean
    #     S2 = (2 * xy_cov + eps) / (x_var + y_var + eps)
    #     S2 = S2.squeeze(-1).squeeze(-1)
    #     return S2.mean()

def prepare_image(image):
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)


# def prepare_salimg(map): # DeepSSIM with saliency calibration
#     salmap = np.array(map).astype(np.float32)
#     salmap = salmap/255
#     smap = salmap[:,:,0]
#     smap[smap == 0] = 0 # It can also be smaller than a certain threshold, such as 0.05, 0.01, etc.
#     smap[smap != 0] = 1
#     smap = torch.from_numpy(smap).float()
#     return smap

if __name__ == '__main__':

    #### DeepSSIM without saliency calibration
    ref = prepare_image(Image.open('butterfly.png').convert("RGB"))
    dist = prepare_image(Image.open('butterfly_shif_0.50.png').convert("RGB"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepSSIM().to(device)
    ref = ref.to(device)
    dist = dist.to(device)
    deepssim_score = model(ref, dist)
    print(deepssim_score)

    #### DeepSSIM with saliency calibration
    # ref = prepare_image(Image.open('butterfly.png').convert("RGB"))
    # dist = prepare_image(Image.open('butterfly_shif_0.50.png').convert("RGB"))
    # salref = prepare_salimg(Image.open('./saliency_maps/butterfly.png').convert("RGB"))
    # saldis = prepare_salimg(Image.open('./saliency_maps/butterfly_shif_0.50.png').convert("RGB"))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = DeepSSIM().to(device)
    # ref = ref.to(device)
    # dist = dist.to(device)
    # salref = salref.to(device)
    # saldis = saldis.to(device)
    # deepssim_score = model(ref, dist, salref, saldis)
    # print(deepssim_score)




