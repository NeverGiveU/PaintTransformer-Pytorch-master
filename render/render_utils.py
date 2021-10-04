from math import remainder
import numpy as np 
from PIL import Image 
from torchvision import transforms
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from PIL import Image 
import matplotlib.pyplot as plt 
import cv2 
import math 


def read_image(img_pth, img_type="RGB", h=None, w=None):
    img = Image.open(img_pth).convert(img_type)
    if h is not None and w is not None:
        img = img.resize((w, h), resample=Image.ANTIALIAS)
    img = np.array(img)
    t = transforms.Compose([transforms.ToTensor()])
    return t(img).unsqueeze(0)


def tensor2numpy(x, denormalize=False):
    y = x.permute(1, 2, 0).squeeze(-1)
    if denormalize:
        y = y * 0.5 + 0.5
    return y 


class Erosion2d(nn.Module):
    def __init__(self, m=1):
        super(Erosion2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]

    def forward(self, x):
        b, c, h, w = x.size()
        x_padded = F.pad(x, pad=self.pad, mode="constant", value=1e9) # (b,c,h+2m,w+2m)
        x_unfolded = F.unfold(x_padded, kernel_size=2*self.m+1, stride=1, padding=0) # (b,c*(2m+1)*(2m+1), hw)
        x_unfolded = x_unfolded.view(b, c, -1, h, w)
        x_folded = torch.min(x_unfolded, dim=2)[0]
        return x_folded


class Dilation2d(nn.Module):
    def __init__(self, m=1):
        super(Dilation2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]
    
    def forward(self, x):
        b, c, h, w = x.size()
        x_padded = F.pad(x, pad=self.pad, mode="constant", value=-1e9) # (b,c,h+2m,w+2m)
        x_unfolded = F.unfold(x_padded, kernel_size=2*self.m+1, stride=1, padding=0) # (b,c,h,w)->(b,c*(2m+1)*(2m+1), hw)
        x_unfolded = x_unfolded.view(b, c, -1, h, w)
        x_folded = torch.max(x_unfolded, dim=2)[0]
        return x_folded


## a demo for testing `Dilation2d` and `Erosion2d`
# img_pth = "../samples/brush/brush_large_vertical.png"
# img = Image.open(img_pth).convert("RGB")
# t = transforms.ToTensor()
# x = t(img).unsqueeze(0)
# dilation2d = Dilation2d(m=5)
# erosion2d = Erosion2d(m=5)
# x_dilated = dilation2d(x)
# x_eroded = erosion2d(x)
# plt.subplot(131)
# plt.imshow(tensor2numpy(x[0])), plt.title("Original")
# plt.subplot(132)
# plt.imshow(tensor2numpy(x_dilated[0])), plt.title("Dilated")
# plt.subplot(133)
# plt.imshow(tensor2numpy(x_eroded[0])), plt.title("Eroded")
# plt.show()


def preprocess(arr, w=512, h=512):
    arr = cv2.resize(arr, (w, h), cv2.INTER_NEAREST)
    arr = arr.transpose((2, 0, 1))
    tsr = torch.from_numpy(arr).unsqueeze(0).astype("float32") / 255.0
    return tsr


def pad(tsr, H, W):
    b, c, h, w = tsr.size()
    pad_h = (H-h) // 2
    pad_w = (W-w) // 2
    remainder_h = (H-h) % 2
    remainder_w = (W-w) % 2
    tsr_padded = F.pad(tsr, pad=[pad_h, pad_h+remainder_h, pad_w, pad_w+remainder_w])
    return tsr_padded


def param2stroke(param, H, W, meta_brushes):
    b = param.size(0) # (N,8)
    param_list = torch.split(param, 1, dim=1) # get the first 8 elements
    x0, y0, w, h, theta = [item.squeeze(-1) for item in param_list[:5]]
    
    sin_theta = torch.sin(math.pi * theta)
    cos_theta = torch.cos(math.pi * theta)
    # index = torch.full((b,), -1).int()
    ones = torch.ones((b,))
    zeros = torch.zeros((b,))

    # index[h>w] = 0
    # index[h<=w]= 1
    index = torch.where(h<=w, ones, zeros).long()
    meta_brushes_resized = F.interpolate(meta_brushes, (H, W))
    brush = meta_brushes_resized[index]
    
    warp_00 = cos_theta / w
    warp_01 = sin_theta * H / (W * w)
    warp_02 = (1 - 2 * x0) * cos_theta / w + (1 - 2 * y0) * sin_theta * H / (W * w)
    warp_10 = -sin_theta * W / (H * h)
    warp_11 = cos_theta / h
    warp_12 = (1 - 2 * y0) * cos_theta / h - (1 - 2 * x0) * sin_theta * W / (H * h)
    warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
    warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
    warp = torch.stack([warp_0, warp_1], dim=1)
    grid = F.affine_grid(warp, (b, 3, H, W))
    brush = F.grid_sample(brush, grid)

    return brush 


## a demo for testing `param2stroke`
# brush_large_vertical = read_image("../samples/brush/brush_large_vertical.png", "L")
# brush_large_horizontal = read_image("../samples/brush/brush_large_horizontal.png", "L")
# meta_brushes = torch.cat([brush_large_vertical, brush_large_horizontal], dim=0) # (2,1,394,394)

# param = torch.FloatTensor([[0.61263520, 0.57211965, 0.29923382, 0.30190831, 0.44374609, 1.        , 1.        , 1.        ]])
# H = W = 512
# brush = param2stroke(param, H, W, meta_brushes)
# plt.imshow(tensor2numpy(brush[0]), cmap="gray")
# plt.show()