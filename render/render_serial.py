import time 
from PIL import Image 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import cv2 
from .render_utils import *
import math


__P__ = 32  # patch size
__N__ = 8   # #strokes per set


def  stroke_net_predict(tgt_patches, cvs_patches, patch_size, painter, stroke_num, channel_num):
    # (b,c,h,w) -> (b,c*__P__*__P__,(nh)(nw)) -> (b*nh*nw,c,__P__,__P__)
    # N = b*nh*nw
    tgt_patches = tgt_patches.permute(0,2,1).view(-1, channel_num, patch_size, patch_size) # (N,__N__,#param)
    cvs_patches = cvs_patches.permute(0,2,1).view(-1, channel_num, patch_size, patch_size) # (N,__N__,1)
    # predict strokes
    param, score = painter(tgt_patches, cvs_patches) # (N,__N__,5), (N,__N__,1)
    # print(score, score.sum())
    score = (score > 0)                              # binary decision vector
    # print(score.sum())
    # sample colors
    grid = param[:, :, :2].view(param.size(0)*stroke_num, 1, 1, 2) # (N*__N__,1,1,2)
    tgt_patches_tmp = tgt_patches.unsqueeze(1).repeat(
        1, stroke_num, 1, 1, 1
    ).view(
        tgt_patches.size(0)*stroke_num, channel_num, patch_size, patch_size
    )                                                              # (N*__N__,C,__P__,__P__)
    color = F.grid_sample(tgt_patches_tmp, 2*grid-1, align_corners=False).view(
        tgt_patches.size(0), stroke_num, channel_num
    )                                                              # (N,__N__,C)
    param = torch.cat([param, color], axis=-1)                     # (N,__N__,#param+C), e.g., #param==5, C=3.
    param = param.view(-1, painter.n_param_stroke+channel_num)     # (N*__N__,#param+C)

    score = score.view(-1)
    param[:, :2] = param[:, :2] / 1.25 + 0.1
    param[:, 2:4] = param[:, 2:4] / 1.25
    return param, score


def get_single_layer_lists(param, score,
                           target,
                           render_size_x, render_size_y, # both are in [640, 320, 160, 80, 40]
                           nh, nw, # both are in [1, 2, 4, 8, 16]
                           meta_brushes, dilation, erosion,
                           stroke_num):
    # N = b*nh*nw
    valid_foregrounds = param2stroke(param, render_size_y, render_size_x, meta_brushes) # (N*__N__,1,render_size_y,render_size_x)
    valid_alphas = (valid_foregrounds > 0).float()
    valid_foregrounds = valid_foregrounds.view(-1, stroke_num, 1, render_size_y, render_size_x) # (N, __N__, 1, render_size_y, render_size_x)
    valid_alphas = valid_alphas.view(-1, stroke_num, 1, render_size_y, render_size_x)           # (N, __N__, 1, render_size_y, render_size_x)

    # for forground, dilate it for larger regions
    valid_foregrounds_tmp = [dilation(valid_foregrounds[:, i, :, :, :]) for i in range(stroke_num)]
    valid_foregrounds = torch.stack(valid_foregrounds_tmp, dim=1)
    valid_foregrounds = valid_foregrounds.view(-1, 1, render_size_y, render_size_x)             # (N*__N__,1,render_size_x,render_size_y)
    # for alpha, erode it for smaller regions,
    # s.t. the msaked valid foreground regions are narrow w/o black artifacts
    valid_alphas_tmp = [erosion(valid_alphas[:, i, :, :, :]) for i in range(stroke_num)] 
    valid_alphas = torch.stack(valid_alphas_tmp, dim=1)
    valid_alphas = valid_alphas.view(-1, 1, render_size_y, render_size_x)                       # (N*__N__,1,render_size_x,render_size_y)
    
    # re-project to full resolution
    patch_y = 4*render_size_y // 5 # 512, 256, 128, 64, 32
    patch_x = 4*render_size_x // 5 # 512, 256, 128, 64, 32
    
    B, C, H, W = target.size()
    tgt_patches = target.view(B, C, nh, H//nh, nw, W//nw)    # (b,C,nh,H/nh,nw,H//nw)
    tgt_patches = tgt_patches.permute(0,2,4,1,3,5)[0]        # (b,)(nh,nw,C,H//nh,W//nw)
    # from now on, just take patches for the single sample as a batch !!!
    

    inner_fores = valid_foregrounds[:, :, render_size_y//10:9*render_size_y//10, render_size_x//10:9*render_size_x//10] # (N*__N__,1,patch_y,patch_x)
    #                                     64:576(512), 32:288(256), 16:144(128), 8:72(64), 4:36(32)
    print(inner_fores.size())
    inner_alphas = valid_alphas[:, :, render_size_y//10:9*render_size_y//10, render_size_x//10:9*render_size_x//10]     # (N*__N__,1,patch_y,patch_x)
    inner_fores = inner_fores.view(nh*nw, stroke_num, 1, patch_y, patch_x)              # (b,)(nh*nw,__N__,1,patch_y,patch_x)
    inner_alphas=inner_alphas.view(nh*nw, stroke_num, 1, patch_y, patch_x)              # (b,)(nh*nw,__N__,1,patch_y,patch_x)
    inner_tpatch= tgt_patches.contiguous().view(nh*nw,             C, patch_y, patch_x).unsqueeze(1) # (b,)(nh*nw,1,C,patch_y,patch_x)
    
    # error
    R = param[:, 5] # (N*__N__,)
    G = param[:, 6] # (N*__N__,)
    B = param[:, 7] # (N*__N__,)
    R = R.view(-1, stroke_num).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # (b,)(N,__N__,1,1,1)
    G = G.view(-1, stroke_num).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # (b,)(N,__N__,1,1,1)
    B = B.view(-1, stroke_num).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # (b,)(N,__N__,1,1,1)
    error_R = R * inner_fores - inner_tpatch[:, :, 0:1, :, :]
    error_G = G * inner_fores - inner_tpatch[:, :, 1:2, :, :]
    error_B = B * inner_fores - inner_tpatch[:, :, 2:3, :, :]
    error = torch.abs(error_R) + torch.abs(error_G) + torch.abs(error_B)    # (b,)(N,1,1,patch_y,patch_x)

    error = error*inner_alphas                                              # (b,)(N,__N__,1,patch_y,patch_x)
    error = torch.sum(error, dim=(2,3,4)) / torch.sum(inner_alphas, dim=(2,3,4))
    error = error.view(-1)                                                  # (b,)(N,__N__) -> (b,)(N*__N__)
    # zeros = torch.zeros_like(error, device=error.device)
    # error_list = torch.where(score, error, zeros)
    # error_list = list(error_list)
    error_list = []
    # print(error_list)

    xid_list = []
    yid_list = []
    valid_foregrounds_list = []
    valid_alphas_list = []
    valid_param_list = []
    # score = score.view(B, -1)                                # (N*__N__) -> (b,nh*nw*__N__)
    for idx, decision in enumerate(np.array(score.data.cpu())):
        if decision:
            valid_foregrounds_list.append(valid_foregrounds[idx])
            valid_alphas_list.append(valid_alphas[idx])
            valid_param_list.append(param[idx])
            error_list.append(error[idx])

            idx = idx // stroke_num # the unfolded index of the patch
            x_id = idx % nw         # the folded w-index of the patch
            idx = idx // nw 
            y_id = idx % nh         # the folded h-index of the patch
            xid_list.append(x_id)
            yid_list.append(y_id)
    
    return torch.Tensor(xid_list), torch.Tensor(yid_list),\
        torch.stack(valid_foregrounds_list, dim=0), torch.stack(valid_alphas_list, dim=0),\
        torch.stack(error_list, dim=0), torch.stack(valid_param_list, dim=0)
    

def get_single_stroke_on_full_image_A(x_id, y_id, 
                                      valid_foreground, valid_alpha, valid_param, 
                                      target, 
                                      render_size_x, render_size_y, patch_x, patch_y):
    patch_y_num = target.size(2) // patch_y
    patch_x_num = target.size(3) // patch_x
    
    _, C, _, _ = target.size()
    brush = valid_foreground.unsqueeze(0)  # (b,C,H,W) -> (1,b,C,H,W)
    color_map = valid_param[5:].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    brush = brush.repeat(1, C, 1, 1)
    brush = color_map * brush 
    
    # padding
    pad_l = x_id * patch_x
    pad_r = (patch_x_num-x_id-1) * patch_x
    pad_t = y_id * patch_y
    pad_d = (patch_y_num-y_id-1) * patch_y
    tmp_foreground = F.pad(brush, pad=(pad_l, pad_r, pad_t, pad_d))
    tmp_foreground = tmp_foreground[:, :, render_size_y//10:-render_size_y//10, render_size_x//10:-render_size_x//10]
    
    tmp_alpha = F.pad(valid_alpha.unsqueeze(0), pad=(pad_l, pad_r, pad_t, pad_d))
    tmp_alpha = tmp_alpha[:, :, render_size_y//10:-render_size_y//10, render_size_x//10:-render_size_x//10]
    return tmp_foreground, tmp_alpha


def get_single_stroke_on_full_image_B(x_id, y_id, 
                                      valid_foreground, valid_alpha, valid_param, 
                                      target, 
                                      render_size_x, render_size_y, patch_x, patch_y):
    x_expand = patch_x // 2 + render_size_x // 10
    y_expand = patch_y // 2 + render_size_y // 10
    
    pad_l = x_id * patch_x
    pad_r = target.size(3) + 2*x_expand - (x_id*patch_x+render_size_x)
    pad_t = y_id * patch_y
    pad_d = target.size(2) + 2*y_expand - (y_id*patch_y+render_size_y)
    _, C, _, _ = target.size()
    brush = valid_foreground.unsqueeze(0).repeat(1, C, 1, 1)
    color_map = valid_param[5:].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    brush = brush * color_map

    tmp_foreground = F.pad(brush, pad=(pad_l, pad_r, pad_t, pad_d))
    tmp_foreground = tmp_foreground[:, :, y_expand:-y_expand, x_expand:-x_expand]
    tmp_alpha = F.pad(valid_alpha.unsqueeze(0), pad=(pad_l, pad_r, pad_t, pad_d))
    tmp_alpha = tmp_alpha[:, :, y_expand:-y_expand, x_expand:-x_expand]
    
    return tmp_foreground, tmp_alpha


def render_serial(target, canvas, painter, meta_brushes):
    assert target.size() == canvas.size()
    b, C, H, W = target.size()
    assert b == 1, "One and only one sample for each inference."
    
    __K__ = max(math.ceil(math.log2(max(H, W)/__P__)), 0)
    # print(__K__) >> 4

    erosion = Erosion2d(m=1)
    dilation = Dilation2d(m=1)

    frames_per_layer = [20, 20, 30, 40, 60, 60]
    painting_sequence = []

    with torch.no_grad():
        for layer in range(__K__+1):
            # if layer != __K__:
            #     continue
            t0 = time.time()
            layer_size = __P__ * (2**layer) # 32, 64, 128, 256, 512
            
            # resize
            tgt = F.interpolate(target, (layer_size, layer_size))
            cvs = F.interpolate(canvas, (layer_size, layer_size))
            # split to patches
            tgt_patches = F.unfold(tgt, kernel_size=(__P__, __P__), stride=(__P__, __P__)) # (b,c*__P__*__P__,nh*nw)
            cvs_patches = F.unfold(cvs, kernel_size=(__P__, __P__), stride=(__P__, __P__)) # (b,c*__P__*__P__,nh*nw)
            # the # of patches along each axis
            _, _, h, w = tgt.size()
            nh = (h - __P__) // __P__ + 1  # 1, 2, 4, 8, 16
            nw = (w - __P__) // __P__ + 1  # 1, 2, 4, 8, 16
            # 
            render_size_y = int(1.25 * H // nh) # 640, 320, 160, 80, 40
            render_size_x = int(1.25 * W // nw) # 640, 320, 160, 80, 40

            # generate strokes on window type A
            param, score = stroke_net_predict(tgt_patches, cvs_patches, __P__, painter, __N__, C)
            target_expanded = target # used for color sampling of `A` version
            wA_xid_list, wA_yid_list, wA_fore_list, wA_alpha_list, wA_error_list, wA_params = \
                get_single_layer_lists(param, score, target, render_size_x, render_size_y, nh, nw,
                                       meta_brushes, dilation, erosion, __N__)
            
            # generate strokes on window type B
            tgt = F.pad(tgt, pad=(__P__//2, __P__//2, __P__//2, __P__//2))
            cvs = F.pad(cvs, pad=(__P__//2, __P__//2, __P__//2, __P__//2))
            tgt_patches = F.unfold(tgt, kernel_size=(__P__, __P__), stride=(__P__, __P__))
            cvs_patches = F.unfold(cvs, kernel_size=(__P__, __P__), stride=(__P__, __P__))
            
            nh += 1
            nw += 1
            param, score = stroke_net_predict(tgt_patches, cvs_patches, __P__, painter, __N__, C)
            
            patch_y = 4 * render_size_y // 5 
            patch_x = 4 * render_size_x // 5 
            target_expanded = F.pad(target, (patch_x//2, patch_x//2, patch_y//2, patch_y//2)) # used for color sampling of `B` version
            wB_xid_list, wB_yid_list, wB_fore_list, wB_alpha_list, wB_error_list, wB_params = \
                get_single_layer_lists(param, score, target_expanded, render_size_x, render_size_y, nh, nw,
                                       meta_brushes, dilation, erosion, __N__)

            # rank strokes and plot strokes one by one
            numA = len(wA_error_list)
            numB = len(wB_error_list) 

            total_error_list = torch.cat((wA_error_list,wB_error_list), dim=0)
            sort_list = torch.argsort(total_error_list)

            sample = 0
            samples = np.linspace(0, len(sort_list) - 2, frames_per_layer[layer]).astype(int)
            
            for ii in sort_list:
                ii = int(ii)
                if ii < numA:
                    x_id = int(wA_xid_list[ii].data.item())
                    y_id = int(wA_yid_list[ii].data.item())
                    valid_foreground = wA_fore_list[ii]
                    valid_alpha = wA_alpha_list[ii]
                    valid_param = wA_params[ii]
                    tmp_foreground, tmp_alpha = get_single_stroke_on_full_image_A(
                        x_id, y_id, valid_foreground, valid_alpha, valid_param, target, render_size_x, render_size_y, patch_x, patch_y)
                else:
                    x_id = int(wB_xid_list[ii-numA].data.item())
                    y_id = int(wB_yid_list[ii-numA].data.item())
                    valid_foreground = wB_fore_list[ii-numA]
                    valid_alpha = wB_alpha_list[ii-numA]
                    valid_param = wB_params[ii-numA] 
                    tmp_foreground, tmp_alpha = get_single_stroke_on_full_image_B(
                        x_id, y_id, valid_foreground, valid_alpha, valid_param, target, render_size_x, render_size_y, patch_x, patch_y)
                
                canvas = tmp_foreground * tmp_alpha + (1 - tmp_alpha) * canvas
                if sample in samples:
                    saveframe = (np.array(canvas).squeeze().transpose([1,2,0])[:,:,::-1] * 255).astype(np.uint8)
                    painting_sequence.append(saveframe)
                    #saveframe = cv2.resize(saveframe, (ow, oh))
                sample += 1

            print("layer %d cost: %.02f" %(layer, time.time() - t0))
        
        saveframe = (np.array(canvas).squeeze().transpose([1,2,0])[:,:,::-1] * 255).astype(np.uint8)
        painting_sequence.append(saveframe)
    return painting_sequence