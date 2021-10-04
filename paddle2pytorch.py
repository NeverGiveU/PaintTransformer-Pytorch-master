import torch
import paddle
from network import Painter
import numpy as np 


## mapping rules
general_mapping = {
    "running_mean":"_mean",
    "running_var":"_variance",
    "multihead_attn":"cross_attn",
}

customed_mapping = {
    "query_pos_embedding":"query_pos",
    "row_embedding":"row_embed",
    "col_embedding":"col_embed",
    "enc_target":"enc_img",
    "enc_canvas":"enc_canvas",
    "linear_score":"linear_decider"
}


ckpt_pth = "./others/PaintTransformer-main/inference/paint_best.pdparams"
ckpt = paddle.load(ckpt_pth)
# for k in ckpt:
#     print(k, ckpt[k].shape)

painter = Painter(5, 8, 256)
state_dict = painter.state_dict()
# for k in state_dict:
#     print(k, state_dict[k].data.size())


## parameters
state_dict["query_pos_embedding"].data = torch.from_numpy(np.array(ckpt["query_pos"]))
state_dict["row_embedding"].data = torch.from_numpy(np.array(ckpt["row_embed"]))
state_dict["col_embedding"].data = torch.from_numpy(np.array(ckpt["col_embed"]))

## encoder
state_dict["enc_target.1.weight"] = torch.from_numpy(np.array(ckpt["enc_img.1.weight"]))
state_dict["enc_target.1.bias"] = torch.from_numpy(np.array(ckpt["enc_img.1.bias"]))
state_dict["enc_target.2.weight"] = torch.from_numpy(np.array(ckpt["enc_img.2.weight"]))
state_dict["enc_target.2.bias"] = torch.from_numpy(np.array(ckpt["enc_img.2.bias"]))
state_dict["enc_target.2.running_mean"] = torch.from_numpy(np.array(ckpt["enc_img.2._mean"]))
state_dict["enc_target.2.running_var"] = torch.from_numpy(np.array(ckpt["enc_img.2._variance"]))
# state_dict["enc_target.2.num_batches_tracked"] = torch.from_numpy(np.array(ckpt[""]))
state_dict["enc_target.5.weight"] = torch.from_numpy(np.array(ckpt["enc_img.5.weight"]))
state_dict["enc_target.5.bias"] = torch.from_numpy(np.array(ckpt["enc_img.5.bias"]))
state_dict["enc_target.6.weight"] = torch.from_numpy(np.array(ckpt["enc_img.6.weight"]))
state_dict["enc_target.6.bias"] = torch.from_numpy(np.array(ckpt["enc_img.6.bias"]))
state_dict["enc_target.6.running_mean"] = torch.from_numpy(np.array(ckpt["enc_img.6._mean"]))
state_dict["enc_target.6.running_var"] = torch.from_numpy(np.array(ckpt["enc_img.6._variance"]))
# state_dict["enc_target.6.num_batches_tracked"] = torch.from_numpy(np.array(ckpt[""]))
state_dict["enc_target.9.weight"] = torch.from_numpy(np.array(ckpt["enc_img.9.weight"]))
state_dict["enc_target.9.bias"] = torch.from_numpy(np.array(ckpt["enc_img.9.bias"]))
state_dict["enc_target.10.weight"] = torch.from_numpy(np.array(ckpt["enc_img.10.weight"]))
state_dict["enc_target.10.bias"] = torch.from_numpy(np.array(ckpt["enc_img.10.bias"]))
state_dict["enc_target.10.running_mean"] = torch.from_numpy(np.array(ckpt["enc_img.10._mean"]))
state_dict["enc_target.10.running_var"] = torch.from_numpy(np.array(ckpt["enc_img.10._variance"]))
# state_dict["enc_target.10.num_batches_tracked"] = torch.from_numpy(np.array(ckpt[""]))
state_dict["enc_canvas.1.weight"] = torch.from_numpy(np.array(ckpt["enc_canvas.1.weight"]))
state_dict["enc_canvas.1.bias"] = torch.from_numpy(np.array(ckpt["enc_canvas.1.bias"]))
state_dict["enc_canvas.2.weight"] = torch.from_numpy(np.array(ckpt["enc_canvas.2.weight"]))
state_dict["enc_canvas.2.bias"] = torch.from_numpy(np.array(ckpt["enc_canvas.2.bias"]))
state_dict["enc_canvas.2.running_mean"] = torch.from_numpy(np.array(ckpt["enc_canvas.2._mean"]))
state_dict["enc_canvas.2.running_var"] = torch.from_numpy(np.array(ckpt["enc_canvas.2._variance"]))
# state_dict["enc_canvas.2.num_batches_tracked"] = torch.from_numpy(np.array(ckpt[""]))
state_dict["enc_canvas.5.weight"] = torch.from_numpy(np.array(ckpt["enc_canvas.5.weight"]))
state_dict["enc_canvas.5.bias"] = torch.from_numpy(np.array(ckpt["enc_canvas.5.bias"]))
state_dict["enc_canvas.6.weight"] = torch.from_numpy(np.array(ckpt["enc_canvas.6.weight"]))
state_dict["enc_canvas.6.bias"] = torch.from_numpy(np.array(ckpt["enc_canvas.6.bias"]))
state_dict["enc_canvas.6.running_mean"] = torch.from_numpy(np.array(ckpt["enc_canvas.6._mean"]))
state_dict["enc_canvas.6.running_var"] = torch.from_numpy(np.array(ckpt["enc_canvas.6._variance"]))
# state_dict["enc_canvas.6.num_batches_tracked"] = torch.from_numpy(np.array(ckpt[""]))
state_dict["enc_canvas.9.weight"] = torch.from_numpy(np.array(ckpt["enc_canvas.9.weight"]))
state_dict["enc_canvas.9.bias"] = torch.from_numpy(np.array(ckpt["enc_canvas.9.bias"]))
state_dict["enc_canvas.10.weight"] = torch.from_numpy(np.array(ckpt["enc_canvas.10.weight"]))
state_dict["enc_canvas.10.bias"] = torch.from_numpy(np.array(ckpt["enc_canvas.10.bias"]))
state_dict["enc_canvas.10.running_mean"] = torch.from_numpy(np.array(ckpt["enc_canvas.10._mean"]))
state_dict["enc_canvas.10.running_var"] = torch.from_numpy(np.array(ckpt["enc_canvas.10._variance"]))
# state_dict["enc_canvas.10.num_batches_tracked"] = torch.from_numpy(np.array(ckpt[""]))

## adaptor
state_dict["conv.weight"].data = torch.from_numpy(np.array(ckpt["conv.weight"]))
state_dict["conv.bias"].data = torch.from_numpy(np.array(ckpt["conv.bias"]))

## transformer encoder
for i in [0, 1, 2]:
    k_prefix = "transformer.encoder.layers.{}.".format(i)
    state_dict[k_prefix+"self_attn.in_proj_weight"] = torch.cat([
        torch.from_numpy(np.array(ckpt[k_prefix+"self_attn.q_proj.weight"])).permute(1, 0),
        torch.from_numpy(np.array(ckpt[k_prefix+"self_attn.k_proj.weight"])).permute(1, 0),
        torch.from_numpy(np.array(ckpt[k_prefix+"self_attn.v_proj.weight"])).permute(1, 0),
    ], dim=0)
    state_dict[k_prefix+"self_attn.in_proj_bias"] = torch.cat([
        torch.from_numpy(np.array(ckpt[k_prefix+"self_attn.q_proj.bias"])),
        torch.from_numpy(np.array(ckpt[k_prefix+"self_attn.k_proj.bias"])),
        torch.from_numpy(np.array(ckpt[k_prefix+"self_attn.v_proj.bias"])),
    ], dim=0)
    for k in [k_prefix+"self_attn.out_proj.weight",
              k_prefix+"self_attn.out_proj.bias",
              k_prefix+"linear1.weight",
              k_prefix+"linear1.bias",
              k_prefix+"linear2.weight",
              k_prefix+"linear2.bias",
              k_prefix+"norm1.weight",
              k_prefix+"norm1.bias",
              k_prefix+"norm2.weight",
              k_prefix+"norm2.bias"]:
        if ("linear" in k or "proj" in k) and "weight" in k:
            state_dict[k].data = torch.from_numpy(np.array(ckpt[k])).permute(1, 0)
        else:
            state_dict[k].data = torch.from_numpy(np.array(ckpt[k]))
#
state_dict["transformer.encoder.norm.weight"].data = torch.from_numpy(np.array(ckpt["transformer.encoder.norm.weight"]))
state_dict["transformer.encoder.norm.bias"].data = torch.from_numpy(np.array(ckpt["transformer.encoder.norm.bias"]))

## transformer decoder
for i in [0, 1, 2]:
    k_prefix = "transformer.decoder.layers.{}.".format(i)
    state_dict[k_prefix+"self_attn.in_proj_weight"].data = torch.cat([
        torch.from_numpy(np.array(ckpt[k_prefix+"self_attn.q_proj.weight"])).permute(1, 0),
        torch.from_numpy(np.array(ckpt[k_prefix+"self_attn.k_proj.weight"])).permute(1, 0),
        torch.from_numpy(np.array(ckpt[k_prefix+"self_attn.v_proj.weight"])).permute(1, 0),
    ], dim=0)
    state_dict[k_prefix+"self_attn.in_proj_bias"].data = torch.cat([
        torch.from_numpy(np.array(ckpt[k_prefix+"self_attn.q_proj.bias"])),
        torch.from_numpy(np.array(ckpt[k_prefix+"self_attn.k_proj.bias"])),
        torch.from_numpy(np.array(ckpt[k_prefix+"self_attn.v_proj.bias"])),
    ], dim=0)
    state_dict[k_prefix+"multihead_attn.in_proj_weight"].data = torch.cat([
        torch.from_numpy(np.array(ckpt[k_prefix+"cross_attn.q_proj.weight"])).permute(1, 0),
        torch.from_numpy(np.array(ckpt[k_prefix+"cross_attn.k_proj.weight"])).permute(1, 0),
        torch.from_numpy(np.array(ckpt[k_prefix+"cross_attn.v_proj.weight"])).permute(1, 0),    
    ], dim=0)
    state_dict[k_prefix+"multihead_attn.in_proj_bias"].data = torch.cat([
        torch.from_numpy(np.array(ckpt[k_prefix+"cross_attn.q_proj.bias"])),
        torch.from_numpy(np.array(ckpt[k_prefix+"cross_attn.k_proj.bias"])),
        torch.from_numpy(np.array(ckpt[k_prefix+"cross_attn.v_proj.bias"])),    
    ], dim=0)
    state_dict[k_prefix+"multihead_attn.out_proj.weight"].data = torch.from_numpy(np.array(ckpt[k_prefix+"cross_attn.out_proj.weight"])).permute(1, 0)   
    state_dict[k_prefix+"multihead_attn.out_proj.bias"].data = torch.from_numpy(np.array(ckpt[k_prefix+"cross_attn.out_proj.bias"]))
    for k in [k_prefix+"self_attn.out_proj.weight", 
              k_prefix+"self_attn.out_proj.bias", 
              k_prefix+"linear1.weight", 
              k_prefix+"linear1.bias", 
              k_prefix+"linear2.weight", 
              k_prefix+"linear2.bias", 
              k_prefix+"norm1.weight", 
              k_prefix+"norm1.bias", 
              k_prefix+"norm2.weight", 
              k_prefix+"norm2.bias", 
              k_prefix+"norm3.weight", 
              k_prefix+"norm3.bias"]:
        if ("linear" in k or "proj" in k) and "weight" in k:
            state_dict[k].data = torch.from_numpy(np.array(ckpt[k])).permute(1, 0)
        else:
            state_dict[k].data = torch.from_numpy(np.array(ckpt[k]))
#
state_dict["transformer.decoder.norm.weight"].data = torch.from_numpy(np.array(ckpt["transformer.decoder.norm.weight"]))
state_dict["transformer.decoder.norm.bias"].data = torch.from_numpy(np.array(ckpt["transformer.decoder.norm.bias"]))

##
state_dict["linear_param.0.weight"].data = torch.from_numpy(np.array(ckpt["linear_param.0.weight"])).permute(1, 0)
state_dict["linear_param.0.bias"].data = torch.from_numpy(np.array(ckpt["linear_param.0.bias"]))
state_dict["linear_param.2.weight"].data = torch.from_numpy(np.array(ckpt["linear_param.2.weight"])).permute(1, 0)
state_dict["linear_param.2.bias"].data = torch.from_numpy(np.array(ckpt["linear_param.2.bias"]))
state_dict["linear_param.4.weight"].data = torch.from_numpy(np.array(ckpt["linear_param.4.weight"])).permute(1, 0)
state_dict["linear_param.4.bias"].data = torch.from_numpy(np.array(ckpt["linear_param.4.bias"]))
state_dict["linear_score.weight"].data = torch.from_numpy(np.array(ckpt["linear_decider.weight"])).permute(1, 0)
state_dict["linear_score.bias"].data = torch.from_numpy(np.array(ckpt["linear_decider.bias"]))


painter.load_state_dict(state_dict)
torch.save(painter.state_dict(), "paint_best.pth")