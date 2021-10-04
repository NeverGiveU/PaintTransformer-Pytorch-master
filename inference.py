import torch 
import numpy as np 
from network import Painter 
import os 
import time 
from render.render_utils import read_image
from render.render_serial import * 


if __name__ == "__main__":
    ## files
    input_path = "samples/inputs/darling.jpg"
    output_dir = "samples/outputs/"
    
    ## roots
    fname = os.path.basename(input_path).split(".")[0]
    save_dir = os.path.join(output_dir, fname)
    if os.path.exists(save_dir) is not True:
        os.mkdir(save_dir)

    ## image
    h = 512
    w = 512 
    target = read_image(input_path, h=h, w=w)
    canvas = torch.zeros_like(target)

    ## brush
    brush_large_vertical = read_image("./samples/brush/brush_large_vertical.png", "L")
    brush_large_horizontal = read_image("./samples/brush/brush_large_horizontal.png", "L")
    meta_brushes = torch.cat([brush_large_vertical, brush_large_horizontal], dim=0) # (2,1,394,394)

    ## model
    painter = Painter(5, 8, 256, 8, 3, 3)
    painter.load_state_dict(torch.load("./paint_best.pth"))
    painter.eval()
    
    ## begin
    t0 = time.time()
    # 1. serial
    painting_sequence = render_serial(target, canvas, painter, meta_brushes)
    print("total frame:", len(painting_sequence))
    for idx, frame in enumerate(painting_sequence):
        cv2.imwrite(os.path.join(save_dir, '%03d.png'%idx), frame)
    # 2. parallel

    print("Total inference time: {}s.".format(time.time()-t0))



    
