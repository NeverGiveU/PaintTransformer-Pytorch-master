import torch
import paddle
from network import Painter
from network_paddle import Painter as Painter_paddle
import numpy as np 


painter_paddle = Painter_paddle(5, 8, 256)
painter_paddle.set_state_dict(paddle.load("./others/PaintTransformer-main/inference/paint_best.pdparams"))
painter_paddle.eval()

painter = Painter(5, 8, 256)
painter.load_state_dict(torch.load("./paint_best.pth"))
painter.eval()


x_paddle = paddle.randn([1, 3, 32, 32])
c_paddle = paddle.randn([1, 3, 32, 32])
x = torch.from_numpy(np.array(x_paddle))
c = torch.from_numpy(np.array(c_paddle))
y = painter(x, c)
y_paddle = painter_paddle(x_paddle, c_paddle)
print(y)
print(y_paddle)