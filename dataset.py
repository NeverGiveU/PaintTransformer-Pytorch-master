from matplotlib import cm
import numpy as np 
import os 
from PIL import Image 
import cairosvg
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt 
from torchvision import transforms
import torch 


def linear_equation(x1, y1, x2, y2):
    '''
    ax + by + c = 0.0
    '''
    if y1 == y2:
        a = 0.
        b = 1.
        c = -y1
    elif x1 == x2:
        a = 1.
        b = 0.
        c = -x1
    else:
        k = (y2-y1)/(x2-x1)
        b = y1 - k*x1 
        a = k 
        c = b 
        b = -1.
    return a, b, c 



class RandomArchDataset(Dataset):
    def __init__(self, line_width=1, mode="Mc", height=256, width=256) -> None:
        super(RandomArchDataset, self).__init__()
        self.line_width = line_width
        self.mode = mode 
        self.height = height
        self.width = width

        self.svg_header = '<?xml version="1.0" encoding="utf-8"?>\
                <svg version="1.1" id="layer" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"\
	             viewBox="0 0 {} {}" style="enable-background:new 0 0 {} {};" xml:space="preserve">'.format(self.width, self.height, self.width, self.height) + \
            '<style type="text/css">.st0{fill' + ':none;stroke:#000000;stroke-miterlimit:10;stroke-width:{};'.format(self.line_width)+'}</style>'
        self.svg_tailer = '</svg>'

        self.t = transforms.Compose([transforms.Resize((self.height, self.width)), 
                                     transforms.ToTensor()])

    def control_points(self, tag, x0, y0):
        if tag == 'z':
            return 'z', []
        elif tag == 'c':
            ## to make sure a convew will be obtained, sample the first 2 intermediate points
            x1, y1 = random.uniform(self.width/16, self.width/4), random.uniform(self.height/8, self.height/4)
            if random.random() > 0.5:
                x1 = -x1 
            if random.random() > 0.5:
                y1 = -y1 
            x1 += x0 
            y1 += y0 

            x2, y2 = x0, y0 
            while abs(x0-x2) < 1e-2 or abs(x0-y2) < 1e-2:
                x2, y2 = random.uniform(self.width/16, self.width/4), random.uniform(self.height/8, self.height/4)
                if random.random() > 0.5:
                    x2 = -x2 
                if random.random() > 0.5:
                    y2 = -y2 
                x2 += x0 
                y2 += y0 
            # to sample the 4th point,
            # 1) 根据 第 1、3 个点确定一条直线
            a, b, c = linear_equation(x1, y1, x2, y2)
            # 2) 根据 第 2 个点确定第 4 个点在直线的上方还是下方
            r1 = a*x1 + b*y1 + c 
            if r1 >= 0:
                r3 = 1
                while r3 >= 0:
                    x3, y3 = random.uniform(self.width/16, self.width/4), random.uniform(self.height/8, self.height/4)
                    if random.random() > 0.5:
                        x3 = -x3 
                    if random.random() > 0.5:
                        y3 = -y3
                    x3 += x0 
                    y3 += y0 
                    r3 = a*x3 + b*y3 + c 
            else:
                r3 = -1
                while r3 <= 0:
                    x3, y3 = random.uniform(self.width/16, self.width/4), random.uniform(self.height/8, self.height/4)
                    if random.random() > 0.5:
                        x3 = -x3 
                    if random.random() > 0.5:
                        y3 = -y3
                    x3 += x0 
                    y3 += y0 
                    r3 = a*x3 + b*y3 + c 
            ## plotting
            plt.plot([x0, x1, x2, x3, x0], [y0, y1, y2, y3, y0])
            plt.plot(x0, y0, 'r.')
            plt.plot(x1, y1, 'g.')
            plt.plot(x2, y2, 'b.')
            plt.plot(x3, y3, 'y.')

            s = 'c'
            s += ',{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f}'.format(x1-x0, y1-y0, x2-x0, y2-y0, x3-x0, y3-y0)
            return s, [(x1-x0)/self.width, (y1-y0)/self.height, (x2-x0)/self.width, (y2-y0)/self.height, (x3-x0)/self.width, (y3-y0)/self.height]
        else:
            return "", []
    
    def reset_scale(self, height, width):
        self.height = height
        self.width = width
        self.t = transforms.Compose([transforms.Resize((self.height, self.width)), 
                                     transforms.ToTensor()])

    def __getitem__(self, index=0, plotting=False):
        points = []
        path = '<path class="st0" d="'
        x0, y0 = random.uniform(0.0, self.width-0.1), random.uniform(0.0, self.height-0.1)
        path += "{}{:.1f},{:.1f}".format('M', x0, y0)
        points.append(x0/self.width)
        points.append(y0/self.height)

        for tag in self.mode[1:]:
            p1, p2 = self.control_points(tag, x0, y0)
            path += p1 
            points.extend(p2)
        path += '"/>'
        
        svg_str = self.svg_header + path + self.svg_tailer
        fh = open("./samples/sample.svg", "w", encoding="utf-8")
        fh.write(svg_str)
        fh.close()

        cairosvg.svg2png(url="./samples/sample.svg", write_to="./samples/sample.png")
        sample = Image.open("./samples/sample.png").convert("RGBA")
        # agba2gray
        array = np.array(sample)
        alpha = array[:, :, 3:4]
        alpha = alpha.astype(np.float32)/255
        alpha = 1-alpha 
        ones = np.ones_like(array[:, :, :3], dtype=np.float32)
        ones = (ones*alpha*255).astype(np.uint8)
        array = ones + array[:, :, :3]

        sample = Image.fromarray(array).convert("L")

        tensor = self.t(sample)
        
        if plotting:
            plt.imshow(sample, cmap="gray")
            plt.show()
        return torch.Tensor(points), tensor

# random_arch_dataset = RandomArchDataset()
# while True:
#     p, t = random_arch_dataset.__getitem__(plotting=True)
#     print(p)
