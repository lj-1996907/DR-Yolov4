
#预测

import torch
import torch.nn as nn
import numpy as np
from yolo import YOLO
from PIL import Image
import torch.backends.cudnn as cudnn
from nets.yolo4 import YoloBody


    
yolo = YOLO()
#yolo = torch.nn.DataParallel(yolo).cuda()
#yolo = torch.nn.DataParallel(yolo).cuda()






#model = torch.nn.DataParallel(model).cuda()
while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image)
        r_image.show()
