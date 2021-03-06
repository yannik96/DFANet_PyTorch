from model.dfanet import DFANet
import numpy as np
from collections import OrderedDict
import torchvision.transforms as transforms
from sklearn import preprocessing
import cv2
import torch


cityscapes_color_dict = {
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (107, 142, 35),
    9: (152, 251, 152),
    10: (70, 130, 180),
    11: (220, 20, 60),
    12: (255, 0, 0),
    13: (0, 0, 142),
    14: (0, 0, 70),
    15: (0, 60, 100),
    16: (0, 80, 100),
    17: (0, 0, 230),
    18: (119, 11, 32),
    19: (0, 0, 0)
}

mask_to_colormap = np.vectorize(lambda x: cityscapes_color_dict[x])


class DFANetPlugin(object):
    def __init__(self, im_height, im_width, use_cuda, model_url='./Cityscapes_best.pth.tar', opacity=0.4):
        super().__init__()
        self.name = "DFANet"
        self.im_height = im_height
        self.im_width = im_width
        self.use_cuda = use_cuda
        self.opacity = opacity

        self.model = DFANet(pretrained=False, pretrained_backbone=False)

        state_dict = torch.load(model_url)["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' from nn.DataParallel
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)

        self.model.eval()
        if use_cuda:
            self.model.cuda()

    def process(self, image):
        x_n = cv2.resize(image, dsize=(1024, 1024)).transpose((2, 0, 1))
        #mean0 = np.sum(x_n[0, :, :]) / (1024**2)
        #mean1 = np.sum(x_n[1, :, :]) / (1024 ** 2)
        #mean2 = np.sum(x_n[2, :, :]) / (1024 ** 2)
        x_n = x_n.astype(np.float64)
        x_n[0, :, :] = preprocessing.scale(x_n[0, :, :])
        x_n[1, :, :] = preprocessing.scale(x_n[1, :, :])
        x_n[2, :, :] = preprocessing.scale(x_n[2, :, :])
        #normal = transforms.Normalize(mean=[mean0, mean1, mean2], std=[1, 1, 1])
        x = torch.from_numpy(x_n).type('torch.FloatTensor').view(1, 3, 1024, 1024)
        if self.use_cuda:
            x = x.cuda()
        _, mask = self.model(x).max(1)
        if self.use_cuda:
            mask = mask.cpu()
        mask = mask.numpy()
        colormap = mask_to_colormap(mask)
        colormap = np.array(colormap)
        colormap = colormap.reshape(3, 1024, 1024)
        colormap = colormap.transpose(1, 2, 0)
        colormap = cv2.resize(colormap, dsize=(2048, 1024), interpolation=cv2.INTER_NEAREST)
        output = self.opacity * colormap + (1 - self.opacity) * image
        return output

    def release(self):
        del self.model
