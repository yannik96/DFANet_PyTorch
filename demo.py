import torch
from datasets.cityscapes import Cityscapes
import torchvision.datasets as datasets
import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms
from plugin import DFANetPlugin
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image
import cv2


def main():
    demo()

def demo():
    net_plugin = DFANetPlugin(2048, 1024, False)

    image = cv2.imread("")  # path of image

    output = net_plugin.process(np.array(image))

    cv2.imwrite("demo.png", output)


if __name__ == '__main__':
    main()
