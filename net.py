import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim
import torchvision
from torchvision.transforms import *
from torchvision.transforms import functional

import numpy as np
from collections import defaultdict
from scipy.stats import entropy

import pickle
import ipdb
import traceback

import os
from PIL import Image, ImageCms
import time
from skimage import io, color

from wideresnet import WideResNet


def PIL_To_Tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    See ``ToTensor`` for more details.
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    """
    if not(functional._is_pil_image(pic) or functional._is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return img.float().div(255)

    # if accimage is not None and isinstance(pic, accimage.Image):
    #     nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
    #     pic.copyto(nppic)
    #     return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()

    img = img.float()

    return img


class ImageNet(Dataset):

    def __init__(self, data_root_path, transform=None):
        """

        :param data_root_path: the root path of the images
        :param data_paths_and_labels: a file that contains the path of each image from root and its label
        :param transform: data augmentation, must include ToTensor to convert to PyTorch image format
        """

        self.srgb_profile = ImageCms.createProfile("sRGB")
        self.lab_profile = ImageCms.createProfile("LAB", colorTemp=6500)

        self.rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(self.srgb_profile, self.lab_profile, "RGB",
                                                                         "LAB")

        self.data_root_path = data_root_path
        self.image_list = os.listdir(data_root_path)
        self.transform = transform

        self.image_paths = [data_root_path + image for image in self.image_list]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = image.convert(mode='RGB')

        lab_image = ImageCms.applyTransform(image, self.rgb2lab_transform)

        if self.transform:
            lab_image = self.transform(lab_image)

        # return just lightness image and full image (q-value quantized, H by W by Q, 5-hot)

        return lab_image[0,:,:], lab_image
        # return np.asarray(lab_image)[:,:,0], np.asarray(lab_image)


# def bin_prestige():
if __name__=='__main__':

    # Need the line: return np.asarray(lab_image)[:,:,0], np.asarray(lab_image)
    # Do not use the transform to convert to tensor
    bins = defaultdict(int)
    images = ImageNet("ILSVRC2012_img_val/", transform=Resize((256,256)))

    for i in range(len(images.image_paths)):

        try:
            print(i)
            lab = images[i][1]
            ab_points = zip(lab[:,:,1].flatten().tolist(),lab[:,:,2].flatten().tolist())
            for (a, b) in ab_points:
                bins[(round(a, -1),round(b, -1))] += 1
        except:
            print(i, "FAILED")
            traceback.print_exc()
            pass


    ab2bin = {loc:idx for (idx,loc) in list(enumerate(list(bins)))}
    bin_counts = dict(bins)

    with open('ab2bin.p', 'wb') as handle:
        pickle.dump(ab2bin, handle)

    with open('bin_counts.p', 'wb') as handle:
        pickle.dump(bin_counts, handle)

class SabrinaNet(nn.Module):

    def __init__(self):

        super(SabrinaNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # downsample, 128 by 128
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # downsample, 64 by 64
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # downsample, 32 by 32
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=17), # upsample, 64 by 64
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )

        self.conv_layers = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv6,
            self.conv7
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return x


def train_sabrina(sabrina, epochs, cuda_available):

    train_data_transform = Compose([
        Resize((256,256)),
        PIL_To_Tensor()
    ])

