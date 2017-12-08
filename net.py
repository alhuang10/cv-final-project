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

import math
import os
from PIL import Image, ImageCms
from scipy.ndimage import gaussian_filter1d
import time
from skimage import io, color

from wideresnet import WideResNet

with open('ab2bin.p', 'rb') as f:
    ab2bin = pickle.load(f)
with open('bin_counts.p', 'rb') as f:
    bin_counts = pickle.load(f)
with open('bin_probs.p', 'rb') as f:
    bin_probs = pickle.load(f)
with open('block_to_q_vector_mapping.p', 'rb') as f:
    block_to_q_vector_mapping = pickle.load(f)


def pil_to_tensor(pic):
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

class PILToTensor(object):

    def __call__(self, pic):

        return pil_to_tensor(pic)

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

        ab_portion_image = lab_image[1:,:,:]
        # TODO finish labeling

        ground_truth_encoding = np.zeros([310, 256, 256])

        for i in range(256):
            for j in range(256):

                a = ab_portion_image[0,i,j]
                b = ab_portion_image[1,i,j]
                ground_truth_encoding[:, i, j] = block_to_q_vector_mapping[(a,b)]


        ground_truth_encoding = torch.FloatTensor(ground_truth_encoding)

        return lab_image[0,:,:].view(1,256,256), ground_truth_encoding  # just Lightness image and ground truth soft 5-hot encoding
        # return np.asarray(lab_image)[:,:,0], np.asarray(lab_image)


    def create_soft_encoding(self, ab_tuple, ab2bin):
        '''
        Takes in an ab pair and returns a Q length vector with 5 closest points/bins weighted using Gaussian kernel
        :param ab_tuple:
        :param ab2bin: mapping from ab to bin index
        :return:
        '''
        pass


def test_func(a, b, ab2bin):

    sigma = 5
    output_vector = [0] * len(ab2bin)

    distances = [((a_center, b_center),
                  np.linalg.norm([a-a_center, b-b_center])) for (a_center, b_center) in ab2bin.keys()]

    distances.sort(key=lambda x: x[1])

    for i in range(5):

        (a_center,b_center), distance = distances[i]

        index = ab2bin[(a_center,b_center)]

        # Weight using a gaussian kernel with sigma=5
        output_vector[index] = math.exp(-1*pow(distance,2) / (2*pow(sigma,2)))

    return output_vector


v_test_func = np.vectorize(test_func)



def find_block_to_output_vector_mapping():
    block_to_output_vector_mapping = {}

    for a in range(256):
        for b in range(256):
            print(a,b)

            block_to_output_vector_mapping[(a,b)] = test_func(a,b,ab2bin)

    ipdb.set_trace()

def bin_prestige():

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

def nn_prestige():
    nn = {}
    for (a,b) in [(i,j) for i in range(256) for j in range(256)]:
        nearest_bin = (round(a,-1),round(b,-1))
        if not nearest_bin in ab2bin: # this ab value is not in-gamut for the dataset, so no need to compute nn
            continue
        neighbors_ab = [nearest_bin] # closest neighbor is its bin
        d = 1
        while len(neighbors_ab) > 5:
            valid_neighbors_d = []
            valid_neighbors_d.extend([(10*d,10*a) for a in range(-d,d+1)])
            valid_neighbors_d.extend([(-10*d,10*a) for a in range(-d,d+1)])
            valid_neighbors_d.extend([(10*a,10*d) for a in range(-d,d+1)])
            valid_neighbors_d.extend([(10*a,-10*d) for a in range(-d,d+1)])

            valid_neighbors_d = [(x,y) for (x,y) in valid_neighbors_d if (x,y) in ab2bin]

            distances = [((x,y), np.linalg.norm([a-x,b-y])) for (x,y) in valid_neighbors_d]
            distances.sort(key=lambda x:x[1])
            neighbors_ab.extend([x[0] for x in distances][:min(5-len(neighbors_ab),5)]) # take the bin locs with the smallest dists
            d += 1

        nn[(a,b)] = neighbors_ab

    with open('nearest_neighbors.p', 'wb') as handle:
        pickle.dump(nn, handle)

def get_weight_vector():

    ab_to_weights = {}
    lamb = 0.5
    Q = len(bin_probs)

    for (a,b), p in bin_probs.items():
        weight_value = 1/((1-lamb)*p + lamb/Q)

        ab_to_weights[(a,b)] = weight_value

    probability_vector = [0]*Q
    weight_vector = [0]*Q

    for (a,b), index in ab2bin.items():
        weight_vector[index] = ab_to_weights[(a,b)]
        probability_vector[index] = bin_probs[(a,b)]

    normalization_factor = np.dot(probability_vector, weight_vector)

    # Normalize so weighting factor is one on expectation
    weight_vector = [weight/normalization_factor for weight in weight_vector]

    return weight_vector

    # probs = gaussian_filter1d(probs,5)
    # bin_probs_smoothed = {bins[i]:probs[i] for i in range(len(bin_probs))}
    # with open('bin_probs_smoothed.p','wb') as handler:
    #     pickle.dump(bin_probs_smoothed, handler)
    # with open('bin_probs_smoothed.p', 'rb') as f:
    #     bin_probs_smoothed = pickle.load(f)

# for a given image, softmax predictions in Q -> bin numbers
def get_annealed_means(z_hat, temp):
    # softmax Q predictions given in h x w x Q numpy array
    Q = z_hat.shape[2]
    denom = np.sum(np.exp(np.log(z_hat)/temp),axis=2)
    denom = np.dstack([denom] * Q) # repeat along third dimension 
    z_annealed = (np.exp(np.log(z_hat)/temp))/denom # still h x w x Q

    bins = list(ab2bin.values()) # vector of size Q
    return np.rint(np.matmul(z_annealed,bins)) # take the mean, round to nearest bin 

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
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # upsample, 64 by 64
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )

        self.conv8 = nn.Conv2d(256, 310, kernel_size=3, stride=1, padding=1)


        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(310, 310, kernel_size=4, stride=2, padding=1), # upsample 128 by 128
            nn.ConvTranspose2d(310, 310, kernel_size=4, stride=2, padding=1)  # upsample 256 by 256
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
        x = self.conv8(x)
        x = self.conv9(x)

        return x


def train_sabrina(sabrina, epochs, cuda_available):

    train_data_transform = Compose([
        Resize((256,256)),
        PIL_To_Tensor()
    ])


if __name__=='__main__':

    # with open('ab2bin.p', 'rb') as f:
    #     ab2bin = pickle.load(f)
    #
    # x = test_func(0,0, ab2bin)

    sabrina = SabrinaNet()

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print("Using CUDA")
        sabrina.cuda()


    train_data_transform = Compose([
        Resize((256,256)),
        PILToTensor()
    ])

    training_batch_size = 4

    print("Creating DataLoader")

    train_images = ImageNet("ILSVRC2012_img_val/", transform=train_data_transform)
    # train_images = ImageNet("bin_test/", transform=train_data_transform)
    trainloader = torch.utils.data.DataLoader(train_images, batch_size=training_batch_size,shuffle=False, num_workers=2)

    print("Generating weight vector")

    optimizer = optim.SGD(sabrina.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

    cross_entropy_weight_vector = get_weight_vector()
    cross_entropy_weight_vector = torch.Tensor(cross_entropy_weight_vector)

    if use_cuda:
        cross_entropy_weight_vector = cross_entropy_weight_vector.cuda()

    softmax = torch.nn.Softmax(dim=2)
    bce = torch.nn.BCELoss(weight=cross_entropy_weight_vector)

    print("here")

    for i, data in enumerate(trainloader):

        print(i)

        lightness_images, ground_truth_encodings = data

        if use_cuda:
            lightness_images, ground_truth_encodings \
                = Variable(lightness_images.cuda()), Variable(ground_truth_encodings.cuda())
        else:
            lightness_images, ground_truth_encodings = Variable(lightness_images), Variable(ground_truth_encodings)

        output = sabrina(lightness_images)

        output = output.view(training_batch_size,310,256*256).permute(0,2,1)
        ground_truth_encodings = ground_truth_encodings.view(training_batch_size,310,256*256).permute(0,2,1)

        optimizer.zero_grad()

        output = softmax(output)

        loss = bce(output, ground_truth_encodings)
        loss.backward()

        print(loss)

        optimizer.step()

        # torch.save(sabrina.state_dict(), "sabrina_model_weights_{num}".format(num=i))







