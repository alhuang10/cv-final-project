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



def train_fox(foxnet, epochs, cuda_available):

    # 16 limit for 28 layer - 10 wide network seems like
    training_batch_size = 128
    validation_batch_size = 10

    channel_mean = torch.Tensor([.4543, .4362, .4047])
    # channel_std = torch.Tensor([.2274, .2244, .2336])
    channel_std = torch.ones(3)

    train_data_transform = Compose([
        Resize(256),
        ToTensor()
    ])

    val_data_transform = Compose([
        TenCrop(112),  # Crops PIL image into four corners, central crop, and flipped version
        Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
        Lambda(lambda crops: torch.stack([Normalize(channel_mean, channel_std)(crop) for crop in crops]))
    ])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(foxnet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.25, verbose=True, patience=5)

    # 32 with FoxNet
    trainset = Places("data/images/", "ground_truth/train.txt", transform=train_data_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=training_batch_size,
                                              shuffle=True, num_workers=4)

    # 10 with FoxNet
    valset = Places("data/images/", "ground_truth/val.txt", transform=val_data_transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=validation_batch_size,
                                            shuffle=False, num_workers=8)

    best_validation_acc = 0
    best_model_wts = None

    print("Beginning Training")

    training_accuracies = []
    validation_accuracies = []
    entropies = []

    validation_softmax = torch.nn.Softmax(dim=2)

    for epoch in range(epochs):

        running_loss = 0

        train_top5_right = 0
        train_top5_wrong = 0

        current_time = time.time()

        ### Start of training code
        foxnet.train(True)
        for i, data in enumerate(trainloader, 0):

            if i % 10 == 0:
                time_to_run = time.time() - current_time
                current_time = time.time()
                print("Training:", i, time_to_run)

            input_images, labels = data

            if cuda_available:
                input_images, labels = Variable(input_images.cuda()), Variable(labels.cuda())
            else:
                input_images, labels = Variable(input_images), Variable(labels)

            optimizer.zero_grad()

            outputs = foxnet(input_images)
            loss = criterion(outputs, labels)
            loss.backward()

            running_loss += loss.data[0]
            optimizer.step()

            # Keep track of training score
            _, top_5_indices = torch.topk(outputs, 5)
            num_correct, num_incorrect = find_top_5_error(labels, top_5_indices)
            train_top5_right += num_correct
            train_top5_wrong += num_incorrect

            # Print stats every 1000
            if i % 500 == 499:
                print('[%d, %5d] average loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

        training_acc = train_top5_right / (train_top5_right+train_top5_wrong)
        training_accuracies.append(training_acc)
        print("Epoch {e}: Training Accuracy: {acc}".format(e=epoch + 1, acc=training_acc))
        print("Training accuracies so far:", training_accuracies)
        ### End of training code

        ### Start of Validation code
        foxnet.train(False)
        # Calculate validation accuracy after each epoch
        val_top5_right = 0
        val_top5_wrong = 0

        validation_loss = 0

        current_time = time.time()

        # Make prediction for validation set and test set by taking a 10 crop, and taking top 5 from the sum of the 10
        # Takes about 2 minutes to run, kind of slow
        for i, val_data in enumerate(valloader):

            if i % 50 == 0:
                time_to_run = time.time() - current_time
                current_time = time.time()
                print("Validation:", i, time_to_run)

            input_images, labels = val_data

            input_images = input_images.view(validation_batch_size*10, 3, 112, 112)  # First dimension is batch_size * 10

            if cuda_available:
                input_images, labels = Variable(input_images.cuda()), Variable(labels.cuda())
            else:
                input_images, labels = Variable(input_images), Variable(labels)

            output = foxnet(input_images)
            output = output.view(validation_batch_size, 10, 100)  # Each index into first dimension is a single one of the 10 predictions
            output = validation_softmax(output)

            combined_output = torch.sum(output, dim=1)  # Average the 10 predictions

            loss = criterion(combined_output, labels)

            validation_loss += loss.data[0]

            _, top_5_indices = torch.topk(combined_output, 5)
           
            num_correct, num_incorrect = find_top_5_error(labels, top_5_indices)

            val_top5_right += num_correct
            val_top5_wrong += num_incorrect

        validation_acc = val_top5_right/(val_top5_right+val_top5_wrong)
        validation_accuracies.append(validation_acc)
        print("Epoch {e}: Validation Accuracy: {acc}".format(e=epoch+1, acc=validation_acc))
        print("Epoch {e}: Validation Loss: {loss}".format(e=epoch+1, loss=validation_loss))
        print("Validation accuracies so far:", validation_accuracies)

        if validation_acc > best_validation_acc:
            best_validation_acc = validation_acc
            best_model_wts = foxnet.state_dict()

            torch.save(best_model_wts, "current_best_model_weights")

        # Adjust the learning rate when the validation loss or accuracy plateaus
        scheduler.step(validation_acc)

        # Save the weights for each epoch
        torch.save(foxnet.state_dict(), "model_weights/epoch_{num}_model_weights".format(num=epoch+1))

        ent = evaluate_foxnet(fox, use_cuda, epoch=epoch, folder="dropout_trial_submission_files/")
        entropies.append(ent)
        print("Entropies:", entropies)


def evaluate_foxnet(foxnet, cuda_available, epoch=0, folder="./"):

    # Disable dropout at eval time
    foxnet.train(False)

    channel_mean = torch.Tensor([.4543, .4362, .4047])
    # channel_std = torch.Tensor([.2274, .2244, .2336])
    channel_std = torch.ones(3)

    test_data_transform = Compose([
        TenCrop(112),
        Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
        Lambda(lambda crops: torch.stack([Normalize(channel_mean, channel_std)(crop) for crop in crops]))
    ])

    testset = Places("data/images/", "ground_truth/test.txt", transform=test_data_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=8)

    counts = defaultdict(int)
    predictions = []

    test_softmax = torch.nn.Softmax(dim=1)

    for i, test_data in enumerate(testloader):

        # ipdb.set_trace()

        if i % 500 == 0:
            print("Test:", i)

        input_images, labels = test_data

        # Remove the redundant batch_size dimension
        input_images = torch.squeeze(input_images)

        if cuda_available:
            input_images = Variable(input_images.cuda())
        else:
            input_images = Variable(input_images)

        output = foxnet(input_images)
        output = test_softmax(output)

        combined_output = torch.sum(output, dim=0)

        _, top_5_indices = torch.topk(combined_output, 5)

        predictions.append(list(top_5_indices.cpu().data.numpy()))

    # Ensuring the test predictions are reasonable by keeping track of counts and doing entropy calc
    x = np.array(predictions)
    x = x.flatten()

    for num in x:
        counts[num] += 1

    print(counts)

    values = list(counts.values())
    ent = entropy(values)
    print("Entropy:", ent)
    # End of counting and entropy

    # Write the output to submission file format
    image_paths = []
    with open("ground_truth/test.txt", "r") as f:
        for line in f:
            image_path, _ = line.rstrip().split(" ")
            image_paths.append(image_path)

    with open(folder + "submission_file_epoch_{e}.txt".format(e=epoch), "w") as f:
        for image_path, top_5_prediction in zip(image_paths, predictions):

            f.write(image_path + " " + " ".join(map(str, top_5_prediction)))
            f.write("\n")

    return ent


if __name__ == '__main__':

    fox = WideResNet(depth=34, num_classes=100, widen_factor=3, dropRate=0.4)

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print("Using CUDA")
        fox.cuda()

    epochs = 250

    train_fox(fox, epochs, use_cuda)
    # evaluate_foxnet(fox, use_cuda)
