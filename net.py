import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim
import torchvision
from torchvision.transforms import *
import numpy as np
from collections import defaultdict
from scipy.stats import entropy

import os
from PIL import Image
import time

from wideresnet import WideResNet

class Places(Dataset):

    def __init__(self, data_root_path, data_paths_and_labels, transform=None):
        """

        :param data_root_path: the root path of the images
        :param data_paths_and_labels: a file that contains the path of each image from root and its label
        :param transform: data augmentation, must include ToTensor to convert to PyTorch image format
        """
        self.data_root_path = data_root_path
        self.image_path_list = []
        self.labels_list = []
        self.transform = transform

        with open(data_paths_and_labels, 'r') as f:
            for line in f:
                image_path, label = line.rstrip().split(' ')
                self.image_path_list.append(os.path.join(self.data_root_path, image_path))
                self.labels_list.append(int(label))

                # if label is None:
                #     self.labels_list.append(-1)  # Test set doesn't have labels
                # else:
                #     self.labels_list.append(int(label))

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):

        image_path = self.image_path_list[idx]
        # image = io.imread(image_path)  # Using skimage
        image = Image.open(image_path)  # Using PIL
        label = self.labels_list[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

        # sample = {'image': image, 'label': label}
        #
        # if self.transform:
        #     image = self.transform(sample)
        #
        #
        # return sample['image'], sample['label']


class FoxNet(nn.Module):

    def __init__(self, num_classes=100):

        super(FoxNet, self).__init__()

        self.classifier_input_size = None

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


def train_fox(foxnet, epochs, cuda_available):

    # 16 limit for 28 layer - 10 wide network seems like
    training_batch_size = 128
    validation_batch_size = 10

    channel_mean = torch.Tensor([.4543, .4362, .4047])
    # channel_std = torch.Tensor([.2274, .2244, .2336])
    channel_std = torch.ones(3)

    train_data_transform = Compose([
        RandomCrop(112),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(channel_mean, channel_std),
        ColorAugmentation(112)
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
