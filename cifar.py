import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import cv2
from transformers import CLIPModel
import torch



class CifarDataset(Dataset):

    def __init__(self, ROOT_DIR, transform=None, train=0, size=32, interpolation="bicubic", flip_p=0.5, training_sample = 4096, test_sample=512):
        # training dataset

        if train == 0:
            self.image_input = np.load(ROOT_DIR + "normal_data.npy", allow_pickle=True)
            self.label_input = np.load(ROOT_DIR + "normal_label.npy", allow_pickle=True)
            input_len = len(self.label_input)
            self.sample_index = random.sample(range(input_len), training_sample)

        ## validation dataset
        elif train == 1: ##
            self.image_input = np.load(ROOT_DIR + "normal_validate_data.npy", allow_pickle=True)
            self.label_input = np.load(ROOT_DIR + "normal_validate_label.npy", allow_pickle=True)
            input_len = len(self.label_input)
            self.sample_index = random.sample(range(input_len), test_sample)
        ### test dataset
        elif train == 2:
            self.image_input = np.load(ROOT_DIR + "abnormal_data.npy", allow_pickle=True)
            self.label_input = np.load(ROOT_DIR + "abnormal_label.npy", allow_pickle=True)
            input_len = len(self.label_input)
            self.sample_index = random.sample(range(input_len), test_sample)

        else:
            self.image_input = np.load(ROOT_DIR + "normal_test_data.npy", allow_pickle=True)
            self.label_input = np.load(ROOT_DIR + "normal_test_label.npy", allow_pickle=True)
            input_len = len(self.label_input)
            self.sample_index = random.sample(range(input_len), test_sample)

        self.size = size

        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))


        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        normalize]) if not transform else transform




    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        example = dict()


        image = self.image_input[self.sample_index[idx], :].reshape((32,32,3))

        if self.transform:
            image = self.transform(image)





        return image


class Cifar_train(CifarDataset):
    def __init__(self, folder_number = 0, **kwargs):
        folder = "./paperdata_cifar/" + str(folder_number) + "/"
        super().__init__(ROOT_DIR=folder, train=0, **kwargs)


class Cifar_test(CifarDataset):
    def __init__(self, folder_number = 0, **kwargs):
        folder = "paperdata_cifar/" + str(folder_number) + "/"
        super().__init__(ROOT_DIR=folder, train=2, **kwargs)

class Cifar_test_normal(CifarDataset):
    def __init__(self, folder_number = 0, **kwargs):
        folder = "paperdata_cifar/" + str(folder_number) + "/"
        super().__init__(ROOT_DIR=folder, train=3, **kwargs)

class Cifar_validate(CifarDataset):
    def __init__(self, folder_number = 0, **kwargs):
        folder = "paperdata_cifar/" + str(folder_number) + "/"
        super().__init__(ROOT_DIR=folder, train=1, **kwargs)
