# -*- coding:utf-8 -*-
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import numpy as np

class MyData(data.Dataset):

    def __init__(self, root, transforms=None, train=True):
        '''
        Get images, divide into train/val set
        '''

        self.train = train
        self.images_root = root

        self._read_txt_file()

        if transforms is None:
            normalize = T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2023, 0.1994, 0.2010])
            #self.transforms = T.Compose([T.ToTensor(), normalize])
            if not train:
                self.transforms = T.Compose([
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Pad(4),
                    T.RandomCrop(32),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])
    def _read_txt_file(self):
        self.images_path = []
        self.images_labels = []

        if self.train:
            txt_file = os.path.join(self.images_root, "train.txt")
        else:
            txt_file = os.path.join(self.images_root, "test.txt")

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                self.images_path.append(item[0])
                self.images_labels.append(item[1])


    def __getitem__(self, index):
        '''
        return the data of one image
        '''
        img_path = os.path.join(self.images_root, self.images_path[index])
        label = self.images_labels[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, int(label)

    def __len__(self):
        return len(self.images_path)
class MyData_one(data.Dataset):

    def __init__(self, root, transforms=None, train=True):
        '''
        Get images, divide into train/val set
        '''

        self.train = train
        self.images_root = root

        self._read_txt_file()

        if transforms is None:
            #normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            normalize = T.Normalize(mean=[0.5], std=[0.5])
            #self.transforms = T.Compose([T.ToTensor(), normalize])
            if not train:
                self.transforms = T.Compose([
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Pad(4),
                    T.RandomCrop(28),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])
    def _read_txt_file(self):
        self.images_path = []
        self.images_labels = []

        if self.train:
            txt_file = os.path.join(self.images_root, "train.txt")
        else:
            txt_file = os.path.join(self.images_root, "test.txt")

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(',')
                self.images_path.append(item[0])
                self.images_labels.append(item[1])

    def __getitem__(self, index):
        '''
        return the data of one image
        '''
        img_path = os.path.join(self.images_root, self.images_path[index])
        label = self.images_labels[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, int(label)

    def __len__(self):
        return len(self.images_path)

class MyData_imagenet(data.Dataset):

    def __init__(self, root, transforms=None, train=True):
        '''
        Get images, divide into train/val set
        '''

        self.train = train
        self.images_root = root

        self._read_txt_file()

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            #self.transforms = T.Compose([T.ToTensor(), normalize])
            if not train:
                self.transforms = T.Compose([
                    T.Resize([224,224]),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize([256,256]),
                    T.CenterCrop([224,224]),
                    T.ToTensor(),
                    normalize
                ])
    def _read_txt_file(self):
        self.images_path = []
        self.images_labels = []

        if self.train:
            txt_file = os.path.join('/data/rdd/ImageNet2012/', "train.txt")
        else:
            txt_file = os.path.join('/data/rdd/ILSVRC2012_small/', "val.txt")

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                self.images_path.append(item[0])
                self.images_labels.append(item[1])


    def __getitem__(self, index):
        '''
        return the data of one image
        '''
        
        img_path = os.path.join(self.images_root, self.images_path[index])
                
        label = self.images_labels[index]
        data = Image.open(img_path).convert('RGB')
        if len(np.array(data).shape) != 3:
            data = data.convert('RGB')
        data = self.transforms(data)
        return data, int(label)

    def __len__(self):
        return len(self.images_path)