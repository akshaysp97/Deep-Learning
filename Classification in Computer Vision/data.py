
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:19:25 2020

@author: vineethellore
"""

from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random

import os
import numpy as np
import os.path as osp
import json
from tqdm import tqdm
from PIL import Image

import tensorflow
from utils import Config
import time


class polyvore_dataset:
    def __init__(self):
        if Config['aws']:
            self.root_dir = Config['root_path_aws']
        else:
            self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.transforms = self.get_data_transforms()
        self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test, self.classes = self.create_dataset()

    def get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomVesrticalFlip(),
                # transforms.RandomRotation(45),
                # transforms.ColorJitter(
                #     brightness=0.4,
                #     contrast=0.4,
                #     saturation=0.4,
                # ),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        return data_transforms

    def create_dataset(self):
        X_train = [];
        y_train = []
        with open(os.path.join(self.root_dir, Config['compatible_train']), 'r') as meta:
            for line in tqdm(meta):
                y_train.append(line[0])
                X_train.append(line[2:].replace('\n', ''))
        X_valid = [];
        y_valid = []
        with open(os.path.join(self.root_dir, Config['compatible_valid']), 'r') as meta:
            for line in tqdm(meta):
                y_valid.append(line[0])
                X_valid.append(line[2:].replace('\n', ''))
        X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=0.5)

        c = list(zip(X_train, y_train))
        random.shuffle(c)
        X_train, y_train = zip(*c)
        return X_train, X_valid, X_test, y_train, y_valid, y_test, 1


class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, dataset, dataset_size, params):
        self.batch_size = params['batch_size']
        self.shuffle = params['shuffle']
        self.n_classes = params['n_classes']
        self.X, self.y, self.transform = dataset
        self
        if Config['aws']:
            self.root_dir = Config['root_path_aws']
        else:
            self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        X0, y = self.__data_generation(indexes, 0)
        X1, y = self.__data_generation(indexes, 1)
        X0, X1, y = np.stack(X0), np.stack(X1), np.stack(y)
        # return [np.moveaxis(X0, 1, 3), np.moveaxis(X1, 1, 3)], tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)
        return [np.moveaxis(X0, 1, 3), np.moveaxis(X1, 1, 3)], np.array(y)

    def __data_generation(self, indexes, k):
        X = [];
        y = []
        for idx in indexes:
            file_path = osp.join(self.image_dir, self.X[idx].split(" ")[k] + ".jpg")
            X.append(self.transform(Image.open(file_path)))
            y.append(float(self.y[idx]))
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


if __name__ == '__main__':
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_valid, X_test, y_train, y_valid, y_test, n_classes = \
        dataset.X_train, dataset.X_valid, dataset.X_test, dataset.y_train, \
        dataset.y_valid, dataset.y_test, dataset.classes

    if Config['debug']:
        k = Config['debug_size']
        train_set = (X_train[:k], y_train[:k], transforms['train'])
        valid_set = (X_valid[:k], y_valid[:k], transforms['test'])
        test_set = (X_test[:k], y_test[:k], transforms['test'])
        dataset_size = {'train': k, 'test': k}
    else:
        train_set = (X_train, y_train, transforms['train'])
        valid_set = (X_valid, y_valid, transforms['test'])
        test_set = (X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    params = {'batch_size': Config['batch_size'],
              'n_classes': n_classes,
              'shuffle': True
              }

    train_generator = DataGenerator(train_set, dataset_size, params)
    valid_generator = DataGenerator(valid_set, dataset_size, params)
    test_generator = DataGenerator(test_set, dataset_size, params)
    # train_generator.indexes
