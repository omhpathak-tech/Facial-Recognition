import requests 
import tarfile 
import os 
import numpy as np 
import cv2 
import tensorflow as tf
from tensorflow import keras
import imgaug as ia
from imgaug import augmenters as iaa
from typing import List 
import dlib
from models.face_recognition.align import AlignDlib

class LfwDataGenerator(keras.utils.Sequence):
    def __init__(self, pairs_file_name, batch_size, anchor_shape = (96, 96), n_channels = 3, img_aug = False, shuffle = True):
        self.pairs_file_name = pairs_file_name
        self.batch_size = batch_size
        
        self.anchor_shape = anchor_shape 
        self.n_channels = n_channels 
        self.img_aug = img_aug 
        self.shuffle = shuffle 
        self.positive_pairs = []
        self.negative_pairs = []
        self.alignment = AlignDlib('models/landmarks.dat')
        # process positive and negative pairs 
        with open(pairs_file_name, 'rt') as f:
            for line in f:
                tokens = line.split()
                if len(tokens) == 3:
                    self.positive_pairs.append(
                    [(tokens[0], tokens[1]), (tokens[0], tokens[2])]
                    )
                elif len(tokens) == 4:
                    self.negative_pairs.append(
                    [(tokens[0], tokens[1]), (tokens[2]), (tokens[3])]
                    )
        self.on_epoch_end()
    def on_epoch_end(self):
    # reset indexes  after each epoch 
        self.pos_indexes = np.arange(len(self.positive_pairs))
        self.neg_indexes = np.arange(len(self.negative_pairs))
        if self.shuffle == True:
            np.random.shuffle(self.pos_indexes)
            np.random.shuffle(self.neg_indexes)
        
    def __len__(self):
        # return the number of batches per epoch
        return int(np.floor(len(self.pos_indexes) / self.batch_size))

    def get_image_path(self, name, id):
    # this helper function returns the path to the image of name_id.jpg
        path_name = os.path.join("data/lfw", name)
        return os.path.join(path_name, name) + "_" + f"{id}".zfill(4) + ".jpg"
    
    def augment_images(self, img_batch):
        seq = iaa.Sequential([
            iaa.Crop(px = (0, 16)), 
            iaa.Fliplr(0.5), 
            iaa.GaussianBlur(sigma = (0, 1.0)), 
            # Strengthen or weaken contrast in each image
            iaa.LinearContrast((0.75, 1.5)), # blur images with a sigma of 0 to 3.0
        ])
        return seq(images = img_batch)
    def __getitem__(self, index):
        # slice indexes of batch_size 
        pos_indexes = self.pos_indexes[index * self.batch_size:(index + 1) * self.batch_size]
        neg_indexes = self.neg_indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # init output (numpy) arrays 
        anchor_img_arr = np.empty((self.batch_size, *self.anchor_shape, self.n_channels), dtype = np.float32)
        pos_img_arr = np.empty((self.batch_size, *self.anchor_shape, self.n_channels), dtype = np.float32)
        neg_img_arr = np.empty((self.batch_size, *self.anchor_shape, self.n_channels), dtype = np.float32)
        
        # get the images from name and id from the indexes 
        pos_pairs_batch = [self.positive_pairs[j] for j in pos_indexes]
        neg_pairs_batch = [self.negative_pairs[k] for k in neg_indexes]
        
        for i, pos_pair in enumerate(pos_pairs_batch):
            # process anchor image 
            pos_name, pos_id_1 = pos_pair[0]
            img = cv2.imread(self.get_image_path(pos_name, pos_id_1))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.align_image(img)
            img = img.astype('float32')
            img = img / 255.0
            anchor_img_arr[i,] = img 
            
            # process positive image 
            pos_name, pos_id_2 = pos_pair[1]
            img = cv2.imread(self.get_image_path(pos_name, pos_id_2))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.align_image(img)
            img = img.astype('float32')
            img = img / 255.0
            pos_img_arr[i,] = img 
            
            # process negative image 
            neg_pair = neg_pairs_batch[i]
            neg_name, neg_id = neg_pair[0]
            if pos_name == neg_name:
                neg_name, neg_id = neg_pair[1]
            img = cv2.imread(self.get_image_path(neg_name, neg_id))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.align_image(img)
            img = img.astype('float32')
            img = img / 255.0
            neg_img_arr[i,] = img 
        
        if self.img_aug:
            # apply image augmentation
            anchor_img_arr = self.augment_images(anchor_img_arr)
            pos_img_arr = self.augment_images(pos_img_arr)
            neg_img_arr = self.augment_images(neg_img_arr)
            
        return [anchor_img_arr, pos_img_arr, neg_img_arr], None
    
    def align_image(self, img):
        alignment = AlignDlib('models/landmarks.dat')
        bb = alignment.getLargestFaceBoundingBox(img)
        if bb is None:
            return cv2.resize(img, self.anchor_shape)
        else:
            return alignment.align(self.anchor_shape[0], img, bb, landmarkIndices = AlignDlib.OUTER_EYES_AND_NOSE)