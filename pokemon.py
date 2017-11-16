import os 
import glob
import numpy as np
import six
import cPickle
import tensorflow as tf
from scipy.ndimage import imread
from scipy.misc import imresize
import scipy.io as sio

def normalize_img(X):
    Xnorm = np.copy(X).astype(np.float64)
    Xnorm = Xnorm / 127.5 - 1.
    return Xnorm

class PokemonDataset():
    """
    Pokemon dataset, 26436 images resized to 80x80
    Download dataset: https://s3-us-west-2.amazonaws.com/twairball.datasets.pokemon/pokemon.tgz
    """
    def __init__(self, path):
        self.path = path
        
        # split directory of images into 2 lists
        img_files = glob.glob(path + "/*")
        self.train_files = np.array(img_files[:26000])
        self.test_files = np.array(img_files[26000:])

        # properties
        self.x_dim = [80, 80, 3]
        self.num_classes = 1
        self.dataset_size = len(self.train_files)

    def get_batch(self, batch_size, array):
        imgs = np.random.choice(array, size=batch_size)
        new_image_batch = []
        for img in imgs:
            im = imread(img)
            X = normalize_img(im)
            new_image_batch.append(X)

        new_image_batch = np.array(new_image_batch)
        labels_batch = np.ones((batch_size,1,1)) # dummy labels
        return new_image_batch, labels_batch

    def next_batch(self, batch_size, class_id=None):
        return self.get_batch(batch_size, self.train_files)
    
    def test_batch(self, batch_size):
        return self.get_batch(batch_size, self.test_files)

    def get_test_set(self):
        return self.test_batch(len(self.test_files))
