import cPickle as pickle
import os.path
import pdb

import scipy.io as sio
import cv2
import numpy as np

from keras_models.imagenet_utils import preprocess_input


# wrote this to load in and hold the dataset so I can share that across challenges.
class data_holder:

    # true for color, false for greyscale. Color is needed for implementation of convnet
    def load_training(self, is_color):
        train_names = self.filenames['trainImNames']
        train_data = []
        train_labels = []

        #dev flag is for working out bugs on a subset of data.
        if self.environment == 'dev':
            for i in range(2):
                loaded_images = [cv2.imread('dataset/' + train_names[i,j][0].split('397')[1], is_color) for j in range(train_names.shape[1])]
                if is_color:
                    resized_images = [cv2.resize(img.astype('float'), (224, 224), interpolation=cv2.INTER_CUBIC) for img in loaded_images]
                else:
                    resized_images = [cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC) for img in loaded_images]
                train_data.extend(resized_images)
                train_labels.extend([i]*train_names.shape[1])
        else:
            for i in range(train_names.shape[0]):
                loaded_images = [cv2.imread('dataset/' + train_names[i,j][0].split('397')[1], is_color) for j in range(train_names.shape[1])]
                print self.index_to_class[i]

                if is_color:
                    resized_images = [cv2.resize(img.astype('float'), (224, 224), interpolation=cv2.INTER_CUBIC) for img in loaded_images if img is not None]
                else:
                    resized_images = [cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC) for img in loaded_images if img is not None]
                train_data.extend(resized_images)
                train_labels.extend([i]*len(resized_images))

        return train_data, train_labels

    def load_test(self, is_color):
        test_names = self.test_2_filenames['test2ImNames']
        test_data = []
        test_labels = []

        if self.environment == 'dev':
            for i in range(2):
                loaded_images = [cv2.imread('dataset/' + test_names[i,j][0].split('397')[1], is_color) for j in range(test_names.shape[1])]
                if is_color:
                    resized_images = [cv2.resize(img.astype('float'), (224, 224), interpolation=cv2.INTER_CUBIC) for img in loaded_images]
                else:
                    resized_images = [cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC) for img in loaded_images]
                test_data.extend(resized_images)
                test_labels.extend([i]*test_names.shape[1])
        else:
            for i in range(test_names.shape[0]):
                loaded_images = [cv2.imread('dataset/' + test_names[i,j][0].split('397')[1], is_color) for j in range(test_names.shape[1])]
                if is_color:
                    resized_images = [cv2.resize(img.astype('float'), (224, 224), interpolation=cv2.INTER_CUBIC) for img in loaded_images if img is not None]
                else:
                    resized_images = [cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC) for img in loaded_images if img is not None]
                test_data.extend(resized_images)
                test_labels.extend([i]*len(resized_images))

        return test_data, test_labels

    #preparation for running data with CNN
    def prep_for_vgg(self):
        self.test_data = [preprocess_input(np.expand_dims(self.test_data[i], axis=0)) for i in range(len(self.test_data))]

    #series of methods to pickle datasets; used so I don't have to go through the computationally expensive
    #step of extracting features from a CNN more than once.
    def is_pickled(self, filename):
        return os.path.isfile(self.PICKLE_FOLDER + '/' + filename)

    def load_pickled_data(self, filename):
        read_file = open(self.PICKLE_FOLDER + '/' + filename, 'rb')

        pickled_data = pickle.load(read_file)
        read_file.close()
        vgg_train_feats = pickled_data[0]
        vgg_test_feats = pickled_data[1]
        self.training_labels = pickled_data[2]

        return vgg_train_feats, vgg_test_feats

    def pickle_data(self, vgg_train_feats, vgg_test_feats, filename):

        if not os.path.isdir(self.PICKLE_FOLDER):
            os.mkdir(self.PICKLE_FOLDER)

        write_file = open(self.PICKLE_FOLDER + '/' + filename, 'wb')

        to_write = [vgg_train_feats, vgg_test_feats, self.training_labels, self.test_labels]

        pickle.dump(to_write, write_file)
        write_file.close()

    def __init__(self, environment, challenge_no):
        self.environment = environment
        self.challenge_no = challenge_no

        self.filenames = sio.loadmat('filenames.mat')
        self.test_2_filenames = sio.loadmat('test2ImNames.mat')
        self.index_to_class = self.filenames['classnames'][0]
        self.PICKLE_FOLDER = 'pickled'
        self.TRAINING_FILE = '/vgg_train_data'
        self.class_labels = [label[0] for label in self.index_to_class]

        self.training_data, self.training_labels = self.load_training(challenge_no == 2)
        self.test_data, self.test_labels = self.load_test(challenge_no == 2)






