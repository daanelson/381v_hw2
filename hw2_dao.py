import cPickle as pickle
import os.path

import scipy.io as sio
import cv2
import numpy as np

from keras_models.imagenet_utils import preprocess_input

class data_holder:

    # true for color, false for greyscale. Color is needed for implementation of convnet
    def load_training(self, color_or_greyscale):
        train_names = self.filenames['trainImNames']
        train_data = []
        train_labels = []

        if self.environment == 'dev':
            for i in range(2):
                loaded_images = [cv2.imread('dataset/' + train_names[i,j][0].split('397')[1], color_or_greyscale) for j in range(train_names.shape[1])]
                resized_images = [cv2.resize(img.astype('float'), (224, 224), interpolation=cv2.INTER_CUBIC) for img in loaded_images]
                train_data.extend(resized_images)
                train_labels.extend([i]*train_names.shape[1])
        else:
            for i in range(train_names.shape[0]):
                loaded_images = [cv2.imread(train_names[i,j][0],0) for j in range(train_names.shape[1])]
                train_data.extend(loaded_images)
                train_labels.extend([i]*train_names.shape[1])
            # process them accordingly - mean subtraction and greyscale

        return train_data, train_labels

    def load_test(self, color_or_greyscale):
        test_names = self.filenames['test1ImNames']
        test_data = []
        test_labels = []

        if self.environment == 'dev':
            for i in range(2):
                loaded_images = [cv2.imread('dataset/' + test_names[i,j][0].split('397')[1], color_or_greyscale) for j in range(test_names.shape[1])]
                resized_images = [cv2.resize(img.astype('float'), (224, 224), interpolation=cv2.INTER_CUBIC) for img in loaded_images]
                test_data.extend(resized_images)
                test_labels.extend([i]*test_names.shape[1])
        else:
            for i in range(test_names.shape[0]):
                loaded_images = [cv2.imread(test_names[i,j][0],0) for j in range(test_names.shape[1])]
                test_data.extend(loaded_images)
                test_labels.extend([i]*test_names.shape[1])

        return test_data, test_labels

    def prep_for_vgg(self):
        self.training_data = [preprocess_input(np.expand_dims(self.training_data[i], axis=0)) for i in range(len(self.training_data))]
        self.test_data = [preprocess_input(np.expand_dims(self.test_data[i], axis=0)) for i in range(len(self.test_data))]

    def is_pickled(self, filename):
        return os.path.isfile(self.PICKLE_FOLDER + '/' + filename)

    def load_pickled_data(self, filename):
        read_file = open(self.PICKLE_FOLDER + '/' + filename, 'rb')

        pickled_data = pickle.load(read_file)
        read_file.close()
        vgg_train_feats = pickled_data[0]
        vgg_test_feats = pickled_data[1]
        self.training_labels = pickled_data[2]
        self.test_data = pickled_data[3]

        return vgg_train_feats, vgg_test_feats,

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

        if environment == 'dev':
            prefix = ''
        else:
            prefix = '/projects/cs381V.grauman/'
        self.filenames = sio.loadmat(prefix + 'filenames.mat')

        self.training_data, self.training_labels = self.load_training(challenge_no == 2)
        self.test_data, self.test_labels = self.load_test(challenge_no == 2)
        self.index_to_class = self.filenames['classnames'][0]

        self.PICKLE_FOLDER = 'pickled'
        self.TRAINING_FILE = '/vgg_train_data'



