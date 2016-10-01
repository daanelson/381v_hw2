# Dan Nelson
# CS 381V
# Challenge 2 code
# Reference: https://github.com/fchollet/deep-learning-models

import pdb
from keras_models.vgg16 import VGG16
from keras.models import Model

from sklearn import svm
from sklearn.metrics import confusion_matrix

import hw2_dao
import confusionmatrix

if __name__ == '__main__':
    CLASSIFICATION_TASK = 2
    DATA_FILE = "challenge_2_data"
    NEW_DATA_FILE = "challenge_2_data_2"

    data_holder = hw2_dao.data_holder('n_dev', CLASSIFICATION_TASK)

    print('Loading Pre-Computed Features')
    vgg_training_features, _ = data_holder.load_pickled_data(DATA_FILE)

    data_holder.prep_for_vgg()

    #Extract last fully connected layer of VGG16 trained on imagenet data as per https://arxiv.org/pdf/1409.1556v6.pdf
    vgg_model = VGG16(weights='imagenet', include_top=True)
    feature_extractor = Model(input=vgg_model.input, output=vgg_model.get_layer('fc2').output)

    #Infer features from model
    print('Generating Features')
    vgg_test_features = [feature_extractor.predict(x).squeeze() for x in data_holder.test_data]

    data_holder.pickle_data(vgg_training_features, vgg_test_features, NEW_DATA_FILE)

    #Train SVM; sklearn uses 1 v 1 SVM algorithm
    print('Training SVM')
    clf = svm.SVC()
    clf.fit(vgg_training_features, data_holder.training_labels)

    #Accuracy! More metrics may be forthcoming
    print('Scoring SVM')
    score = clf.score(vgg_test_features, data_holder.test_labels)
    print score

    preds = clf.predict(vgg_test_features)

    confmat = confusion_matrix(data_holder.test_labels, preds)
    confusionmatrix.plot_confusion_matrix(confmat, data_holder.class_labels, 'challenge_2_data_2')







