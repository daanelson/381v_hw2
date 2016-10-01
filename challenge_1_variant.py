import hw2_dao
import cv2
import numpy as np
import cPickle as pickle
import pdb
import confusionmatrix
from sklearn.metrics import confusion_matrix

from sklearn.cluster import KMeans
from sklearn import svm


def pickle_something(thing, thing_name):
    file_to_save = open("pickled/challenge_1/" + thing_name, "wb")
    pickle.dump(thing, file_to_save)
    file_to_save.close()

if __name__ == '__main__':
    #controls whether histograms are normalized or not
    DENSITY = True
    FEATS_TO_CLUSTER = 20000

    print('Loading Data')
    data_holder = hw2_dao.data_holder('not_dev',1)
    np.random.seed(12345)

    # extract dense SIFT features from that sweet, sweet data
    dense_extractor = cv2.FeatureDetector_create("Dense")
    dense_extractor.setInt('initXyStep', 8)
    dense_keypoints = [dense_extractor.detect(img) for img in data_holder.training_data]

    sift = cv2.SIFT()
    zipped_sift_feats = [sift.compute(img_kp[0], img_kp[1]) for img_kp in zip(data_holder.training_data, dense_keypoints)]

    kp, dense_sift_feats = zip(*zipped_sift_feats)
    kmeans_sift_feats = np.asarray(dense_sift_feats)
    array_sift_feats = np.asarray(dense_sift_feats)
    kmeans_sift_feats = kmeans_sift_feats.reshape(array_sift_feats.shape[0]*array_sift_feats.shape[1], 128)
    np.random.shuffle(kmeans_sift_feats)


    print('Computing clusters')
    # bag of features model, 200 clusters as per pyramid match
    kmeans = KMeans(n_clusters=100).fit(kmeans_sift_feats[:FEATS_TO_CLUSTER])
    pickle_something(kmeans, 'kmeans_classifier')

    per_image_clusters = [kmeans.predict(sift_feats) for sift_feats in array_sift_feats]
    # histogram calculation per image
    image_hists = [np.histogram(clustered_feats, bins=200, range=(-0.5, 199.5), density=DENSITY)[0] for clustered_feats in per_image_clusters]

    # SVM
    print('Fitting Classifier')
    clf = svm.SVC()
    clf.fit(image_hists, data_holder.training_labels)
    pickle_something(clf, 'hist_svm')

    # Testing classifier
    dense_test_keypoints = [dense_extractor.detect(img) for img in data_holder.test_data]
    zipped_test_sift_feats = [sift.compute(img_kp[0], img_kp[1]) for img_kp in zip(data_holder.test_data, dense_test_keypoints)]
    _, dense_test_sift_feats = zip(*zipped_test_sift_feats)
    array_test_sift_feats = np.asarray(dense_test_sift_feats)
    per_test_image_clusters = [kmeans.predict(sift_feats) for sift_feats in array_test_sift_feats]
    test_hists = [np.histogram(clustered_feats, bins=200, range=(-0.5, 199.5), density=DENSITY)[0] for clustered_feats in per_test_image_clusters]


    print('Score')
    score = clf.score(test_hists, data_holder.test_labels)
    print score

    preds = clf.predict(test_hists)
    confmat = confusion_matrix(data_holder.test_labels, preds)

    confusionmatrix.plot_confusion_matrix(confmat, data_holder.class_labels, 'challenge_1')







