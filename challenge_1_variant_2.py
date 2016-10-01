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
    file_to_save = open("pickled/challenge_1_variant/" + thing_name, "wb")
    pickle.dump(thing, file_to_save)
    file_to_save.close()

if __name__ == '__main__':
    #controls whether histograms are normalized or not
    DENSITY = True
    FEATS_TO_CLUSTER = 20000

    print('Loading Data')
    data_holder = hw2_dao.data_holder('not_dev',1)
    np.random.seed(12345)

    # use SIFT to detect keypoints for sparse representation
    sift = cv2.SIFT()
    feat_holder = []
    kmeans_feats = []
    dense_extractor = cv2.FeatureDetector_create("Dense")
    dense_extractor.setInt('initXyStep', 8)


    for img in data_holder.training_data:
        img_feats = sift.detectAndCompute(img, None)[1]

        # fallback for the one picture which detects ZERO sift features
        if img_feats == None:
            dense_keypoints = dense_extractor.detect(img)
            img_feats= sift.compute(img, dense_keypoints)

        kmeans_feats.extend(img_feats)
        feat_holder.append(img_feats)

    np.random.shuffle(kmeans_feats)

    print('Computing clusters')
    # bag of features model, 200 clusters as per pyramid match
    kmeans = KMeans(n_clusters=200).fit(kmeans_feats[:FEATS_TO_CLUSTER])
    pickle_something(kmeans, 'variant_kmeans_classifier')


    per_image_clusters = [kmeans.predict(sift_feats) for sift_feats in feat_holder]
    # histogram calculation per image
    image_hists = [np.histogram(clustered_feats, bins=200, range=(-0.5, 199.5), density=DENSITY)[0] for clustered_feats in per_image_clusters]

    # SVM
    print('Fitting Classifier')
    clf = svm.SVC()
    clf.fit(image_hists, data_holder.training_labels)
    pickle_something(clf, 'variant_hist_svm')

    test_feat_holder = []
    # Testing classifier
    for img in data_holder.test_data:
        img_feats = sift.detectAndCompute(img, None)[1]
        if img_feats == None:
            dense_keypoints = dense_extractor.detect(img)
            img_feats= sift.compute(img, dense_keypoints)
        test_feat_holder.append(img_feats)

    per_test_image_clusters = [kmeans.predict(sift_feats) for sift_feats in test_feat_holder]
    test_hists = [np.histogram(clustered_feats, bins=200, range=(-0.5, 199.5), density=DENSITY)[0] for clustered_feats in per_test_image_clusters]

    print('Score')
    score = clf.score(test_hists, data_holder.test_labels)
    print score

    preds = clf.predict(test_hists)
    confmat = confusion_matrix(data_holder.test_labels, preds)

    confusionmatrix.plot_confusion_matrix(confmat, data_holder.class_labels, 'challenge_1_variant')







