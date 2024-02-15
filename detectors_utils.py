from copy import deepcopy
import faiss
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize


class Mahalanobis(object):
    def __init__(self, normalize_on=False, standardize_on=False, num_clusters=5):
        self.normalize_on = normalize_on
        self.standardize_on = standardize_on
        self.num_clusters = num_clusters  # the number of K-means clusters = num of classes

    def fit(self, feats, labels=None):
        dim = feats.shape[1]
        feats = np.array(feats)
        if labels is None:
            supervised = False
        else:
            supervised = True

        self.mean = np.mean(feats, axis=0, keepdims=True)
        self.std = np.std(feats, axis=0, keepdims=True)
        feats = self.preprocess_features(feats)
        # clustering
        if supervised:
            self.num_clusters = len(np.unique(labels))
        else:
            # link: https://github.com/inspire-group/SSD/blob/main/eval_ssdk.py
            if self.num_clusters > 1:
                kmeans = faiss.Kmeans(d=feats.shape[1], k=self.num_clusters, niter=100, verbose=False, gpu=False)
                kmeans.train(np.array(feats))
                labels = np.array(kmeans.assign(feats)[1])
            else:
                labels = np.zeros(len(feats))

        self.centers = np.zeros(shape=(self.num_clusters, dim))
        cov = np.zeros(shape=(self.num_clusters, dim, dim))
        for k in tqdm(range(self.num_clusters), desc='k-means clustering', unit="cluster (class)"):
            X_k = np.array(feats[labels == k])
            self.centers[k] = np.mean(X_k, axis=0)
            cov[k] = np.cov(X_k.T,bias=True)

        if supervised:
            shared_cov = cov.mean(axis=0)
            self.shared_icov = np.linalg.pinv(shared_cov)
        else:
            self.icov = np.zeros(shape=(self.num_clusters, dim, dim))
            self.shared_icov = None
            for k in tqdm(range(self.num_clusters)):
                self.icov[k] = np.linalg.pinv(cov[k])

    def score(self, feats, return_distance=False):
        feats = np.array(feats)
        feats = self.preprocess_features(feats)

        if self.shared_icov is not None:
            M = self.shared_icov
            U = self.centers
            md = (np.matmul(feats, M) * feats).sum(axis=1)[:, None] \
                 + ((np.matmul(U, M) * U).sum(axis=1).T)[None, :] \
                 - 2 * np.matmul(np.matmul(feats, M), U.T)
        else:
            md = []
            for k in tqdm(range(self.num_clusters)):
                delta_k = feats - self.centers[k][None, :]
                md.append(
                    (np.matmul(delta_k, self.icov[k]) * delta_k).sum(axis=1))
            md = np.array(md).T

        out = md.min(axis=1)
        if return_distance:
            return out

        # return np.exp(-(out/2048) / 2)
        return np.exp(-(out / feats.shape[1]) / 2)
        # return np.exp(-out / 2)

    def preprocess_features(self, feats):
        if self.normalize_on:
            feats = normalize(feats, axis=1)  # normalize

        if self.standardize_on:
            feats = (feats - self.mean) / (self.std + 1e-8)  # standardize

        return feats

    def compute_mahalanobis_score(self, x, center, icov):
        delta = x - center
        ms = (np.matmul(delta, icov) * delta).sum(axis=1)
        return np.maximum(ms, 0)
