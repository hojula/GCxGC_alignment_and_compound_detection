from interface import OODDetector
import torch
from detectors_utils import Mahalanobis
import numpy as np


class MahalanobisOODDetector(OODDetector):

    def setup(self, args, id_train_model_outputs):
        print('Setting up Mahalanobis OOD detector...')
        train_feas = id_train_model_outputs['feas']
        train_labels = id_train_model_outputs['labels_true']
        self.mahalanobis = Mahalanobis(normalize_on=False, standardize_on=False)
        self.mahalanobis.fit(train_feas, train_labels)

    def infer(self, model_outputs):
        feas = model_outputs['feas']
        if isinstance(feas, np.ndarray):
            feas = torch.from_numpy(feas)
        device = feas.device
        scores = torch.from_numpy(self.mahalanobis.score(feas, return_distance=False)).to(device)
        return scores
