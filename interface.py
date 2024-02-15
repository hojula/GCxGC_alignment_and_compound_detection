from abc import ABC, abstractmethod
from typing import Dict, Tuple
import torch


class ClassificationModel(ABC):

    @abstractmethod
    def set_model(self, args):
        """
        Set up the model architecture and configurations.
        """
        self._data_root_path = args.data_root_path
        self._id_dataset_name = args.id_dataset_name
        self._ood_dataset_name = args.ood_dataset_name
        self._num_classes = args.num_classes
        self._k_fold_number = args.k_fold_number
        self._model_parameters = args.model_parameters
        self._device = args.device
        self._verbose = args.verbose
        self._seed = args.seed
        self._folds = ['id_train', 'id_test', 'id', 'ood']
        self._retention_times = args.retention_times
        self._trained_models_save_dir_path = args.trained_models_save_dir_path

    @abstractmethod
    def preprocess_data(self, data):
        return data

    @abstractmethod
    def load_id_data(self):
        pass

    @abstractmethod
    def load_ood_data(self):
        pass

    @abstractmethod
    def train(self, model, train_data, test_data):
        pass

    @abstractmethod
    def train_k_folds(self):
        pass

    @abstractmethod
    def extract_features(self, data):
        pass

    @abstractmethod
    def get_model_outputs(self):
        pass

    @abstractmethod
    def get_id_classification_scores(self):
        pass

    @abstractmethod
    def predict_from_logits(self, logits):
        pass

class OODDetector(ABC):

    @abstractmethod
    def setup(self, args, id_train_model_outputs: Dict[str, torch.Tensor]):
        pass

    @abstractmethod
    def infer(self, model_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass


