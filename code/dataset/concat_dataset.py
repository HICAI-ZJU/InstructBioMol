from torch.utils.data import ConcatDataset, Dataset
from torch_geometric.data import Data, Batch
import torch

from .collate_func import collate_func
from .data_catalog import DatasetCatalog
from .json_dataset import JSONDataset
from .lmdb_dataset import LMDBDataset

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from exp_utils import filter_logging

import logging

logger = logging.getLogger(__name__)




class MyConcatDataset(Dataset):
    def __init__(self, dataset_name_list):
        super(MyConcatDataset, self).__init__()

        _datasets = []

        catalog = DatasetCatalog()
        for dataset_idx, dataset_name in enumerate(dataset_name_list):
            dataset_dict = getattr(catalog, dataset_name)
            if dataset_dict['data_format'] == 'JSON':
                dataset = JSONDataset(**dataset_dict, data_name=dataset_name)
            elif dataset_dict['data_format'] == 'LMDB':
                dataset = LMDBDataset(**dataset_dict, data_name=dataset_name)
            else:
                raise NotImplementedError
            _datasets.append(dataset)
            logger.info(f"DATASET-{dataset_name}: {len(dataset):,}")
        self.datasets = ConcatDataset(_datasets)

    def __len__(self):
        return self.datasets.__len__()

    def __getitem__(self, item):
        return self.datasets.__getitem__(item)

    def collate(self, instances):
        return collate_func(instances)

