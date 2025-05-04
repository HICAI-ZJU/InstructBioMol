import copy
import os
import torch
import numpy as np
import json
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from .preprocess_smiles import process_smiles
from .load_sdf import load_sdf_data
from .collate_func import collate_func

from tqdm import tqdm
import pandas as pd
import lmdb
import pickle


class LMDBDataset(Dataset):

    def __init__(self,
                 data_name: str,
                 data_path: str,
                 id_key: str,
                 input_modality: str,
                 target_modality: str,
                 input_seq_key: str,  # for input
                 target_seq_key: str,  # for output
                 input_enc_seq_key: str,  # for input qformer encoder
                 input_enc_fp_key: str,  # for fingerprint
                 instruction: str,
                 conf_path: str,
                 conf_key: str,
                 data_format: str):
        super(LMDBDataset, self).__init__()

        self.data_path = data_path
        self.data_name = data_name

        self.env = lmdb.open(self.data_path, readonly=True, lock=False)
        self.keys = self._get_keys()

        self.instruction = instruction
        self.input_modality = input_modality
        self.target_modality = target_modality

        self.id_key = id_key
        self.input_seq_key = input_seq_key
        self.target_seq_key = target_seq_key
        self.input_enc_seq_key = input_enc_seq_key
        self.input_enc_fp_key = input_enc_fp_key
        self.conf_path = conf_path
        self.conf_key = conf_key

    def _get_keys(self):
        keys = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                keys.append(key)
        return keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key_bytes = self.keys[index]
        with self.env.begin() as txn:
            value_bytes = txn.get(key_bytes)
            data = pickle.loads(value_bytes)

        data_id = data[self.id_key]
        if self.conf_path == '':
            # 3Di token
            conf = data[self.conf_key]
        else:
            conf = os.path.join(self.conf_path, data[self.conf_key])

        return {'input_seqs': data[self.input_seq_key],
                'target_seqs': data[self.target_seq_key],
                'input_enc_seqs': data[self.input_enc_seq_key],
                'input_enc_fps': data[self.input_enc_fp_key],
                'input_modality': self.input_modality,
                'target_modality': self.target_modality,
                'instructions': self.instruction,
                'id': data_id,
                'conf': conf,
                'data_name': self.data_name}

    def collate(self, instances):
        return collate_func(instances)