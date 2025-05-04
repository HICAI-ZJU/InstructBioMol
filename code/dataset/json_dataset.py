import copy
import os
import torch
import numpy as np
import json
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import pandas as pd
from .preprocess_smiles import process_smiles
from .load_sdf import load_sdf_data
from .collate_func import collate_func


class JSONDataset(Dataset):
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
        super(JSONDataset, self).__init__()

        self.data_path = data_path
        self.data_name = data_name

        self.data_id_list = []
        self.conformation_list = []
        self.input_seq_list = []  # for input
        self.target_seq_list = []  # for output
        self.input_enc_seq_list = []  # for input qformer encoder
        self.input_enc_fp_list = []

        self.instruction = instruction
        self.input_modality = input_modality
        self.target_modality = target_modality

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for datapoint in data:
            data_id = datapoint[id_key]
            self.input_seq_list.append(datapoint[input_seq_key])
            self.target_seq_list.append(datapoint[target_seq_key])
            self.input_enc_seq_list.append(datapoint[input_enc_seq_key])
            self.input_enc_fp_list.append(datapoint[input_enc_fp_key])
            self.data_id_list.append(data_id)
            if self.instruction == '':
                # for some downstream tasks
                self.instruction = datapoint['instruction']
            if conf_path == '':
                # 3Di token
                self.conformation_list.append(datapoint[conf_key])
            else:
                # sdf file
                self.conformation_list.append(os.path.join(conf_path, datapoint[conf_key]))

    def __len__(self):
        return len(self.input_seq_list)

    def __getitem__(self, i):

        return {'input_seqs': self.input_seq_list[i],
                'target_seqs': self.target_seq_list[i],
                'input_enc_seqs': self.input_enc_seq_list[i],
                'input_enc_fps': self.input_enc_fp_list[i],
                'input_modality': self.input_modality,
                'target_modality': self.target_modality,
                'instructions': self.instruction,
                'id': self.data_id_list[i],
                'conf': self.conformation_list[i],
                'data_name': self.data_name}

    def collate(self, instances):
        return collate_func(instances)