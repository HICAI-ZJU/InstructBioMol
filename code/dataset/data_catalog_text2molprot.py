from torch.utils.data import ConcatDataset, Dataset
from torch_geometric.data import Data, Batch
import torch

import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from exp_utils import filter_logging

import logging

logger = logging.getLogger(__name__)



def collate_func4text2molprot(instances):


    input_seqs, target_protein, target_molecule, instructions, ids, data_name = tuple(
        [instance[key] for instance in instances] for key in ("input_seqs",
                                                              "target_protein",
                                                              "target_molecule",
                                                              "instructions",
                                                              "id",
                                                              "data_name"))

    data_name = data_name[0]

    return None, {'input_seqs': input_seqs,
            'target_protein': target_protein,
            'target_molecule': target_molecule,
            'instructions': instructions,
            'id': ids,
            'data_name': data_name}



class DatasetText2MolProtCatalog:
    def __init__(self):
        self.text_2_protein_molecule_train = {
            'data_path': 'data/text2protmol/bindingdb-train+desc.json',
            'id_key': 'PID',
            'text_key': 'Protein_desc',
            'protein_key': 'protein',
            'molecule_key': 'selfies',
            'instruction': 'Please generate the corresponding protein sequence based on the following description of the protein and generate a molecule that can interact with this protein.'
        }

        self.text_2_protein_molecule_test = {
            'data_path': 'data/text2protmol/bindingdb-test+desc-filter-100.json',
            'id_key': 'PID',
            'text_key': 'Protein_desc',
            'protein_key': 'protein',
            'molecule_key': 'selfies',
            'instruction': 'Please generate the corresponding protein sequence based on the following description of the protein and generate a molecule that can interact with this protein.'
        }


class JSONDataset4Text2MolProt(Dataset):
    def __init__(self,
                 data_name: str,
                 data_path: str,
                 id_key: str,
                 text_key: str,
                 protein_key: str,
                 molecule_key: str,
                 instruction: str):
        super(JSONDataset4Text2MolProt, self).__init__()

        self.data_path = data_path
        self.data_name = data_name

        self.data_id_list = []
        self.text_list = []
        self.protein_list = []
        self.molecule_list = []

        self.instruction = instruction


        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for datapoint in data:
            data_id = datapoint[id_key]

            self.data_id_list.append(data_id)
            self.text_list.append(datapoint[text_key])
            self.protein_list.append(datapoint[protein_key])
            self.molecule_list.append(datapoint[molecule_key])


    def __len__(self):
        return len(self.data_id_list)

    def __getitem__(self, i):

        return {
                'input_seqs': self.text_list[i],
                'target_protein': self.protein_list[i],
                'target_molecule': self.molecule_list[i],
                'instructions': self.instruction,
                'id': self.data_id_list[i],
                'data_name': self.data_name
                }
    def collate(self, instances):
        return collate_func4text2molprot(instances)

class MyConcatText2MolProtDataset(Dataset):
    def __init__(self, dataset_name_list):
        super(MyConcatText2MolProtDataset, self).__init__()

        _datasets = []

        catalog = DatasetText2MolProtCatalog()
        for dataset_idx, dataset_name in enumerate(dataset_name_list):
            dataset_dict = getattr(catalog, dataset_name)
            dataset = JSONDataset4Text2MolProt(**dataset_dict, data_name=dataset_name)
            _datasets.append(dataset)
            logger.info(f"DATASET-{dataset_name}: {len(dataset):,}")
        self.datasets = ConcatDataset(_datasets)

    def __len__(self):
        return self.datasets.__len__()

    def __getitem__(self, item):
        return self.datasets.__getitem__(item)

    def collate(self, instances):
        return collate_func4text2molprot(instances)

