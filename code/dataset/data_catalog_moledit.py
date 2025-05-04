from torch.utils.data import ConcatDataset, Dataset
from torch_geometric.data import Data, Batch
import torch

import json
import os
import sys
from .preprocess_smiles import process_smiles
from .load_sdf import load_sdf_data, collate_sdf

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from exp_utils import filter_logging

import logging

logger = logging.getLogger(__name__)



def collate_func4moledit(instances):


    input_seqs, target_molecule, input_enc_fps, input_enc_seqs, instructions, confs, ids, data_name = tuple(
        [instance[key] for instance in instances] for key in ("input_seqs",
                                                              "target_molecule",
                                                              "input_enc_fps",
                                                              "input_enc_seqs",
                                                              "instructions",
                                                              "conf",
                                                              "id",
                                                              "data_name"))

    data_name = data_name[0]

    smiles = input_enc_seqs
    graph_data_list = []
    for s in smiles:
        x, edge_index, edge_attr = process_smiles(s)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph_data_list.append(data)
    batch_graph = Batch.from_data_list(graph_data_list)

    sdf_list = []
    for sdf_path in confs:
        z, coords = load_sdf_data(sdf_path)
        sdf_list.append({'z': z, 'pos': coords})
    geoformer_input = collate_sdf(sdf_list)

    return (batch_graph, geoformer_input), {'input_seqs': input_seqs,
                                            'target_molecule': target_molecule,
                                            'input_enc_fps': input_enc_fps,
                                            'input_enc_seqs': input_enc_seqs,
                                            'instructions': instructions,
                                            'conf': confs,
                                            'id': ids,
                                            'data_name': data_name}



class DatasetMolEditCatalog:
    def __init__(self):
        self.moledit_101 = {
            'data_path': 'data/moledit/json/101-12141216.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/101-12141216',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule more soluble in water.'
        }


        self.moledit_102 = {
            'data_path': 'data/moledit/json/102-12141216.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/102-12141216',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule less soluble in water.'
        }

        self.moledit_103 = {
            'data_path': 'data/moledit/json/103-12141216.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/103-12141216',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule more like a drug.'
        }

        self.moledit_104 = {
            'data_path': 'data/moledit/json/104-12141216.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/104-12141216',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule less like a drug.'
        }

        self.moledit_105 = {
            'data_path': 'data/moledit/json/105-12141216.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/105-12141216',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule higher permeability.'
        }

        self.moledit_106 = {
            'data_path': 'data/moledit/json/106-12141216.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/106-12141216',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule lower permeability.'
        }

        self.moledit_107 = {
            'data_path': 'data/moledit/json/107-12141216.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/107-12141216',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule with more hydrogen bond acceptors.'
        }

        self.moledit_108 = {
            'data_path': 'data/moledit/json/108-12141216.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/108-12141216',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule with more hydrogen bond donors.'
        }

        self.moledit_201 = {
            'data_path': 'data/moledit/json/201-12141216.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/201-12141216',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule more soluble in water and more hydrogen bond acceptors.'
        }
        self.moledit_202 = {
            'data_path': 'data/moledit/json/202-12141216.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/202-12141216',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule less soluble in water and more hydrogen bond acceptors.'
        }
        self.moledit_203 = {
            'data_path': 'data/moledit/json/203-12141216.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/203-12141216',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule more soluble in water and more hydrogen bond donors.'
        }
        self.moledit_204 = {
            'data_path': 'data/moledit/json/204-12141216.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/204-12141216',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule less soluble in water and more hydrogen bond donors.'
        }
        self.moledit_205 = {
            'data_path': 'data/moledit/json/205-12141216.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/205-12141216',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule more soluble in water and higher permeability.'
        }
        self.moledit_206 = {
            'data_path': 'data/moledit/json/206-12141216.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/206-12141216',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule more soluble in water and lower permeability.'
        }
        # ==========================test
        self.moledit_test_101 = {
            'data_path': 'data/moledit/json/test.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/test',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule more soluble in water.'
        }

        self.moledit_test_102 = {
            'data_path': 'data/moledit/json/test.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/test',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule less soluble in water.'
        }

        self.moledit_test_103 = {
            'data_path': 'data/moledit/json/test.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/test',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule more like a drug.'
        }

        self.moledit_test_104 = {
            'data_path': 'data/moledit/json/test.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/test',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule less like a drug.'
        }

        self.moledit_test_105 = {
            'data_path': 'data/moledit/json/test.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/test',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule higher permeability.'
        }

        self.moledit_test_106 = {
            'data_path': 'data/moledit/json/test.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/test',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule lower permeability.'
        }

        self.moledit_test_107 = {
            'data_path': 'data/moledit/json/test.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/test',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule with more hydrogen bond acceptors.'
        }

        self.moledit_test_108 = {
            'data_path': 'data/moledit/json/test.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/test',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule with more hydrogen bond donors.'
        }

        self.moledit_test_201 = {
            'data_path': 'data/moledit/json/test.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/test',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule more soluble in water and more hydrogen bond acceptors.'
        }
        self.moledit_test_202 = {
            'data_path': 'data/moledit/json/test.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/test',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule less soluble in water and more hydrogen bond acceptors.'
        }

        self.moledit_test_203 = {
            'data_path': 'data/moledit/json/test.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/test',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule more soluble in water and more hydrogen bond donors.'
        }
        self.moledit_test_204 = {
            'data_path': 'data/moledit/json/test.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/test',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule less soluble in water and more hydrogen bond donors.'
        }

        self.moledit_test_205 = {
            'data_path': 'data/moledit/json/test.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/test',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule more soluble in water and higher permeability.'
        }
        self.moledit_test_206 = {
            'data_path': 'data/moledit/json/test.json',
            'id_key': 'cid',
            'input_selfies_key': 'input_selfies',
            'target_selfies_key': 'target_selfies',
            'input_smiles_key': 'input_smiles',
            'target_smiles_key': 'target_smiles',
            'input_fp_key': 'input_fingerprint',
            'conf_path': 'data/moledit/sdf/test',
            'conf_key': 'input_sdf',
            'instruction': 'Make the molecule more soluble in water and lower permeability.'
        }

class JSONDataset4MolEdit(Dataset):
    def __init__(self,
                 data_name: str,
                 data_path: str,
                 id_key: str,
                 input_selfies_key: str,
                 target_selfies_key: str,
                 input_smiles_key: str,
                 target_smiles_key: str,
                 input_fp_key: str,
                 conf_path: str,
                 conf_key: str,
                 instruction: str
                 ):
        super(JSONDataset4MolEdit, self).__init__()

        self.data_path = data_path
        self.data_name = data_name

        self.data_id_list = []
        self.input_selfies_list = []
        self.target_selfies_list = []
        self.input_smiles_list = []
        self.target_smiles_list = []
        self.input_fp_list = []
        self.conformation_list = []

        self.instruction = instruction


        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for datapoint in data:
            data_id = datapoint[id_key]
            self.data_id_list.append(data_id)
            self.input_selfies_list.append(datapoint[input_selfies_key])
            self.input_smiles_list.append(datapoint[input_smiles_key])
            self.target_selfies_list.append(datapoint[target_selfies_key])
            self.target_smiles_list.append(datapoint[target_smiles_key])
            self.input_fp_list.append(datapoint[input_fp_key])

            self.conformation_list.append(os.path.join(conf_path, datapoint[conf_key]))



    def __len__(self):
        return len(self.data_id_list)

    def __getitem__(self, i):

        return {
                'input_seqs': self.input_selfies_list[i],
                'target_molecule': self.target_selfies_list[i],
                'input_enc_fps': self.input_fp_list[i],
                'input_enc_seqs': self.input_smiles_list[i],
                'instructions': self.instruction,
                'conf': self.conformation_list[i],
                'id': self.data_id_list[i],
                'data_name': self.data_name
                }
    
    def collate(self, instances):
        return collate_func4moledit(instances)

class MyConcatMolEditProtDataset(Dataset):
    def __init__(self, dataset_name_list):
        super(MyConcatMolEditProtDataset, self).__init__()

        _datasets = []

        catalog = DatasetMolEditCatalog()
        for dataset_idx, dataset_name in enumerate(dataset_name_list):
            dataset_dict = getattr(catalog, dataset_name)

            dataset = JSONDataset4MolEdit(**dataset_dict, data_name=dataset_name)
            _datasets.append(dataset)
            logger.info(f"DATASET-{dataset_name}: {len(dataset):,}")
        self.datasets = ConcatDataset(_datasets)

    def __len__(self):
        return self.datasets.__len__()

    def __getitem__(self, item):
        return self.datasets.__getitem__(item)

    def collate(self, instances):
        return collate_func4moledit(instances)

