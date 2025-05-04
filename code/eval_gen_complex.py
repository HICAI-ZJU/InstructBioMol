import argparse
import os
import logging
from collections import defaultdict
import json

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
from rdkit import Chem

# from evaluation.score import generate_and_save_sdf
from evaluation.diffdock import call_diffdock_csv
from evaluation.protein import all_characters_are_amino_acids
from exp_utils import initialize_exp, get_dump_path, describe_model, set_seed, filter_logging

logger = logging.getLogger(__name__)


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--exp_name', default='complex-generation', type=str)
    parser.add_argument('--exp_id', default='eval-score-debug', type=str)
    parser.add_argument('--dump_path', default='dump', type=str)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--data_file', required=True, type=str)
    parser.add_argument('--diffdock_path', default='/home/zhangqiang/userdata/zhuangxiang/code/DiffDock-L', type=str)
    parser.add_argument('--mode', default='p2m', choices=['p2m', 'm2p'], type=str)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def run(args):
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # write to csv
    csv_file_path = os.path.join(args.dump_path, 'generation.csv')
    with open(csv_file_path, 'w') as f:
        f.write('complex_name,protein_path,ligand_description,protein_sequence\n')

    for data in tqdm(data_list):
        data_id = data['id']

        generation = data['generation']
        if args.debug:
            generation = generation[:2]
        if args.mode == 'p2m':
            pdb_file = f'data/pdb-conf/bindingdb/AF-{data_id}-F1-model_v4.pdb'
            for idx, gen_mol in enumerate(generation):
                try:
                    mol = Chem.MolFromSmiles(gen_mol)
                    if mol is None:
                        continue
                    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                except:
                    continue
                with open(csv_file_path, 'a') as f:
                    f.write(f'{data_id}-{idx},{pdb_file},{canonical_smiles},\n')
        else:
            with open('data/protein-molecule/gorhea-molecule-to-protein-test.json', 'r', encoding='utf-8') as f:
                ref_data_list = json.load(f)
            cid2smiles = {data['CID']: data['smiles'] for data in ref_data_list}
            for idx, gen_protein in enumerate(generation):
                if not all_characters_are_amino_acids(gen_protein):
                    continue
                with open(csv_file_path, 'a') as f:
                    f.write(f'{data_id}-{idx},,{cid2smiles[data_id]},{gen_protein}\n')

    call_diffdock_csv(csv_file=csv_file_path,
                      diffdock_path=args.diffdock_path,
                      dump_path=os.path.join(args.dump_path, 'generation'),
                      gpu=args.gpu)


def main():
    args = parser_args()

    initialize_exp(args)
    dump_path = get_dump_path(args)
    args.dump_path = dump_path
    set_seed(args.random_seed)
    run(args)



if __name__ == '__main__':

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()