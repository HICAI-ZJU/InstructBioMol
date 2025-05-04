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

from evaluation.esp.esp import ESP_predicton
from evaluation.protein import all_characters_are_amino_acids

from exp_utils import initialize_exp, get_dump_path, describe_model, set_seed, filter_logging


logger = logging.getLogger(__name__)



def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--exp_name', default='eval-ESP', type=str)
    parser.add_argument('--exp_id', default='eval-score-debug', type=str)
    parser.add_argument('--dump_path', default='dump', type=str)
    parser.add_argument('--random_seed', default=0, type=int)

    parser.add_argument('--data_file', default='', type=str)

    return parser.parse_args()


def run(args):

    test_data_list = json.load(open('data/protein-molecule/gorhea-molecule-to-protein-test-filtered.json', 'r'))
    cid2smiles = dict()
    for data in test_data_list:
        cid = data['CID']
        smiles = data['smiles']
        cid2smiles[cid] = smiles


    load_data_list = json.load(open(args.data_file, 'r'))
    all_smiles_list = []
    all_protein_list = []

    for data in load_data_list:
        cid = data['id']
        smiles = cid2smiles[cid]
        generation = []
        for gen_protein in data['generation']:
            if all_characters_are_amino_acids(gen_protein):
                generation.append(gen_protein)
                all_smiles_list.append(smiles)
                all_protein_list.append(gen_protein)


    df = ESP_predicton(substrate_list=all_smiles_list,
                       enzyme_list=all_protein_list)
    avg_score = df.groupby('substrate')['Prediction'].max().mean()

    logger.info(f"ESP Score={avg_score:.4f}")

    with open(os.path.join(args.dump_path, 'score.csv'), 'w', encoding='utf-8') as f:
        f.write(f'smiles,protein,score\n')

    for smiles, protein, score in zip(all_smiles_list,
                                      all_protein_list,
                                      df['Prediction'].tolist()):
        with open(os.path.join(args.dump_path, 'score.csv'), 'a', encoding='utf-8') as f:
            f.write(f'{smiles},{protein},{score:.4f}\n')



def main():
    args = parser_args()
    initialize_exp(args)
    dump_path = get_dump_path(args)
    args.dump_path = dump_path
    set_seed(args.random_seed)
    run(args)



if __name__ == '__main__':

    main()
