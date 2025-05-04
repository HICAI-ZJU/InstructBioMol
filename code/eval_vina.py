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
from Bio.PDB import PDBParser



from evaluation.docking import calculate_qvina
from exp_utils import initialize_exp, get_dump_path, describe_model, set_seed, filter_logging


logger = logging.getLogger(__name__)


def extract_mol_from_sdf(sdf_file):

    if not os.path.exists(sdf_file):
        return None
    try:
        mol = Chem.SDMolSupplier(sdf_file)[0]
        smiles = Chem.MolToSmiles(mol)
    except:
        return None
    return smiles


def extract_sequence_from_pdb(pdb_file):

    def three_to_one(three_letter_code):

        aa_dict = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        return aa_dict.get(three_letter_code, 'X')

    if not os.path.exists(pdb_file):
        return ''

    parser = PDBParser()

    structure = parser.get_structure('protein_structure', pdb_file)


    model = structure[0]


    sequence = []


    for chain in model:

        for residue in chain:

            if residue.get_id()[0] == ' ':

                sequence.append(residue.resname)


    one_letter_sequence = "".join([three_to_one(res) for res in sequence])
    return one_letter_sequence




def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--exp_name', default='eval-vina', type=str)
    parser.add_argument('--exp_id', default='eval-score-debug', type=str)
    parser.add_argument('--dump_path', default='dump', type=str)
    parser.add_argument('--random_seed', default=0, type=int)

    parser.add_argument('--folder', default='', type=str)
    parser.add_argument('--mode', default='p2m', type=str)
    parser.add_argument('--eval_n', default=100, type=int)
    parser.add_argument('--cpu', default=16, type=int)

    return parser.parse_args()


def run(args):
    id2gt = defaultdict(list)

    if args.mode == 'p2m':
        test_data_list = json.load(open('data/protein-molecule/protein-to-molecule-bindingdb-test.json', 'r'))
        for data in test_data_list:
            pid = data['PID']
            smiles = data['smiles']
            id2gt[pid].append(smiles)
    else:
        test_data_list = json.load(open('data/protein-molecule/gorhea-molecule-to-protein-test.json', 'r'))
        for data in test_data_list:
            cid = data['CID']
            protein = data['protein']
            id2gt[cid].append(protein)

    vina_list = []
    vina_top1_list, vina_top5_list, vina_top10_list = [], [], []

    result_dict = dict()

    for data_id, groundtruth in id2gt.items():
        tmp_score_list = []
        tmp_gen_list = []
        if args.mode == 'p2m':
            pdb_file = f'data/pdb-conf/bindingdb/AF-{data_id}-F1-model_v4.pdb'
            for complex_name in range(args.eval_n):

                sdf_file = os.path.join(args.folder, f'{data_id}-{complex_name}', 'rank1.sdf')
                tmp_path = os.path.join(args.dump_path, 'generation', f'{data_id}', f'{complex_name}')
                smiles = extract_mol_from_sdf(sdf_file)

                if smiles is None:
                    continue

                if not os.path.exists(sdf_file):
                    vina_score = None
                vina_score = calculate_qvina(receptor_file=pdb_file,
                                                ligand_file=sdf_file,
                                                output_path=tmp_path,
                                                ref_ligand_file=sdf_file,
                                                num_cpu=args.cpu)

                if vina_score is None:
                    continue
                tmp_gen_list.append(smiles)
                tmp_score_list.append(vina_score)
                logger.info(f'[{data_id}] [{smiles}]: {vina_score}')

            # tmp_score_list = [s for s in tmp_score_list if s is not np.nan]
            if len(tmp_score_list) > 0:
                logger.info(f'tmp-avg={np.nanmean(tmp_score_list):.5f}')
                sort_tmp_score_list = np.sort(tmp_score_list).tolist()
                vina_list.extend(tmp_score_list)
                vina_top1_list.extend(sort_tmp_score_list[:1])
                vina_top5_list.extend(sort_tmp_score_list[:5])
                vina_top10_list.extend(sort_tmp_score_list[:10])

                result_dict[data_id] = {
                    'groundtruth': id2gt[data_id],
                    'generation_smiles': tmp_gen_list,
                    'score': tmp_score_list
                }
            logger.info(f"avg={np.nanmean(vina_list):.5f}, med={np.nanmedian(vina_list):.5f}")
            logger.info(f"top1-avg={np.nanmean(vina_top1_list):.5f}, top1-med={np.nanmedian(vina_top1_list):.5f}")
            logger.info(f"top5-avg={np.nanmean(vina_top5_list):.5f}, top5-med={np.nanmedian(vina_top5_list):.5f}")
            logger.info(f"top10-avg={np.nanmean(vina_top10_list):.5f}, top10-med={np.nanmedian(vina_top10_list):.5f}")
    #
        else:
            for complex_name in range(args.eval_n):

                sdf_file = os.path.join(args.folder, f'{data_id}-{complex_name}', 'rank1.sdf')
                tmp_path = os.path.join(args.dump_path, 'generation', f'{data_id}', f'{complex_name}')
                pdb_file = os.path.join(args.folder, f'{data_id}-{complex_name}', f'{data_id}-{complex_name}_esmfold.pdb')
                protein = extract_sequence_from_pdb(pdb_file)
                if not os.path.exists(sdf_file):
                    vina_score = None

                vina_score = calculate_qvina(receptor_file=pdb_file,
                                                 ligand_file=sdf_file,
                                                 output_path=tmp_path,
                                                 ref_ligand_file=sdf_file,
                                                 num_cpu=args.cpu)
                if vina_score is None:
                    continue
                tmp_gen_list.append(protein)
                tmp_score_list.append(vina_score)
                logger.info(f'[{data_id}] [{protein}]: {vina_score}')

            # tmp_score_list = [s for s in tmp_score_list if s is not np.nan]
            if len(tmp_score_list) > 0:
                logger.info(f'tmp-avg={np.nanmean(tmp_score_list):.5f}')
                sort_tmp_score_list = np.sort(tmp_score_list).tolist()
                vina_list.extend(sort_tmp_score_list[:1])

                result_dict[data_id] = {
                    'groundtruth': id2gt[data_id],
                    'generation_protein': tmp_gen_list,
                    'score': tmp_score_list
                }
            logger.info(f"avg={np.nanmean(vina_list):.5f}, med={np.nanmedian(vina_list):.5f}")


    format_result = json.dumps(result_dict, indent=4, ensure_ascii=False)
    with open(os.path.join(args.dump_path, 'generation.json'), 'w') as f:
        f.write(format_result)

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
