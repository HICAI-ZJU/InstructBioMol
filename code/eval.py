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
import selfies

from load_config import load_config
from model.unimodel import UniModel
from dataset.json_dataset import JSONDataset
from dataset.lmdb_dataset import LMDBDataset
from dataset.data_catalog import DatasetCatalog

from evaluation.protein import eval_protein, eval_protein_set
from evaluation.text import eval_text
from evaluation.mol import eval_mol
from evaluation.moses import get_all_metrics
from exp_utils import initialize_exp, get_dump_path, describe_model, set_seed, filter_logging

logger = logging.getLogger(__name__)


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--mode', type=str, default='validation', help='train or test or validation')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--exp_name', default='run', type=str)
    parser.add_argument('--exp_id', default='eval', type=str)
    parser.add_argument('--dump_path', default='dump', type=str)
    parser.add_argument('--random_seed', default=0, type=int)

    # model configurations
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval_mode', default=1, type=int)
    parser.add_argument('--datatype', default='bf16', type=str, choices=['bf16', 'half', 'float'])
    # model configurations
    parser.add_argument('--dataset_name', default=None, type=str)
    parser.add_argument('--val_dataset_name_list', default=None, nargs='+')
    parser.add_argument('--load_ckpt_path_list', default=None, nargs='+')


    parser.add_argument('--generate_bs', default=2, type=int)
    parser.add_argument('--generate_N', default=1, type=int)  # total
    parser.add_argument('--generate_n', default=1, type=int)

    parser.add_argument('--generate_top_p', default=0.1, type=float)
    parser.add_argument('--generate_t', default=1., type=float)
    parser.add_argument('--generate_num_beams', default=None, type=int)
    parser.add_argument('--generate_min_new_tokens', default=None, type=int)
    parser.add_argument('--generate_max_new_tokens', default=None, type=int)


    return parser.parse_args()


class Runner:
    def __init__(self, args):
        self.args = args
        self.model = self.load_model()
        self.dataset = self.load_dataset(self.args['dataset_name'])
        self.instruction = self.dataset.instruction

        self.tgt_modality = self.dataset.target_modality

        logger.info(f"TARGET MODALITY = {self.tgt_modality}")
        logger.info(f"==============================================================")
        for k, v in self.args.items():
            logger.info(f"{k}: {v}")
        logger.info(f"==============================================================")
        describe_model(self.model, args['dump_path'])

    @torch.no_grad()
    def run(self):
        logger.info(f"TOTAL={len(self.dataset)}")
        if self.args['eval_mode'] == 1:
            eval_score = self.eval_for_mode_1()
        elif self.args['eval_mode'] == 2:
            self.eval_for_mode_2()
        else:
            raise NotImplementedError

    def eval_for_mode_1(self):

        groundtruth_list = []
        generation_list = []

        dataloader = DataLoader(
            self.dataset,
            collate_fn=self.dataset.collate,
            batch_size=self.args['generate_bs']
        )

        with open(os.path.join(self.args['dump_path'], 'generate.csv'), 'w', encoding='utf-8') as f:
            f.write(f"description,generation,seq\n")

        for batch in tqdm(dataloader):
            input_batch, inputs = batch[0], batch[1]
            generate_result = self.model.generate(input_batch, inputs, self.tgt_modality,
                                                  top_p=self.args['generate_top_p'],
                                                  t=self.args['generate_t'],
                                                  num_beams=self.args['generate_num_beams'])  # list

            generation_list.extend(generate_result)
            if self.tgt_modality == 'text':
                groundtruth_list.extend(inputs['target_seqs'])
                description = inputs['target_seqs']

            elif self.tgt_modality == 'molecule':
                groundtruth_list.extend(inputs['input_enc_seqs'])
                description = inputs['input_seqs']

            elif self.tgt_modality == 'protein':
                groundtruth_list.extend(inputs['target_seqs'])
                description = inputs['input_seqs']

            else:
                raise NotImplementedError


            seq = inputs['input_enc_seqs']
            with open(os.path.join(self.args['dump_path'], 'generate.csv'), 'a', encoding='utf-8') as f:
                for d, g, s in zip(description, generate_result, seq):
                    # TEXT, GENERATION, MOL/PROT
                    f.write(f'"{d}","{g}",{s}\n')

            if self.args['debug'] and len(groundtruth_list) >= 10:
                break

        if self.tgt_modality == 'text':
            eval_score = eval_text(groundtruth_list=groundtruth_list,
                                   generation_list=generation_list)
        elif self.tgt_modality == 'molecule':
            eval_score = eval_mol(groundtruth_list=groundtruth_list,
                                  generation_list=generation_list)
        elif self.tgt_modality == 'protein':
            eval_score = eval_protein(groundtruth_list=groundtruth_list,
                                      generation_list=generation_list)
        else:
            raise NotImplementedError

        for k, v in eval_score.items():
            logger.info(f"{k}: {v:.5f}")

        data_list = []
        for groundtruth, generation in zip(groundtruth_list, generation_list):
            data_list.append({'id': groundtruth,
                              'groundtruth': groundtruth,
                              'generation': generation})
        # DUMP JSON DATA
        formatted_data = json.dumps(data_list, indent=4, ensure_ascii=False)

        with open(os.path.join(self.args['dump_path'], 'generate.json'), "w", encoding='utf-8') as f:
            f.write(formatted_data)

        return eval_score

    def eval_for_mode_2(self):

        if self.args['dataset_name'] == 'molecule_to_protein_gorhea_test':
            all_data_list = json.load(open('data/protein-molecule/gorhea-molecule-to-protein-test.json', 'r'))
            all_groundtruth_dict = defaultdict(list)
            for data in all_data_list:
                cid = data['CID']
                all_groundtruth_dict[cid].append(data['protein'])
        elif self.args['dataset_name'] == 'protein_to_molecule_bindingdb_test':
            all_data_list = json.load(open('data/protein-molecule/protein-to-molecule-bindingdb-test.json', 'r'))
            all_groundtruth_dict = defaultdict(list)
            for data in all_data_list:
                pid = data['PID']
                all_groundtruth_dict[pid].append(data['smiles'])
        else:
            raise NotImplementedError


        # p2m or m2p
        dataloader = DataLoader(
            self.dataset,
            collate_fn=self.dataset.collate,
            batch_size=self.args['generate_bs']
        )
        data_list = []
        # ground_truth_dict = defaultdict(list)  # input_seq -> gt
        generation_dict = defaultdict(list)  # input_seq -> generation
        data_id_dict = dict()  # input_seq -> data_id

        with open(os.path.join(self.args['dump_path'], 'generate.csv'), 'w', encoding='utf-8') as f:
            f.write(f"id,input,generation\n")

        for batch in tqdm(dataloader):
            input_batch, inputs = batch[0], batch[1]
            for _ in range(self.args['generate_N'] // self.args['generate_n']):
                generate_result = self.model.generate(input_batch, inputs, self.tgt_modality,
                                                      num_return_sequences=self.args['generate_n'],
                                                      top_p=self.args['generate_top_p'],
                                                      t=self.args['generate_t'],
                                                      min_new_tokens=self.args['generate_min_new_tokens'],  #
                                                      max_new_tokens=self.args['generate_max_new_tokens'])  # list

                idx = 0
                for data_id, input_seq in zip(inputs['ids'], inputs['input_enc_seqs']):
                    data_id_dict[input_seq] = data_id
                    for _ in range(self.args['generate_n']):
                        generation_dict[input_seq].append(generate_result[idx])

                        with open(os.path.join(self.args['dump_path'], 'generate.csv'), 'a', encoding='utf-8') as f:
                            f.write(f'{data_id},{input_seq},{generate_result[idx]}\n')

                        idx += 1

            if self.args['debug'] and len(generation_dict.keys()) >= 2:
                break

        eval_result_dict = defaultdict(list)
        for input_seq in generation_dict.keys():
            data_id = data_id_dict[input_seq]
            groundtruth = all_groundtruth_dict[data_id_dict[input_seq]]
            generation = generation_dict[input_seq]
            data_list.append({'id': data_id,
                              # 'seq': input_seq,
                              'groundtruth': groundtruth,
                              'generation': generation})

            if self.tgt_modality == 'protein':
                score = eval_protein_set(generation_list=generation,
                                        groundtruth_list=groundtruth)
            else:
                score = get_all_metrics(gen=generation,
                                        test=groundtruth)
            for k, v in score.items():
                eval_result_dict[k].append(v)                

        for k, v in eval_result_dict.items():
            logger.info(f"{k}-avg: {np.nanmean(v):.5f}, {k}-med: {np.nanmedian(v):.5f}")


        # DUMP JSON DATA
        formatted_data = json.dumps(data_list, indent=4, ensure_ascii=False)

        with open(os.path.join(self.args['dump_path'], 'generate.json'), "w", encoding='utf-8') as f:
            f.write(formatted_data)


    def load_dataset(self, dataset_name):
        catalog = DatasetCatalog()
        dataset_dict = getattr(catalog, dataset_name)
        if dataset_dict['data_format'] == 'JSON':
            dataset = JSONDataset(**dataset_dict, data_name=dataset_name)
        elif dataset_dict['data_format'] == 'LMDB':
            dataset = LMDBDataset(**dataset_dict, data_name=dataset_name)
        else:
            raise NotImplementedError
        return dataset

    def load_model(self):
        if self.args['datatype'] == 'bf16':
            model = UniModel(self.args).bfloat16().cuda()
        elif self.args['datatype'] == 'half':
            model = UniModel(self.args).half().cuda()
        elif self.args['datatype'] == 'float':
            model = UniModel(self.args).float().cuda()
        else:
            raise ValueError

        for ckpt_path in self.args['load_ckpt_path_list']:
            delta_ckpt = torch.load(os.path.join(ckpt_path, 'pytorch_model.bin'), map_location=torch.device('cuda'))
            model.load_state_dict(delta_ckpt, strict=False)
            for k, _ in delta_ckpt.items():
                logger.info(f"loading {k} from {ckpt_path}")
        model = model.eval()

        return model


def main():
    args = parser_args()
    # if args.local_rank == 0:
    initialize_exp(args)
    dump_path = get_dump_path(args)
    set_seed(args.random_seed)

    args = vars(args)
    config = load_config(args)

    for k, v in config.items():
        if k not in args.keys() or args[k] is None:
            args[k] = v

    args['dump_path'] = dump_path

    runner = Runner(args)
    runner.run()


if __name__ == '__main__':
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    assert not tf.config.get_visible_devices('GPU')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
