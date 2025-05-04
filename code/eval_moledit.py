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

from model.unimodel_moledit import UniModel4MolEdit
from dataset.data_catalog_moledit import DatasetMolEditCatalog, JSONDataset4MolEdit
from evaluation.mol_edit import evaluate_molecule, task2threshold_list
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
    parser.add_argument('--load_lora_ckpt_path_list', default=None, nargs='+')

    # lora
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--lora_r', default=32, type=int)
    parser.add_argument('--lora_alpha', default=32, type=int)
    parser.add_argument('--lora_dropout', default=0.1, type=float)

    parser.add_argument('--generate_bs', default=2, type=int)
    parser.add_argument('--generate_N', default=1, type=int)  # total
    parser.add_argument('--generate_n', default=1, type=int)

    parser.add_argument('--task_id', default=101, type=int)
    parser.add_argument('--thres_mode', default=0, type=int)

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


        logger.info(f"==============================================================")
        for k, v in self.args.items():
            logger.info(f"{k}: {v}")
        logger.info(f"==============================================================")
        describe_model(self.model, args['dump_path'])


    @torch.no_grad()
    def run(self):

        dataloader = DataLoader(
            self.dataset,
            collate_fn=self.dataset.collate,
            batch_size=self.args['generate_bs']
        )
        


        with open(os.path.join(self.args['dump_path'], 'generate.csv'), 'w', encoding='utf-8') as f:
            f.write(f"id,input,gen_molecule\n")

        generation_dict = defaultdict(list)  # cid -> mol
        cid2mol = dict()
        for batch in tqdm(dataloader):
            input_batch, inputs = batch[0], batch[1]
            for _ in range(self.args['generate_N'] // self.args['generate_n']):
                generate_result = self.model.generate(input_batch, inputs,
                                                      num_return_sequences=self.args['generate_n'],
                                                      top_p=self.args['generate_top_p'],
                                                      t=self.args['generate_t'],
                                                      min_new_tokens=self.args['generate_min_new_tokens'],  #
                                                      max_new_tokens=self.args['generate_max_new_tokens'])  # list

                idx = 0
                for data_id, input_seq in zip(inputs['id'], inputs['input_enc_seqs']):
                    cid2mol[data_id] = input_seq
                    for _ in range(self.args['generate_n']):
                        generation_dict[data_id].append(generate_result[idx])

                        with open(os.path.join(self.args['dump_path'], 'generate.csv'), 'a', encoding='utf-8') as f:
                            f.write(f'{data_id},{input_seq},{generate_result[idx]}\n')

                        idx += 1
                
                    
        
        data_list = []
        total_cnt = 0
        acc_cnt = 0
        for cid in cid2mol.keys():
            input_mol = cid2mol[cid]
            gen_mols = generation_dict[cid]
            for gen in gen_mols:
                total_cnt += 1
                input_v, output_v, acc = evaluate_molecule(input_mol, gen, task_id=self.args['task_id'],
                                                           threshold_list=task2threshold_list[self.args['task_id']][self.args['thres_mode']])
                if acc:
                    acc_cnt += 1
                data_list.append({
                    'cid': cid,
                    'input_mol': input_mol,
                    'output_mol': gen,
                    'acc': acc,
                    'input_v': input_v,
                    'output_v': output_v
                })
        logger.info(f"TOTAL={total_cnt}, ACC-CNT={acc_cnt}, ACC={acc_cnt/total_cnt:.4f}")
        # DUMP JSON DATA
        formatted_data = json.dumps(data_list, indent=4, ensure_ascii=False)

        with open(os.path.join(self.args['dump_path'], 'generate.json'), "w", encoding='utf-8') as f:
            f.write(formatted_data)


    def load_dataset(self, dataset_name):
        catalog = DatasetMolEditCatalog()
        dataset_dict = getattr(catalog, dataset_name)

        dataset = JSONDataset4MolEdit(**dataset_dict, data_name=dataset_name)
        return dataset

    def load_model(self):
        if self.args['datatype'] == 'bf16':
            model = UniModel4MolEdit(self.args).bfloat16().cuda()
        elif self.args['datatype'] == 'half':
            model = UniModel4MolEdit(self.args).half().cuda()
        elif self.args['datatype'] == 'float':
            model = UniModel4MolEdit(self.args).float().cuda()
        else:
            raise ValueError
        for ckpt_path in self.args['load_ckpt_path_list']:
            delta_ckpt = torch.load(os.path.join(ckpt_path, 'pytorch_model.bin'), map_location=torch.device('cuda'))
            model.load_state_dict(delta_ckpt, strict=False)
            for k, _ in delta_ckpt.items():
                logger.info(f"loading {k} from {ckpt_path}")
        if self.args['lora']:
            model.init_lora()
            for ckpt_path in self.args['load_lora_ckpt_path_list']:
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
