import argparse
import os
import logging
import datetime
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from exp_utils import initialize_exp, get_dump_path, describe_model, set_seed, filter_logging
from load_config import load_config
from dataset.concat_dataset import MyConcatDataset
from dataset.sampler import DistributedWeightedAccumulateMultiDatasetBatchSampler


from model.unimodel import UniModel
from dsagent import DeepSpeedAgent

logger = logging.getLogger(__name__)


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--mode', type=str, default='train', help='train or test or validation')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--exp_name', default='run', type=str)
    parser.add_argument('--exp_id', default='test', type=str)
    parser.add_argument('--dump_path', default='dump', type=str)
    parser.add_argument('--random_seed', default=0, type=int)

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--logging_step', default=None, type=int)
    parser.add_argument('--max_length', default=None, type=int)


    parser.add_argument('--eval_step', default=None, type=int)
    parser.add_argument('--warmup_step', default=None, type=int)
    parser.add_argument('--total_epochs', default=None, type=int)
    parser.add_argument('--total_steps', default=None, type=int)
    parser.add_argument('--bs_per_gpu', default=None, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=None, type=int)
    parser.add_argument('--lr', default=None, type=float)

    parser.add_argument('--llama_ckpt_path', default=None, type=str)
    parser.add_argument('--dataset_name_list', default=None, nargs='+')
    parser.add_argument('--val_dataset_name_list', default=None, nargs='+')
    parser.add_argument('--dataset_selected_prob', default=None, nargs='+')
    parser.add_argument('--load_ckpt_path_list', default=None, nargs='+')
    parser.add_argument('--restore_ds', action='store_true')
    parser.add_argument('--restore_lr', action='store_true')

    return parser.parse_args()


def initialize_distributed(args):
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    args['master_port'] = os.getenv('MASTER_PORT', '6000')
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
    args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
    device = args['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend='nccl')


class Runner:
    def __init__(self, args):
        self.args = args
        self.init_ds()
        self.model = UniModel(args)

        self.train_dataset, self.iter_, _ = self.load_train_dataset()
        self.agent, self.total_steps = self.load_agent()
        torch.distributed.barrier()
        writer_path = os.path.join(args['dump_path'], 'tensorboard')
        os.makedirs(writer_path, exist_ok=True)
        self.writer = SummaryWriter(writer_path)
        filter_logging(f"==============================================================", logger, args['local_rank'])
        for key, value in self.args.items():
            filter_logging(f"{key}: {value}", logger, args['local_rank'])
        filter_logging(f"==============================================================", logger, args['local_rank'])
        if args['local_rank'] == 0:
            describe_model(self.model, args['dump_path'])

    def run(self):
        pbar = tqdm(total=self.total_steps)
        current_step = 0
        current_optimization_step = 0
        losses_list = []
        losses_dict = defaultdict(list)
        dataset_select_n_dict = defaultdict(int)

        for epoch_i in range(self.args['total_epochs']):

            for batch in self.iter_:
                loss, data_name = self.agent.train_model(
                    batch=batch,
                    current_step=current_step,
                    pbar=pbar
                )
                losses_list.append(loss)
                losses_dict[data_name].append(loss)
                dataset_select_n_dict[data_name] += 1

                current_step += 1
                if current_step % self.args['dschf'].config['gradient_accumulation_steps'] == 0:
                    current_optimization_step += 1

                if self.args['local_rank'] == 0 and \
                        current_step % self.args['dschf'].config['gradient_accumulation_steps'] == 0 and \
                        current_optimization_step % self.args['logging_step'] == 0:
                    lr = self.agent.get_lr()
                    rate = pbar.format_dict['rate']
                    remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
                    remaining = str(datetime.timedelta(seconds=remaining))
                    logger.info(f'[!] step:{current_optimization_step}; progress: {round(pbar.n / pbar.total, 5)}; '
                                f'remaining time: {remaining}; loss: {np.mean(losses_list):.5f}; lr: {lr:.8f}')
                    self.writer.add_scalar('loss', np.mean(losses_list), current_optimization_step)
                    self.writer.add_scalar('lr', lr, current_optimization_step)
                    for k, v in losses_dict.items():
                        if len(v) != 0:
                            self.writer.add_scalar(f'loss-{k}', np.mean(v), current_optimization_step)
                    self.writer.add_scalars('select', dataset_select_n_dict, current_optimization_step)

                    losses_list = []
                    losses_dict = defaultdict(list)

                if current_step % self.args['dschf'].config['gradient_accumulation_steps'] == 0 and \
                        current_optimization_step % self.args['eval_step'] == 0:
                    self.agent.save_model(self.args['dump_path'], current_optimization_step)

                if self.args['debug'] and current_step == 10:
                    self.agent.save_model(self.args['dump_path'], current_optimization_step)
                    return

                if current_optimization_step == self.args['total_steps']:
                    self.agent.save_model(self.args['dump_path'], current_optimization_step)
                    return

                if self.agent.get_lr() == 0. and current_optimization_step > self.args['warmup_step']:
                    self.agent.save_model(self.args['dump_path'], current_optimization_step)
                    return


    def load_train_dataset(self):

        if self.args['local_rank'] > 0:
            torch.distributed.barrier()
        concat_data = MyConcatDataset(self.args['dataset_name_list'])
        if self.args['local_rank'] == 0:
            torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        batch_size = self.args['world_size'] * self.args['dschf'].config['train_micro_batch_size_per_gpu']
        gradient_accumulation_steps = self.args['dschf'].config['gradient_accumulation_steps']
        sampler = torch.utils.data.RandomSampler(concat_data)
        batch_sampler = DistributedWeightedAccumulateMultiDatasetBatchSampler(dataset=concat_data,
                                                                              sampler=sampler,
                                                                              batch_size=batch_size,
                                                                              drop_last=True,
                                                                              rank=rank,
                                                                              world_size=world_size,
                                                                              gradient_accumulation_steps=gradient_accumulation_steps,
                                                                              selected_prob=self.args['dataset_selected_prob'])
        iter_ = DataLoader(
            concat_data,
            batch_sampler=batch_sampler,
            collate_fn=lambda x: concat_data.collate(x),
            num_workers=4,
            pin_memory=True
        )
        return concat_data, iter_, sampler

    def load_agent(self):
        if self.args['total_steps'] is None:
            train_num = max([_cur_dataset.__len__() for _cur_dataset in self.train_dataset.datasets.datasets]) * len(
                self.train_dataset.datasets.datasets)
            length = self.args['total_epochs'] * train_num // self.args['world_size'] // self.args['dschf'].config[
                'train_micro_batch_size_per_gpu']
            total_steps = self.args['total_epochs'] * train_num // self.args['dschf'].config['train_batch_size']
            self.args['total_steps'] = total_steps
        else:
            length = self.args['total_steps'] * self.args['dschf'].config['gradient_accumulation_steps']
            # update epoch num
            one_epoch_num = max([_cur_dataset.__len__() for _cur_dataset in self.train_dataset.datasets.datasets]) * \
                            len(self.train_dataset.datasets.datasets)
            epochs = (self.args['total_steps'] * self.args['dschf'].config['train_batch_size']) // one_epoch_num + 1
            self.args['total_epochs'] = epochs
        self.args['dschf'].config['scheduler']['params']['total_num_steps'] = self.args['total_steps']
        self.args['dschf'].config['scheduler']['params']['warmup_num_steps'] = max(10, self.args['warmup_step'])
        agent = DeepSpeedAgent(self.model, self.args)
        torch.distributed.barrier()
        return agent, length

    def init_ds(self):

        if self.args.get('lr', None) is not None:
            self.args['dschf'].config['optimizer']['params']['lr'] = self.args['lr']
            self.args['dschf'].config['scheduler']['params']['warmup_max_lr'] = self.args['lr']
        if self.args.get('bs_per_gpu', None) is not None:
            self.args['dschf'].config['train_micro_batch_size_per_gpu'] = self.args['bs_per_gpu']
        if self.args.get('gradient_accumulation_steps', None) is not None:
            self.args['dschf'].config['gradient_accumulation_steps'] = self.args['gradient_accumulation_steps']

        self.args['dschf'].config['train_batch_size'] = self.args['world_size'] * self.args['dschf'].config['train_micro_batch_size_per_gpu'] * self.args['dschf'].config['gradient_accumulation_steps']


def main():
    args = parser_args()
    # if args.local_rank == 0:
    initialize_exp(args)
    dump_path = get_dump_path(args)
    set_seed(args.random_seed)

    args = vars(args)
    config = load_config(args)
    # args.update(config)
    for k, v in config.items():
        if k not in args.keys() or args[k] is None:
            args[k] = v

    # TODO
    initialize_distributed(args)
    args['ds_config_path'] = f'config/dsconfig.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf
    args['dump_path'] = dump_path

    runner = Runner(args)
    runner.run()


if __name__ == '__main__':
    main()
