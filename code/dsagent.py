import os.path
import json
import types
import logging
from collections import OrderedDict
import datetime

import torch
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
from exp_utils import filter_logging

logger = logging.getLogger(__name__)


class DeepSpeedAgent:

    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model

        for path in self.args['load_ckpt_path_list']:
            self.load_parameters(path)
        
        if args['lora']:
            self.model.init_lora()
        self.print_model_parameters()

        # ds_params, self.total_steps = self.set_hyper_parameters()
        self.ds_engine, self.llamaimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=filter(lambda p: p.requires_grad, self.model.parameters()),
            config_params=self.args['dschf'].config,
            dist_init_required=True,
            args=types.SimpleNamespace(**args)
        )

        if self.args['restore_ds'] and len(self.args['load_ckpt_path_list']) > 0:
            for path in self.args['load_ckpt_path_list']:
                self.ds_engine.load_checkpoint(load_dir=path,
                                               load_module_strict=False,
                                               load_lr_scheduler_states=self.args['restore_lr'],
                                               load_optimizer_states=True)

        filter_logging(f"==============================================================", logger, args['local_rank'])
        for key, value in self.args['dschf'].config.items():
            filter_logging(f"{key}: {value}", logger, args['local_rank'])
        filter_logging(f"==============================================================", logger, args['local_rank'])


    def get_lr(self):
        return self.ds_engine.lr_scheduler.get_lr()[0]

    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()
        loss = self.ds_engine(batch[0], batch[1])

        self.ds_engine.backward(loss)
        self.ds_engine.step()
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}')
        pbar.update(1)

        return loss.item(), batch[1]['data_name']

    def save_model(self, path, current_step='final'):
        """
            this function also save the trainable parameters and specific name parameters
        """
        dump_path = os.path.join(path, 'ckpt', f'ckpt-{current_step}')

        torch.distributed.barrier()
        if self.args['local_rank'] == 0:
            # checkpoint = OrderedDict()
            # # for k, v in self.ds_engine.module.state_dict().items():
            # for k, v in self.ds_engine.module.named_parameters():
            #     if v.requires_grad:
            #         checkpoint[k] = v.detach()

            filter_logging(f"SAVING checkpoint to {dump_path}", logger, self.args['local_rank'])

            os.makedirs(dump_path, exist_ok=True)

            if not self.args['lora']:
                torch.save(self.ds_engine.module.state_dict(), f'{dump_path}/pytorch_model.bin')
                # save DS Engine
                self.ds_engine.save_checkpoint(dump_path, tag=current_step)
            else:
                checkpoint = OrderedDict()
                # for k, v in self.ds_engine.module.state_dict().items():
                for k, v in self.ds_engine.module.named_parameters():
                    if v.requires_grad:
                        checkpoint[k] = v.detach()
                torch.save(checkpoint, f'{dump_path}/pytorch_model.bin')
        # save tokenizer
        self.model.llama_tokenizer.save_pretrained(dump_path)
        # save configuration
        self.model.llama_model.config.save_pretrained(dump_path)
        filter_logging(f'[!] SAVING model config into {dump_path}', logger, self.args['local_rank'])

    def print_model_parameters(self, use_4bit=False):
        """
            Prints the number of trainable parameters in the model.
        """
        trainable_param = 0
        all_param = 0
        input_projector_param = 0
        llama_param = 0
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            if 'enc_proj_dict' in name:
                input_projector_param += num_params

            if 'llama' in name:
                llama_param += num_params

            all_param += num_params
            if param.requires_grad:
                trainable_param += num_params
                filter_logging(f"trainable params: {name}", logger, self.args['local_rank'])

        filter_logging(
            f"all params: {all_param:,d} || trainable params: {trainable_param:,d} || trainable%: {100 * trainable_param / all_param}",
            logger, self.args['local_rank'])
        filter_logging(f'encoder projector params: {input_projector_param:,d} || llama params: {llama_param:,d}',
            logger, self.args['local_rank'])

    def load_parameters(self, path):
        if path == '':
            return
        if os.path.exists(os.path.join(path, 'pytorch_model.bin')):
            filter_logging(f'loading parameters from {path}', logger, self.args['local_rank'])
            delta_ckpt = torch.load(f'{path}/pytorch_model.bin', map_location=torch.device('cuda'))
            checkpoint = delta_ckpt
            for k, v in checkpoint.items():
                filter_logging(f"loading {k} from {path}", logger, self.args['local_rank'])
            self.model.load_state_dict(checkpoint, strict=False)


