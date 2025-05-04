import os
import logging
import re
from typing import List
import sys
import json

import torch
from torch import nn
import numpy as np
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig
import selfies
from peft import LoraConfig, TaskType, get_peft_model

from .encoder_projector import EncoderProjector

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from exp_utils import filter_logging

logger = logging.getLogger(__name__)


class UniModel4MolEdit(nn.Module):
    def __init__(self, args):
        super(UniModel4MolEdit, self).__init__()
        self.args = args
        self.max_length = args['max_length']

        self.device = torch.cuda.current_device()

        # llama MODEL
        self.llama_ckpt_path = args['llama_ckpt_path']
        # self.llama_model = LlamaForCausalLM.from_pretrained(self.llama_ckpt_path)
        self.llama_model = LlamaForCausalLM(LlamaConfig.from_pretrained(self.llama_ckpt_path))
        self.llama_tokenizer = AutoTokenizer.from_pretrained(self.llama_ckpt_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = 'left'
        filter_logging(f"LOADING llama from {self.llama_ckpt_path}", logger, self.args['local_rank'])


        # if args['lora']:

        
        # if args['add_token']:
        #     # # ADD TOKENS
        #     self._add_molecule_protein_token()
        #     self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

        # initialize Encoder Projector
        self.enc_proj_dict = nn.ModuleDict({
            'mol_enc': EncoderProjector(args=self.args,
                                           input_modality='molecule',
                                           hidden_size=self.args['projector_hidden_size'],
                                           output_size=self.llama_model.config.hidden_size,
                                           encoder_layer=self.args['projector_enc_layer'],
                                           decoder_layer=self.args['projector_dec_layer'],
                                           num_query_tokens=self.args['enc_num_mol_tokens'],
                                           num_fp=self.args['num_mol_fp']),
            'prot_enc': EncoderProjector(args=self.args,
                                            input_modality='protein',
                                            hidden_size=self.args['projector_hidden_size'],
                                            output_size=self.llama_model.config.hidden_size,
                                            encoder_layer=self.args['projector_enc_layer'],
                                            decoder_layer=self.args['projector_dec_layer'],
                                            num_query_tokens=self.args['enc_num_prot_tokens'],
                                            num_fp=self.args['num_prot_fp'])
        })



        # self.input_embedding = self.llama_model.get_input_embeddings()
        # self.llama_model.model.embed_tokens

    def init_lora(self):
        filter_logging("Lora tuning the LLaMa ...", logger, self.args['local_rank'])
        # add the lora module
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.args['lora_r'],
            lora_alpha=self.args['lora_alpha'],
            lora_dropout=self.args['lora_dropout'],
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        )
        self.llama_model = get_peft_model(self.llama_model, peft_config)

        if self.args['lora']:
            for param in self.enc_proj_dict.parameters():
                param.requires_grad = False

    def prepare_inputs_ids(self, input_batch, inputs):
        """

        """

        begin_special_token = '<MOL>'
        end_special_token = '</MOL>'
        begin_seq_special_token = '<SELFIES>'
        end_seq_special_token = '</SELFIES>'

        p_before_text_list = [self.llama_tokenizer.bos_token + i + begin_special_token for i in inputs['instructions']]
        p_before_inputs_ids = self.llama_tokenizer(p_before_text_list,
                                                    add_special_tokens=False,
                                                    max_length=self.max_length,
                                                    truncation=True).input_ids
        seq_inputs_ids = self.llama_tokenizer(inputs['input_seqs'],
                                                add_special_tokens=False,
                                                max_length=self.max_length,
                                                truncation=True).input_ids
        begin_seq_special_ids = self.llama_tokenizer(
            [end_special_token  + begin_seq_special_token for _ in inputs['input_seqs']],
            add_special_tokens=False).input_ids
        end_seq_special_ids = self.llama_tokenizer([end_seq_special_token for _ in inputs['input_seqs']],
                                                    add_special_tokens=False).input_ids
        # </Special> + <SEQ_Special> + SEQ + </SEQ_Special>
        p_end_inputs_ids = [id1 + id2 + id3 for id1, id2, id3 in zip(begin_seq_special_ids,
                                                                        seq_inputs_ids,
                                                                        end_seq_special_ids)]


        input_enc_fp_list = []
        for fp in inputs['input_enc_fps']:
            fp_tensor = torch.zeros(self.args['num_mol_fp'])
            fp_tensor[fp] = 1
            input_enc_fp_list.append(fp_tensor)
        input_enc_fps = torch.stack(input_enc_fp_list).to(self.llama_model.dtype).to(self.device)
        input_enc_fps = input_enc_fps

        bio_chem_embeds = self.enc_proj_dict['mol_enc'](input_batch, input_enc_fps, self.llama_model.dtype)

        return p_before_inputs_ids, bio_chem_embeds, p_end_inputs_ids

    def prepare_output_target(self, inputs):
        """
        """

        mol_begin_special_token = '<SELFIES>'
        mol_end_special_token = '</SELFIES>'

        # <SELFIES>
        labels_before = self.llama_tokenizer([mol_begin_special_token for _ in range(len(inputs['target_molecule']))],
                                                add_special_tokens=False).input_ids
        # </SELFIES> <EOS>
        labels_end = self.llama_tokenizer([mol_end_special_token + self.llama_tokenizer.eos_token
                                            for _ in range(len(inputs['target_molecule']))],
                                            add_special_tokens=False).input_ids
        #  SELFIES
        mol_labels = self.llama_tokenizer(inputs['target_molecule'],
                                        add_special_tokens=False,
                                        truncation=True,
                                        max_length=self.max_length).input_ids
    

        labels = [b + l + e for b, l, e in zip(labels_before, mol_labels, labels_end)]

        return labels

    def training_multi_modality(self, input_batch, inputs, return_logits=True):
        # input_modality = inputs['input_modality']

        target_labels = self.prepare_output_target(inputs)  # List of List
        attention_mask = []
        labels = []
        inputs_list = []

        p_before_inputs_ids, bio_chem_embeds, p_end_inputs_ids = self.prepare_inputs_ids(input_batch,
                                                                                            inputs)
        p_before_inputs_ids = torch.tensor(p_before_inputs_ids, dtype=torch.long).to(self.device)
        # p_end_inputs_ids = torch.tensor(p_end_inputs_ids, dtype=torch.long).to(self.device)
        if self.args['lora']:
            prefix_inputs_embeds = torch.cat([self.llama_model.model.model.embed_tokens(p_before_inputs_ids),
                                            bio_chem_embeds],
                                            dim=1)
        else:
            prefix_inputs_embeds = torch.cat([self.llama_model.model.embed_tokens(p_before_inputs_ids),
                                            bio_chem_embeds],
                                            dim=1)
        prefix_len = prefix_inputs_embeds.shape[1]
        batch_max_len = max(len(pe) + len(l) for pe, l in zip(p_end_inputs_ids, target_labels))
        label_idx = []  # record the index of label in each input
        for p_end_inputs_id, label in zip(p_end_inputs_ids, target_labels):
            this_len = len(label) + len(p_end_inputs_id)
            this_input = p_end_inputs_id + label + [self.llama_tokenizer.pad_token_id] * (batch_max_len - this_len)
            this_attention_mask = [1] * prefix_len + [1] * this_len + [0] * (batch_max_len - this_len)
            this_label = [-100] * (prefix_len + len(p_end_inputs_id)) + label + [-100] * (batch_max_len - this_len)
            inputs_list.append(this_input)
            attention_mask.append(this_attention_mask)
            labels.append(this_label)
            label_idx.append(prefix_len + len(p_end_inputs_id))
        inputs_ids = torch.tensor(inputs_list, dtype=torch.long).to(self.device)  # [bs, l]
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(self.device)
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        if self.args['lora']:
            inputs_embeds = torch.cat([prefix_inputs_embeds,
                                    self.llama_model.model.model.embed_tokens(inputs_ids)],
                                    dim=1)
        else:
            inputs_embeds = torch.cat([prefix_inputs_embeds,
                                    self.llama_model.model.embed_tokens(inputs_ids)],
                                    dim=1)


        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
        )  # odict_keys(['loss', 'logits', 'past_key_values', 'hidden_states'])
        loss = outputs.loss


        return loss, None



    def forward(self, input_batch, inputs):

        loss, _ = self.training_multi_modality(input_batch, inputs, return_logits=False)
        return loss


    def generate(self, input_batch, inputs,
                 num_return_sequences=1,
                 top_p=0.1,
                 t=1.,
                 num_beams=None,
                 min_new_tokens=None,
                 max_new_tokens=None):
        self.llama_tokenizer.padding_side = "left"

  
        p_before_input_ids, bio_chem_embeds, p_end_input_ids = self.prepare_inputs_ids(input_batch,
                                                                                        inputs)
        max_end_ids = max(len(p) for p in p_end_input_ids)
        p_before_input_ids = torch.tensor(p_before_input_ids, dtype=torch.long).to(self.device)
        input_embed_list = []
        attention_mask_list = []
        for p_before_input_id, bio_chem_embed, p_end_input_id in zip(p_before_input_ids,
                                                                        bio_chem_embeds,
                                                                        p_end_input_ids):
            pad_token_num = max_end_ids - len(p_end_input_id)
            this_pad_tokens = torch.tensor([self.llama_tokenizer.pad_token_id] * pad_token_num,
                                            dtype=torch.long).to(self.device)
            # before PAD + BEFORE

            if self.args['lora']:
                this_before_input_embed = torch.cat([self.llama_model.model.model.embed_tokens(this_pad_tokens),
                                                    self.llama_model.model.model.embed_tokens(p_before_input_id)], dim=0)
                p_end_input_id = torch.tensor(p_end_input_id, dtype=torch.long).to(self.device)
                this_end_input_embed = self.llama_model.model.model.embed_tokens(p_end_input_id)
            else:
                this_before_input_embed = torch.cat([self.llama_model.model.embed_tokens(this_pad_tokens),
                                                    self.llama_model.model.embed_tokens(p_before_input_id)], dim=0)
                p_end_input_id = torch.tensor(p_end_input_id, dtype=torch.long).to(self.device)
                this_end_input_embed = self.llama_model.model.embed_tokens(p_end_input_id)
            this_input_embed = torch.cat([this_before_input_embed, bio_chem_embed, this_end_input_embed], dim=0)
            attention_mask_list.append([0] * pad_token_num + [1] * (this_input_embed.shape[0] - pad_token_num))
            input_embed_list.append(this_input_embed)

        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long).to(self.device)
        inputs_embeds = torch.stack(input_embed_list, dim=0)

        if num_beams is None:
            # do sample
            kwargs = {'top_p': top_p,
                      'temperature': t,
                      'do_sample': True,
                      'min_new_tokens': min_new_tokens}
        else:
            kwargs = {'num_beams': num_beams,
                      'min_new_tokens': min_new_tokens}

        output = self.llama_model.generate(inputs_embeds=inputs_embeds,
                                           attention_mask=attention_mask,
                                           max_new_tokens=self.max_length if max_new_tokens is None else max_new_tokens,
                                           # top_p=top_p,
                                           # top_k=inputs['top_k'],
                                           # temperature=1.0,
                                           # do_sample=True,
                                           num_return_sequences=num_return_sequences,
                                           # num_beams=inputs['num_beams'],
                                           use_cache=True,
                                           # stopping_criteria=stopping_criteria,
                                           # output_hidden_states=True,
                                           return_dict_in_generate=True,
                                           **kwargs)
        output_ids = output.sequences

        generation_result = self.llama_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        generation_result = [g.replace(' ', '') for g in generation_result]
        output_sequences = []


        pattern = re.compile(r'<SELFIES>(.*?)</SELFIES>')
        for g in generation_result:
            matches = re.findall(pattern, g)
            try:
                output_sequence = selfies.decoder(matches[0].replace(' ', ''))
            except:
                output_sequence = ''
            output_sequences.append(output_sequence)


        return output_sequences
