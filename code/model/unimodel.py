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

from .encoder_projector import EncoderProjector

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from exp_utils import filter_logging

logger = logging.getLogger(__name__)


class UniModel(nn.Module):
    def __init__(self, args):
        super(UniModel, self).__init__()
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


    def prepare_inputs_ids(self, input_batch, inputs, input_modality):
        """
        :param input_batch
        :param inputs: {'input_seqs': input_seqs,
                        'target_seqs': target_seqs,
                        'input_enc_seqs': input_enc_seqs,
                        'input_modality': input_modality,
                        'target_modality': target_modality,
                        'instructions': instructions,
                        'data_name': data_name}
        :param input_modality
        :return:
        """
        if input_modality == 'text':
            # bos + instructions + descriptions -> token id -> inputs_embeds
            inputs_text_list = [self.llama_tokenizer.bos_token + i + d for (i, d) in
                                zip(inputs['instructions'], inputs['input_seqs'])]
            token_result = self.llama_tokenizer(inputs_text_list,
                                                truncation=True,
                                                max_length=self.max_length,
                                                add_special_tokens=False)  # List[List]
            input_ids = token_result.input_ids
            return input_ids
        else:
            if input_modality == 'molecule':
                begin_special_token = '<MOL>'
                end_special_token = '</MOL>'
                begin_seq_special_token = '<SELFIES>'
                end_seq_special_token = '</SELFIES>'
            elif input_modality == 'protein':
                begin_special_token = '<PROT>'
                end_special_token = '</PROT>'
                begin_seq_special_token = '<FASTA>'
                end_seq_special_token = '</FASTA>'
                inputs['input_seqs'] = [''.join([f'<p>{a}' for a in seq]) for seq in inputs['input_seqs']]
            else:
                raise NotImplementedError
            # <bos> + Instruction + <Special>
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

            if input_modality == 'molecule':
                input_enc_fp_list = []
                for fp in inputs['input_enc_fps']:
                    fp_tensor = torch.zeros(self.args['num_mol_fp'])
                    fp_tensor[fp] = 1
                    input_enc_fp_list.append(fp_tensor)
                input_enc_fps = torch.stack(input_enc_fp_list).to(self.llama_model.dtype).to(self.device)
                input_enc_fps = input_enc_fps
                bio_chem_embeds = self.enc_proj_dict['mol_enc'](input_batch, input_enc_fps, self.llama_model.dtype)
            elif input_modality == 'protein':
                input_enc_fp_list = []
                for fp in inputs['input_enc_fps']:
                    fp_tensor = torch.zeros(self.args['num_prot_fp'])
                    fp_tensor[fp] = 1
                    input_enc_fp_list.append(fp_tensor)
                input_enc_fps = torch.stack(input_enc_fp_list).to(self.llama_model.dtype).to(self.device)
                bio_chem_embeds = self.enc_proj_dict['prot_enc'](input_batch, input_enc_fps, self.llama_model.dtype)
            else:
                raise NotImplementedError

            return p_before_inputs_ids, bio_chem_embeds, p_end_inputs_ids

    def prepare_output_target(self, inputs):
        """

        :param inputs: {'input_seqs': input_seqs,
                        'target_seqs': target_seqs,
                        'input_enc_seqs': input_enc_seqs,
                        'input_modality': input_modality,
                        'target_modality': target_modality,
                        'instructions': instructions,
                        'data_name': data_name}
        :return: labels, attention_mask
        """
        target_modality = inputs['target_modality']
        if target_modality == 'text':
            targets = self.llama_tokenizer([d + self.llama_tokenizer.eos_token for d in inputs['target_seqs']],
                                           truncation=True,
                                           max_length=self.max_length,
                                           add_special_tokens=False)
            labels = targets.input_ids
        else:
            if target_modality == 'molecule':
                begin_special_token = '<SELFIES>'
                end_special_token = '</SELFIES>'
            elif target_modality == 'protein':
                begin_special_token = '<FASTA>'
                end_special_token = '</FASTA>'
                inputs['target_seqs'] = [''.join([f'<p>{a}' for a in seq]) for seq in inputs['target_seqs']]
            else:
                raise NotImplementedError

            # <Special>
            labels_before = self.llama_tokenizer([begin_special_token for _ in range(len(inputs['target_seqs']))],
                                                 add_special_tokens=False).input_ids
            # </Special> <EOS>
            labels_end = self.llama_tokenizer([end_special_token + self.llama_tokenizer.eos_token
                                               for _ in range(len(inputs['target_seqs']))],
                                              add_special_tokens=False).input_ids
            # target-seqs
            labels = self.llama_tokenizer(inputs['target_seqs'],
                                          add_special_tokens=False,
                                          truncation=True,
                                          max_length=self.max_length).input_ids
            labels = [b + l + e for b, l, e in zip(labels_before, labels, labels_end)]
        return labels

    def training_multi_modality(self, input_batch, inputs):
        input_modality = inputs['input_modality']

        target_labels = self.prepare_output_target(inputs)  # List of List
        attention_mask = []
        labels = []
        inputs_list = []

        if self.args['debug'] and self.training:
            filter_logging(f"============================{inputs['data_name']}============================", logger,
                           self.args['local_rank'])
            filter_logging(f"TARGET-IDS {target_labels}", logger, self.args['local_rank'])
            filter_logging(f"TARGET {self.llama_tokenizer.batch_decode(target_labels, skip_special_tokens=False)}",
                           logger, self.args['local_rank'])

        if input_modality == 'text':
            input_ids = self.prepare_inputs_ids(input_batch, inputs, input_modality)
            batch_max_len = max([len(i + l) for i, l in zip(input_ids, target_labels)])
            for input_id, label in zip(input_ids, target_labels):
                this_len = len(input_id) + len(label)
                this_input = input_id + label + [self.llama_tokenizer.pad_token_id] * (batch_max_len - this_len)
                this_attention_mask = [1] * this_len + [0] * (batch_max_len - this_len)
                this_label = [-100] * len(input_id) + label + [-100] * (batch_max_len - this_len)
                inputs_list.append(this_input)
                attention_mask.append(this_attention_mask)
                labels.append(this_label)
            inputs_ids = torch.tensor(inputs_list, dtype=torch.long).to(self.device)  # [bs, l]
            attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(self.device)
            labels = torch.tensor(labels, dtype=torch.long).to(self.device)
            inputs_embeds = self.llama_model.model.embed_tokens(inputs_ids)  # [bs, l, d]

            if self.args['debug'] and self.training:
                filter_logging(f"INPUTS-IDS {inputs_ids.tolist()}", logger, self.args['local_rank'])
                filter_logging(f"INPUTS {self.llama_tokenizer.batch_decode(inputs_ids, skip_special_tokens=False)}",
                               logger, self.args['local_rank'])
                filter_logging(f"LABELS {labels.tolist()}", logger, self.args['local_rank'])
                filter_logging(f"ATTENTION-MASK {attention_mask.tolist()}", logger, self.args['local_rank'])
        else:
            p_before_inputs_ids, bio_chem_embeds, p_end_inputs_ids = self.prepare_inputs_ids(input_batch,
                                                                                             inputs,
                                                                                             input_modality)
            p_before_inputs_ids = torch.tensor(p_before_inputs_ids, dtype=torch.long).to(self.device)

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
            inputs_embeds = torch.cat([prefix_inputs_embeds,
                                       self.llama_model.model.embed_tokens(inputs_ids)],
                                      dim=1)

            if self.args['debug'] and self.training:
                filter_logging(f"P-BEFORE-INPUTS-IDS {p_before_inputs_ids.tolist()}", logger, self.args['local_rank'])
                filter_logging(
                    f"P-BEFORE-LABEL-INPUTS {self.llama_tokenizer.batch_decode(p_before_inputs_ids, skip_special_tokens=False)}",
                    logger, self.args['local_rank'])
                filter_logging(f"P-END-INPUTS-IDS {p_end_inputs_ids}", logger, self.args['local_rank'])
                filter_logging(
                    f"P-END-INPUTS {self.llama_tokenizer.batch_decode(p_end_inputs_ids, skip_special_tokens=False)}",
                    logger, self.args['local_rank'])
                filter_logging(f"PEND-LABEL-INPUTS-IDS {inputs_ids.tolist()}", logger, self.args['local_rank'])
                filter_logging(
                    f"PEND-LABEL-INPUTS {self.llama_tokenizer.batch_decode(inputs_ids, skip_special_tokens=False)}",
                    logger, self.args['local_rank'])
                filter_logging(f"LABELS {labels.tolist()}", logger, self.args['local_rank'])
                filter_logging(f"ATTENTION-MASK {attention_mask.tolist()}", logger, self.args['local_rank'])

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
        )  # odict_keys(['loss', 'logits', 'past_key_values', 'hidden_states'])
        loss = outputs.loss

        return loss


    def forward(self, input_batch, inputs):
        loss = self.training_multi_modality(input_batch, inputs)
        return loss



    def generate(self, input_batch, inputs, target_modality,
                 num_return_sequences=1,
                 top_p=0.1,
                 t=1.,
                 num_beams=None,
                 min_new_tokens=None,
                 max_new_tokens=None):
        self.llama_tokenizer.padding_side = "left"
        input_modality = inputs['input_modality']

        if input_modality == 'text':
            inputs_text_list = [i + d for (i, d) in zip(inputs['instructions'], inputs['input_seqs'])]
            token_result = self.llama_tokenizer(inputs_text_list,
                                                truncation=True,
                                                padding=True,
                                                max_length=self.max_length,
                                                add_special_tokens=True,
                                                return_tensors='pt')
            input_ids = token_result.input_ids.to(self.device)
            attention_mask = token_result.attention_mask.to(self.device)
            inputs_embeds = self.llama_model.model.embed_tokens(input_ids)

        else:
            p_before_input_ids, bio_chem_embeds, p_end_input_ids = self.prepare_inputs_ids(input_batch,
                                                                                           inputs,
                                                                                           input_modality)
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
                                           num_return_sequences=num_return_sequences,
                                           use_cache=True,
                                           return_dict_in_generate=True,
                                           **kwargs)
        output_ids = output.sequences

        generation_result = self.llama_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if self.args['debug']:
            filter_logging(f"{generation_result}", logger, self.args['local_rank'])

        if target_modality == 'text':
            output_sequences = [g.replace('"', '') for g in generation_result]
        else:
            generation_result = [g.replace(' ', '') for g in generation_result]
            output_sequences = []
            if target_modality == 'molecule':
                pattern = re.compile(r'<SELFIES>(.*?)</SELFIES>')
                for g in generation_result:
                    matches = re.findall(pattern, g)
                    try:
                        output_sequence = selfies.decoder(matches[0].replace(' ', ''))
                    except:
                        output_sequence = ''
                    output_sequences.append(output_sequence)

            elif target_modality == 'protein':
                for g in generation_result:
                    pattern = re.compile(r'<FASTA>(.*?)</FASTA>')
                    g = g.replace('<p>', '')
                    matches = re.findall(pattern, g)
                    if len(matches) == 0:
                        # output_sequence = ''
                        pattern = re.compile(r'<FASTA>(.*)')
                        matches = re.findall(pattern, g)
                        if len(matches) == 0:
                            output_sequence = ''
                        else:
                            output_sequence = matches[0].replace(' ', '')
                    else:
                        output_sequence = matches[0].replace(' ', '')
                    output_sequences.append(output_sequence)
            else:
                raise NotImplementedError

        return output_sequences
