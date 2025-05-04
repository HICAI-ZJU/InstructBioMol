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


class UniModel4Text2ProtMol(nn.Module):
    def __init__(self, args):
        super(UniModel4Text2ProtMol, self).__init__()
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
        # if input_modality == 'text':
            # bos + instructions + descriptions -> token id -> inputs_embeds
        inputs_text_list = [self.llama_tokenizer.bos_token + i + d for (i, d) in
                            zip(inputs['instructions'], inputs['input_seqs'])]
        token_result = self.llama_tokenizer(inputs_text_list,
                                            truncation=True,
                                            max_length=self.max_length,
                                            add_special_tokens=False)  # List[List]
        input_ids = token_result.input_ids
        return input_ids

    def prepare_output_target(self, inputs):
        """

        :param inputs: {'input_seqs': input_seqs,
                        'target_seqs': target_seqs,
                        'input_enc_seqs': input_enc_seqs,
                        'input_modality': input_modality,
                        'target_modality': target_modality,
                        'instructions': instructions,
                        'data_name': data_name}
        :return: labels, attention_mask
        """

        mol_begin_special_token = '<SELFIES>'
        mol_end_special_token = '</SELFIES>'
        # elif target_modality == 'protein':
        prot_begin_special_token = '<FASTA>'
        prot_end_special_token = '</FASTA>'
        inputs['target_protein'] = [''.join([f'<p>{a}' for a in seq]) for seq in inputs['target_protein']]
        # else:
        #     raise NotImplementedError

        # <FASTA>
        prot_labels_before = self.llama_tokenizer([prot_begin_special_token for _ in range(len(inputs['target_protein']))],
                                                add_special_tokens=False).input_ids
        # </FASTA>
        prot_labels_end = self.llama_tokenizer([prot_end_special_token
                                            for _ in range(len(inputs['target_protein']))],
                                            add_special_tokens=False).input_ids
        #  PROTEIN
        prot_labels = self.llama_tokenizer(inputs['target_protein'],
                                        add_special_tokens=False,
                                        truncation=True,
                                        max_length=self.max_length).input_ids

        # <SELFIES>
        mol_labels_before = self.llama_tokenizer([mol_begin_special_token for _ in range(len(inputs['target_molecule']))],
                                                add_special_tokens=False).input_ids
        # </SELFIES> <EOS>
        mol_labels_end = self.llama_tokenizer([mol_end_special_token + self.llama_tokenizer.eos_token
                                            for _ in range(len(inputs['target_molecule']))],
                                            add_special_tokens=False).input_ids
        #  SELFIES
        mol_labels = self.llama_tokenizer(inputs['target_molecule'],
                                        add_special_tokens=False,
                                        truncation=True,
                                        max_length=self.max_length).input_ids
    

        # labels = [b + l + e for b, l, e in zip(labels_before, labels, labels_end)]
        labels = [bp + lp + ep + bm + lm + em for bp, lp, ep, bm, lm, em in zip(prot_labels_before,
                                                                                prot_labels,
                                                                                prot_labels_end,
                                                                                mol_labels_before,
                                                                                mol_labels,
                                                                                mol_labels_end)]
        return labels

    def training_multi_modality(self, input_batch, inputs, return_logits=True):
        # input_modality = inputs['input_modality']

        target_labels = self.prepare_output_target(inputs)  # List of List
        attention_mask = []
        labels = []
        inputs_list = []


        # if input_modality == 'text':
        input_ids = self.prepare_inputs_ids(input_batch, inputs)
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
        if self.args['lora']:
            inputs_embeds = self.llama_model.model.model.embed_tokens(inputs_ids)  # [bs, l, d]
        else:
            inputs_embeds = self.llama_model.model.embed_tokens(inputs_ids)  # [bs, l, d]


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
        # input_modality = inputs['input_modality']

 
        inputs_text_list = [i + d for (i, d) in zip(inputs['instructions'], inputs['input_seqs'])]
        token_result = self.llama_tokenizer(inputs_text_list,
                                            truncation=True,
                                            padding=True,
                                            max_length=self.max_length,
                                            add_special_tokens=True,
                                            return_tensors='pt')
        input_ids = token_result.input_ids.to(self.device)
        attention_mask = token_result.attention_mask.to(self.device)
        if self.args['lora']:
            inputs_embeds = self.llama_model.model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = self.llama_model.model.embed_tokens(input_ids)


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

        fasta_pattern = r"<FASTA>(.*?)</FASTA>"
        selfies_pattern = r"<SELFIES>(.*?)</SELFIES>"


        for gen in generation_result:
            fasta_match = re.search(fasta_pattern, gen)
            selfies_match = re.search(selfies_pattern, gen)

            if fasta_match:
                protein = fasta_match.group(1)
                protein = protein.replace('<p>', '').replace(' ', '')
            else:
                protein = ''

            if selfies_match:
                molecule = selfies_match.group(1)
                try:
                    molecule = selfies.decoder(molecule.replace(' ', ''))
                except:
                    molecule = ''
            else:
                molecule = ''
            # import pdb; pdb.set_trace()
            output_sequences.append((protein, molecule))

        return output_sequences
