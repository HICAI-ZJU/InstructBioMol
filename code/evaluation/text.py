from transformers import BertTokenizerFast, BertModel
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import numpy as np
import os
import csv
import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity


def eval_text(generation_list,
                  groundtruth_list,

                  text_trunc_length=512):
    text_tokenizer = BertTokenizerFast.from_pretrained('data/eval_assist/allenai/scibert_scivocab_uncased')

    references = []
    hypotheses = []
    meteor_scores = []
    rouge_scores = []
    # text2mol_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    for (generation, groundtruth) in zip(generation_list, groundtruth_list):
        gt_tokens = text_tokenizer.tokenize(groundtruth, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = text_tokenizer.tokenize(generation, truncation=True, max_length=text_trunc_length,
                                             padding='max_length')
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

        rs = scorer.score(generation, groundtruth)
        rouge_scores.append(rs)



    bleu2 = corpus_bleu(references, hypotheses, weights=(.5, .5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25, .25, .25, .25))
    _meteor_score = np.mean(meteor_scores)
    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])

    return {'BLEU-2': bleu2,
            'BLEU-4': bleu4,
            'ROUGE-1': rouge_1,
            'ROUGE-2': rouge_2,
            'ROUGE-L': rouge_l,
            'METEOR': _meteor_score}

