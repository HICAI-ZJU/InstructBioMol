from Bio import pairwise2
from Bio.Seq import Seq
from Bio.Align import substitution_matrices
from nltk.metrics import edit_distance
from Levenshtein import distance as lev

import numpy as np
import multiprocessing as mp
from itertools import product
import concurrent.futures


def all_characters_are_amino_acids(s):
    if s == '':
        return False

    amino_acids = [
        "A", "C", "D", "E", "F",
        "G", "H", "I", "K", "L",
        "M", "N", "P", "Q", "R",
        "S", "T", "V", "W", "Y"
    ]
    return all(char in amino_acids for char in s)


def percentage_identity(seq1, seq2):
    # Assuming seq1 and seq2 are strings representing protein sequences
    length = min(len(seq1), len(seq2))  # Choose the minimum length
    identical_residues = sum(a == b for a, b in zip(seq1[:length], seq2[:length]))

    if length == 0:
        return 0  # Avoid division by zero if both sequences are empty

    identity = 2 * identical_residues / (len(seq1) + len(seq2))
    return identity


def similarity_matrix_score(seq1, seq2):
    substitution_matrix = substitution_matrices.load('BLOSUM45')
    score = sum(substitution_matrix.get((a, b), substitution_matrix.get((b, a))) for a, b in zip(seq1, seq2))
    score = 2 * score / (len(seq1) + len(seq2))
    return score


def alignment_similarity(seq1, seq2):
    alignments = pairwise2.align.localxx(seq1, seq2, score_only=True)
    # similarity = alignments / len(seq2)
    similarity = (alignments * 2) / (len(seq1) + len(seq2))
    return similarity



def process_pair(generation, groundtruth):
    result = {
        'identity': None,
        'align': None,
        'matrix_score': None
    }

    if not all_characters_are_amino_acids(generation):
        generation = ''

    if generation == '':
        return None

    if not all_characters_are_amino_acids(groundtruth):
        groundtruth = ''
    if groundtruth == '':
        return None

    result['identity'] = percentage_identity(generation, groundtruth)
    result['align'] = alignment_similarity(generation, groundtruth)
    result['matrix_score'] = similarity_matrix_score(generation, groundtruth)

    return result


def eval_protein(generation_list,
                 groundtruth_list,
                 cpu=8):
    bad_num = 0
    identity_list = []
    align_list = []
    matrix_score_list = []


    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu) as executor:
        futures = [executor.submit(process_pair, gen, gt) for gen, gt in zip(generation_list, groundtruth_list)]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is None:
                bad_num += 1
            else:
                identity_list.append(result['identity'])
                align_list.append(result['align'])
                matrix_score_list.append(result['matrix_score'])
    return {'IDENTITY': np.mean(identity_list),
            'BLOSUM': np.mean(matrix_score_list),
            'ALIGN': np.mean(align_list),
            'VALIDITY': 1 - bad_num / len(generation_list)}


def eval_protein_set(generation_list,
                     groundtruth_list,
                     cpu=8):
    #
    bad_num = 0

    valid_generation_list = list()
    for gen in generation_list:
        if not all_characters_are_amino_acids(gen):
            gen = ''
        if gen == '':
            bad_num += 1
            continue
        valid_generation_list.append(gen)


    pairs = list(product(groundtruth_list, valid_generation_list))

    with mp.Pool(cpu) as pool:
        identity_results = pool.starmap(percentage_identity, pairs)
        align_results = pool.starmap(alignment_similarity, pairs)
        matrix_results = pool.starmap(similarity_matrix_score, pairs)

    identity = max(identity_results)
    align = max(align_results)
    matrix_score = max(matrix_results)

    return {'IDENTITY': identity,
            'ALIGN': align,
            'BLOSUM': matrix_score,
            'VALIDITY': 1 - bad_num / len(generation_list)}

