import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from Levenshtein import distance as lev
from fcd import get_fcd, load_ref_model, canonical_smiles
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger

from rdkit.Chem.QED import qed
from moses.metrics.SA_Score import sascorer

RDLogger.DisableLog('rdApp.*')


def eval_mol(generation_list,
             groundtruth_list):
    bleu_references = []
    bleu_hypotheses = []
    levs = []
    num_exact = 0
    bad_mols = 0
    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []

    fcd_model = load_ref_model()
    canon_gt_smis = []
    canon_ot_smis = []
    for (generation, groundtruth) in zip(generation_list, groundtruth_list):
        if generation == '':
            bad_mols += 1
            continue

        try:
            generation = Chem.MolToSmiles(Chem.MolFromSmiles(generation))
            groundtruth = Chem.MolToSmiles(Chem.MolFromSmiles(groundtruth))

            gt_tokens = [c for c in groundtruth]
            out_tokens = [c for c in generation]

            bleu_references.append([gt_tokens])
            bleu_hypotheses.append(out_tokens)

            m_out = Chem.MolFromSmiles(generation)
            m_gt = Chem.MolFromSmiles(groundtruth)

            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt):
                num_exact += 1

            MACCS_sims.append(
                DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(m_gt), MACCSkeys.GenMACCSKeys(m_out),
                                                  metric=DataStructs.TanimotoSimilarity))
            RDK_sims.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(m_gt), Chem.RDKFingerprint(m_out),
                                                              metric=DataStructs.TanimotoSimilarity))
            morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(m_gt, 2),
                                                              AllChem.GetMorganFingerprint(m_out, 2)))
        except Exception as e:
            bad_mols += 1

        levs.append(lev(generation, groundtruth))
        canon_gt_smis.append(groundtruth)
        canon_ot_smis.append(generation)

    canon_gt_smis = [w for w in canonical_smiles(canon_gt_smis) if w is not None]
    canon_ot_smis = [w for w in canonical_smiles(canon_ot_smis) if w is not None]

    fcd_sim_score = get_fcd(canon_gt_smis, canon_ot_smis, fcd_model)

    bleu_score = corpus_bleu(bleu_references, bleu_hypotheses)
    exact_score = num_exact / len(generation_list)
    levenshtein_score = np.mean(levs)
    validity_score = 1 - bad_mols / len(generation_list)
    maccs_sims_score = np.mean(MACCS_sims)
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)

    return {'BLEU': bleu_score,
            'EXACT': exact_score,
            'LEVENSHTEIN': levenshtein_score,
            'MACCS_FTS': maccs_sims_score,
            'RDK_FTS': rdk_sims_score,
            'MORGAN_FTS': morgan_sims_score,
            'FCD': fcd_sim_score,
            'VALIDITY': validity_score}

