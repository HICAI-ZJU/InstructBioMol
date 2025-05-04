import os
from rdkit import Chem
from typing import List, Dict, Any, Tuple
import torch


def load_sdf_data(sdf_file):
    sdf_supplier = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=False)  
    mol = sdf_supplier[0]  # 

    if mol is not None:
        z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        coords = mol.GetConformer().GetPositions()
        z = [x if x < 100 else 0 for x in z]
    else:
        z = [1]
        coords = [[0, 0, 0]]
    z, coords = torch.tensor(z), torch.tensor(coords)
    return z, coords


def pad_feats(feats: torch.Tensor, max_node: int) -> torch.Tensor:
    N, *_ = feats.shape
    if N > max_node:
        feats_padded = feats[:max_node]
    else:
        feats_padded = torch.zeros([max_node, *_], dtype=feats.dtype)
        feats_padded[:N] = feats
    return feats_padded


def collate_sdf(features: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    MAX_NODE = 32
    max_node = max(feat["z"].shape[0] for feat in features)
    max_node = min(max_node, MAX_NODE)

    z = torch.stack(
        [pad_feats(feat["z"], max_node) for feat in features]
    )
    pos = torch.stack(
        [pad_feats(feat["pos"], max_node) for feat in features]
    )

    return z, pos
