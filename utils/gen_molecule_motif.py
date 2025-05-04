from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def collect_molecule_motif_idx(smiles):
     mol = Chem.MolFromSmiles(smiles)
     fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useFeatures=True)
     fp = list(fp)
     idx_list =  np.where(list(fp))[0].tolist()
     return idx_list


seq = 'C[C@]12CCC(=O)C=C1CC[C@@H]3[C@@H]2C(=O)C[C@]\\\\4([C@H]3CC/C4=C/C(=O)OC)C'
ids = collect_molecule_motif_idx(seq)
print(ids)