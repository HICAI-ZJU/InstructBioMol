from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures
import torch

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def process_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    Chem.Kekulize(mol)

    xs = []
    for atom in mol.GetAtoms():
        x = []
        x.append(allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum()))
        x.append(allowable_features['possible_chirality_list'].index(atom.GetChiralTag()))
        xs.append(x)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 2)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(allowable_features['possible_bonds'].index(bond.GetBondType()))
        e.append(allowable_features['possible_bond_dirs'].index(bond.GetBondDir()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 2)

    # Sort indices.
    if edge_index.numel() > 0:
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return x, edge_index, edge_attr