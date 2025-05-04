import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
import shutil
import pickle
import os
from os.path import join
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import warnings

warnings.filterwarnings("ignore")

df_metabolites = pd.read_pickle('data/eval_assist/esp_data/additional_data/all_substrates.pkl')

save_folder = join("data/eval_assist/esp_data", "temp_met", "GNN_input_data")


def metabolite_preprocessing(metabolite_list):
    # removing duplicated entries and creating a pandas DataFrame with all metabolites
    df_met = pd.DataFrame(data={"metabolite": list(set(metabolite_list))})
    df_met["type"], df_met["ID"] = np.nan, np.nan

    # each metabolite should be either a KEGG ID, InChI string, or a SMILES:
    for ind in df_met.index:
        df_met["ID"][ind] = "metabolite_" + str(ind)
        met = df_met["metabolite"][ind]
        if type(met) != str:
            df_met["type"][ind] = "invalid"
            print(
                ".......Metabolite string '%s' could be neither classified as a valid KEGG ID, InChI string or SMILES string." % met)
        elif is_KEGG_ID(met):
            df_met["type"][ind] = "KEGG"
        elif is_InChI(met):
            df_met["type"][ind] = "InChI"
        elif is_SMILES(met):
            df_met["type"][ind] = "SMILES"
        else:
            df_met["type"][ind] = "invalid"
            print(
                ".......Metabolite string '%s' could be neither classified as a valid KEGG ID, InChI string or SMILES string" % met)
    df_met = calculate_atom_and_bond_feature_vectors(df_met)
    N_max = maximal_number_of_atoms(df_met=df_met)
    df_met = calculate_input_matrices(df_met=df_met, N_max=70)
    df_met = get_substrate_representations(df=df_met, N_max=70)
    shutil.rmtree(join("data/eval_assist/esp_data", "temp_met"))
    return (df_met)


def get_representation_input(cid_list):
    XE = ();
    X = ();
    A = ();
    UniRep = ();
    extras = ();
    # Generate data
    for i in range(len(cid_list)):
        cid = cid_list[i]
        X = X + (np.load(join(save_folder, cid + '_X.npy')),);
        XE = XE + (np.load(join(save_folder, cid + '_XE.npy')),);
        A = A + (np.load(join(save_folder, cid + '_A.npy')),);
    return (XE, X, A)


def get_substrate_representations(df, N_max):
    model = GNN(D=100, N=N_max, F1=32, F2=10).to(device)
    model.load_state_dict(torch.load(join("data/eval_assist/esp_data", "GNN", "Pytorch_GNN_with_pretraining")))
    model.eval()
    df["substrate_rep"] = ""

    i = 0
    n = len(df)

    while i * 64 <= n:
        cid_all = list(df["ID"])

        if (i + 1) * 64 <= n:
            XE, X, A = get_representation_input(cid_all[i * 64:(i + 1) * 64])
            XE = torch.tensor(np.array(XE), dtype=torch.float32).to(device)
            X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
            A = torch.tensor(np.array(A), dtype=torch.float32).to(device)
            representations = model.get_GNN_rep(XE, X, A, N_max).cpu().detach().numpy()
            df["substrate_rep"][i * 64:(i + 1) * 64] = list(representations[:, :100])
        else:
            XE, X, A = get_representation_input(cid_all[i * 64:(i + 1) * 64])
            XE = torch.tensor(np.array(XE), dtype=torch.float32).to(device)
            X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
            A = torch.tensor(np.array(A), dtype=torch.float32).to(device)
            representations = model.get_GNN_rep(XE, X, A, N_max).cpu().detach().numpy()
            df["substrate_rep"][-len(representations):] = list(representations[:, :100])
            break
        i += 1
    return (df)


def maximal_number_of_atoms(df_met):
    N_max = np.max(df_met["number_atoms"].loc[df_met["successfull"]]) + 1
    if N_max > 70:
        print(
            ".......The biggest molecule has over 70 atoms (%s). This will slow down the process of calculating the metabolite representations." % N_max)
    return (N_max)


def is_KEGG_ID(met):
    # a valid KEGG ID starts with a "C" or "D" followed by a 5 digit number:
    if len(met) == 6 and met[0] in ["C", "D"]:
        try:
            int(met[1:])
            return (True)
        except:
            pass
    return (False)


def is_SMILES(met):
    m = Chem.MolFromSmiles(met, sanitize=False)
    if m is None:
        return (False)
    else:
        try:
            Chem.SanitizeMol(m)
        except:
            print('.......Metabolite string "%s" is in SMILES format but has invalid chemistry' % met)
            return (False)
    return (True)


def is_InChI(met):
    m = Chem.inchi.MolFromInchi(met, sanitize=False)
    if m is None:
        return (False)
    else:
        try:
            Chem.SanitizeMol(m)
        except:
            print('.......Metabolite string "%s" is in InChI format but has invalid chemistry' % met)
            return (False)
    return (True)


# Create dictionaries for the bond features:
dic_bond_type = {'AROMATIC': np.array([0, 0, 0, 1]), 'DOUBLE': np.array([0, 0, 1, 0]),
                 'SINGLE': np.array([0, 1, 0, 0]), 'TRIPLE': np.array([1, 0, 0, 0])}

dic_conjugated = {0.0: np.array([0]), 1.0: np.array([1])}

dic_inRing = {0.0: np.array([0]), 1.0: np.array([1])}

dic_stereo = {'STEREOANY': np.array([0, 0, 0, 1]), 'STEREOE': np.array([0, 0, 1, 0]),
              'STEREONONE': np.array([0, 1, 0, 0]), 'STEREOZ': np.array([1, 0, 0, 0])}

##Create dictionaries, so the atom features can be easiliy converted into a numpy array

# all the atomic numbers with a total count of over 200 in the data set are getting their own one-hot-encoded
# vector. All the otheres are lumped to a single vector.
dic_atomic_number = {0.0: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 1.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     3.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 4.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     5.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 6.0: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
                     7.0: np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), 8.0: np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
                     9.0: np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), 11.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     12.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 13.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     14.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 15.0: np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
                     16.0: np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]), 17.0: np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
                     19.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 20.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     23.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 24.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     25.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 26.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     27.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 28.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     29.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 30.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     32.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 33.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     34.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 35.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
                     37.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 38.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     42.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 46.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     47.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     48.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 50.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     51.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 52.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     53.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 54.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     56.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 57.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     74.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     78.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 79.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     80.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 81.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     82.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 83.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     86.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 88.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                     90.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 94.0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])}

# There are only 5 atoms in the whole data set with 6 bonds and no atoms with 5 bonds. Therefore I lump 4, 5 and 6 bonds
# together
dic_num_bonds = {0.0: np.array([0, 0, 0, 0, 1]), 1.0: np.array([0, 0, 0, 1, 0]),
                 2.0: np.array([0, 0, 1, 0, 0]), 3.0: np.array([0, 1, 0, 0, 0]),
                 4.0: np.array([1, 0, 0, 0, 0]), 5.0: np.array([1, 0, 0, 0, 0]),
                 6.0: np.array([1, 0, 0, 0, 0])}

# Almost alle charges are -1,0 or 1. Therefore I use only positiv, negative and neutral as features:
dic_charge = {-4.0: np.array([1, 0, 0]), -3.0: np.array([1, 0, 0]), -2.0: np.array([1, 0, 0]),
              -1.0: np.array([1, 0, 0]),
              0.0: np.array([0, 1, 0]), 1.0: np.array([0, 0, 1]), 2.0: np.array([0, 0, 1]), 3.0: np.array([0, 0, 1]),
              4.0: np.array([0, 0, 1]), 5.0: np.array([0, 0, 1]), 6.0: np.array([0, 0, 1])}

dic_hybrid = {'S': np.array([0, 0, 0, 0, 1]), 'SP': np.array([0, 0, 0, 1, 0]), 'SP2': np.array([0, 0, 1, 0, 0]),
              'SP3': np.array([0, 1, 0, 0, 0]), 'SP3D': np.array([1, 0, 0, 0, 0]), 'SP3D2': np.array([1, 0, 0, 0, 0]),
              'UNSPECIFIED': np.array([1, 0, 0, 0, 0])}

dic_aromatic = {0.0: np.array([0]), 1.0: np.array([1])}

dic_H_bonds = {0.0: np.array([0, 0, 0, 1]), 1.0: np.array([0, 0, 1, 0]), 2.0: np.array([0, 1, 0, 0]),
               3.0: np.array([1, 0, 0, 0]), 4.0: np.array([1, 0, 0, 0]), 5.0: np.array([1, 0, 0, 0]),
               6.0: np.array([1, 0, 0, 0])}

dic_chirality = {'CHI_TETRAHEDRAL_CCW': np.array([1, 0, 0]), 'CHI_TETRAHEDRAL_CW': np.array([0, 1, 0]),
                 'CHI_UNSPECIFIED': np.array([0, 0, 1])}


def calculate_atom_and_bond_feature_vectors(df_met):
    df_met["successfull"] = True
    df_met["metabolite_similarity_score"] = np.nan
    df_met["metabolite_identical_ID"] = np.nan
    df_met["#metabolite in training set"] = np.nan
    df_met["number_atoms"] = 0
    df_met["LogP"], df_met["MW"] = np.nan, np.nan
    # Creating a temporary directory to save data for metabolites

    df_count_met = pd.read_csv(join("data/eval_assist/esp_data", "additional_data", "all_training_metabolites.csv"), sep="\t")

    try:
        os.mkdir(join("data/eval_assist/esp_data", "temp_met"))
        os.mkdir(join("data/eval_assist/esp_data", "temp_met", "mol_feature_vectors"))
    except FileExistsError:
        shutil.rmtree(join("data/eval_assist/esp_data", "temp_met"))
        os.mkdir(join("data/eval_assist/esp_data", "temp_met"))
        os.mkdir(join("data/eval_assist/esp_data", "temp_met", "mol_feature_vectors"))

    for ind in df_met.index:
        ID, met_type, met = df_met["ID"][ind], df_met["type"][ind], df_met["metabolite"][ind]
        if met_type == "invalid":
            mol = None
        elif met_type == "KEGG":
            try:
                mol = Chem.MolFromMolFile(join("data/eval_assist/esp_data", "mol-files", met + ".mol"))
            except:
                print(
                    ".......Mol file for KEGG ID '%s' is not available. Try to enter InChI string or SMILES for the metabolite instead." % met)
                mol = None
        elif met_type == "InChI":
            mol = Chem.inchi.MolFromInchi(met)
        elif met_type == "SMILES":
            mol = Chem.MolFromSmiles(met)

        if mol is None:
            df_met["successfull"][ind] = False
        else:
            df_met["number_atoms"][ind] = mol.GetNumAtoms()
            df_met["MW"][ind] = Descriptors.ExactMolWt(mol)
            df_met["LogP"][ind] = Crippen.MolLogP(mol)
            calculate_atom_feature_vector_for_mol(mol, ID)
            calculate_bond_feature_vector_for_mol(mol, ID)
            df_met["metabolite_similarity_score"][ind], df_met["metabolite_identical_ID"][
                ind] = calculate_metabolite_similarity(df_metabolites=df_metabolites,
                                                       mol=mol)
            if not pd.isnull(df_met["metabolite_identical_ID"][ind]):
                df_met["#metabolite in training set"][ind] = \
                list(df_count_met["count"].loc[df_count_met["ID"] == df_met["metabolite_identical_ID"][ind]])[0]
            else:
                df_met["#metabolite in training set"][ind] = 0
    return (df_met)


def calculate_atom_feature_vector_for_mol(mol, mol_ID):
    # get number of atoms N
    N = mol.GetNumAtoms()
    atom_list = []
    for i in range(N):
        features = []
        atom = mol.GetAtomWithIdx(i)
        features.append(atom.GetAtomicNum()), features.append(atom.GetDegree()), features.append(atom.GetFormalCharge())
        features.append(str(atom.GetHybridization())), features.append(atom.GetIsAromatic()), features.append(
            atom.GetMass())
        features.append(atom.GetTotalNumHs()), features.append(str(atom.GetChiralTag()))
        atom_list.append(features)
    with open(join("data/eval_assist/esp_data", "temp_met", "mol_feature_vectors",
                   mol_ID + "-atoms.txt"), "wb") as fp:  # Pickling
        pickle.dump(atom_list, fp)


def calculate_bond_feature_vector_for_mol(mol, mol_ID):
    N = mol.GetNumBonds()
    bond_list = []
    for i in range(N):
        features = []
        bond = mol.GetBondWithIdx(i)
        features.append(bond.GetBeginAtomIdx()), features.append(bond.GetEndAtomIdx()),
        features.append(str(bond.GetBondType())), features.append(bond.GetIsAromatic()),
        features.append(bond.IsInRing()), features.append(str(bond.GetStereo()))
        bond_list.append(features)
    with open(join("data/eval_assist/esp_data", "temp_met", "mol_feature_vectors",
                   mol_ID + "-bonds.txt"), "wb") as fp:  # Pickling
        pickle.dump(bond_list, fp)


def concatenate_X_and_E(X, E, N, F=32 + 10):
    XE = np.zeros((N, N, F))
    for v in range(N):
        x_v = X[v, :]
        for w in range(N):
            XE[v, w, :] = np.concatenate((x_v, E[v, w, :]))
    return (XE)


def calculate_input_matrices(df_met, N_max):
    try:
        os.mkdir(save_folder)
    except:
        pass

    for ind in df_met.index:
        if df_met["successfull"][ind]:
            met_ID = df_met["ID"][ind]
            extras = np.array([df_met["MW"][ind], df_met["LogP"][ind]])
            [XE, X, A] = create_input_data_for_GNN_for_substrates(substrate_ID=met_ID, N_max=N_max, print_error=True)
            if not A is None:
                np.save(join(save_folder, met_ID + '_X.npy'), X)  # feature matrix of atoms/nodes
                np.save(join(save_folder, met_ID + '_XE.npy'), XE)  # feature matrix of atoms/nodes and bonds/edges
                np.save(join(save_folder, met_ID + '_A.npy'), A)
                np.save(join(save_folder, met_ID + '_extras.npy'), extras)
            else:
                df_met["successfull"][ind] = False
    return (df_met)


def create_input_data_for_GNN_for_substrates(substrate_ID, N_max, print_error=False):
    try:
        x = create_atom_feature_matrix(mol_name=substrate_ID, N=N_max)
        if not x is None:
            a, e = create_bond_feature_matrix(mol_name=substrate_ID, N=N_max)
            a = np.reshape(a, (N_max, N_max, 1))
            xe = concatenate_X_and_E(x, e, N=N_max)
            return ([np.array(xe), np.array(x), np.array(a)])
        else:
            if print_error:
                print(".......Could not create input for substrate ID %s" % substrate_ID)
            return (None, None, None)
    except:
        return (None, None, None)


def create_bond_feature_matrix(mol_name, N):
    '''create adjacency matrix A and bond feature matrix/tensor E'''
    try:
        with open(join("data/eval_assist/esp_data", "temp_met", "mol_feature_vectors",
                       mol_name + "-bonds.txt"), "rb") as fp:  # Unpickling
            bond_features = pickle.load(fp)
    except FileNotFoundError:
        return (None)
    A = np.zeros((N, N))
    E = np.zeros((N, N, 10))
    for i in range(len(bond_features)):
        line = bond_features[i]
        start, end = line[0], line[1]
        A[start, end] = 1
        A[end, start] = 1
        e_vw = np.concatenate((dic_bond_type[line[2]], dic_conjugated[line[3]],
                               dic_inRing[line[4]], dic_stereo[line[5]]))
        E[start, end, :] = e_vw
        E[end, start, :] = e_vw
    return (A, E)


def create_atom_feature_matrix(mol_name, N):
    try:
        with open(join("data/eval_assist/esp_data", "temp_met", "mol_feature_vectors",
                       mol_name + "-atoms.txt"), "rb") as fp:  # Unpickling
            atom_features = pickle.load(fp)
    except FileNotFoundError:
        return (None)
    X = np.zeros((N, 32))
    if len(atom_features) >= N:
        print("More than %s (%s) atoms in molcuele %s" % (N, len(atom_features), mol_name))
        return (None)
    for i in range(len(atom_features)):
        line = atom_features[i]
        try:
            atomic_number_mapping = dic_atomic_number[line[0]]
        except KeyError:
            atomic_number_mapping = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        x_v = np.concatenate((atomic_number_mapping, dic_num_bonds[line[1]], dic_charge[line[2]],
                              dic_hybrid[line[3]], dic_aromatic[line[4]], np.array([line[5] / 100.]),
                              dic_H_bonds[line[6]], dic_chirality[line[7]]))
        X[i, :] = x_v
    return (X)


class GNN(nn.Module):
    def __init__(self, D=50, N=70, F1=32, F2=10, droprate=0.2):
        super(GNN, self).__init__()
        self.N = N
        self.F1 = F1
        self.F2 = F2
        F = F1 + F2
        self.F = F
        # first head
        self.Wi = nn.Parameter(torch.empty((1, 1, F, D), requires_grad=True).to(device))
        self.Wm1 = nn.Parameter(torch.empty((1, 1, D, D), requires_grad=True).to(device))
        self.Wm2 = nn.Parameter(torch.empty((1, 1, D, D), requires_grad=True).to(device))
        self.Wa = nn.Parameter(torch.empty((1, D + F1, D), requires_grad=True).to(device))
        nn.init.normal_(self.Wa), nn.init.normal_(self.Wm1), nn.init.normal_(self.Wm2), nn.init.normal_(self.Wi)

        self.OnesN_N = torch.tensor(np.ones((N, N)), dtype=torch.float32, requires_grad=False).to(device)
        self.Ones1_N = torch.tensor(np.ones((1, N)), dtype=torch.float32, requires_grad=False).to(device)
        self.BN1 = nn.BatchNorm2d(D).to(device)
        self.BN2 = nn.BatchNorm2d(D).to(device)

        self.D = D
        # seconda head
        # self.BN2_esm1b = nn.BatchNorm1d(64).to(device)

        self.BN3 = nn.BatchNorm1d(D + 50).to(device)
        self.linear1 = nn.Linear(D + 50, 32).to(device)
        self.linear2 = nn.Linear(32, 1).to(device)

        # dropout_layer
        self.drop_layer = nn.Dropout(p=droprate)

    def forward(self, XE, X, A, ESM1b):
        X = X.view((-1, N, 1, F1))
        H0 = nn.ReLU()(torch.matmul(XE, self.Wi))  # W*XE
        # only get neighbors in each row: (elementwise multiplication)
        M1 = torch.mul(H0, A)
        M1 = torch.transpose(M1, dim0=1, dim1=2)
        M1 = torch.matmul(self.OnesN_N, M1)
        M1 = torch.add(M1, -torch.transpose(H0, dim0=1, dim1=2))
        M1 = torch.mul(M1, A)
        H1 = torch.add(H0, torch.matmul(M1, self.Wm1))
        H1 = torch.transpose(H1, dim0=1, dim1=3)
        H1 = nn.ReLU()(self.BN1(H1))
        H1 = torch.transpose(H1, dim0=1, dim1=3)

        M2 = torch.mul(H1, A)
        M2 = torch.transpose(M2, dim0=1, dim1=2)
        M2 = torch.matmul(self.OnesN_N, M2)
        M2 = torch.add(M2, -torch.transpose(H1, dim0=1, dim1=2))
        M2 = torch.mul(M2, A)
        H2 = torch.add(H0, torch.matmul(M2, self.Wm2))
        H2 = torch.transpose(H2, dim0=1, dim1=3)
        H2 = nn.ReLU()(self.BN2(H2))
        H2 = torch.transpose(H2, dim0=1, dim1=3)

        M_v = torch.mul(H2, A)
        M_v = torch.matmul(self.Ones1_N, M_v)
        XM = torch.cat((X, M_v), 3)
        H = nn.ReLU()(torch.matmul(XM, self.Wa))
        h = torch.matmul(self.Ones1_N, torch.transpose(H, dim0=1, dim1=2))
        h = self.drop_layer(h.view((-1, self.D)))

        h = torch.cat((h, ESM1b), 1)
        h = nn.ReLU()(self.linear1(self.BN3(h)))
        y = nn.Sigmoid()(self.linear2(h))
        return (y)

    def get_GNN_rep(self, XE, X, A, N):
        X = X.view((-1, self.N, 1, self.F1))
        H0 = nn.ReLU()(torch.matmul(XE, self.Wi))  # W*XE
        # only get neighbors in each row: (elementwise multiplication)
        M1 = torch.mul(H0, A)
        M1 = torch.transpose(M1, dim0=1, dim1=2)
        M1 = torch.matmul(self.OnesN_N, M1)
        M1 = torch.add(M1, -torch.transpose(H0, dim0=1, dim1=2))
        M1 = torch.mul(M1, A)
        H1 = torch.add(H0, torch.matmul(M1, self.Wm1))
        H1 = torch.transpose(H1, dim0=1, dim1=3)
        H1 = nn.ReLU()(self.BN1(H1))
        H1 = torch.transpose(H1, dim0=1, dim1=3)

        M2 = torch.mul(H1, A)
        M2 = torch.transpose(M2, dim0=1, dim1=2)
        M2 = torch.matmul(self.OnesN_N, M2)
        M2 = torch.add(M2, -torch.transpose(H1, dim0=1, dim1=2))
        M2 = torch.mul(M2, A)
        H2 = torch.add(H0, torch.matmul(M2, self.Wm2))
        H2 = torch.transpose(H2, dim0=1, dim1=3)
        H2 = nn.ReLU()(self.BN2(H2))
        H2 = torch.transpose(H2, dim0=1, dim1=3)

        M_v = torch.mul(H2, A)
        M_v = torch.matmul(self.Ones1_N, M_v)
        XM = torch.cat((X, M_v), 3)
        H = nn.ReLU()(torch.matmul(XM, self.Wa))
        h = torch.matmul(self.Ones1_N, torch.transpose(H, dim0=1, dim1=2))
        h = h.view((-1, self.D))
        return (h)


def calculate_metabolite_similarity(df_metabolites, mol):
    fp = Chem.RDKFingerprint(mol)

    fps = list(df_metabolites["Sim_FP"])
    IDs = list(df_metabolites["ID"])
    similarity_vector = np.zeros(len(fps))
    for i in range(len(fps)):
        similarity_vector[i] = DataStructs.FingerprintSimilarity(fp, fps[i])
    if max(similarity_vector) == 1:
        k = np.argmax(similarity_vector)
        ID = IDs[k]
    else:
        ID = np.nan
    return (max(similarity_vector), ID)