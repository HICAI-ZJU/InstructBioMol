import os
from subprocess import check_output, CalledProcessError
import shutil
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from tqdm import tqdm
import numpy as np

from vina import Vina
import AutoDockTools
import contextlib
from meeko import MoleculePreparation
from meeko import obutils
import subprocess
from openbabel import pybel

def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)

    return wrapper


def convert_pdbqt_to_sdf(pdbqt_file, sdf_file):
    mol = next(pybel.readfile("pdbqt", pdbqt_file))
    mol.removeh()
    mol.write("sdf", sdf_file, overwrite=True)


class PrepProt(object):
    def __init__(self, pdb_file):
        self.prot = pdb_file

    def del_water(self, dry_pdb_file):  # optional
        with open(self.prot) as f:
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HETATM')]
            dry_lines = [l for l in lines if not 'HOH' in l]

        with open(dry_pdb_file, 'w') as f:
            f.write(''.join(dry_lines))
        self.prot = dry_pdb_file

    def addH(self, prot_pqr):  # call pdb2pqr
        self.prot_pqr = prot_pqr
        subprocess.Popen(['pdb2pqr30', '--ff=AMBER', self.prot, self.prot_pqr],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()

    def get_pdbqt(self, prot_pdbqt):
        prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
        subprocess.Popen(['python3', prepare_receptor, '-r', self.prot_pqr, '-o', prot_pdbqt],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()


class PrepLig(object):
    def __init__(self, input_mol, mol_format):
        if mol_format == 'smi':
            self.ob_mol = pybel.readstring('smi', input_mol)
        elif mol_format == 'sdf':
            self.ob_mol = next(pybel.readfile(mol_format, input_mol))
        else:
            raise ValueError(f'mol_format {mol_format} not supported')

    def addH(self, path, polaronly=False, correctforph=True, PH=7):
        self.ob_mol.OBMol.AddHydrogens(polaronly, correctforph, PH)
        obutils.writeMolecule(self.ob_mol.OBMol, f'{path}/tmp_h.sdf')

    def gen_conf(self):
        sdf_block = self.ob_mol.write('sdf')
        rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
        AllChem.EmbedMolecule(rdkit_mol, Chem.rdDistGeom.ETKDGv3())
        self.ob_mol = pybel.readstring('sdf', Chem.MolToMolBlock(rdkit_mol))
        obutils.writeMolecule(self.ob_mol.OBMol, 'conf_h.sdf')

    @supress_stdout
    def get_pdbqt(self, lig_pdbqt=None):
        preparator = MoleculePreparation()
        preparator.prepare(self.ob_mol.OBMol)
        if lig_pdbqt is not None:
            preparator.write_pdbqt_file(lig_pdbqt)
            return
        else:
            return preparator.write_pdbqt_string()



def calculate_qvina(receptor_file, ligand_file, output_path, ref_ligand_file, num_cpu=16):
    try:
        mol = Chem.MolFromMolFile(ref_ligand_file, sanitize=False)

        pos = mol.GetConformer(0).GetPositions()
        center = np.mean(pos, 0)


        os.makedirs(output_path, exist_ok=True)
        ligand_pdbqt = os.path.join(output_path, 'ligand.pdbqt')
        protein_pqr = os.path.join(output_path, 'receptor.pqr')
        protein_pdbqt = os.path.join(output_path, 'receptor.pdbqt')


        lig = PrepLig(ligand_file, 'sdf')
        lig.addH(output_path)
        lig.get_pdbqt(ligand_pdbqt)

        prot = PrepProt(receptor_file)
        prot.addH(protein_pqr)
        prot.get_pdbqt(protein_pdbqt)

        command = ['qvina2',
               '--cpu', str(num_cpu),
               '--seed', '0',
               '--center_x', str(center[0]),
               '--center_y', str(center[1]),
               '--center_z', str(center[2]),
               '--size_x', '30',
               '--size_y', '30',
               '--size_z', '30',
               '--receptor', protein_pdbqt,
               '--ligand', ligand_pdbqt]
        score = None
        stream = check_output(command, universal_newlines=True)
        shutil.copy2(receptor_file, os.path.join(output_path, 'receptor.pdb'))
        with open(os.path.join(output_path, 'ligand_out.pdbqt'), 'r') as f:
            for line in f.readlines():
                if line.startswith("REMARK VINA RESULT:"):
                    score = float(line.split()[3])
                    break
        return score
    except:
        return None

