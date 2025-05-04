class DatasetCatalog:
    def __init__(self):
        self.molecule_to_text_pubchem_iupac = {
            'data_path': 'data/molecule-text/PubChem-IUPAC-LMDB-demo',
            'data_format': 'LMDB',
            'id_key': 'CID',
            'input_modality': 'molecule',
            'target_modality': 'text',
            'input_seq_key': 'selfies',
            'target_seq_key': 'iupac',
            'input_enc_seq_key': 'canonical_smiles',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': 'data/sdf/pubchem',
            'conf_key': 'SDF_file',
            'instruction': 'Give the IUPAC name of the following molecule.'
        }
        self.text_to_molecule_pubchem_iupac = {
            'data_path': 'data/molecule-text/PubChem-IUPAC-LMDB-demo',
            'data_format': 'LMDB',
            'id_key': 'CID',
            'input_modality': 'text',
            'target_modality': 'molecule',
            'input_seq_key': 'description',
            'target_seq_key': 'selfies',
            'input_enc_seq_key': 'canonical_smiles',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': 'data/sdf/pubchem',
            'conf_key': 'SDF_file',
            'instruction': 'Generate a molecule in SELFIES that fits the provided IUPAC name.',
        }

        self.molecule_to_text_chebi_train = {
            'data_path': 'data/molecule-text/ChEBI-train.json',
            'data_format': 'JSON',
            'id_key': 'CID',
            'input_modality': 'molecule',
            'target_modality': 'text',
            'input_seq_key': 'selfies',
            'target_seq_key': 'description',
            'input_enc_seq_key': 'canonical_smiles',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': 'data/sdf/chebi',
            'conf_key': 'SDF_file',
            'instruction': 'Provide a caption for the molecule below.'
        }


        self.molecule_to_text_chebi_test = {
            'data_path': 'data/molecule-text/ChEBI-test.json',
            'data_format': 'JSON',
            'id_key': 'CID',
            'input_modality': 'molecule',
            'target_modality': 'text',
            'input_seq_key': 'selfies',
            'target_seq_key': 'description',
            'input_enc_seq_key': 'canonical_smiles',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': 'data/sdf/chebi',
            'conf_key': 'SDF_file',
            'instruction': 'Provide a caption for the molecule below.'
        }



        self.text_to_molecule_chebi_train = {
            'data_path': 'data/molecule-text/ChEBI-train.json',
            'data_format': 'JSON',
            'id_key': 'CID',
            'input_modality': 'text',
            'target_modality': 'molecule',
            'input_seq_key': 'description',
            'target_seq_key': 'selfies',
            'input_enc_seq_key': 'canonical_smiles',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': 'data/sdf/chebi',
            'conf_key': 'SDF_file',
            'instruction': 'Generate a molecule in SELFIES that fits the provided description.',
        }


        self.text_to_molecule_chebi_test = {
            'data_path': 'data/molecule-text/ChEBI-test.json',
            'data_format': 'JSON',
            'id_key': 'CID',
            'input_modality': 'text',
            'target_modality': 'molecule',
            'input_seq_key': 'description',
            'target_seq_key': 'selfies',
            'input_enc_seq_key': 'canonical_smiles',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': 'data/sdf/chebi',
            'conf_key': 'SDF_file',
            'instruction': 'Generate a molecule in SELFIES that fits the provided description.',
        }


        self.protein_to_text_trembl_name = {
            'data_path': 'data/protein-text/TrEMBL-LMDB-name-uniref50',
            'data_format': 'LMDB',
            'id_key': 'protein_id',
            'input_modality': 'protein',
            'target_modality': 'text',
            'input_seq_key': 'seq',
            'target_seq_key': 'desc',
            'input_enc_seq_key': 'seq',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': '',
            'conf_key': 'saprot_seq',
            'instruction': 'What is the official name of this protein?'
        }

        self.protein_to_text_trembl_func = {
            'data_path': 'data/protein-text/TrEMBL-LMDB-func-uniref50',
            'data_format': 'LMDB',
            'id_key': 'protein_id',
            'input_modality': 'protein',
            'target_modality': 'text',
            'input_seq_key': 'seq',
            'target_seq_key': 'desc',
            'input_enc_seq_key': 'seq',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': '',
            'conf_key': 'saprot_seq',
            'instruction': 'What is the function of this protein?'
        }

        self.protein_to_text_trembl_loc = {
            'data_path': 'data/protein-text/TrEMBL-LMDB-loc-uniref50',
            'data_format': 'LMDB',
            'id_key': 'protein_id',
            'input_modality': 'protein',
            'target_modality': 'text',
            'input_seq_key': 'seq',
            'target_seq_key': 'desc',
            'input_enc_seq_key': 'seq',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': '',
            'conf_key': 'saprot_seq',
            'instruction': 'What is the subcellular location of this protein?'
        }

        self.protein_to_text_trembl_family = {
            'data_path': 'data/protein-text/TrEMBL-LMDB-family-uniref50',
            'data_format': 'LMDB',
            'id_key': 'protein_id',
            'input_modality': 'protein',
            'target_modality': 'text',
            'input_seq_key': 'seq',
            'target_seq_key': 'desc',
            'input_enc_seq_key': 'seq',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': '',
            'conf_key': 'saprot_seq',
            'instruction': 'What is the protein family that this protein belongs to?'
        }

        self.text_to_protein_trembl = {
            'data_path': 'data/protein-text/TrEMBL-LMDB-combine-uniref50',
            'data_format': 'LMDB',
            'id_key': 'protein_id',
            'input_modality': 'text',
            'target_modality': 'protein',
            'input_seq_key': 'desc',
            'target_seq_key': 'seq',
            'input_enc_seq_key': 'seq',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': '',
            'conf_key': 'saprot_seq',
            'instruction': 'Generate a protein matching the following description.'
        }

        self.protein_to_text_swissprot_train_name = {
            'data_path': 'data/protein-text/swissprot-name-train.json',
            'data_format': 'JSON',
            'id_key': 'protein_id',
            'input_modality': 'protein',
            'target_modality': 'text',
            'input_seq_key': 'seq',
            'target_seq_key': 'description',
            'input_enc_seq_key': 'seq',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': '',
            'conf_key': 'saprot_seq',
            'instruction': 'What is the official name of this protein?'
        }

        self.protein_to_text_swissprot_train_func = {
            'data_path': 'data/protein-text/swissprot-func-train.json',
            'data_format': 'JSON',
            'id_key': 'protein_id',
            'input_modality': 'protein',
            'target_modality': 'text',
            'input_seq_key': 'seq',
            'target_seq_key': 'description',
            'input_enc_seq_key': 'seq',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': '',
            'conf_key': 'saprot_seq',
            'instruction': 'What is the function of this protein?'
        }

        self.protein_to_text_swissprot_train_loc = {
            'data_path': 'data/protein-text/swissprot-loc-train.json',
            'data_format': 'JSON',
            'id_key': 'protein_id',
            'input_modality': 'protein',
            'target_modality': 'text',
            'input_seq_key': 'seq',
            'target_seq_key': 'description',
            'input_enc_seq_key': 'seq',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': '',
            'conf_key': 'saprot_seq',
            'instruction': 'What is the subcellular location of this protein?'
        }

        self.protein_to_text_swissprot_train_family = {
            'data_path': 'data/protein-text/swissprot-family-train.json',
            'data_format': 'JSON',
            'id_key': 'protein_id',
            'input_modality': 'protein',
            'target_modality': 'text',
            'input_seq_key': 'seq',
            'target_seq_key': 'description',
            'input_enc_seq_key': 'seq',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': '',
            'conf_key': 'saprot_seq',
            'instruction': 'What is the protein family that this protein belongs to?'
        }

        self.text_to_protein_swissprot_train = {
            'data_path': 'data/protein-text/swissprot-combine-train.json',
            'data_format': 'JSON',
            'id_key': 'protein_id',
            'input_modality': 'text',
            'target_modality': 'protein',
            'input_seq_key': 'description',
            'target_seq_key': 'seq',
            'input_enc_seq_key': 'seq',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': '',
            'conf_key': 'saprot_seq',
            'instruction': 'Design a protein matching the following description.'
        }

        self.protein_to_text_swissprot_test_name = {
            'data_path': 'data/protein-text/swissprot-name-test.json',
            'data_format': 'JSON',
            'id_key': 'protein_id',
            'input_modality': 'protein',
            'target_modality': 'text',
            'input_seq_key': 'seq',
            'target_seq_key': 'description',
            'input_enc_seq_key': 'seq',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': '',
            'conf_key': 'saprot_seq',
            'instruction': 'What is the official name of this protein?'
        }

        self.protein_to_text_swissprot_test_func = {
            'data_path': 'data/protein-text/swissprot-func-test.json',
            'data_format': 'JSON',
            'id_key': 'protein_id',
            'input_modality': 'protein',
            'target_modality': 'text',
            'input_seq_key': 'seq',
            'target_seq_key': 'description',
            'input_enc_seq_key': 'seq',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': '',
            'conf_key': 'saprot_seq',
            'instruction': 'What is the function of this protein?'
        }

        self.protein_to_text_swissprot_test_loc = {
            'data_path': 'data/protein-text/swissprot-loc-test.json',
            'data_format': 'JSON',
            'id_key': 'protein_id',
            'input_modality': 'protein',
            'target_modality': 'text',
            'input_seq_key': 'seq',
            'target_seq_key': 'description',
            'input_enc_seq_key': 'seq',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': '',
            'conf_key': 'saprot_seq',
            'instruction': 'What is the subcellular location of this protein?'
        }

        self.protein_to_text_swissprot_test_family = {
            'data_path': 'data/protein-text/swissprot-family-test.json',
            'data_format': 'JSON',
            'id_key': 'protein_id',
            'input_modality': 'protein',
            'target_modality': 'text',
            'input_seq_key': 'seq',
            'target_seq_key': 'description',
            'input_enc_seq_key': 'seq',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': '',
            'conf_key': 'saprot_seq',
            'instruction': 'What is the protein family that this protein belongs to?'
        }

        self.text_to_protein_swissprot_test = {
            'data_path': 'data/protein-text/swissprot-combine-test.json',
            'data_format': 'JSON',
            'id_key': 'protein_id',
            'input_modality': 'text',
            'target_modality': 'protein',
            'input_seq_key': 'description',
            'target_seq_key': 'seq',
            'input_enc_seq_key': 'seq',
            'input_enc_fp_key': 'fingerprint',
            'conf_path': '',
            'conf_key': 'saprot_seq',
            'instruction': 'Design a protein matching the following description.'
        }

        # =====================molecule-protein=====================


        self.protein_to_molecule_bindingdb_train = {
            'data_path': 'data/protein-molecule/protein-and-molecule-bindingdb-train.json',
            'data_format': 'JSON',
            'id_key': 'PID',
            'input_modality': 'protein',
            'target_modality': 'molecule',
            'input_seq_key': 'protein',
            'target_seq_key': 'selfies',
            'input_enc_seq_key': 'protein',
            'input_enc_fp_key': 'protein_fingerprint',
            'conf_path': '',
            'conf_key': 'saprot_seq',
            'instruction': 'Generate a drug molecule binding to the target protein.'
        }

        self.protein_to_molecule_bindingdb_test = {
            'data_path': 'data/protein-molecule/protein-to-molecule-bindingdb-test-filtered.json',
            'data_format': 'JSON',
            'id_key': 'PID',
            'input_modality': 'protein',
            'target_modality': 'molecule',
            'input_seq_key': 'protein',
            'target_seq_key': 'selfies',
            'input_enc_seq_key': 'protein',
            'input_enc_fp_key': 'protein_fingerprint',
            'conf_path': '',
            'conf_key': 'saprot_seq',
            'instruction': 'Generate a drug molecule binding to the target protein.'
        }

        self.molecule_to_protein_gorhea_train = {
            'data_path': 'data/protein-molecule/gorhea-molecule-and-protein-train.json',
            'data_format': 'JSON',
            'id_key': 'CID',
            'input_modality': 'molecule',
            'target_modality': 'protein',
            'input_seq_key': 'selfies',
            'target_seq_key': 'protein',
            'input_enc_seq_key': 'smiles',
            'input_enc_fp_key': 'molecule_fingerprint',
            'conf_path': 'data/sdf/gorhea',
            'conf_key': 'SDF_file',
            'instruction': 'Generate an enzyme that can catalyze for the given substrate.'
        }


        self.molecule_to_protein_gorhea_test = {
            'data_path': 'data/protein-molecule/gorhea-molecule-to-protein-test-filtered.json',
            'data_format': 'JSON',
            'id_key': 'CID',
            'input_modality': 'molecule',
            'target_modality': 'protein',
            'input_seq_key': 'selfies',
            'target_seq_key': 'protein',
            'input_enc_seq_key': 'smiles',
            'input_enc_fp_key': 'molecule_fingerprint',
            'conf_path': 'data/sdf/gorhea',
            'conf_key': 'SDF_file',
            'instruction': 'Generate an enzyme that can catalyze for the given substrate.'
        }

