from torch_geometric.data import Data, Batch
import torch

from .preprocess_smiles import process_smiles
from .load_sdf import load_sdf_data, collate_sdf


def collate_func(instances):
    input_seqs, target_seqs, input_enc_seqs, input_enc_fps, input_modality, target_modality, instructions, data_ids, confs, data_name = tuple(
        [instance[key] for instance in instances] for key in ("input_seqs",
                                                              "target_seqs",
                                                              "input_enc_seqs",
                                                              "input_enc_fps",
                                                              "input_modality",
                                                              "target_modality",
                                                              "instructions",
                                                              "id",
                                                              "conf",
                                                              "data_name"))
    input_modality = input_modality[0]
    target_modality = target_modality[0]
    data_name = data_name[0]


    if input_modality == 'text':
        return None, {'input_seqs': input_seqs,
                        'target_seqs': target_seqs,
                        'input_enc_seqs': input_enc_seqs,
                        'input_enc_fps': input_enc_fps,
                        'input_modality': input_modality,
                        'target_modality': target_modality,
                        'instructions': instructions,
                        'ids': data_ids,
                        'data_name': data_name}
    elif input_modality == 'molecule':
        smiles = input_enc_seqs
        graph_data_list = []
        for s in smiles:
            x, edge_index, edge_attr = process_smiles(s)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            graph_data_list.append(data)
        batch_graph = Batch.from_data_list(graph_data_list)

        sdf_list = []
        for sdf_path in confs:
            z, coords = load_sdf_data(sdf_path)
            sdf_list.append({'z': z, 'pos': coords})
        geoformer_input = collate_sdf(sdf_list)
        return (batch_graph, geoformer_input), {'input_seqs': input_seqs,
                                                'target_seqs': target_seqs,
                                                'input_enc_seqs': input_enc_seqs,
                                                'input_enc_fps': input_enc_fps,
                                                'input_modality': input_modality,
                                                'target_modality': target_modality,
                                                'instructions': instructions,
                                                'ids': data_ids,
                                                'data_name': data_name}
    elif input_modality == 'protein':
        return (input_enc_seqs, confs), {'input_seqs': input_seqs,
                                            'target_seqs': target_seqs,
                                            'input_enc_seqs': input_enc_seqs,
                                            'input_enc_fps': input_enc_fps,
                                            'input_modality': input_modality,
                                            'target_modality': target_modality,
                                            'instructions': instructions,
                                            'ids': data_ids,
                                            'data_name': data_name}
    else:
        raise NotImplementedError
