import torch
from torch import nn
from transformers import EsmModel, EsmTokenizer
from torch_geometric.utils import to_dense_batch

from .mol_gnn import GNN_Encoder
from .geoformer import GeoFormer


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Molecule1dEncoder(nn.Module):
    def __init__(self):
        super(Molecule1dEncoder, self).__init__()
        self.encoder = self.init_encoder()
        self.eval()
        self.train = disabled_train

    def init_encoder(self):
        ckpt_path = 'pretrained_ckpt/supervised_contextpred.pth'
        encoder = GNN_Encoder(load_path=ckpt_path)
        for name, param, in encoder.named_parameters():
            param.requires_grad = False
        encoder.eval()
        encoder.train = disabled_train
        return encoder

    def get_dim(self):
        return self.encoder.emb_dim

    @torch.no_grad()
    def forward(self, graph_batch, device, dtype):
        graph_batch = graph_batch.to(device)
        graph_emb, graph_mask = self.encoder(graph_batch)
        if graph_emb.dtype != torch.float32:
            self.encoder = self.encoder.float()
            graph_emb, graph_mask = self.encoder(graph_batch)
        graph_emb = graph_emb.detach().to(dtype)
        return graph_emb, graph_mask


class Molecule3dEncoder(nn.Module):
    def __init__(self):
        super(Molecule3dEncoder, self).__init__()
        self.encoder = self.init_encoder()
        self.eval()
        self.train = disabled_train

    def get_dim(self):
        return 512

    def init_encoder(self):
        encoder = GeoFormer()
        encoder.eval()
        for name, param, in encoder.named_parameters():
            param.requires_grad = False
        encoder.train = disabled_train
        return encoder

    @torch.no_grad()
    def forward(self, data, device, dtype):
        z, coords = data
        z, coords = z.to(device), coords.to(device).to(dtype)
        batch_node, batch_mask = self.encoder(z, coords)
        return batch_node, batch_mask


class Protein1dEncoder(nn.Module):
    def __init__(self):
        super(Protein1dEncoder, self).__init__()
        self.encoder, self.tokenizer = self.init_encoder()
        self.eval()
        self.train = disabled_train

    def get_dim(self):
        return self.encoder.config.hidden_size

    def init_encoder(self):
        ckpt_path = 'pretrained_ckpt/esm2_t12_35M_UR50D'
        encoder = EsmModel.from_pretrained(ckpt_path)
        # input_modality_dim = input_modality_encoder.config.hidden_size

        tokenizer = EsmTokenizer.from_pretrained(ckpt_path)

        for name, param, in encoder.named_parameters():
            param.requires_grad = False
        encoder.eval()
        encoder.train = disabled_train
        return encoder, tokenizer

    @torch.no_grad()
    def forward(self, protein, device, dtype):
        inputs = self.tokenizer(protein, padding='longest', truncation=True,
                                max_length=512, return_tensors='pt').to(device)
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        output = output.last_hidden_state[:, :-1, :]
        output = output.detach().to(dtype)
        return output, attention_mask[:, :-1].bool()


class Protein3dEncoder(nn.Module):
    def __init__(self):
        super(Protein3dEncoder, self).__init__()
        self.encoder, self.tokenizer = self.init_encoder()
        self.eval()
        self.train = disabled_train

    def get_dim(self):
        return self.encoder.config.hidden_size

    def init_encoder(self):
        ckpt_path = 'pretrained_ckpt/SaProt_35M_AF2'
        encoder = EsmModel.from_pretrained(ckpt_path)
        # input_modality_dim = input_modality_encoder.config.hidden_size

        tokenizer = EsmTokenizer.from_pretrained(ckpt_path)

        for name, param, in encoder.named_parameters():
            param.requires_grad = False
        encoder.eval()
        encoder.train = disabled_train
        return encoder, tokenizer

    @torch.no_grad()
    def forward(self, protein, device, dtype):
        inputs = self.tokenizer(protein, padding='longest', truncation=True,
                                max_length=512, return_tensors='pt').to(device)
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        output = output.last_hidden_state[:, :-1, :]
        output = output.detach().to(dtype)
        return output, attention_mask[:, :-1].bool()
