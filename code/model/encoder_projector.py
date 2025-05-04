import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Molecule1dEncoder, Protein1dEncoder, Molecule3dEncoder, Protein3dEncoder


class EncoderProjector(nn.Module):
    def __init__(self,
                 args,
                 input_modality,
                 hidden_size,
                 encoder_layer,
                 decoder_layer,
                 output_size,
                 num_query_tokens,
                 num_fp):
        super(EncoderProjector, self).__init__()
        self.args = args
        self.input_modality = input_modality
        # self.input_modality_encoder, self.input_modality_dim = self.init_input_modality_encoder(input_modality)
        self.input_encoder_1, self.input_encoder_2, self.input_dim_1, self.input_dim_2 = self.init_input_modality_encoder(input_modality)

        self.encoder_linear_1 = nn.Sequential(nn.LayerNorm(self.input_dim_1),
                                              nn.Linear(self.input_dim_1, hidden_size))
        self.encoder_linear_2 = nn.Sequential(nn.LayerNorm(self.input_dim_2),
                                              nn.Linear(self.input_dim_2, hidden_size))
        self.out_linear = nn.Linear(hidden_size, output_size)

        self.transformer = nn.Transformer(d_model=hidden_size,
                                          num_encoder_layers=encoder_layer,
                                          num_decoder_layers=decoder_layer,
                                          batch_first=True)
        self.fp_embedding = nn.Linear(num_fp, hidden_size, bias=False)
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens - 1, hidden_size)
        )


        self.cur_device = torch.cuda.current_device()  # int!

    @classmethod
    def init_input_modality_encoder(cls, input_modality):
        if input_modality == 'molecule':
            encoder1 = Molecule1dEncoder()
            encoder2 = Molecule3dEncoder()
        elif input_modality == 'protein':
            encoder1 = Protein1dEncoder()
            encoder2 = Protein3dEncoder()
        else:
            raise NotImplementedError
        return encoder1, encoder2, encoder1.get_dim(), encoder2.get_dim()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.cur_device)) == 1).transpose(0, 1)
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(self.fp_embedding.weight.dtype)
        return mask

    def create_mask(self, src: torch.Tensor, tgt: torch.Tensor):
        src_seq_len = src.shape[1]  # [B, L, D]
        tgt_seq_len = tgt.shape[1]


        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=self.cur_device).bool()
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.cur_device).type(torch.bool)  # [L, L]

        return src_mask, tgt_mask

    def forward(self, input_batch, input_fp, dtype):

        input_batch1, input_batch2 = input_batch


        input_emb1, input_mask1 = self.input_encoder_1(input_batch1, self.cur_device, dtype)
        input_emb2, input_mask2 = self.input_encoder_2(input_batch2, self.cur_device, dtype)

        input_emb1 = self.encoder_linear_1(input_emb1)
        input_emb2 = self.encoder_linear_2(input_emb2)
        input_modality_emb = torch.cat([input_emb1, input_emb2], dim=1)
        input_modality_mask = torch.cat([input_mask1, input_mask2], dim=1)


        # input_modality_mask is True => have value,  is False => empty
        # src_key_padding_mask is False => have value , is True => empty
        # =>If a BoolTensor is provided, the positions with the value of True will be ignored while the position with the value of False will be unchanged.
        src_key_padding_mask = ~input_modality_mask  # [B, L]


        query_input = self.query_tokens.repeat(input_modality_emb.shape[0], 1, 1)

        src_mask, tgt_mask = self.create_mask(input_modality_emb, query_input)
        transformer_output = self.transformer(src=input_modality_emb,
                                              tgt=query_input,
                                              src_mask=src_mask,
                                              tgt_mask=tgt_mask,
                                              src_key_padding_mask=src_key_padding_mask)  # [B, N, D]
        transformer_output = self.out_linear(transformer_output)
        return transformer_output





