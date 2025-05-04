from typing import Optional
from collections import OrderedDict
import torch
from einops import rearrange, repeat
from torch import nn
from transformers import PreTrainedModel

from .geoformer_config import GeoformerConfig
from .geoformer_layers import (CosineCutoff,
                               ExpNormalSmearing,
                               VecLayerNorm,
                               act_class_mapping)


class GeoFormer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = GeoformerConfig()

        self.geo_encoder = GeoformerEncoder(config=config)
        state_dict = torch.load('pretrained_ckpt/geoformer.ckpt', map_location="cpu")
        # self.load_state_dict(state_dict)
        #
        # new_sd = OrderedDict()
        # for k, v in sd.items():
        #     new_sd['.'.join(k.split('.')[1:])] = v
        #     # del sd[k]
        missing, unexpected = self.load_state_dict(state_dict=state_dict, strict=False)
        print(f"Restored with {len(missing)} missing and {len(unexpected)} unexpected keys")
        # if len(missing) > 0:
        print(f"Missing Keys: {missing}")
        print(f"Unexpected Keys: {unexpected}")
        # import pdb; pdb.set_trace()

    def forward(self, z, coords):
        return self.geo_encoder(z, coords)


class GeoformerMultiHeadAttention(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(GeoformerMultiHeadAttention, self).__init__(*args, **kwargs)

        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = config.embedding_dim // config.num_attention_heads
        if not (
                self.head_dim * config.num_attention_heads == self.embedding_dim
        ):
            raise AssertionError(
                "The embedding_dim must be divisible by num_heads."
            )

        self.act = act_class_mapping[config.activation_function]()
        self.cutoff = CosineCutoff(config.cutoff)

        self.dropout_module = nn.Dropout(
            p=config.attention_dropout, inplace=False
        )

        self.k_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.v_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dk_proj = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.du_update_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.du_norm = VecLayerNorm(
            self.embedding_dim, trainable=False, norm_type=config.norm_type
        )
        self.dihedral_proj = nn.Linear(
            self.embedding_dim, 2 * self.embedding_dim, bias=False
        )
        self.edge_attr_update = nn.Linear(
            self.embedding_dim, self.embedding_dim
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.dk_proj.weight)
        self.dk_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.du_update_proj.weight)
        self.du_update_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.dihedral_proj.weight)
        nn.init.xavier_uniform_(self.edge_attr_update.weight)
        self.edge_attr_update.bias.data.fill_(0.0)

    def forward(
            self,
            x: torch.Tensor,  # (B, N, F)
            vec: Optional[torch.Tensor],  # (B, N, N, 3)
            dist: Optional[torch.Tensor],  # (B, N, N)
            edge_attr: Optional[torch.Tensor],  # (B, N, N, F)
            key_padding_mask: Optional[torch.Tensor] = None,  # (B, N)
            **kwargs,
    ):
        q = rearrange(
            self.q_proj(x), "b n (h d) -> (b h) n d", h=self.num_heads
        )  # (BH, N, D)
        k = rearrange(
            self.k_proj(x), "b n (h d) -> (b h) n d", h=self.num_heads
        )  # (BH, N, D)
        v = rearrange(
            self.v_proj(x), "b n (h d) -> (b h) n d", h=self.num_heads
        )  # (BH, N, D)
        dk = rearrange(
            self.act(self.dk_proj(edge_attr)),
            "b n m (h d) -> (b h) n m d",
            h=self.num_heads,
        )  # (BH, N, N, D)

        attn_weights = ((q.unsqueeze(-2) * k.unsqueeze(-3)) * dk).sum(
            dim=-1
        )  # (BH, N, N)

        if key_padding_mask is not None:
            attn_weights = rearrange(
                attn_weights, "(b h) n m -> b h n m", h=self.num_heads
            )
            attn_weights = attn_weights.masked_fill(
                rearrange(key_padding_mask, "b n m -> b () n m"),
                0.0,
            )
            attn_weights = rearrange(attn_weights, "b h n m -> (b h) n m")

        attn_scale = repeat(
            self.cutoff(dist), "b n m -> b h n m", h=self.num_heads
        )  # (BH, N, N)
        attn_scale = rearrange(
            attn_scale, "b h n m -> (b h) n m", h=self.num_heads
        )  # (BH, N, N)
        attn_probs = self.act(attn_weights) * attn_scale  # (BH, N, N)

        attn_per_nodes = attn_probs.unsqueeze(-1) * v.unsqueeze(
            -3
        )  # (BH, N, N, D)
        attn_per_nodes = rearrange(
            attn_per_nodes, "(b h) n m d -> b n m (h d)", h=self.num_heads
        )  # (B, N, N, F)
        attn = attn_per_nodes.sum(dim=2)  # (B, N, F)

        du = (
                self.du_update_proj(attn_per_nodes)
                .masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
                .unsqueeze(-2)
                * vec.unsqueeze(-1)
        ).sum(
            dim=-3
        )  # (B, N, 3, F)
        du = self.du_norm(du)  # (B, N, 3, F)
        ws, wt = torch.split(
            self.dihedral_proj(du), self.embedding_dim, dim=-1
        )  # (B, N, 3, F)
        ipe = (wt.unsqueeze(1) * ws.unsqueeze(2)).sum(dim=-2)  # (B, N, N, F)
        ipe = self.act(self.edge_attr_update(edge_attr)) * ipe  # (B, N, N, F)

        return attn, ipe


class GeoformerAttnBlock(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(GeoformerAttnBlock, self).__init__(*args, **kwargs)

        self.embedding_dim = config.embedding_dim
        self.dropout_module = nn.Dropout(p=config.dropout, inplace=False)

        self.act = act_class_mapping[config.activation_function]()

        self.self_attn = GeoformerMultiHeadAttention(config)

        self.ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, config.ffn_embedding_dim),
            self.act,
            nn.Dropout(p=config.activation_dropout, inplace=False),
            nn.Linear(config.ffn_embedding_dim, self.embedding_dim),
        )

        self.attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.self_attn.reset_parameters()
        nn.init.xavier_uniform_(self.ffn[0].weight)
        self.ffn[0].bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.ffn[3].weight)
        self.ffn[3].bias.data.fill_(0.0)
        self.attn_layer_norm.reset_parameters()
        self.final_layer_norm.reset_parameters()

    def forward(
            self,
            x: torch.Tensor,  # (B, N, F)
            vec: torch.Tensor,  # (B, N, N, 3)
            dist: torch.Tensor,  # (B, N, N)
            edge_attr: torch.Tensor,  # (B, N, N, ?)
            key_padding_mask: Optional[
                torch.Tensor
            ],  # [padding, cutoff] (B, N, N)
            **kwargs,
    ):
        # attention
        dx, dedge_attr = x, edge_attr
        x, edge_attr = self.self_attn(
            x=x,
            vec=vec,
            dist=dist,
            edge_attr=edge_attr,
            key_padding_mask=key_padding_mask,
        )

        x = self.dropout_module(x)
        x = x + dx
        x = self.attn_layer_norm(x)

        # ipe update
        edge_attr = edge_attr + dedge_attr

        # ffn
        dx = x
        x = self.ffn(x)
        x = self.dropout_module(x)
        x = x + dx

        x = self.final_layer_norm(x)

        return x, edge_attr


class GeoformerEncoder(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(GeoformerEncoder, self).__init__(*args, **kwargs)

        self.pad_token_id = config.pad_token_id
        self.embedding_dim = config.embedding_dim
        self.cutoff = config.cutoff

        self.embedding = nn.Embedding(
            config.max_z, self.embedding_dim, padding_idx=self.pad_token_id
        )
        self.distance_expansion = ExpNormalSmearing(
            cutoff=config.cutoff,
            num_rbf=config.num_rbf,
            trainable=config.rbf_trainable,
        )
        self.dist_proj = nn.Linear(config.num_rbf, self.embedding_dim)
        self.act = act_class_mapping[config.activation_function]()

        self.layers = nn.ModuleList(
            [GeoformerAttnBlock(config) for _ in range(config.num_layers)]
        )

        self.x_in_layernorm = nn.LayerNorm(self.embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        nn.init.xavier_uniform_(self.dist_proj.weight)
        self.dist_proj.bias.data.fill_(0.0)
        for layer in self.layers:
            layer.reset_parameters()
        self.x_in_layernorm.reset_parameters()

    def forward(
            self,
            z: torch.Tensor,  # (B, N)
            pos: torch.Tensor,  # (B, N, 3)
            **kwargs,
    ):
        B, N, *_ = z.shape
        # generate mask
        padding_mask = z == self.pad_token_id  # (B, N)
        pos_mask = ~(
                padding_mask.unsqueeze(1) | padding_mask.unsqueeze(2)
        )  # (B, N, N)
        dist = torch.norm(
            pos.unsqueeze(1) - pos.unsqueeze(2), dim=-1
        )  # (B, N, N)
        loop_mask = torch.eye(N, dtype=torch.bool, device=dist.device)
        loop_mask = repeat(loop_mask, "n m -> b n m", b=B)  # (B, N, N)
        dist = dist.masked_fill(loop_mask, 0.0)  # (B, N, N)
        adj_mask = (dist < self.cutoff) & pos_mask  # (B, N, N)
        loop_adj_mask = ~loop_mask & adj_mask  # (B, N, N)

        vec = (pos.unsqueeze(1) - pos.unsqueeze(2)) / (
                dist.unsqueeze(-1) + 1e-8
        )  # (B, N, N, 3)
        vec = vec.masked_fill(
            ~loop_adj_mask.unsqueeze(-1), 0.0
        )  # (B, N, N, 3)

        key_padding_mask = (
            (~adj_mask)
                .masked_fill(padding_mask.unsqueeze(-1), False)
                .masked_fill(padding_mask.unsqueeze(-2), True)
        )

        x = self.embedding(z)  # (B, N, F)
        x = self.x_in_layernorm(x)
        edge_attr = self.distance_expansion(dist)  # (B, N, N, num_rbf)
        edge_attr = self.act(self.dist_proj(edge_attr))  # (B, N, N, F)
        edge_attr = edge_attr.masked_fill(
            ~adj_mask.unsqueeze(-1), 0.0
        )  # (B, N, N, F)

        for layer in self.layers:
            x, edge_attr = layer(
                x=x,
                vec=vec,
                dist=dist,
                edge_attr=edge_attr,
                key_padding_mask=key_padding_mask,
            )

        return x, ~padding_mask
