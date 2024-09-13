import copy
from torch import Tensor
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math
from typing import Optional

from torchsummary import summary
def generate_graph(y, k=0.3):
    with torch.no_grad():
        N, B, C = y.shape
        k = (int)(len(y) * k)
        y = y.permute(1, 0, 2)
        dist_matrix = torch.cdist(y, y) / (C ** 0.5)
        dist_matrix = dist_matrix[0] + torch.eye(y.size(1), dtype=dist_matrix.dtype,
                                                 device=dist_matrix.device) * torch.max(dist_matrix)
        # dist_matrix = dist_matrix[:, :N]
        # get local density
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        graph_matrix = torch.zeros_like(dist_matrix)
        ones = torch.ones_like(dist_matrix)
        graph_matrix.scatter_(-1, index_nearest, ones)

        return graph_matrix[:N, :N]

def position(x):
    length, d_model = x.size()
    length = length+1
    pe = torch.zeros((length, d_model), device=x.device)
    # d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2, device=x.device)*-(math.log(10000.0)/d_model))
    pos = torch.arange(0, length, device=x.device).unsqueeze(1)
    pe[:, 0::2] = torch.sin(pos.float()*div_term)
    pe[:, 1::2] = torch.cos(pos.float()*div_term)
    return pe




class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_y = y_embed[:, :, :, None] / dim_t
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = pos_y.permute(0, 3, 1, 2).contiguous()

        return pos

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, graph_pos=None, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)


        attno = self.dropout(F.softmax(attn, dim=-1))
        if graph_pos is not None:
            attn = attno * graph_pos.permute(2, 1, 0).unsqueeze(0)

        output = torch.matmul(attn, v)

        return output, attno
        # return output, attn / (torch.sum(attn, dim=-1, keepdim=True)+1e-6)

class GraphMultiheadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head

        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v, bias=False)
        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, graph_pos=None, mask=None):
        q = q.permute(1,0,2)
        k = k.permute(1,0,2)
        v = v.permute(1,0,2)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, graph_pos=graph_pos, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        q = q.permute(1,0,2)


        return q, attn


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, h_dim, nhead, norm, topk, usenum):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.topk = topk
        self.usenum = usenum
        self.near_embeddings = nn.Embedding(usenum, h_dim)
        self.pe_layer = nn.Sequential(
            nn.Conv2d(h_dim, h_dim // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h_dim // 2, nhead, 3, 1, 1),  # for quantize
            nn.Sigmoid()
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):

        x = src
        bs, c, h, w = x.shape
        pe = self.pe_layer(x)
        graph_pos = pe.flatten(2)
        graph_pos = torch.abs(graph_pos.permute(2, 0, 1) - graph_pos.permute(0, 2, 1))
        x = x.flatten(2).permute(2, 0, 1)
        attn = []
        for i, layer in enumerate(self.layers):
            graph = generate_graph(x, self.topk)
            near = torch.sum(graph, dim=0)
            near = near / torch.max(near) * (self.usenum - 1)
            # near = (near-torch.min(near))/(torch.max(near)-torch.min(near)) * (self.usenum-1)
            # near = torch.clamp(near, max=self.usenum-1)
            pos = self.near_embeddings(near.int())
            x, att = layer(x, graph_pos=graph_pos, src_mask=mask, src_key_padding_mask=src_key_padding_mask,
                           pos=pos.unsqueeze(1))
            attn.append(att)

        if self.norm is not None:
            x = self.norm(x)
        output = x.permute(1, 2, 0).view(bs, c, h, w)

        return output, attn


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = GraphMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, graph_pos: Optional[Tensor] = None, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)

        src2, attn = self.self_attn(q, k, src, graph_pos=graph_pos, mask=src_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

if __name__ == '__main__':
    # 参数设置
    batch_size = 64
    channels = 512
    height = 8
    width = 8
    num_layers = 4
    num_heads = 8
    topk = 0.3
    usenum = 5

    # 创建随机输入
    src = torch.rand(batch_size, channels, height, width)

    # 创建位置嵌入层
    encoder_layer = TransformerEncoderLayer(d_model=channels, nhead=num_heads)
    transformer_encoder = TransformerEncoder(
        encoder_layer=encoder_layer,
        num_layers=num_layers,
        h_dim=channels,
        nhead=num_heads,
        norm=nn.LayerNorm(channels),
        topk=topk,
        usenum=usenum
    )
    # summary(transformer_encoder.to('cuda'),(channels, height,width),batch_size,device='cuda')
    # 前向传播
    output, attn = transformer_encoder(src)


    print(output.shape)
    # 输出形状检查
    assert output.shape == (batch_size, channels, height,
                            width), f"Expected output shape {(batch_size, channels, height, width)}, but got {output.shape}"
    print("Output shape is correct:", output.shape)

    # 检查注意力列表的长度
    assert len(attn) == num_layers, f"Expected attention length {num_layers}, but got {len(attn)}"
    print("Attention length is correct:", len(attn))