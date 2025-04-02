import torch
from torch import nn
from .embed import PositionalEmbedding, TemporalEmbedding

from einops import reduce, rearrange

class ConvEncoder(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.output_dims = output_dims

        kernels = [1,2,4,8,16,32,64,128]
        self.kernels = kernels
        self.tfd = nn.ModuleList([nn.Conv1d(input_dims, output_dims, k, padding=k-1) for k in kernels])

    def forward(self, x):  # x: B x T x d1 x Ch
        b,t,d1,Ch=x.size()
        x = x.transpose(1, 2)  # B x d1 x T x Ch
        x = torch.reshape(x, (b*d1,t,Ch))  # B*d1 x T x Ch
        x = x.transpose(1, 2)  # B*d1 x Ch x T

        trend = []
        for idx, mod in enumerate(self.tfd):
            out = mod(x)  # b d t
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            trend.append(out.transpose(1, 2))  # b t d
        trend = reduce(
            rearrange(trend, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
        )
        x = trend
        x = torch.reshape(x, (b,d1,t,-1))  # B x d1 x T x Co
        x = x.transpose(1, 2)  # B x T x d1 x Co
        return x


class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=320, n_covariate_cols=None, d1=None, freq='h', out_mode='mean', embedding=2):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.out_mode = out_mode
        self.embedding = embedding
        self.n_covariate_cols = n_covariate_cols
        d2 = input_dims - n_covariate_cols

        self.input_fc_d1 = nn.Linear(d1, hidden_dims)
        self.input_fc_d2 = nn.Linear(d2, hidden_dims)
        self.position_embedding = PositionalEmbedding(d_model=hidden_dims)
        self.temporal_embedding = TemporalEmbedding(d_model=hidden_dims, embed_type='fixed', freq=freq)
        self.conv_encoder = ConvEncoder(hidden_dims, int(output_dims/2))


    def forward(self, x):  # x: B x T x d1 x input_dims(n_covariate_cols + channels)
        x_data = x[..., self.n_covariate_cols:]
        x_stamp = x[:, :, 0, :self.n_covariate_cols]

        nan_mask = ~x_data.isnan()
        x_data[~nan_mask] = 0

        # temporal, positional, token embedding
        tem_emb = self.temporal_embedding(x_stamp) # B x T x Ch
        tem_emb = torch.unsqueeze(tem_emb, 2) # B x T x 1 x Ch
        pos_emb = self.position_embedding(x_stamp) # 1 x T x Ch
        pos_emb = torch.unsqueeze(pos_emb, 2) # 1 x T x 1 x Ch

        x_emb_d2 = self.input_fc_d2(x_data)  # B x T x d1 x Ch
        x_emb_d1 = self.input_fc_d1(x_data.transpose(2,3))  # B x T x d2 x Ch


        if self.embedding==0:
            x_emb_d2 = x_emb_d2 + pos_emb.repeat(x_data.size(0), 1, x_data.size(2), 1) + tem_emb.repeat(1, 1, x_data.size(2), 1)
            x_emb_d1 = x_emb_d1 + pos_emb.repeat(x_data.size(0), 1, x_data.size(3), 1) + tem_emb.repeat(1, 1, x_data.size(3), 1)
        elif self.embedding==1:
            x_emb_d2 = x_emb_d2 + pos_emb.repeat(x_data.size(0), 1, x_data.size(2), 1)
            x_emb_d1 = x_emb_d1 + pos_emb.repeat(x_data.size(0), 1, x_data.size(3), 1)
        elif self.embedding==2:
            x_emb_d2 = x_emb_d2 + tem_emb.repeat(1, 1, x_data.size(2), 1)
            x_emb_d1 = x_emb_d1 + tem_emb.repeat(1, 1, x_data.size(3), 1)
        elif self.embedding==3:
            x_emb_d2 = x_emb_d2
            x_emb_d1 = x_emb_d1

        # encoder
        x_emb_d2 = self.conv_encoder(x_emb_d2)  # B x T x d1 x Ch
        x_emb_d1 = self.conv_encoder(x_emb_d1)  # B x T x d2 x Ch

        if self.out_mode=='max':
            x_emb_d2_mean = torch.max(x_emb_d2, axis=2).values # B x T x Ch
            x_emb_d1_mean = torch.max(x_emb_d1, axis=2).values # B x T x Ch
        else:
            x_emb_d2_mean = torch.mean(x_emb_d2, dim=2)  # B x T x Ch
            x_emb_d1_mean = torch.mean(x_emb_d1, dim=2)  # B x T x Ch
        out = torch.concat((x_emb_d2_mean, x_emb_d1_mean), dim=-1)

        return out, x_emb_d2, x_emb_d1
