import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import pdb


class ModuleTimestamping(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, t, sampling_endpoints):
        return self.rnn(t[:sampling_endpoints[-1]])[0][[p - 1 for p in sampling_endpoints]]


class LayerGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon=True):
        super().__init__()
        if epsilon:
            self.epsilon = nn.Parameter(torch.Tensor([[0.0]]))  # assumes that the adjacency matrix includes self-loop
        else:
            self.epsilon = 0.0
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU())

    def forward(self, v, a):
        v_aggregate = torch.sparse.mm(a, v)
        v_aggregate += self.epsilon * v  # assumes that the adjacency matrix includes self-loop
        v_combine = self.mlp(v_aggregate)
        return v_combine


class ModuleMeanReadout(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, node_axis=1):
        return x.mean(node_axis), torch.zeros(size=[1, 1, 1], dtype=torch.float32)


class ModuleSERO(nn.Module):
    def __init__(self, hidden_dim, input_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(hidden_dim, round(upscale * hidden_dim)),
                                   nn.BatchNorm1d(round(upscale * hidden_dim)), nn.GELU())
        self.attend = nn.Linear(round(upscale * hidden_dim), input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_axis=1):
        # assumes shape [... x node x ... x feature]
        x_readout = x.mean(node_axis)
        x_shape = x_readout.shape
        x_embed = self.embed(x_readout.reshape(-1, x_shape[-1]))
        x_graphattention = torch.sigmoid(self.attend(x_embed)).view(*x_shape[:-1], -1)
        permute_idx = list(range(node_axis)) + [len(x_graphattention.shape) - 1] + list(
            range(node_axis, len(x_graphattention.shape) - 1))
        x_graphattention = x_graphattention.permute(permute_idx)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1, 0, 2)


class ModuleGARO(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, upscale=1.0, **kwargs):
        super().__init__()
        self.embed_query = nn.Linear(hidden_dim, round(upscale * hidden_dim))
        self.embed_key = nn.Linear(hidden_dim, round(upscale * hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_axis=1):
        # assumes shape [... x node x ... x feature]
        x_q = self.embed_query(x.mean(node_axis, keepdims=True))
        x_k = self.embed_key(x)
        x_graphattention = torch.sigmoid(
            torch.matmul(x_q, rearrange(x_k, 't b n c -> t b c n')) / np.sqrt(x_q.shape[-1])).squeeze(2)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1, 0, 2)


class ModuleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        x_attend, attn_matrix = self.multihead_attn(x, x, x)
        x_attend = self.dropout1(x_attend)  # no skip connection
        x_attend = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
        return x_attend, attn_matrix


# Percentile class based on
# https://github.com/aliutkus/torchpercentile
class Percentile(torch.autograd.Function):
    def __init__(self):
        super().__init__()

    def __call__(self, input, percentiles):
        return self.forward(input, percentiles)

    def forward(self, input, percentiles):
        input = torch.flatten(input)  # find percentiles for flattened axis
        input_dtype = input.dtype
        input_shape = input.shape
        if isinstance(percentiles, int):
            percentiles = (percentiles,)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles, dtype=torch.double)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles)
        input = input.double()
        percentiles = percentiles.to(input.device).double()
        input = input.view(input.shape[0], -1)
        in_sorted, in_argsort = torch.sort(input, dim=0)
        positions = percentiles * (input.shape[0] - 1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > input.shape[0] - 1] = input.shape[0] - 1
        weight_ceiled = positions - floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        self.save_for_backward(input_shape, in_argsort, floored.long(),
                               ceiled.long(), weight_floored, weight_ceiled)
        result = (d0 + d1).view(-1, *input_shape[1:])
        return result.type(input_dtype)

    def backward(self, grad_output):
        """
        backward the gradient is basically a lookup table, but with weights
        depending on the distance between each point and the closest
        percentiles
        """
        (input_shape, in_argsort, floored, ceiled,
         weight_floored, weight_ceiled) = self.saved_tensors

        # the argsort in the flattened in vector

        cols_offsets = (
                           torch.arange(
                               0, input_shape[1], device=in_argsort.device)
                       )[None, :].long()
        in_argsort = (in_argsort * input_shape[1] + cols_offsets).view(-1).long()
        floored = (
                floored[:, None] * input_shape[1] + cols_offsets).view(-1).long()
        ceiled = (
                ceiled[:, None] * input_shape[1] + cols_offsets).view(-1).long()

        grad_input = torch.zeros((in_argsort.size()), device=self.device)
        grad_input[in_argsort[floored]] += (grad_output
                                            * weight_floored[:, None]).view(-1)
        grad_input[in_argsort[ceiled]] += (grad_output
                                           * weight_ceiled[:, None]).view(-1)

        grad_input = grad_input.view(*input_shape)
        return grad_input


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class VAE(nn.Module):
    def __init__(self, view, channel1=8, channel2=16, channel3=32, d_model=32):
        super(VAE, self).__init__()
        
        self.view = view
        self.d_model = d_model
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channel1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(channel1),
            nn.ReLU(),
            nn.Conv1d(channel1, channel2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(channel2),
            nn.ReLU(),
            nn.Conv1d(channel2, channel3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(channel3),
            nn.ReLU(),
            nn.Conv1d(channel3, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # 潜在空间的均值和方差
        self.fc_mu = nn.Linear(d_model, d_model)
        self.fc_logvar = nn.Linear(d_model, d_model)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(d_model, channel3,  kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(channel3),
            nn.ReLU(),
            nn.ConvTranspose1d(channel3, channel2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(channel2),
            nn.ReLU(),
            nn.ConvTranspose1d(channel2, channel1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(channel1),
            nn.ReLU(),
            nn.ConvTranspose1d(channel1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # 输出在[-1,1]范围内
        )
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        """编码输入数据"""
        h = self.encoder(x).permute(0, 2, 1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z, L):
        """解码潜在变量"""
        x = z.permute(0, 2, 1)
        x = self.decoder(x)
        x = F.adaptive_avg_pool1d(x, output_size=L)
        
        # for layer in self.decoder:
        #     x = layer(x)
        #     if torch.isnan(x).any():
        #         pdb.set_trace()       
        return x
    
    def forward(self, x):
        # pdb.set_trace()
        # time_series (B, 68, 1500)
        B, C, L = x.shape
        
        x = x.reshape(B*C, 1, L)
        
        # 编码
        mu, logvar = self.encode(x)
        
        # 重参数化
        z = self.reparameterize(mu, logvar)
        
        # 解码
        recon_x = self.decode(z, L)
        
        assert (self.view == "t" or self.view == "f" or self.view == "p"), "Not find view"
        recon_loss = None
        if self.view == "t":
            recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
        elif self.view == "f":
            fft_x = torch.fft.fft(x, dim=-1)
            # fft_x = torch.complex(
            #     torch.clamp(fft_x.real, min=-1e8, max=1e8),
            #     torch.clamp(fft_x.imag, min=-1e8, max=1e8)
            # )
            f_x = torch.abs(fft_x)
            # f_x = torch.clamp(f_x, min=1e-8, max=1e8)
            fft_recon_x = torch.fft.fft(recon_x, dim=-1)
            # fft_recon_x = torch.complex(
            #     torch.clamp(fft_recon_x.real, min=-1e8, max=1e8),
            #     torch.clamp(fft_recon_x.imag, min=-1e8, max=1e8)
            # )
            f_recon_x = torch.abs(fft_recon_x)
            # f_recon_x = torch.clamp(f_recon_x, min=1e-8, max=1e8)
            recon_loss = nn.MSELoss(reduction='sum')(f_recon_x, f_x)
        else:
            fft_x = torch.fft.fft(x, dim=-1)
            # fft_x = torch.complex(
            #     torch.clamp(fft_x.real, min=-1e8, max=1e8),
            #     torch.clamp(fft_x.imag, min=-1e8, max=1e8)
            # )
            p_x = torch.angle(fft_x)
            fft_recon_x = torch.fft.fft(recon_x, dim=-1)
            # fft_recon_x = torch.complex(
            #     torch.clamp(fft_recon_x.real, min=-1e8, max=1e8),
            #     torch.clamp(fft_recon_x.imag, min=-1e8, max=1e8)
            # )
            p_recon_x = torch.angle(fft_recon_x)
            recon_loss = nn.MSELoss(reduction='sum')(p_recon_x, p_x)
            
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        out = self.reparameterize(mu, logvar).reshape(B, C, -1, self.d_model).permute(0, 2, 1, 3)
        
        return out, recon_loss, kld
    

    
class E2EBlock(torch.nn.Module):
    def __init__(self, in_planes, planes, roi_num, bias=True):
        super().__init__()
        self.d = roi_num
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d, 3)+torch.cat([b]*self.d, 2)
    
    
class BrainNetCNN(nn.Module):
    """BrainNetCNN: Convolutional neural networks for brain networks; towards predicting neurodevelopment"""
    def __init__(self, node_size, d_model):
        super().__init__()
        self.in_planes = 1
        self.d = node_size

        self.e2econv1 = E2EBlock(94, 32, node_size, bias=True)
        self.e2econv2 = E2EBlock(32, 32, node_size, bias=True)
        self.E2N = torch.nn.Conv2d(32, d_model, (1, self.d))
        self.N2G = torch.nn.Conv2d(d_model, d_model, (self.d, 1))
        self.dense1 = torch.nn.Linear(64, 32)

    def forward(self, node_feature: torch.tensor):
        # pdb.set_trace()
        # node_feature = node_feature.unsqueeze(dim=1)
        out = F.leaky_relu(self.e2econv1(node_feature), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5).squeeze(-1).squeeze(-1)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(
            self.dense1(out), negative_slope=0.33), p=0.5)

        return out


