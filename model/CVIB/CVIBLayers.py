import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import pdb

class UncertaintyWeighting(nn.Module):
    """
    使用不确定性自动学习权重（Kendall et al., 2018）
    每个损失有自己的可学习log方差参数
    """
    def __init__(self, num_losses):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
        
    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + 0.5 * self.log_vars[i]
        return total_loss

def gaussian_kl_divergence(mu1, logvar1, mu2=None, logvar2=None):
    """
    计算两个高斯分布之间的KL散度
    
    参数:
        mu1: torch.Tensor, 形状为 (B, C, L), 第一个高斯分布的均值
        logvar1: torch.Tensor, 形状为 (B, C, L), 第一个高斯分布的对数方差
        mu2: torch.Tensor, 形状为 (B, C, L), 第二个高斯分布的均值
             如果为None，则假设为标准正态分布N(0,1)
        logvar2: torch.Tensor, 形状为 (B, C, L), 第二个高斯分布的对数方差
               如果为None，则假设为标准正态分布N(0,1)
    
    返回:
        kl: torch.Tensor, 形状为 (B, C, L), 逐元素的KL散度
        kl_sum: torch.Tensor, 标量, 所有维度求和后的KL散度
        kl_mean: torch.Tensor, 标量, 所有维度的平均KL散度
    """
    # 确保输入形状一致
    assert mu1.shape == logvar1.shape, "mu1和logvar1形状必须一致"
    
    if mu2 is not None:
        assert mu2.shape == mu1.shape, "mu2必须与mu1形状一致"
        assert logvar2 is not None, "如果提供mu2，必须同时提供logvar2"
        assert logvar2.shape == logvar1.shape, "logvar2必须与logvar1形状一致"
    
    # 计算方差（从logvar转换）
    var1 = torch.exp(logvar1)
    
    if mu2 is None:
        # 与标准正态分布N(0,1)的KL散度
        # KL(q||p) = -0.5 * ∑(1 + log(σ^2) - μ^2 - σ^2)
        kl = -0.5 * (1 + logvar1 - mu1.pow(2) - var1)
    else:
        # 计算第二个分布的方差
        var2 = torch.exp(logvar2)
        
        # 两个高斯分布之间的KL散度公式:
        # KL(q||p) = 0.5 * (log(σ2^2/σ1^2) + (σ1^2 + (μ1-μ2)^2)/σ2^2 - 1)
        # 其中q~N(μ1,σ1^2), p~N(μ2,σ2^2)
        
        # 使用数值稳定的版本
        logvar_ratio = logvar2 - logvar1  # log(σ2^2/σ1^2)
        mu_diff_sq = (mu1 - mu2).pow(2)   # (μ1-μ2)^2
        var_ratio = var1 / var2           # σ1^2/σ2^2
        
        kl = 0.5 * (logvar_ratio + mu_diff_sq / var2 + var_ratio - 1)
    
    return kl

class VAE(nn.Module):
    def __init__(self, view, channel1=8, channel2=16, channel3=32, d_model=32):
        super(VAE, self).__init__()
        
        self.view = view
        self.d_model = d_model
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channel1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(channel1),
            nn.LeakyReLU(negative_slope=0.33),
            nn.Conv1d(channel1, channel2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(channel2),
            nn.LeakyReLU(negative_slope=0.33),
            nn.Conv1d(channel2, channel3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(channel3),
            nn.LeakyReLU(negative_slope=0.33),
            nn.Conv1d(channel3, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(negative_slope=0.33)
            # nn.Conv1d(channel3, d_model, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm1d(d_model),
            # nn.LeakyReLU(negative_slope=0.33)
        )
        
        # 潜在空间的均值和方差
        self.fc_mu = nn.Linear(d_model, d_model)
        self.fc_logvar = nn.Linear(d_model, d_model)
        
        # 解码器
        self.decoder = nn.Sequential(
            # nn.ConvTranspose1d(d_model, channel3,  kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm1d(channel3),
            # nn.LeakyReLU(negative_slope=0.33),
            nn.ConvTranspose1d(d_model, channel3,  kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(channel3),
            nn.LeakyReLU(negative_slope=0.33),
            nn.ConvTranspose1d(channel3, channel2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(channel2),
            nn.LeakyReLU(negative_slope=0.33),
            nn.ConvTranspose1d(channel2, channel1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(channel1),
            nn.LeakyReLU(negative_slope=0.33),
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
    
    def forward(self, x, r_mu, r_logvar):
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
            fft_x = torch.complex(
                torch.clamp(fft_x.real, min=-1e8, max=1e8),
                torch.clamp(fft_x.imag, min=-1e8, max=1e8)
            )
            f_x = torch.abs(fft_x)
            f_x = torch.clamp(f_x, min=1e-8, max=1e8)
            fft_recon_x = torch.fft.fft(recon_x, dim=-1)
            fft_recon_x = torch.complex(
                torch.clamp(fft_recon_x.real, min=-1e8, max=1e8),
                torch.clamp(fft_recon_x.imag, min=-1e8, max=1e8)
            )
            f_recon_x = torch.abs(fft_recon_x)
            f_recon_x = torch.clamp(f_recon_x, min=1e-8, max=1e8)
            recon_loss = nn.MSELoss(reduction='sum')(f_recon_x, f_x)
        else:
            fft_x = torch.fft.fft(x, dim=-1)
            fft_x = torch.complex(
                torch.clamp(fft_x.real, min=-1e8, max=1e8),
                torch.clamp(fft_x.imag, min=-1e8, max=1e8)
            )
            p_x = torch.angle(fft_x)
            fft_recon_x = torch.fft.fft(recon_x, dim=-1)
            fft_recon_x = torch.complex(
                torch.clamp(fft_recon_x.real, min=-1e8, max=1e8),
                torch.clamp(fft_recon_x.imag, min=-1e8, max=1e8)
            )
            p_recon_x = torch.angle(fft_recon_x)
            recon_loss = nn.MSELoss(reduction='sum')(p_recon_x, p_x)
            
        # kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        if r_mu is not None:
            r_mu = r_mu.permute(0, 2, 1, 3).reshape(B*C, -1, self.d_model)
            r_logvar = r_logvar.permute(0, 2, 1, 3).reshape(B*C, -1, self.d_model)
        kld = torch.sum(gaussian_kl_divergence(mu, logvar, r_mu, r_logvar))
        
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

        self.e2econv1 = E2EBlock(1, 32, node_size, bias=True)
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


