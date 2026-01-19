from .CVIBLayers import *
from ..base import BaseConfig, ModelOutputs


class CVIBConfig(BaseConfig):
    def __init__(self,
                 node_size,
                 d_model=128,
                 num_classes=2,
                 num_heads=1,
                 abla_channel=-1,
                 abla_vae=None,
                 num_layers=4,
                 sparsity=30,
                 dropout=0.5,
                 cls_token='sum',
                 readout='sero',
                 window_size=50,
                 window_stride=3,
                 dynamic_length=99,
                 sampling_init=None,
                 integration="add",
                 cor_comput="pearson",
                 ):
        super(CVIBConfig, self).__init__(dropout=dropout)
        self.node_size = node_size
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.abla_channel = abla_channel
        self.abla_vae = abla_vae
        self.num_layers = num_layers
        self.sparsity = sparsity
        self.cls_token = cls_token
        self.readout = readout
        # self.readout = "mean"
        self.clip_grad = 0.0
        self.reg_lambda = 1e-5
        self.vae_alpha = 1e-5
        self.window_size = window_size
        self.window_stride = window_stride
        self.dynamic_length = dynamic_length
        self.sampling_init = sampling_init
        self.integration = integration
        self.cor_comput = cor_comput


class CVIB(nn.Module):
    """
    STAGIN from https://github.com/egyptdj/stagin
    """
    def __init__(self, config: CVIBConfig):
        super().__init__()
        self.config = config
        
        self.vae_configs = {
            "t": ("f", "p"),  # 当abla_vae="t"时，frequency_vae用"f"，phase_vae用"p"
            "f": ("t", "p"),  # 当abla_vae="f"时，frequency_vae用"t"，phase_vae用"p"  
            "p": ("t", "f"),  # 当abla_vae="p"时，frequency_vae用"t"，phase_vae用"f"
        }
        
        if config.abla_vae in self.vae_configs:
            view1, view2 = self.vae_configs[config.abla_vae]
            self.vae1 = VAE(view=view1, d_model=config.d_model//2)
            self.vae2 = VAE(view=view2, d_model=config.d_model//2)
        else:
            # 默认情况：创建三个VAE
            self.vae1 = VAE(view="t", d_model=config.d_model//2)
            self.vae2 = VAE(view="f", d_model=config.d_model//2) 
            self.vae3 = VAE(view="p", d_model=config.d_model//2)

        self.num_classes = config.num_classes
        
        if config.abla_channel >= 0:
            self.bnc = BrainNetCNN(config.node_size-1, config.d_model)
        else:
            self.bnc = BrainNetCNN(config.node_size, config.d_model)


        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        self.dense3 = torch.nn.Linear(32, config.num_classes)

        # self.lossWeight = UncertaintyWeighting(3)
        
        self.last_logit_loss = torch.tensor(1, device='cuda')
        self.last_vae_loss = torch.tensor(1000000, device='cuda')

    def forward(self, time_series, node_feature, labels, r1_mu, r1_logvar, r2_mu, r2_logvar, r3_mu, r3_logvar, train):
        
        #消融脑区2，65，36，55，24
        # pdb.set_trace()
        # (B, 68, 1500)
        if self.config.abla_channel >= 0:
            time_series = torch.cat([time_series[:, :self.config.abla_channel, :], time_series[:, self.config.abla_channel+1:, :]], dim=1)
        
        # pdb.set_trace()

        if self.config.abla_vae in self.vae_configs:
            mu1, rl1, kl1 = self.vae1(time_series, r1_mu, r1_logvar)
            mu2, rl2, kl2 = self.vae2(time_series, r2_mu, r2_logvar)
            a1 = self.batch_channel_pearson(mu1)
            a2 = self.batch_channel_pearson(mu2)
            adj = (a1 + a2) / 2
            adj = torch.mean(adj, dim=1).unsqueeze(1)
            
        else:
            mu1, rl1, kl1 = self.vae1(time_series, r1_mu, r1_logvar)
            mu2, rl2, kl2 = self.vae2(time_series, r2_mu, r2_logvar)
            mu3, rl3, kl3 = self.vae3(time_series, r3_mu, r3_logvar)
            a1 = self.batch_channel_pearson(mu1)
            a2 = self.batch_channel_pearson(mu2)
            a3 = self.batch_channel_pearson(mu3)
            adj = (a1 + a2 + a3) / 3
            adj = torch.mean(adj, dim=1).unsqueeze(1)
            
        out = self.bnc(adj)
        
        logits = F.leaky_relu(self.dense3(out), negative_slope=0.33)
        
        logit_loss = self.loss_fn(logits, labels)

        if self.config.abla_vae in self.vae_configs:
            # kl = (kl1 + kl2) / 2
            # rl = (rl1 + rl2) / 2
            vae_loss = rl1 + rl2 + kl1 + kl2
        else:
            # kl = (kl1 + kl2 + kl3) / 3
            # rl = (rl1 + rl2 + rl3) / 3
            vae_loss = rl1 + rl2 + rl3 + kl1 + kl2 + kl3
        
        loss = logit_loss + (self.last_logit_loss.detach() / self.last_vae_loss.detach()) * vae_loss

        # losses = [logit_loss, kl, rl]
        # loss = self.lossWeight(losses)
        
        self.last_logit_loss = logit_loss
        self.last_vae_loss = vae_loss

        if train:
            if self.config.abla_vae in self.vae_configs:
                return ModelOutputs(logits=logits, loss=loss), mu1, mu2, None
            else:
                return ModelOutputs(logits=logits, loss=loss), mu1, mu2, mu3
        else:
            return ModelOutputs(logits=logits, loss=loss)
        
    def compute_channel_attention(self, x):
        """
        计算每个批次、每个时间步的通道注意力分数。
        参数:
            x: 输入张量，形状为 (B, L, C, F)
        返回:
            注意力分数张量，形状为 (B, L, C, C)
        """
        B, L, C, F = x.shape

        # 调整形状以便于计算注意力：(B*L, C, F)
        x_reshaped = x.reshape(B * L, C, F)

        # 计算注意力分数 (QK^T)，这里使用输入张量作为Q和K
        # (B*L, C, F) * (B*L, F, C) -> (B*L, C, C)
        attention_scores = torch.bmm(x_reshaped, x_reshaped.transpose(1, 2))

        # 可选：缩放注意力分数
        attention_scores = attention_scores / (F ** 0.5)

        # 可选：应用softmax
        attention_scores = torch.tanh(attention_scores)

        # 恢复形状为 (B, L, C, C)
        attention_scores = attention_scores.reshape(B, L, C, C)

        return attention_scores

    def batch_channel_pearson(self, x):
        """
        计算形状为(B, L, C, F)的张量在C维度上的Pearson相关系数

        参数:
            x: 输入张量，形状为(B, L, C, F)

        返回:
            corr: Pearson相关系数矩阵，形状为(B, L, C, C)
        """
        # 计算均值
        mean_x = x.mean(dim=-1, keepdim=True)  # (B, L, C, 1)

        # 中心化数据
        x_centered = x - mean_x  # (B, L, C, F)

        # 计算协方差矩阵
        cov_matrix = torch.matmul(x_centered, x_centered.transpose(-2, -1))  # (B, L, C, C)

        # 计算标准差
        std_x = torch.sqrt(torch.sum(x_centered ** 2, dim=-1, keepdim=True))  # (B, L, C, 1)

        # 计算相关系数矩阵
        corr_matrix = cov_matrix / (std_x @ std_x.transpose(-2, -1))

        # 处理数值稳定性（避免除以零）
        corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        if torch.isnan(corr_matrix).any():
            pdb.set_trace()

        return corr_matrix
    
    


