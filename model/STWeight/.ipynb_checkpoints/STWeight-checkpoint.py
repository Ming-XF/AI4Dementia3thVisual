from .STWeightLayers import *
from ..base import BaseConfig, ModelOutputs


class STWeightConfig(BaseConfig):
    def __init__(self,
                 node_size,
                 d_model=128,
                 num_classes=2,
                 num_heads=1,
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
        super(STWeightConfig, self).__init__(dropout=dropout)
        self.node_size = node_size
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_heads = num_heads
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


class STWeight(nn.Module):
    """
    STAGIN from https://github.com/egyptdj/stagin
    """
    def __init__(self, config: STWeightConfig):
        super().__init__()
        self.config = config
            
        self.time_vae = VAE(view="t", d_model=config.d_model//2)
        self.frequency_vae = VAE(view="f", d_model=config.d_model//2)
        self.phase_vae = VAE(view="p", d_model=config.d_model//2)

        self.num_classes = config.num_classes
        self.sparsity = config.sparsity

        self.percentile = Percentile()
        self.initial_linear = nn.Linear(config.node_size + config.d_model // 2 * 3, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        self.conv_spatial = nn.Conv2d(in_channels=config.d_model, out_channels=config.d_model, kernel_size=3, padding='same', bias=True)
        
        self.conv_time = nn.Conv2d(in_channels=config.d_model, out_channels=config.d_model, kernel_size=3, padding='same', bias=True)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        self.final_linear1 = torch.nn.Linear(config.d_model, config.d_model)
        self.final_linear2 = torch.nn.Linear(config.d_model * config.node_size, config.num_classes)
    
    def forward(self, time_series, node_feature, labels):
        
        # pdb.set_trace()
        time_mu, time_recon_loss, time_kld_loss = self.time_vae(time_series)
        frequency_mu, frequency_recon_loss, frequency_kld_loss = self.frequency_vae(time_series)
        phase_mu, phase_recon_loss, phase_kld_loss = self.phase_vae(time_series)
        # (B, T2, C, D)
        
        B, T, C, D = time_mu.shape
        device = time_mu.device
        
        time_a = self.pearson_correlation(time_mu)
        frequency_a = self.pearson_correlation(frequency_mu)
        phase_a = self.pearson_correlation(phase_mu)
        
        adj = (time_a + frequency_a + phase_a) / 3
        #(B, L, C, C)
        
        v = repeat(torch.eye(C), 'n1 n2 -> b t n1 n2', t=T, b=B).to(device)
        
        h = torch.cat([v, time_mu, frequency_mu, phase_mu], dim=3)
        h = self.initial_linear(h)
        #(B, L, C, D)
        
        spatial_h = torch.einsum('blcc,blcd->blcd', adj, h)
        spatial_h = spatial_h.permute(0, 3, 1, 2)
        spatial_h = self.conv_spatial(spatial_h)
        spatial_h = self.dropout(F.leaky_relu(spatial_h, negative_slope=0.33))
        spatial_h = spatial_h.permute(0, 3, 2, 1)
        B, C, L, D = spatial_h.shape
        #(B, C, L, D)
        
        attns = self.compute_channel_attention(spatial_h)
        time_h = torch.einsum('bcll,bcld->bcld', attns, spatial_h)
        time_h = time_h.permute(0, 3, 1, 2)
        time_h = self.conv_time(time_h)
        time_h = self.dropout(F.leaky_relu(time_h, negative_slope=0.33))
        #(B, D, C, L)
        time_h = time_h.permute(0, 2, 3, 1)
        #(B, C, L, D)
        
        out_h = torch.mean(time_h, dim=2)
        out_h = self.dropout(F.leaky_relu(self.final_linear1(out_h), negative_slope=0.33))
        
        out_h = out_h.reshape(B, -1)
        logits = F.leaky_relu(self.final_linear2(out_h), negative_slope=0.33)
        
        loss = self.loss_fn(logits, labels) + self.config.vae_alpha *(time_recon_loss + frequency_recon_loss + phase_recon_loss + time_kld_loss + time_kld_loss + time_kld_loss)
        
        return ModelOutputs(logits=logits,
                            loss=loss)
            
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

    def pearson_correlation(self, x):
        """
        计算输入张量在C维度上的皮尔逊相关系数矩阵

        参数:
        x: 形状为(B, L, C, F)的张量

        返回:
        corr: 形状为(B, L, C, C)的皮尔逊相关系数矩阵
        """
        # 计算每个特征的平均值
        mean_x = torch.mean(x, dim=-1, keepdim=True)

        # 计算每个特征与平均值的差值
        xm = x - mean_x

        # 计算协方差矩阵的分母(F-1)
        # 这里我们使用F作为分母(总体协方差)，因为PyTorch的cov函数也是这样做的
        # 如果需要样本协方差，可以改为(F-1)
        denominator = x.shape[-1]

        # 计算协方差矩阵
        # 我们需要先将张量重塑为(B*L, C, F)，然后计算协方差
        batch_size, seq_len, num_features, num_samples = x.shape
        x_reshaped = xm.reshape(batch_size * seq_len, num_features, num_samples)

        # 计算协方差矩阵
        cov = torch.bmm(x_reshaped, x_reshaped.transpose(1, 2)) / denominator

        # 计算标准差
        stddev = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2))

        # 计算相关系数矩阵
        stddev_outer = torch.bmm(stddev.unsqueeze(2), stddev.unsqueeze(1))
        corr = cov / (stddev_outer + 1e-8)  # 添加小常数防止除以0

        # 重塑回原始形状
        corr = corr.reshape(batch_size, seq_len, num_features, num_features)

        return corr
    
    def _collate_adjacency(self, t_mu, f_mu, p_mu, sparsity, sparse=True):
        
        # pdb.set_trace()
        a = None
        f_a = None
        p_a = None
        assert (self.config.cor_comput == "pearson"
                or self.config.cor_comput == "attention"
                ), "Not find cor_comput"
        if self.config.cor_comput == "pearson":
            # with torch.no_grad():#存在爆显存问题
            a = self.pearson_correlation(t_mu)
            f_a = self.pearson_correlation(f_mu)
            p_a = self.pearson_correlation(p_mu)
        else:
            a = self.compute_channel_attention(t_mu)
            f_a = self.compute_channel_attention(f_mu)
            p_a = self.compute_channel_attention(p_mu)
        
        i_list = []
        v_list = []
        
        # a2 = []

        # pdb.set_trace()
        for sample, (_dyn_a, _dyn_f_a, _dyn_p_a) in enumerate(zip(a, f_a, p_a)):
            for timepoint, (_a, _f_a, _p_a) in enumerate(zip(_dyn_a, _dyn_f_a, _dyn_p_a)):
                
                assert (self.config.integration == "add"
                        or self.config.integration == "union"
                        or self.config.integration == "intersection"
                       ), "Not find integration"
                
                if self.config.integration == "add":
                    _a = _a + _f_a + _p_a #三种方式：直接求和；取交集；取并集。每种方式添加权重
                    # a2.append(_a.unsqueeze(0))
                    thresholded_a = (_a > self.percentile(_a, 100 - sparsity))
                    
                    _i = thresholded_a.nonzero(as_tuple=False)
                    _v = _a[thresholded_a]
                    
                elif self.config.integration == "union":
                    thresholded_a = (_a > self.percentile(_a, 100 - sparsity))
                    thresholded_f_a = (_f_a > self.percentile(_f_a, 100 - sparsity))
                    thresholded_p_a = (_p_a > self.percentile(_p_a, 100 - sparsity))
                    
                    thresholded_all = thresholded_a | thresholded_f_a | thresholded_p_a
                    
                    _i = thresholded_all.nonzero(as_tuple=False)
                    
                    _a[~thresholded_a] = 0
                    _f_a[~thresholded_f_a] = 0
                    _p_a[~thresholded_p_a] = 0
                    _a = _a + _f_a + _p_a
                    # a2.append(_a.unsqueeze(0))
                    _v = _a[thresholded_all]

                else:
                    thresholded_a = (_a > self.percentile(_a, 100 - sparsity))
                    thresholded_f_a = (_f_a > self.percentile(_f_a, 100 - sparsity))
                    thresholded_p_a = (_p_a > self.percentile(_p_a, 100 - sparsity))
                    
                    thresholded_all = thresholded_a & thresholded_f_a & thresholded_p_a
                    a2.append(thresholded_all)
                    
                    _i = thresholded_all.nonzero(as_tuple=False)
                    
                    _a[~thresholded_all] = 0
                    _f_a[~thresholded_all] = 0
                    _p_a[~thresholded_all] = 0
                    _a = _a + _f_a + _p_a 
                    # a2.append(_a.unsqueeze(0))
                    _v = _a[thresholded_all]
                
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)
        
        a1 = torch.sparse.FloatTensor(_i, _v, (a.shape[0] * a.shape[1] * a.shape[2], a.shape[0] * a.shape[1] * a.shape[3]))
        # a2 = torch.cat(a2, dim=0).reshape(a.shape)
        
        return a1


