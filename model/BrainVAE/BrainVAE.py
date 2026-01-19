from .BrainVAELayers import *
from ..base import BaseConfig, ModelOutputs


class BrainVAEConfig(BaseConfig):
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
        super(BrainVAEConfig, self).__init__(dropout=dropout)
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


class BrainVAE(nn.Module):
    """
    STAGIN from https://github.com/egyptdj/stagin
    """
    def __init__(self, config: BrainVAEConfig):
        super().__init__()
        self.config = config
        # assert config.cls_token in ['sum', 'mean', 'param']
        # if config.readout == 'garo':
        #     readout_module = ModuleGARO
        # elif config.readout == 'sero':
        #     readout_module = ModuleSERO
        # elif config.readout == 'mean':
        #     readout_module = ModuleMeanReadout
        # else:
        #     raise

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

        # self.token_parameter = nn.Parameter(
        #     torch.randn([config.num_layers, 1, 1, config.d_model])) if config.cls_token == 'param' else None

        self.num_classes = config.num_classes
        # self.sparsity = config.sparsity
        
        # self.bnc = BrainNetCNN(config.node_size, config.d_model)
        #消融脑区2，65，36，55，24
        # self.bnc = BrainNetCNN(config.node_size-1, config.d_model)
        
        if config.abla_channel >= 0:
            self.bnc = BrainNetCNN(config.node_size-1, config.d_model)
        else:
            self.bnc = BrainNetCNN(config.node_size, config.d_model)

        # define modules
#         self.percentile = Percentile()
#         self.timestamp_encoder = ModuleTimestamping(config.node_size, config.d_model, config.d_model)
#         self.initial_linear = nn.Linear(config.node_size + config.d_model // 2 * 3, config.d_model)
#         self.gnn_layers = nn.ModuleList()
#         self.readout_modules = nn.ModuleList()
#         self.transformer_modules = nn.ModuleList()
#         self.linear_layers = nn.ModuleList()
#         self.dropout = nn.Dropout(config.dropout)

#         for i in range(config.num_layers):
#             self.gnn_layers.append(LayerGIN(config.d_model, config.d_model, config.d_model))
#             self.readout_modules.append(readout_module(hidden_dim=config.d_model, input_dim=config.node_size, dropout=0.1))
#             self.transformer_modules.append(
#                 ModuleTransformer(config.d_model, 2 * config.d_model, num_heads=1, dropout=0.1))
#             self.linear_layers.append(nn.Linear(config.d_model, config.num_classes))

        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        self.dense3 = torch.nn.Linear(32, config.num_classes)
        
        self.last_logit_loss = torch.tensor(1, device='cuda')
        self.last_vae_loss = torch.tensor(1000000, device='cuda')
        
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
    
    def forward(self, time_series, node_feature, labels):
        
        #消融脑区2，65，36，55，24
        # pdb.set_trace()
        # (B, 68, 1500)
        if self.config.abla_channel >= 0:
            time_series = torch.cat([time_series[:, :self.config.abla_channel, :], time_series[:, self.config.abla_channel+1:, :]], dim=1)
        
        # pdb.set_trace()

        if self.config.abla_vae in self.vae_configs:
            mu1, rl1, kl1 = self.vae1(time_series)
            mu2, rl2, kl2 = self.vae2(time_series)
            a1 = self.batch_channel_pearson(mu1)
            a2 = self.batch_channel_pearson(mu2)
            adj = (a1 + a2) / 2
            adj = torch.mean(adj, dim=1).unsqueeze(1)
            
        else:
            mu1, rl1, kl1 = self.vae1(time_series)
            mu2, rl2, kl2 = self.vae2(time_series)
            mu3, rl3, kl3 = self.vae3(time_series)
            a1 = self.batch_channel_pearson(mu1)
            a2 = self.batch_channel_pearson(mu2)
            a3 = self.batch_channel_pearson(mu3)
            adj = (a1 + a2 + a3) / 3
            adj = torch.mean(adj, dim=1).unsqueeze(1)
            
        # time_mu, time_recon_loss, time_kld_loss = self.time_vae(time_series)
        # frequency_mu, frequency_recon_loss, frequency_kld_loss = self.frequency_vae(time_series)
        # phase_mu, phase_recon_loss, phase_kld_loss = self.phase_vae(time_series)
        # # (B, T2, C, D)

        # time_a = self.batch_channel_pearson(time_mu)
        # frequency_a = self.batch_channel_pearson(frequency_mu)
        # phase_a = self.batch_channel_pearson(phase_mu)
        
        # adj = (time_a + frequency_a + phase_a) / 3
        # adj = (time_a + frequency_a) / 2
        # adj = torch.mean(adj, dim=1).unsqueeze(1)
        
        # pdb.set_trace()
        out = self.bnc(adj)
        
        logits = F.leaky_relu(self.dense3(out), negative_slope=0.33)
        
        logit_loss = self.loss_fn(logits, labels)

        if self.config.abla_vae in self.vae_configs:
            vae_loss = rl1 + rl2 + kl1 + kl2
        else:
            vae_loss = rl1 + rl2 + rl3 + kl1 + kl2 + kl3
        # vae_loss = time_recon_loss + frequency_recon_loss + phase_recon_loss + time_kld_loss + frequency_kld_loss + phase_kld_loss
        # vae_loss = time_recon_loss + frequency_recon_loss + time_kld_loss + frequency_kld_loss
        
        loss = logit_loss + (self.last_logit_loss.detach() / self.last_vae_loss.detach()) * vae_loss
        
        self.last_logit_loss = logit_loss
        self.last_vae_loss = vae_loss
        # loss = logit_alpha * logit_loss + vae_alpha * vae_loss
        
        return ModelOutputs(logits=logits,
                            loss=loss)
        
        
        
        
#         minibatch_size, num_timepoints, num_nodes = time_mu.shape[:3]
        
#         v = repeat(torch.eye(num_nodes), 'n1 n2 -> b t n1 n2', t=num_timepoints,
#                        b=minibatch_size).to(time_series.device)
        
#         # pdb.set_trace()
#         a = self._collate_adjacency(time_mu, frequency_mu, phase_mu, self.sparsity)
        
        
        
#         logits = 0.0
#         reg_ortho = 0.0
#         attention = {'node-attention': [], 'time-attention': []}
#         latent_list = []
#         # pdb.set_trace()
#         # minibatch_size, num_timepoints, num_nodes = B, L, C

#         # time_encoding = self.timestamp_encoder(t, sampling_endpoints)               # T x B x D
#         # time_encoding = repeat(time_encoding, 'b t c -> t b n c', n=num_nodes)      # B x T x N x D
        
#         # pdb.set_trace()
#         # h_bnc = self.bnc(node_feature).unsqueeze(1).repeat(1, v.shape[1], 1, 1)
#         # h_bnc = self.bnc(node_feature)
#         h = torch.cat([v, time_mu, frequency_mu, phase_mu], dim=3)            # B x T x N x (D+N)
#         h = rearrange(h, 'b t n c -> (b t n) c')
#         # pdb.set_trace()
#         h = self.initial_linear(h)                          # (BxTxN) x D
#         # a = self._collate_adjacency(a, self.sparsity)       # (BxTxN) x (BxTxN)
#         for layer, (G, R, T, L) in enumerate(
#                 zip(self.gnn_layers, self.readout_modules, self.transformer_modules, self.linear_layers)):
#             h = G(h, a)         # (BxTxN) x D
#             h_bridge = rearrange(h, '(b t n) c -> t b n c', t=num_timepoints, b=minibatch_size, n=num_nodes)  # TxBxNxD
#             h_readout, node_attn = R(h_bridge, node_axis=2)     # TxBxD, BxTxN
#             if self.token_parameter is not None: h_readout = torch.cat(
#                 [h_readout, self.token_parameter[layer].expand(-1, h_readout.shape[1], -1)])
#             h_attend, time_attn = T(h_readout)      # TxBxD, BxTxT
#             ortho_latent = rearrange(h_bridge, 't b n c -> (t b) n c')      # (TxB)xNxD
#             matrix_inner = torch.bmm(ortho_latent, ortho_latent.permute(0, 2, 1))
#             reg_ortho += (matrix_inner / matrix_inner.max(-1)[0].unsqueeze(-1) - torch.eye(num_nodes, device=matrix_inner.device)).triu().norm(
#                 dim=(1, 2)).mean()
#             latent = self.cls_token(h_attend)       # BxD
#             logits += self.dropout(L(latent))

#             attention['node-attention'].append(node_attn)
#             attention['time-attention'].append(time_attn)
#             latent_list.append(latent)

#         attention['node-attention'] = torch.stack(attention['node-attention'], dim=1).detach().cpu()
#         attention['time-attention'] = torch.stack(attention['time-attention'], dim=1).detach().cpu()
#         latent = torch.stack(latent_list, dim=1)

#         loss = self.loss_fn(logits, labels) + self.config.reg_lambda * reg_ortho + self.config.vae_alpha *(time_recon_loss + frequency_recon_loss + phase_recon_loss + time_kld_loss + time_kld_loss + time_kld_loss)
#         return ModelOutputs(logits=logits,
#                             loss=loss,
#                             hidden_state={'attention': attention,
#                                           'latent': latent,
#                                           'reg_ortho': reg_ortho})
    def cls_token(self, x):
        if self.config.cls_token == 'sum':
            return x.sum(0)
        elif self.config.cls_token == 'mean':
            return x.mean(0)
        elif self.config.cls_token == 'param':
            return x[-1]
        else:
            raise


