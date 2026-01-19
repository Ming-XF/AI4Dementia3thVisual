from .MTSTALayers import *
from ..base import BaseConfig, ModelOutputs

import pdb


class MTSTAConfig(BaseConfig):
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
                 integration="add"
                 ):
        super(MTSTAConfig, self).__init__(dropout=dropout)
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
        self.window_size = window_size
        self.window_stride = window_stride
        self.dynamic_length = dynamic_length
        self.sampling_init = sampling_init
        self.integration = integration


class MTSTA(nn.Module):
    """
    STAGIN from https://github.com/egyptdj/stagin
    """
    def __init__(self, config: MTSTAConfig):
        super().__init__()
        self.config = config
        assert config.cls_token in ['sum', 'mean', 'param']
        if config.readout == 'garo':
            readout_module = ModuleGARO
        elif config.readout == 'sero':
            readout_module = ModuleSERO
        elif config.readout == 'mean':
            readout_module = ModuleMeanReadout
        else:
            raise

        self.token_parameter = nn.Parameter(
            torch.randn([config.num_layers, 1, 1, config.d_model])) if config.cls_token == 'param' else None

        self.num_classes = config.num_classes
        self.sparsity = config.sparsity

        # define modules
        self.percentile = Percentile()
        self.timestamp_encoder = ModuleTimestamping(config.node_size, config.d_model, config.d_model)
        self.initial_linear = nn.Linear(config.node_size + config.d_model, config.d_model)
        self.gnn_layers = nn.ModuleList()
        self.readout_modules = nn.ModuleList()
        self.transformer_modules = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.dropout = nn.Dropout(config.dropout)

        for i in range(config.num_layers):
            self.gnn_layers.append(LayerGIN(config.d_model, config.d_model, config.d_model))
            self.readout_modules.append(readout_module(hidden_dim=config.d_model, input_dim=config.node_size, dropout=0.1))
            self.transformer_modules.append(
                ModuleTransformer(config.d_model, 2 * config.d_model, num_heads=config.num_heads, dropout=0.1))
            self.linear_layers.append(nn.Linear(config.d_model, config.num_classes))

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def _collate_adjacency(self, a, f_a, p_a, sparsity, sparse=True):
        i_list = []
        v_list = []

        # pdb.set_trace()
        for sample, (_dyn_a, _dyn_f_a, _dyn_p_a) in enumerate(zip(a, f_a, p_a)):
            for timepoint, (_a, _f_a, _p_a) in enumerate(zip(_dyn_a, _dyn_f_a, _dyn_p_a)):
                
                assert (self.config.integration == "add"
                        or self.config.integration == "union"
                        or self.config.integration == "intersection"
                       ), "Not find integration"
                
                if self.config.integration == "add":
                    _a = _a + _f_a + _p_a #三种方式：直接求和；取交集；取并集。每种方式添加权重
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
                    _v = _a[thresholded_all]
                    
                else:
                    thresholded_a = (_a > self.percentile(_a, 100 - sparsity))
                    thresholded_f_a = (_f_a > self.percentile(_f_a, 100 - sparsity))
                    thresholded_p_a = (_p_a > self.percentile(_p_a, 100 - sparsity))
                    
                    thresholded_all = thresholded_a & thresholded_f_a & thresholded_p_a
                    
                    _i = thresholded_all.nonzero(as_tuple=False)
                    
                    _a[~thresholded_all] = 0
                    _f_a[~thresholded_all] = 0
                    _p_a[~thresholded_all] = 0
                    _a = _a + _f_a + _p_a 
                    _v = _a[thresholded_all]
                
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)

        return torch.sparse.FloatTensor(_i, _v,
                                        (a.shape[0] * a.shape[1] * a.shape[2], a.shape[0] * a.shape[1] * a.shape[3]))

    def forward(self, v, a, f_a, p_a, t, sampling_endpoints, labels):
        # assumes shape [minibatch x time x node x feature] for v
        # assumes shape [minibatch x time x node x node] for a
        logits = 0.0
        reg_ortho = 0.0
        attention = {'node-attention': [], 'time-attention': []}
        latent_list = []
        minibatch_size, num_timepoints, num_nodes = a.shape[:3]

        time_encoding = self.timestamp_encoder(t, sampling_endpoints)               # T x B x D
        time_encoding = repeat(time_encoding, 'b t c -> t b n c', n=num_nodes)      # B x T x N x D

        h = torch.cat([v, time_encoding], dim=3)            # B x T x N x (D+N)
        h = rearrange(h, 'b t n c -> (b t n) c')
        h = self.initial_linear(h)                          # (BxTxN) x D
        # pdb.set_trace()
        a = self._collate_adjacency(a, f_a, p_a, self.sparsity)       # (BxTxN) x (BxTxN)
        for layer, (G, R, T, L) in enumerate(
                zip(self.gnn_layers, self.readout_modules, self.transformer_modules, self.linear_layers)):
            h = G(h, a)         # (BxTxN) x D
            h_bridge = rearrange(h, '(b t n) c -> t b n c', t=num_timepoints, b=minibatch_size, n=num_nodes)  # TxBxNxD
            h_readout, node_attn = R(h_bridge, node_axis=2)     # TxBxD, BxTxN
            if self.token_parameter is not None: h_readout = torch.cat(
                [h_readout, self.token_parameter[layer].expand(-1, h_readout.shape[1], -1)])
            h_attend, time_attn = T(h_readout)      # TxBxD, BxTxT
            ortho_latent = rearrange(h_bridge, 't b n c -> (t b) n c')      # (TxB)xNxD
            matrix_inner = torch.bmm(ortho_latent, ortho_latent.permute(0, 2, 1))
            reg_ortho += (matrix_inner / matrix_inner.max(-1)[0].unsqueeze(-1) - torch.eye(num_nodes,
                                                                                           device=matrix_inner.device)).triu().norm(
                dim=(1, 2)).mean()

            latent = self.cls_token(h_attend)       # BxD
            logits += self.dropout(L(latent))

            attention['node-attention'].append(node_attn)
            attention['time-attention'].append(time_attn)
            latent_list.append(latent)

        attention['node-attention'] = torch.stack(attention['node-attention'], dim=1).detach().cpu()
        attention['time-attention'] = torch.stack(attention['time-attention'], dim=1).detach().cpu()
        latent = torch.stack(latent_list, dim=1)

        loss = self.loss_fn(logits, labels) + self.config.reg_lambda * reg_ortho
        return ModelOutputs(logits=logits,
                            loss=loss,
                            hidden_state={'attention': attention,
                                          'latent': latent,
                                          'reg_ortho': reg_ortho})

    def cls_token(self, x):
        if self.config.cls_token == 'sum':
            return x.sum(0)
        elif self.config.cls_token == 'mean':
            return x.mean(0)
        elif self.config.cls_token == 'param':
            return x[-1]
        else:
            raise


