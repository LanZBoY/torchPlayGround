import torch
import dgl
from dgl.utils import expand_as_pair
import dgl.function as fn
import torch.nn.functional as F
from dgl.utils import check_eq_shape

class SAGEGConv(torch.nn.Module):
    
    def __init__(self, in_feats, out_feats, aggregator_type, bias = True, norm = None, activation = None) -> None:
        super(SAGEGConv, self).__init__()
        self.src_feats , self.dst_feats = expand_as_pair(in_feats)
        self.out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.activation = activation
        if aggregator_type not in ['mean', 'pool', 'lstm', 'gcn']:
            raise KeyError('Aggregator type {} not supported.'.format(aggregator_type))
        if aggregator_type in ['mean', 'pool', 'lstm']:
            self.fc_self = torch.nn.Linear(self.dst_feats, self.out_feats, bias=bias)
        self.fc_neigh = torch.nn.Linear(self.src_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = torch.nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            torch.nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            torch.nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
    
    def forward(self, graph : dgl.DGLGraph, feat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._aggre_type == 'mean':
                graph.srcdata['h'] = feat_src
                graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
        rst = self.fc_self(feat_src) + self.fc_neigh(h_neigh)
        print(rst)


model = SAGEGConv(4, 2, aggregator_type='mean')
g = dgl.graph(data=(torch.tensor([0]), torch.tensor([1])))
# g = dgl.add_self_loop(g)
g.ndata['x'] = torch.tensor(
    [
        [1., 0., 1., 0.],
        [0., 1., 0., 1.]
    ]
        )
model(g, g.ndata['x'])