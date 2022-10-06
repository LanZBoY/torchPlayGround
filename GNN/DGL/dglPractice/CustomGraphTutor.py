from turtle import forward
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class GraphSage(torch.nn.Module):

    def __init__(self, inFeats, outFeats) -> None:
        super(GraphSage, self).__init__()
        self.inFeats = inFeats
        self.outFeats = outFeats
        self.linear = nn.Linear(in_features= inFeats * 2, out_features=outFeats)

    def forward(self, graph:dgl.DGLGraph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.update_all(message_func=fn.u_add_v('x','x', 'm'), reduce_func=fn.mean('m', 'h_N'))
            print(graph.ndata)
            print(graph.edata)
            h_N = graph.ndata['h_N']
            h_total = torch.concat([h, h_N], dim = 1)
            print(h_total)
            return self.linear(h_total)

model = GraphSage(2, 2)

g = dgl.graph(data=([0, 1],[2, 2]))
g.ndata['x'] = torch.tensor(data=[
    [1., 0.],
    [0., 1.],
    [1., 1.]
    ], dtype=torch.float32)

temp = model(g, g.ndata['x'])

print(temp)