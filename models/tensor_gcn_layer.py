from torch import Tensor
import torch
import torch_geometric.nn as GN
import torch.nn as N
import torch as T

class TensorGCNLayer(N.Module):
    def __init__(self, layer_num,
                       intra_gcn_paras,
                       inter_gcn_paras,
                       device
                       ) -> None:
        super(TensorGCNLayer, self).__init__()

        self.device = device

        # self.intra_gcns = N.ModuleList([GN.GCNConv(**intra_gcn_paras) for i in range(layer_num)])
        self.intra_gcns = GN.GCNConv(**intra_gcn_paras)
        self.inter_gcn = GN.GCNConv(**inter_gcn_paras)

        virtual_edges_t = T.tensor([[s, t] for s in range(4) for t in range(4)], device=self.device).t().contiguous()
        self.virtual_edges = T.cat([ virtual_edges_t+4*i for i in range(128 * 600) ], dim=1)

    def forward(self, x, edge_index):
        nodes_num = int(x.size(0))
        # X = x.expand(4, -1, -1)
        # X = T.cat([x, x, x, x], dim=0)
        # X = X.view(4, nodes_num, -1)
        TX = T.cat([x, x, x, x], dim=0)
        # for i, gcn in enumerate(self.intra_gcns):
        # for j in range(4):
        #     TX[j] = self.intra_gcns(TX[j], edge_index[j])
        # X = T.cat(TX, dim=0)
        new_edge_index = T.cat([edge_index[i]+i*nodes_num for i in range(4)], dim=1)
        X = self.intra_gcns(TX, new_edge_index)
        X = X.view(4, nodes_num, -1)
        # nodes_num = int(x.size(0))
        # virtual_x = [T.tensor([X[0][i].detach().tolist(), X[1][i].detach().tolist(), X[2][i].detach().tolist(), X[3][i].detach().tolist()], device=self.device) for i in range(nodes_num)]
        virtual_x = X.permute(1, 0, 2)
        # nodes_num = int(virtual_x.size(0))
        n = int(virtual_x.size(0))
        copy = int(virtual_x.size(1))
        virtual_x = virtual_x.reshape(n * copy, -1)
        # virtual_edges = T.tensor([[s, t] for s in range(4) for t in range(4)], device=self.device).t().contiguous()
        # virtual_edges = T.cat([ virtual_edges+4*i for i in range(nodes_num) ], dim=1)
        # for i in range(len(virtual_x)):
        #     virtual_x[i] = self.inter_gcn(virtual_x[i], virtual_edges)
        # virtual_edges = self.virtual_edges.clone()
        if nodes_num < 128 * 600:
            virtual_edges = T.tensor([[s, t] for s in range(4) for t in range(4)], device=self.device).t().contiguous()
            tmp_virtual_edges = T.cat([ virtual_edges+4*i for i in range(nodes_num) ], dim=1)
            new_x = self.inter_gcn(virtual_x, tmp_virtual_edges)
        else:
            new_x = self.inter_gcn(virtual_x, self.virtual_edges)
        out = new_x.reshape(n, copy, -1)
        # out = T.tensor([T.sum(t, dim=0).tolist() for t in virtual_x], device=self.device)
        out_y = T.sum(out, dim=1)

        return out_y

    