import torch
from torch import Tensor, device
from torch.nn import Parameter as Param
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor

from torch_geometric.nn.inits import uniform

from get_device import try_device
from torch.nn.functional import pad

class CircleGated4GruLayer(MessagePassing):

    def __init__(self, 
                 out_channels: int, 
                 num_layers: int, 
                 max_node_per_graph,
                 add_self_loops = True,
                 aggr: str = 'add',
                 bias: bool = True, 
                 **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.max_node_per_graph = max_node_per_graph
        self.add_self_loops = add_self_loops

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.weight = Param(Tensor(num_layers, 4, out_channels, out_channels))

        self.rnn_1 = torch.nn.GRUCell(out_channels, out_channels, bias=bias)
        self.rnn_2 = torch.nn.GRUCell(out_channels, out_channels, bias=bias)
        self.rnn_3 = torch.nn.GRUCell(out_channels, out_channels, bias=bias)
        self.rnn_4 = torch.nn.GRUCell(out_channels, out_channels, bias=bias)
    
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn_1.reset_parameters()
        self.rnn_2.reset_parameters()
        self.rnn_3.reset_parameters()
        self.rnn_4.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        nodes_num = int(x.size(0))
        edge_index_1, edge_index_2, edge_index_3, edge_index_4 = self.circle_edges(nodes_num, edge_index)
        # x_1, x_2, x_3, x_4 = x, x, x, x
        for i in range(self.num_layers):
            m_1 = torch.matmul(x, self.weight[i][0])
            m_2 = torch.matmul(x, self.weight[i][1])
            m_3 = torch.matmul(x, self.weight[i][2])
            m_4 = torch.matmul(x, self.weight[i][3])

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            m_1 = self.propagate(edge_index_1, x=m_1, edge_weight=edge_weight, size=None)
            m_2 = self.propagate(edge_index_2, x=m_2, edge_weight=edge_weight, size=None)
            m_3 = self.propagate(edge_index_3, x=m_3, edge_weight=edge_weight, size=None)
            m_4 = self.propagate(edge_index_4, x=m_4, edge_weight=edge_weight, size=None)

            x_1 = self.rnn_1(m_1, x)
            x_2 = self.rnn_2(m_2, x)
            x_3 = self.rnn_3(m_3, x)
            x_4 = self.rnn_4(m_4, x)

            # aggregate
            # print(x[0:nodes_num])
            # print(x[nodes_num:2*nodes_num])
            # print(x[2*nodes_num:3*nodes_num])
            # print(x[3*nodes_num:4*nodes_num])

            x = (x_1 + x_2 + x_3 + x_4)

        return x

    def circle_edges(self, node_num, edge_list):
        assert len(edge_list) == 4
        A1, A2, A3, A4 = edge_list
        # n = node_num
        # n = self.max_node_per_graph

        return torch.cat([A1, A2, A3, A4], dim=1), torch.cat([A2, A3, A4, A1], dim=1), torch.cat([A3, A4, A1, A2], dim=1), torch.cat([A4, A1, A2, A3], dim=1)

    def matrix_transfer(self, edge, i, j):
        # edge: [[i],[j]]
        edge_new = edge.detach().clone()
        edge_new[0]+=i
        edge_new[1]+=j
        return edge_new

    def message(self, x_j: Tensor, edge_weight: OptTensor):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'num_layers={self.num_layers})')


if __name__ == "__main__":

    node_initial_embedding = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=torch.float32)

    ast = torch.tensor([[0,1],[0,2],[1,3],[3,0]])
    cfg = torch.tensor([[2,3],[0,3]])
    dfg = torch.tensor([[0,2],[0,1],[0,3]])
    ncs = torch.tensor([[2,3],[3,1]])

    ast = torch.tensor([[0, 0, 1, 3], [1, 2, 3, 0]])
    cfg = torch.tensor([[2, 0], [3, 3]])
    cfg = pad(cfg, (0, 2))
    dfg = torch.tensor([[0, 0, 0], [2, 1, 3]])
    dfg = pad(dfg, (0, 1))
    ncs = torch.tensor([[2, 3], [3, 1]])
    ncs = pad(ncs, (0, 2))

    edges_index = torch.tensor([item.detach().numpy()  for item in [ast, cfg, dfg, ncs]])
    # edges_index = torch.cat([ast, cfg, dfg, ncs], dim=0)

    # to device
    dev = try_device()
    node_initial_embedding = node_initial_embedding.to(dev)
    edges_index = edges_index.to(dev)

    net = CircleGated4GruLayer(out_channels=4, num_layers=4)
    net.to(dev)

    # forward propagation
    node_hidden_embedding = net(node_initial_embedding, edges_index)

    print("node_hidden_embedding: ", node_hidden_embedding)
    
