import torch
from torch import Tensor, device
from torch.nn import Parameter as Param
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor

from torch_geometric.nn.inits import uniform

from get_device import try_device
from torch.nn.functional import pad

class CircleGatedGraphLayer(MessagePassing):

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

        self.weight = Param(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)
    
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        nodes_num = int(x.size(0))
        circle_edge_index = self.circle_edges(nodes_num, edge_index)

        for i in range(self.num_layers):
            m = torch.matmul(x, self.weight[i])

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            m = torch.cat([m, m, m], dim=0)
            m = self.propagate(circle_edge_index, x=m, edge_weight=edge_weight,
                               size=None)
            x = torch.cat([x, x, x], dim=0)
            x = self.rnn(m, x)

            # aggregate
            # print(x[0:nodes_num])
            # print(x[nodes_num:2*nodes_num])
            # print(x[2*nodes_num:3*nodes_num])
            # print(x[3*nodes_num:4*nodes_num])

            x = x[0:nodes_num] + x[nodes_num:2*nodes_num] + x[2*nodes_num:3*nodes_num] # + x[3*nodes_num:4*nodes_num]

        return x


    def message(self, x_j: Tensor, edge_weight: OptTensor):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'num_layers={self.num_layers})')

    def circle_edges(self, node_num, edge_list):
        assert len(edge_list) == 4
        A1, A2, A3, A4 = edge_list
        # n = node_num
        n = self.max_node_per_graph

        loop_edge_list = []
        loop_edge_list.append(A1)
        loop_edge_list.append(self.matrix_transfer(A3, n, 0))
        loop_edge_list.append(self.matrix_transfer(A4, 2*n, 0))
        # loop_edge_list.append(self.matrix_transfer(A4, 3*n, 0))

        loop_edge_list.append(self.matrix_transfer(A4, 0, n))
        loop_edge_list.append(self.matrix_transfer(A1, n, n))
        loop_edge_list.append(self.matrix_transfer(A3, 2*n, n))

        loop_edge_list.append(self.matrix_transfer(A3, 0, 2*n))
        loop_edge_list.append(self.matrix_transfer(A4, n, 2*n))
        loop_edge_list.append(self.matrix_transfer(A1, 2*n, 2*n))

        # loop_edge_list.append(self.matrix_transfer(A4, 0, n))
        # loop_edge_list.append(self.matrix_transfer(A1, n, n))
        # loop_edge_list.append(self.matrix_transfer(A2, 2*n, n))
        # loop_edge_list.append(self.matrix_transfer(A3, 3*n, n))

        # loop_edge_list.append(self.matrix_transfer(A3, 0, 2*n))
        # loop_edge_list.append(self.matrix_transfer(A4, n, 2*n))
        # loop_edge_list.append(self.matrix_transfer(A1, 2*n, 2*n))
        # loop_edge_list.append(self.matrix_transfer(A2, 3*n, 2*n))

        # loop_edge_list.append(self.matrix_transfer(A2, 0, 3*n))
        # loop_edge_list.append(self.matrix_transfer(A3, n, 3*n))
        # loop_edge_list.append(self.matrix_transfer(A4, 2*n, 3*n))
        # loop_edge_list.append(self.matrix_transfer(A1, 3*n, 3*n))

        return torch.cat(loop_edge_list, dim=1)

    def matrix_transfer(self, edge, i, j):
        # edge: [[i],[j]]
        edge_new = edge.detach().clone()
        edge_new[0]+=i
        edge_new[1]+=j
        return edge_new

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

    net = CircleGatedGraphLayer(out_channels=4, num_layers=4)
    net.to(dev)

    # forward propagation
    node_hidden_embedding = net(node_initial_embedding, edges_index)

    print("node_hidden_embedding: ", node_hidden_embedding)
    
