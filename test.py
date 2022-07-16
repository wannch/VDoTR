import torch
import torch_geometric as tg
from torch_geometric.nn.conv.gated_graph_conv import GatedGraphConv
import torch_sparse as ts
import torch.nn.functional as F
import torch.nn as N
from torch_geometric.data import Data

import pandas as pd

# i = [[0, 1, 1], [2, 0, 2]]
# v =  [7, 8, 9]
# s = torch.sparse_coo_tensor(i, v, (2, 3))
# print(isinstance(s, ts.SparseTensor))
# print(s)

# input.apply_(lambda x: 1 if x >= 0.5 else 0)
# print(input.requires_grad)

# print(input.numel())

# test 1-D conv and maxpool effect
# input = torch.randn((32, 600, 101))
# print(input.shape)
# conv1d_1 = N.Conv1d(in_channels=600, out_channels=50, kernel_size=3, padding=1)
# maxpool_1 = N.MaxPool1d(kernel_size=3, stride=2)
# conv1d_2 = N.Conv1d(in_channels=50, out_channels=20, kernel_size=1, padding=0)
# maxpool_2 = N.MaxPool1d(kernel_size=2, stride=2)
# c1 = conv1d_1(input)
# print(c1.shape)
# m1 = maxpool_1(c1)
# print(m1.shape)

# c2 = conv1d_2(m1)
# print(c2.shape)
# m2 = maxpool_2(c2)
# print(m2.shape)

# x = torch.randn(4, 4)
# print(x.size())
# y = x.view(16)
# print(x.size())
# print(y.size())

# convert dataset target to one-hot
# ds_path = "/home/passwd123/wch/VDoTR/dataset/composite-1"
# ds_path = "/home/passwd123/wch/v_dataset/CWE-119_dir_000"
# ds_df = pd.read_pickle(ds_path)
# a = ds_df["target"][1:5].tolist()
# print(a)
# for x in a:
#     print(type(x))
#     x = [b.item() for b in x]
#     print(x)
# # a = a.detach().apply(lambda x : torch.tensor(x))
# a = torch.tensor(a)
# print(a)
# _, y = torch.where(a == 1)
# print(y)
# ds_df["target"] = ds_df["target"].apply(lambda x : [1, 0] if x == 0 else [0, 1])
# print(ds_df["target"])

# test softmax and binary_cross_entropy
# input = torch.randn(10, 6, requires_grad=True)
# target = F.one_hot(torch.tensor([1, 2, 3, 4, 5, 4, 5, 1, 2, 3]), 6)
# sm_layer = N.Softmax(dim=1)
# y_hat = sm_layer(input)
# # y_hat = torch.argmax(y_hat, dim=1)
# _, y = torch.where(target == 1)
# # print(y)
# # print((y_hat == y).sum().item())
# print(y_hat, y)
# loss = F.cross_entropy(y_hat, y)
# print(loss)

# a = torch.randint(1, 4, (2, 3))
# b = torch.randint_like(a, 1, 4)
# print(a)
# print(b)
# print((a + b) / 2)
# print(a * b)

# x_1 = torch.rand(30, 101)
# x_2 = torch.rand(30, 101)
# x_3 = torch.rand(30, 101)
# n = int(x_1.size(0))
# n_x = [torch.tensor([x_1[i].detach().tolist(), x_2[i].detach().tolist(), x_3[i].detach().tolist()])  for i in range(n)]
# print(n_x)

# v_e = torch.Tensor([[s, t] for s in range(4) for t in range(4)]).t().contiguous()
# print(v_e)

# a = torch.rand(8, 5)
# print(a.repeat(4, 1))
# b = a.expand(4, -1, -1).clone()
# print(b)
# c = b.permute(1, 0, 2)
# print(c)
# d1 = c.size(0)
# d2 = c.size(1)
# d = c.reshape(d1 * d2, -1)
# print(d)
# print(d.shape)

# virtual_edges = torch.tensor([[s, t] for s in range(4) for t in range(4)]).t().contiguous()
# n = 8
# print(virtual_edges)
# virtual_edges = torch.cat([ virtual_edges+4*i for i in range(n) ], dim=1)
# print(virtual_edges)
# print(virtual_edges.shape)

# out = d.reshape(d1, d2, -1)
# print(out)
# print(out.shape)

# out_y = torch.sum(out, dim=1)
# print(out_y)
# print(out_y.shape)

# a = torch.rand(3, 4)
# print(a)
# b = torch.cat([a, a, a, a], dim=0)
# print(b)
# c = b.view(4, 3, -1)
# print(c)

# a = torch.tensor([1,0,0,0])
# if torch.argmax(a) == 0:
#     print("hhh1")

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
y = 1
d = Data(x=x, edge_index=edge_index, y=y)

print(d)
d.y = 2
print(d)
print(d.y)
