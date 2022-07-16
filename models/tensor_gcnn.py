from conv_output import *
from tensor_gcn_layer import *

import torch.nn as N

class TensorGCNN(N.Module):
    def __init__(self,
                 graph_conv_args,
                 conv_args,
                 out_size,
                 emb_size,
                 is_multi=False,
                 n_classes=1):
        super(TensorGCNN, self).__init__()

        self.tensor_gcn_layer = TensorGCNLayer(**graph_conv_args,)
        
        if not is_multi:
            self.output_layer = ConvOutputLayer(**conv_args,
                                                fc_1_size=out_size + emb_size,
                                                fc_2_size=out_size)
        else:
            self.output_layer = ConvMultiOutputLayer(**conv_args,
                                                    fc_1_size=out_size + emb_size,
                                                    fc_2_size=out_size,
                                                    n_classes=n_classes)
    
    def forward(self, data):
        x, ast_edge_index,cfg_edge_index,ddg_edge_index,ncs_edge_index = data.x, data.ast_edge_index,data.cfg_edge_index,data.ddg_edge_index,data.ncs_edge_index
        edge_list = [ast_edge_index, cfg_edge_index, ddg_edge_index, ncs_edge_index]

        circle_ggnn_output = self.tensor_gcn_layer(x, edge_list)
        
        y_hat = self.output_layer(circle_ggnn_output, x)
        
        return y_hat

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))