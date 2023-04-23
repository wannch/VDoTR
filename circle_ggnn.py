from torch.nn import *
from circle_gated_graph_4gru import *
from circle_gated_graph import *
from conv_output import *

class CircleGGNN(Module):

    def __init__(self, gated_graph_args, conv_args, emb_size):
        super(CircleGGNN, self).__init__()
        
        self.circle_ggnn_layer = CircleGatedGraphLayer(**gated_graph_args)

        self.conv_output_layer = ConvOutputLayer(**conv_args,
                                                fc_1_size=gated_graph_args["out_channels"] + emb_size,
                                                fc_2_size=gated_graph_args["out_channels"])
        # self.conv.apply(init_weights)

    def forward(self, data):
        x, ast_edge_index,cfg_edge_index,ddg_edge_index,ncs_edge_index = data.x, data.ast_edge_index,data.cfg_edge_index,data.ddg_edge_index,data.ncs_edge_index
        edge_list=[ast_edge_index, cfg_edge_index, ddg_edge_index, ncs_edge_index]

        circle_ggnn_output = self.circle_ggnn_layer(x,edge_list)
        
        y_hat = self.conv_output_layer(circle_ggnn_output, x)
        #x = self.conv(x, data.x)

        return y_hat

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class MultiCircleGGNN(Module):

    def __init__(self, gated_graph_args, conv_args, emb_size, n_classes):
        super(MultiCircleGGNN, self).__init__()
        
        # self.circle_ggnn_layer = CircleGatedGraphLayer(**gated_graph_args)
        self.circle_ggnn_layer = CircleGated4GruLayer(**gated_graph_args)       # four GRU cells

        self.conv_output_layer = ConvMultiOutputLayer(**conv_args,
                                                fc_1_size=gated_graph_args["out_channels"] + emb_size,
                                                fc_2_size=gated_graph_args["out_channels"],
                                                n_classes=n_classes)
        # self.conv.apply(init_weights)

    def forward(self, data):
        x, ast_edge_index,cfg_edge_index,ddg_edge_index,ncs_edge_index = data.x, data.ast_edge_index,data.cfg_edge_index,data.ddg_edge_index,data.ncs_edge_index
        edge_list=[ast_edge_index, cfg_edge_index, ddg_edge_index, ncs_edge_index]

        circle_ggnn_output = self.circle_ggnn_layer(x,edge_list)
        
        y_hat = self.conv_output_layer(circle_ggnn_output, x)
        #x = self.conv(x, data.x)

        return y_hat

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))