from torch.nn import *
import torch.nn.functional as F
import torch

def get_conv_mp_out_size(in_size, last_channels, mps):
    size = in_size

    for mp in mps:
        size = round((size - mp["kernel_size"]) / mp["stride"] + 1)
        #size = round((size - mp["kernel_size"]) / mp["stride"] + 1)

    # size = size + 1 if size %2 != 0 else size  #%2

    return int(size * last_channels)

class ConvOutputLayer(Module):

    def __init__(self, conv1d_1, conv1d_2, maxpool1d_1, maxpool1d_2, fc_1_size, fc_2_size):
        super(ConvOutputLayer, self).__init__()

        self.conv1d_1_args = conv1d_1
        self.conv1d_2_args = conv1d_2

        self.conv1d_1 = Conv1d(**conv1d_1)
        self.conv1d_2 = Conv1d(**conv1d_2)

        self.mp_1 = MaxPool1d(**maxpool1d_1)
        self.mp_2 = MaxPool1d(**maxpool1d_2)

        last_conv_channels = conv1d_2["out_channels"]
        fc1_size = get_conv_mp_out_size(fc_1_size, last_conv_channels, [maxpool1d_1, maxpool1d_2])
        fc2_size = get_conv_mp_out_size(fc_2_size, last_conv_channels, [maxpool1d_1, maxpool1d_2])

        # Dense layers
        # fc1_size = 1020
        #fc2_size=340
        self.fc1 = Linear(fc1_size, 1)
        self.fc2 = Linear(fc2_size, 1)

        # Dropout
        # self.drop = Dropout(p=0.2)

    def forward(self, hidden, x):
        concat = torch.cat([hidden, x], 1)
        concat_size = hidden.shape[1] + x.shape[1]

        concat = concat.view(-1, self.conv1d_1_args["in_channels"], concat_size)

        middle = self.conv1d_1(concat)
        middle_1 = F.relu(middle)
        Z = self.mp_1(middle_1)
        Z = self.mp_2(self.conv1d_2(Z))

        hidden = hidden.view(-1, self.conv1d_1_args["in_channels"], hidden.shape[1])

        Y = self.mp_1(F.relu(self.conv1d_1(hidden)))
        Y = self.mp_2(self.conv1d_2(Y))

        Z_flatten_size = int(Z.shape[1] * Z.shape[-1])
        Y_flatten_size = int(Y.shape[1] * Y.shape[-1])

        Z = Z.view(-1, Z_flatten_size)
        Y = Y.view(-1, Y_flatten_size)
        res = self.fc1(Z) * self.fc2(Y)
        # res = self.drop(res)

        # res = res.mean(1)
        # print(res, mean)
        
        sig = torch.sigmoid(torch.flatten(res))

        return sig

class ConvMultiOutputLayer(Module):

    def __init__(self, conv1d_1, conv1d_2, maxpool1d_1, maxpool1d_2, fc_1_size, fc_2_size, n_classes):
        super(ConvMultiOutputLayer, self).__init__()

        self.conv1d_1_args = conv1d_1
        self.conv1d_2_args = conv1d_2

        self.conv1d_1 = Conv1d(**conv1d_1)
        self.conv1d_2 = Conv1d(**conv1d_2)

        self.mp_1 = MaxPool1d(**maxpool1d_1)
        self.mp_2 = MaxPool1d(**maxpool1d_2)

        last_conv_channels = conv1d_2["out_channels"]
        fc1_size = get_conv_mp_out_size(fc_1_size, last_conv_channels, [maxpool1d_1, maxpool1d_2])
        fc2_size = get_conv_mp_out_size(fc_2_size, last_conv_channels, [maxpool1d_1, maxpool1d_2])

        # Dense layers
        # fc1_size = 1020
        # fc2_size=340
        # self.fc1 = Linear(fc1_size, n_classes)
        # self.fc2 = Linear(fc2_size, n_classes)
        self.fc1 = Linear(fc1_size, n_classes)
        self.fc2 = Linear(fc2_size, n_classes)
        # self.fc = Linear(1024, n_classes)

        # self.softmax = Softmax(dim=1)

        # Dropout
        # self.drop = Dropout(p=0.2)

    def forward(self, hidden, x):
        concat = torch.cat([hidden, x], 1)
        concat_size = hidden.shape[1] + x.shape[1]

        concat = concat.view(-1, self.conv1d_1_args["in_channels"], concat_size)
        middle = self.conv1d_1(concat)
        middle_1 = F.relu(middle)
        Z = self.mp_1(middle_1)
        Z = self.mp_2(self.conv1d_2(Z))

        hidden = hidden.view(-1, self.conv1d_1_args["in_channels"], hidden.shape[1])

        Y = self.mp_1(F.relu(self.conv1d_1(hidden)))
        Y = self.mp_2(self.conv1d_2(Y))

        Z_flatten_size = int(Z.shape[1] * Z.shape[-1])
        Y_flatten_size = int(Y.shape[1] * Y.shape[-1])

        Z = Z.view(-1, Z_flatten_size)
        Y = Y.view(-1, Y_flatten_size)
        res = self.fc1(Z) * self.fc2(Y)
        # res = self.drop(res)
        
        # sig = torch.sigmoid(torch.flatten(res))
        # out = self.fc(res)
        # sfx = self.softmax(out)

        return res