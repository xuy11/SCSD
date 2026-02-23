import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim: list = [512, 1024, 1024, 1024, 512], act=nn.GELU,
                 norm=nn.BatchNorm1d, dropout_rate=0., final_act=True, final_norm=True, residual=False):
        super(MLP, self).__init__()
        self.act = act
        self.norm = norm
        self.dropout_rate = dropout_rate
        self.residual = residual
        # init layers
        self.mlps = []
        layers = []

        if len(hidden_dim) > 0:
            layers.append(nn.Linear(in_dim, hidden_dim[0]))
            layers.append(self.norm(hidden_dim[0]))
            layers.append(self.act())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            self.mlps.append(nn.Sequential(*layers))
            layers = []
            ##hidden layer
            for i in range(len(hidden_dim) - 1):
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
                layers.append(self.norm(hidden_dim[i + 1]))
                layers.append(self.act())
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))
                self.mlps.append(nn.Sequential(*layers))
                layers = []
            ##output layer
            layers.append(nn.Linear(hidden_dim[-1], out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
        else:
            layers.append(nn.Linear(in_dim, out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
        self.mlps = nn.ModuleList(self.mlps)

    def forward(self, x):
        for layers in self.mlps:
            x = layers(x)
        return x


