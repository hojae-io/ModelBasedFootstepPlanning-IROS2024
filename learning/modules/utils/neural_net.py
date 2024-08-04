import torch.nn as nn
from torch.distributions import Normal
import torch

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        # torch.nn.init.constant_(m.bias, 0)

def create_MLP(num_inputs, num_outputs, hidden_dims, activation,
               dropouts=None):

    activation = get_activation(activation)

    if dropouts is None:
        dropouts = [0]*len(hidden_dims)

    layers = []
    if not hidden_dims:  # handle no hidden layers
        add_layer(layers, num_inputs, num_outputs)
    else:
        add_layer(layers, num_inputs, hidden_dims[0], activation, dropouts[0])
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                add_layer(layers, hidden_dims[i], num_outputs)
            else:
                add_layer(layers, hidden_dims[i], hidden_dims[i+1],
                          activation, dropouts[i+1])
    return nn.Sequential(*layers)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


def add_layer(layer_list, num_inputs, num_outputs, activation=None, dropout=0):
    layer_list.append(nn.Linear(num_inputs, num_outputs))
    if dropout > 0:
        layer_list.append(nn.Dropout(p=dropout))
    if activation is not None:
        layer_list.append(activation)