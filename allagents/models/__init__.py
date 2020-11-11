'''
Pytorch models for continuous control.

All models assume that a feature representing the
current timestep is used in addition to the features
received from the environment.
'''
from all import nn
import torch.nn

# for how to initialise weights
# https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch/49433937#49433937


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.001)


def create_net(input_dim, output_dim, hidden1, hidden2):
    net = nn.Sequential(
        nn.Linear(input_dim, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, output_dim)
    )
    net.apply(init_weights)
    net.float()
    return net


def actor(env, hidden1=400, hidden2=300):
    return create_net(env.state_space.shape[0], env.action_space.shape[0], 48, 48)


def critic(env, hidden1=400, hidden2=300):
    return create_net(env.state_space.shape[0], 1, 48, 48)


def features(state_space_size, hidden1=400):
    net = nn.Sequential(
        nn.Linear(state_space_size + 1, hidden1),
        nn.ReLU(),
    )
    net.apply(init_weights)
    net.float()
    return net
