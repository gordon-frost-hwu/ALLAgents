'''
Pytorch models for continuous control.

All models assume that a feature representing the
current timestep is used in addition to the features
received from the environment.
'''
from all import nn

def actor(action_space_size, hidden1=400, hidden2=300):
    return nn.Sequential(
        # nn.Linear(env.state_space.shape[0], hidden1),
        # nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, action_space_size)
    )

def critic(hidden1=400, hidden2=300):
    return nn.Sequential(
        # nn.Linear(env.state_space.shape[0], hidden1),
        # nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, 1)
    )

def features(state_space_size, hidden1=400):
    return nn.Sequential(
        nn.Linear(state_space_size + 1, hidden1),
        nn.ReLU(),
    )
