from torch.optim import Adam, SGD
from all.approximation import VNetwork, FeatureNetwork
from all.logging import DummyWriter
from all.memory import MyReplayBuffer
from all.policies import DeterministicPolicy
from all.environments.state import State
from torch.nn.functional import mse_loss
import allagents.models as models
from allagents.agents.tocla import TOCLA
from allagents.models import RBFKernel
from itertools import permutations
from numpy import linspace
import torch

def tocla(
        # Common settings
        device="cpu",
        discount_factor=0.1,   # gamma
        sigma=1.0,
        sigma_decay=0.9998,
        lr_v=0.009,
        lr_pi=0.000005,
        trace_decay=0.98,
        # Ten runs
        # lr_v=0.001125209337,
        # lr_pi=0.00259986294,
        # trace_decay=0.5306405172,
        # four runs
        # lr_v=0.0006983937324,
        # lr_pi=0.002654206888,
        # trace_decay=0.5021468202,
        log=True,
        eps=0.01,   # from https://medium.com/autonomous-learning-library/radam-a-new-state-of-the-art-optimizer-for-rl-442c1e830564
        # Replay buffer settings
        replay_buffer_size=350,    # was 4000
        hidden1=400,
        hidden2=300,
):
    """
    True Online Continuous Learning Automation (TOCLA) classic control preset.

    Args:
        device (str): The device to load parameters and buffers onto for this agent.
        discount_factor (float): Discount factor for future rewards.
        lr_v (float): Learning rate for value network.
        lr_pi (float): Learning rate for policy network and feature network.
        eps (float): Stability parameters for the Adam optimizer.
        replay_buffer_size (int): maximum replay buffer size that samples get taken from
    """
    def _tocla(env, writer=DummyWriter()):
        value_model = models.critic(env, hidden1=hidden1, hidden2=hidden2).to(device)
        policy_model = models.actor(env, hidden1=hidden1, hidden2=hidden2).to(device)

        value_optimizer = Adam(value_model.parameters(), lr=lr_v, eps=eps)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi, eps=eps)

        policy = DeterministicPolicy(
            policy_model,
            policy_optimizer,
            env.action_space,
            quiet=not log,
            clip_grad=1.0,
            writer=writer,
            normalise_inputs=True,
            box=env.state_space,
        )

        v = VNetwork(value_model,
                     value_optimizer,
                     quiet=not log,
                     writer=writer,
                     normalise_inputs=True,
                     box=env.state_space,
                     )
        replay_buffer = MyReplayBuffer(replay_buffer_size, device=device)

        features = RBFKernel([[-1.0, 1.0], [-1.0, 1.0]], 41, 0.05)
        r = linspace(-1, 1, 21)
        perms = list(permutations(r, 2))
        states = State(torch.as_tensor(perms, device="cuda", dtype=torch.float32))

        target_values = torch.as_tensor([0 for x in range(len(states))], dtype=torch.float32).cuda()
        # initialise the value function to sensible values
        for i in range(1000):
            values = v(states)
            loss = mse_loss(values, target_values)
            v.reinforce(loss)
        
        target_values = torch.as_tensor([[0] for x in range(len(states))], dtype=torch.float32).cuda()
        # initialise the policy output to sensible values
        for i in range(5000):
            values = policy(states)
            loss = mse_loss(values, target_values)
            policy.reinforce(loss)

        # for perm in perms:
        #     # perm = features(torch.as_tensor(perm, device="cuda", dtype=torch.float32))
        #     # states = State(perm)
        #     states = State(torch.as_tensor(perm, device="cuda", dtype=torch.float32))
        #     # for i in range(200):
        #     values = v(states)
        #     # print("values before: {0}".format(values[0:20]))
        #     target_values = torch.as_tensor([-20 for x in range(values.shape[0])], dtype=torch.float32).cuda()

        #     loss = mse_loss(values, target_values)
        #     v.reinforce(loss)
        
        
        new_values = v.eval(states)
        new_values_policy = policy.eval(states)
        print("values after: {0}".format(new_values[0:20]))
        print("policy values after: {0}".format(new_values_policy[0:20]))

        # TODO - reintroduce TimeFeature wrapper
        return TOCLA(v, policy, replay_buffer, env.action_space,
                     sigma=sigma,
                     sigma_decay=sigma_decay,
                     log=log,
                     trace_decay=trace_decay,
                     writer=writer,
                     discount_factor=discount_factor)
    return _tocla

__all__ = ["tocla"]
