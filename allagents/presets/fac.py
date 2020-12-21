from torch.optim import Adam, SGD
from all.approximation import VNetwork, FeatureNetwork
from all.logging import DummyWriter
from all.memory import ExperienceReplayBuffer
from all.policies import DeterministicPolicy
from itertools import permutations
from numpy import linspace
import allagents.models as models
from allagents.agents.fac import ForwardAC
from all.environments.state import State
import torch

def fac(
        # Common settings
        device="cpu",
        discount_factor=0.99,
        # Adam optimizer settings
        # lr_v=0.001266006463,
        # lr_pi=0.001369217032,
        # trace_decay=0.0413122279,
        lr_v=0.005,
        lr_pi=0.00001,
        trace_decay=0.93,
        sigma=0.05,
        sigma_decay=1.0,
        log=True,
        eps=0.01,   # from https://medium.com/autonomous-learning-library/radam-a-new-state-of-the-art-optimizer-for-rl-442c1e830564
        # Replay buffer settings
        replay_buffer_size=4000,
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
    def _fac(env, writer=DummyWriter()):
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

        r = linspace(-1, 1, 21)
        states = State(torch.as_tensor(list(permutations(r, 2)), dtype=torch.float32).cuda())
        for i in range(200):
            values = v(states)
            # print("values before: {0}".format(values[0:20]))
            target_values = torch.as_tensor([-1000 for x in range(values.shape[0])], dtype=torch.float32).cuda()

            loss = torch.nn.functional.mse_loss(values, target_values)
            v.reinforce(loss)

        new_values = v.eval(states)
        print("values after: {0}".format(new_values[0:20]))

        replay_buffer = ExperienceReplayBuffer(replay_buffer_size, device=device)


        # TODO - reintroduce TimeFeature wrapper
        return ForwardAC(v, policy,
                         replay_buffer,
                         env.action_space,
                         sigma=sigma,
                         sigma_decay=sigma_decay,
                         log=log,
                         trace_decay=trace_decay,
                         writer=writer,
                         discount_factor=discount_factor)
    return _fac

__all__ = ["fac"]
