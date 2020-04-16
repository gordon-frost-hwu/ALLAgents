from torch.optim import Adam, SGD
from all.approximation import VNetwork, FeatureNetwork
from all.logging import DummyWriter
from all.memory import ExperienceReplayBuffer
from all.policies import DeterministicPolicy

import models
from agents.agent import CACLA

def cacla(
        # Common settings
        device="cpu",
        discount_factor=0.99,
        # Adam optimizer settings
        lr_v=1e-3,
        lr_pi=1e-4,
        eps=0.01,   # from https://medium.com/autonomous-learning-library/radam-a-new-state-of-the-art-optimizer-for-rl-442c1e830564
        # Replay buffer settings
        replay_buffer_size=4000
):
    """
    Continuous Actor Critic Learning Automation (CACLA) classic control preset.

    Args:
        device (str): The device to load parameters and buffers onto for this agent.
        discount_factor (float): Discount factor for future rewards.
        lr_v (float): Learning rate for value network.
        lr_pi (float): Learning rate for policy network and feature network.
        eps (float): Stability parameters for the Adam optimizer.
        replay_buffer_size (int): maximum replay buffer size that samples get taken from
    """
    def _cacla(env, writer=DummyWriter()):
        value_model = models.critic(env).to(device)
        policy_model = models.actor(env).to(device)
        # feature_model = models.features(env.state_space.shape[0]).to(device)

        value_optimizer = Adam(value_model.parameters(), lr=lr_v, eps=eps)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi, eps=eps)
        # feature_optimizer = Adam(feature_model.parameters(), lr=lr_pi, eps=eps)
        # feature_optimizer = SGD(feature_model.parameters(), lr=lr_pi, momentum=0.9)

        policy = DeterministicPolicy(
            policy_model,
            policy_optimizer,
            env.action_space,
            clip_grad=1.0,
            writer=writer,
            normalise_inputs=True,
            box=env.state_space,
        )

        v = VNetwork(value_model,
                     value_optimizer,
                     writer=writer,
                     normalise_inputs=True,
                     box=env.state_space,
                     )
        features = None     # FeatureNetwork(feature_model, feature_optimizer, writer=writer, normalize_input=False)
        replay_buffer = ExperienceReplayBuffer(replay_buffer_size, device=device)

        # TODO - reintroduce TimeFeature wrapper
        return CACLA(features, v, policy, replay_buffer, env.action_space, writer=writer, discount_factor=discount_factor)
    return _cacla

__all__ = ["cacla"]
