from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.approximation import VNetwork, FeatureNetwork, PolyakTarget
from all.bodies import TimeFeature
from all.logging import DummyWriter
from all.optim import LinearScheduler
from all.memory import ExperienceReplayBuffer
from all.policies import StochasticPolicy, GaussianPolicy, DeterministicPolicy, SoftDeterministicPolicy
from torch.distributions.normal import Normal

import models
from agent import CACLA

def cacla(
        # Common settings
        device="cpu",
        discount_factor=0.99,
        # Adam optimizer settings
        lr_v=1e-5,
        lr_pi=1e-5,
        eps=1e-5,
        polyak_rate=0.005,
        # Replay buffer settings
        replay_buffer_size=5
):
    """
    Vanilla Actor-Critic classic control preset.

    Args:
        device (str): The device to load parameters and buffers onto for this agent.
        discount_factor (float): Discount factor for future rewards.
        lr_v (float): Learning rate for value network.
        lr_pi (float): Learning rate for policy network and feature network.
        eps (float): Stability parameters for the Adam optimizer.
    """
    def _cacla(env, writer=DummyWriter()):
        value_model = models.critic().to(device)
        policy_model = models.actor(env.action_space.shape[0]).to(device)
        feature_model = models.features(env.state_space.shape[0]).to(device)

        value_optimizer = Adam(value_model.parameters(), lr=lr_v, eps=eps)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi, eps=eps)
        feature_optimizer = Adam(feature_model.parameters(), lr=lr_pi, eps=eps)
        # value_optimizer = SGD(value_model.parameters(), lr=lr_v, momentum=0.9)
        # policy_optimizer = SGD(policy_model.parameters(), lr=lr_pi, momentum=0.9)
        # feature_optimizer = SGD(feature_model.parameters(), lr=lr_pi, momentum=0.9)

        # old arguments to GaussianPolicy
        # clip_grad = clip_grad,
        # writer = writer,
        # scheduler = CosineAnnealingLR(
        #     policy_optimizer,
        #     final_anneal_step
        # ),
        # policy = StochasticPolicy(
        #     policy_model,
        #     policy_optimizer,
        #     Normal(0, 0.1)
        # )
        policy = DeterministicPolicy(
            policy_model,
            policy_optimizer,
            env.action_space,
            clip_grad=1.0,
            writer=writer,
            normalise_inputs=False,
            # target=PolyakTarget(polyak_rate),
            # scheduler=CosineAnnealingLR(
            #     policy_optimizer,
            #     final_anneal_step
            # ),
        )

        v = VNetwork(value_model, value_optimizer, writer=writer)
        # policy = SoftmaxPolicy(policy_model, policy_optimizer, writer=writer)
        features = FeatureNetwork(feature_model, feature_optimizer, normalize_input=False)
        replay_buffer = ExperienceReplayBuffer(replay_buffer_size, device=device)

        return TimeFeature(CACLA(features, v, policy, replay_buffer, env.action_space, writer=writer, discount_factor=discount_factor))
    return _cacla

__all__ = ["cacla"]
