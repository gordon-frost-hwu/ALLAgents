from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.approximation import VNetwork, FeatureNetwork
from all.bodies import TimeFeature
from all.logging import DummyWriter
from all.optim import LinearScheduler
from all.policies import StochasticPolicy, GaussianPolicy, DeterministicPolicy
from torch.distributions.normal import Normal

import models
from agent import CACLA

def cacla(
        # Common settings
        device="cpu",
        discount_factor=0.99,
        # Adam optimizer settings
        lr_v=5e-4,
        lr_pi=1e-4,
        eps=1e-5,
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
        # was running
        # feature_model, value_model, policy_model = models.fc_actor_critic(env)
        # feature_model = models.fc_relu_features(env)
        # feature_model.to(device)
        # value_model.to(device)
        # policy_model = models.fc_deterministic_policy(env)
        # policy_model.to(device)
        value_model = models.critic(env).to(device)
        policy_model = models.actor(env).to(device)

        # value_model = models.fc_value_head().to(device)
        # policy_model = models.fc_policy_head(env).to(device)
        # feature_model = models.fc_relu_features(env).to(device)

        value_optimizer = Adam(value_model.parameters(), lr=lr_v, eps=eps)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi, eps=eps)
        # feature_optimizer = Adam(feature_model.parameters(), lr=lr_pi, eps=eps)
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
            clip_grad=1.0
        )

        v = VNetwork(value_model, value_optimizer, writer=writer)
        # policy = SoftmaxPolicy(policy_model, policy_optimizer, writer=writer)
        # features = FeatureNetwork(feature_model, feature_optimizer)

        return CACLA(v, policy, env.action_space, writer=writer, discount_factor=discount_factor)
    return _cacla


# def cacla(
#         # Common settings
#         device="cuda",
#         discount_factor=0.98,
#         last_frame=2e6,
#         # Adam optimizer settings
#         lr=3e-4,  # Adam learning rate
#         eps=1e-5,  # Adam stability
#         # Loss scaling
#         entropy_loss_scaling=0.01,
#         value_loss_scaling=0.5,
#         # Training settings
#         clip_grad=0.5,
#         clip_initial=0.2,
#         clip_final=0.01,
#         epochs=20,
#         minibatches=4,
#         # Batch settings
#         n_envs=32,
#         n_steps=128,
#         # GAE settings
#         lam=0.95,
# ):
#     """
#     PPO continuous control preset.
#
#     Args:
#         device (str): The device to load parameters and buffers onto for this agent.
#         discount_factor (float): Discount factor for future rewards.
#         last_frame (int): Number of frames to train.
#         lr (float): Learning rate for the Adam optimizer.
#         eps (float): Stability parameters for the Adam optimizer.
#         entropy_loss_scaling (float): Coefficient for the entropy term in the total loss.
#         value_loss_scaling (float): Coefficient for the value function loss.
#         clip_grad (float): The maximum magnitude of the gradient for any given parameter. Set to 0 to disable.
#         clip_initial (float): Value for epsilon in the clipped PPO objective function at the beginning of training.
#         clip_final (float): Value for epsilon in the clipped PPO objective function at the end of training.
#         epochs (int): Number of times to iterature through each batch.
#         minibatches (int): The number of minibatches to split each batch into.
#         n_envs (int): Number of parallel actors.
#         n_steps (int): Length of each rollout.
#         lam (float): The Generalized Advantage Estimate (GAE) decay parameter.
#     """
#     def _cacla(envs, writer=DummyWriter()):
#         final_anneal_step = last_frame * epochs * minibatches / (n_steps * n_envs)
#         env = envs[0]
#
#         feature_model, value_model, policy_model = fc_actor_critic(env)
#         feature_model.to(device)
#         value_model.to(device)
#         policy_model.to(device)
#
#         feature_optimizer = Adam(
#             feature_model.parameters(), lr=lr, eps=eps
#         )
#         value_optimizer = Adam(value_model.parameters(), lr=lr, eps=eps)
#         policy_optimizer = Adam(policy_model.parameters(), lr=lr, eps=eps)
#
#         features = FeatureNetwork(
#             feature_model,
#             feature_optimizer,
#             clip_grad=clip_grad,
#             scheduler=CosineAnnealingLR(
#                 feature_optimizer,
#                 final_anneal_step
#             ),
#             writer=writer
#         )
#         v = VNetwork(
#             value_model,
#             value_optimizer,
#             loss_scaling=value_loss_scaling,
#             clip_grad=clip_grad,
#             writer=writer,
#             scheduler=CosineAnnealingLR(
#                 value_optimizer,
#                 final_anneal_step
#             ),
#         )
#         policy = GaussianPolicy(
#             policy_model,
#             policy_optimizer,
#             env.action_space,
#             clip_grad=clip_grad,
#             writer=writer,
#             scheduler=CosineAnnealingLR(
#                 policy_optimizer,
#                 final_anneal_step
#             ),
#         )
#
#         return TimeFeature(CACLA(
#             features,
#             v,
#             policy,
#             epsilon=LinearScheduler(
#                 clip_initial,
#                 clip_final,
#                 0,
#                 final_anneal_step,
#                 name='clip',
#                 writer=writer
#             ),
#             epochs=epochs,
#             minibatches=minibatches,
#             n_envs=n_envs,
#             n_steps=n_steps,
#             discount_factor=discount_factor,
#             lam=lam,
#             entropy_loss_scaling=entropy_loss_scaling,
#             writer=writer,
#         ))
#
#     return _cacla, n_envs


__all__ = ["cacla"]
