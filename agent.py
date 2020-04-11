import torch
from torch.nn.functional import mse_loss
from all.logging import DummyWriter
from torch.distributions.normal import Normal
from all.agents import Agent
from all import nn
import models
import numpy as np

class CACLA(Agent):
    '''
    Vanilla Actor-Critic (VAC).
    VAC is an implementation of the actor-critic alogorithm found in the Sutton and Barto (2018) textbook.
    This implementation tweaks the algorithm slightly by using a shared feature layer.
    It is also compatible with the use of parallel environments.
    https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf

    Args:
        features (FeatureNetwork): Shared feature layers.
        v (VNetwork): Value head which approximates the state-value function.
        policy (DeterministicPolicy): Policy head which outputs an action distribution.
        buffer (ExperienceBuffer): buffer for experience replay.
        discount_factor (float): Discount factor for future rewards.
        n_envs (int): Number of parallel actors/environments
        n_steps (int): Number of timesteps per rollout. Updates are performed once per rollout.
        writer (Writer): Used for logging.
    '''
    def __init__(self, features, v, policy, buffer, action_space,
                 discount_factor=0.99,
                 sigma=1.0,
                 sigma_decay=0.9995,
                 sigma_min=0.1,
                 n_iter=100,
                 minibatch_size=32,
                 update_frequency=200,
                 replay_start_size=100,
                 writer=DummyWriter()):
        self.features = features
        self.v = v
        self.policy = policy
        self.replay_buffer = buffer
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        self.writer = writer
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.n_iter = n_iter
        self._features = None
        self._distribution = None
        self._action = None
        self._exploration = None
        self._state = None
        self._action_low = torch.tensor(action_space.low, device=policy.device)
        self._action_high = torch.tensor(action_space.high, device=policy.device)
        print("action low: {0}".format(self._action_low))
        print("action high: {0}".format(self._action_high))
        self.log_prob = None
        self._last_features = None
        self._frames_seen = 0
        self.update_frequency = update_frequency
        self.replay_start_size = replay_start_size
        # self.gaussian = Normal(0, noise * torch.tensor((action_space.high - action_space.low) / 2).to(policy.device))

    def _normal(self, output):
        # return Normal(output, torch.tensor([self.sigma]).to('cuda'))
        # noise = self.sigma * torch.tensor((self._action_high - self._action_low) / 2).to('cuda')
        self.writer.add_scalar("sigma", self.sigma)
        return Normal(output, self.sigma)

    def act(self, state, reward):
        self._train(state, reward)
        # self.writer.add_scalar("state/0", state.raw[0][0])
        # self.writer.add_scalar("state/1", state.raw[0][1])
        # self.writer.add_scalar("state/2", state.raw[0][2])
        # self.writer.add_scalar("state/3", state.raw[0][3])
        # print("state: {0}".format(state.raw))
        self.replay_buffer.store(self._state, self._action, reward, state)

        self._state = state
        self._action = self._choose_action(state)

        return self._action

    def _choose_action(self, state):
        self._features = self.features(state)

        # if self._last_features is not None:
        #     print("features: {0}".format(self._features.raw - self._last_features))
        # self._distribution = Normal(self.policy(self._features), 0)

        # deterministic_action, self.log_prob = self.policy.eval(self._features)
        # print("det action: {0}".format(deterministic_action.data))
        # print("log prob: {0}".format(self.log_prob))
        deterministic_action = self.policy.eval(self._features)

        self.writer.add_scalar("action/det", deterministic_action)

        self._distribution = self._normal(deterministic_action)

        # TODO -
        # if self._frames_seen % 5 == 0 or self._frames_seen == 1:
        stochastic_action = self._distribution.sample()

        self._exploration = stochastic_action - deterministic_action
        # print("exploration: {0}".format(self._exploration))

        # self.log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        # self.log_prob = self.log_prob.sum(1)

        # print("deterministic action: {0}".format(deterministic_action.data))
        # print("exploration   action: {0}".format(action.data))

        # stochastic_action = deterministic_action + self._exploration

        # print(type(self.writer))
        # print("stochastic action: {0}".format(self._action.data))
        stochastic_action = torch.max(stochastic_action, self._action_low)
        stochastic_action = torch.min(stochastic_action, self._action_high)
        self.writer.add_scalar("action/sto", stochastic_action)
        return stochastic_action

    def _train(self, states, rewards):
        if self._should_train():
            for i in range(self.n_iter):
                # sample from replay buffer
                (states, actions, rewards, next_states, _) = self.replay_buffer.sample(self.minibatch_size)
                # dummy_network = models.actor(1).to("cuda")
                # dummy_features = self._dummy_features().to("cuda")
                # # buffer_size = len(self.replay_buffer)
                #
                # print("====================")
                # print(len(self.replay_buffer.buffer))
                # for param in dummy_network.parameters():
                #     print(param.grad)
                # for (states, actions, rewards, next_states) in self.replay_buffer.iterate_backwards():
                #     print(states.raw)
                #     features = dummy_features(states.raw)
                #     action = dummy_network(features)
                #     for param in dummy_network.parameters():
                #         print(param.grad)
                #     print("--------------------")
                #
                # exit(0)
                # print("state: {0}".format(states.raw))
                # forward pass
                features = self.features(states)
                # pi_features = self.features(state)
                values = self.v(features)
                # pi_values = self.v(pi_features)
                if not torch.isnan(values[0]):
                    self.writer.add_scalar("statevalue", values[0])

                # compute targets
                targets = rewards + self.discount_factor * self.v.target(self.features.target(next_states))
                # targets = rewards + self.discount_factor * values.detach()

                if not torch.isnan(targets[0]):
                    self.writer.add_scalar("target", targets[0])

                greedy_actions = self.policy(features)
                self.log_prob = self._distribution.log_prob(greedy_actions)
                # self.update_actor_ac(values=values, targets=targets)
                self.update_actor_cacla(values=values, targets=targets, greedy_actions=greedy_actions, actions=actions)

                # print("policy loss: {0}".format(policy_loss.data))
                # debugging
                # self.writer.add_loss('policy_gradient', policy_gradient_loss.detach())
                # self.writer.add_loss('entropy', entropy_loss.detach())
                self.update_critic(values=values, targets=targets)

            if self.sigma > self.sigma_min:
                self.sigma *= self.sigma_decay

    def update_actor_ac(self, targets, values):
        # advantages = targets[0] - values.detach()[0]
        advantages = targets - values.detach()
        # print("type of adv and log_prob: {0} {1}".format(type(advantages), type(self.log_prob)))
        # self.writer.add_scalar("advantages", advantages)
        # compute losses
        # print(advantages)
        # print(self.log_prob)
        policy_loss = -(advantages * self.log_prob).mean()
        policy_loss = policy_loss.requires_grad_(True)       # if uncommented, policy output always zero!!
        # print(policy_loss.data)
        if not torch.isnan(policy_loss):
            self.policy.reinforce(policy_loss)
        else:
            print("policy loss is NaN")

    def update_actor_cacla(self, targets, values, greedy_actions, actions):
        advantages = targets - values.detach()
        exploration = actions - greedy_actions

        idx = torch.where(advantages > 0.0)

        policy_loss = (exploration[idx] * self.log_prob[idx]).mean()
        # policy_loss = policy_loss.clone().detach().requires_grad_(True)       # if uncommented, policy output always zero!!
        # print(policy_loss.data)
        if len(idx) > 0:
            if not torch.isnan(policy_loss):
                self.policy.reinforce(policy_loss)
            else:
                print("policy loss is NaN")

    def update_critic(self, values, targets):
        # for param in self.policy.model.parameters():
        #     print(param.data)
        # print("policy weights before: {0}".format(self.policy.model.parameters()))
        # backward pass
        value_loss = mse_loss(values, targets)
        self.v.reinforce(value_loss)
        self.features.reinforce()
        # idx += 1

    def _should_train(self):
        self._frames_seen += 1
        return self._frames_seen > self.replay_start_size and self._frames_seen % self.update_frequency == 0
