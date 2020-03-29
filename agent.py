import torch
from torch.nn.functional import mse_loss
from all.logging import DummyWriter
from torch.distributions.normal import Normal
from all.agents import Agent

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
    def __init__(self, v, policy, buffer, action_space,
                 discount_factor=1,
                 noise=2.0,
                 minibatch_size=32,
                 update_frequency=32,
                 replay_start_size=32,
                 writer=DummyWriter()):
        self.v = v
        self.policy = policy
        self.replay_buffer = buffer
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        self.writer = writer
        self.noise = noise
        self._features = None
        self._distribution = None
        self._action = None
        self._state = None
        self._action_low = torch.tensor(action_space.low, device=policy.device)
        self._action_high = torch.tensor(action_space.high, device=policy.device)
        self.log_prob = None
        self._last_features = None
        self._frames_seen = 0
        self.update_frequency = update_frequency
        self.replay_start_size = replay_start_size
        # self.gaussian = Normal(0, noise * torch.tensor((action_space.high - action_space.low) / 2).to(policy.device))

    def _normal(self, output):
        return Normal(output, torch.tensor([self.noise]).to('cuda'))
        # return Normal(output, self.noise)   # * torch.tensor((self._action_high - self._action_low) / 2))

    def act(self, state, reward):
        self.replay_buffer.store(self._state, self._action, reward, state)

        # print("state: {0}".format(state.raw))
        self._train(state, reward)
        self._state = state

        # if self._last_features is not None:
        #     print("features: {0}".format(self._features.raw - self._last_features))
        # self._distribution = Normal(self.policy(self._features), 0)

        deterministic_action = self.policy(state)
        self.writer.add_scalar("action/det", deterministic_action)
        normal = self._normal(deterministic_action)
        if self.noise > 0.1:
            self.noise *= 0.999998
        self.writer.add_scalar("sigma", self.noise)
        action = normal.sample()
        self.log_prob = normal.log_prob(action)
        # self.log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        # self.log_prob = self.log_prob.sum(1)

        # print("deterministic action: {0}".format(deterministic_action.data))
        # print("exploration   action: {0}".format(action.data))

        # print("variance {0}".format(self._distribution.variance))
        self._action = action
        # print(type(self.writer))
        # print("stochastic action: {0}".format(self._action.data))
        self.writer.add_scalar("action/sto", self._action)
        # print("")
        return self._action

    def _train(self, state, reward):
        if self._should_train():
            # sample from replay buffer
            # (states, actions, rewards, next_states, _) = self.replay_buffer.sample(self.minibatch_size)
            buffer_size = len(self.replay_buffer)
            idx = 0
            for (states, actions, rewards, next_states) in self.replay_buffer.iterate_backwards():
                # print("state: {0}".format(states.raw))
                # forward pass
                values = self.v(states)
                # if not torch.isnan(values):
                #     self.writer.add_scalar("statevalue", values[0])

                # compute targets
                targets = reward + self.discount_factor * self.v.target(states)
                advantages = targets - values.detach()
                # print("type of adv and log_prob: {0} {1}".format(type(advantages), type(self.log_prob)))
                # self.writer.add_scalar("advantages", advantages)
                # compute losses
                policy_loss = -(advantages * self.log_prob).mean()

                policy_loss = policy_loss.clone().detach().requires_grad_(True)       # if uncommented, policy output always zero!!
                # print(policy_loss.data)
                if not torch.isnan(policy_loss):
                    self.writer.add_loss('pg', policy_loss)
                    self.policy.reinforce(policy_loss)
                else:
                    print("policy loss is NaN")

                # print("policy loss: {0}".format(policy_loss.data))
                # debugging
                # self.writer.add_loss('policy_gradient', policy_gradient_loss.detach())
                # self.writer.add_loss('entropy', entropy_loss.detach())

                # for param in self.policy.model.parameters():
                #     print(param.data)
                # print("policy weights before: {0}".format(self.policy.model.parameters()))
                # backward pass
                value_loss = mse_loss(values, targets)
                self.v.reinforce(value_loss)
                idx += 1

    def _should_train(self):
        self._frames_seen += 1
        return self._frames_seen > self.replay_start_size and self._frames_seen % self.update_frequency == 0
