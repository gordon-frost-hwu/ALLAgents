import torch
from torch.nn.functional import mse_loss
from all.logging import DummyWriter
from torch.distributions.normal import Normal
from all.agents import Agent

class TOCLA(Agent):
    '''
    True Online Continuous Learning Automation (TOCLA)
    CACLA is an implementation of the CACLA alogorithm found in van Hasselt and Wiering (2007).
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.75.7658&rep=rep1&type=pdf
    This implementation tweaks the algorithm slightly by only updating weights at the end of an episode
    and using a replay buffer to collect interaction samples. At the end of the episode, the replay buffer
    gets sampled n_iter times to train the critic, and then separately another n_iter times to train actor
    A shared feature layer can also be used, but successful results did not need it

    Args:
        features (FeatureNetwork): Shared feature layers. If None, raw environment state vector is used
        v (VNetwork): Value head which approximates the state-value function.
        policy (DeterministicPolicy): Policy head which outputs an action distribution.
        buffer (ExperienceBuffer): buffer for experience replay.
        discount_factor (float): Discount factor for future rewards.
        sigma (float): initial variance of Normal distribution used for exploration
        sigma_decay (float): decay rate for exploration
        sigma_min (float): minimum value for exploration
        n_iter (int): Number of times to sample the replay buffer when training
        minibatch_size (int): Number of timesteps sampled per n_iter of training
        writer (Writer): Used for logging.
    '''
    def __init__(self, features, v, policy, buffer, action_space,
                 discount_factor=0.99,
                 sigma=1.0,
                 sigma_decay=0.9995,
                 sigma_min=0.1,
                 n_iter=100,
                 minibatch_size=32,
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
        self._action = None
        self._state = None
        self._action_low = torch.tensor(action_space.low, device=policy.device)
        self._action_high = torch.tensor(action_space.high, device=policy.device)

    def _normal(self, output):
        self.writer.add_scalar("sigma", self.sigma)
        return Normal(output, self.sigma)

    def act(self, state, reward):
        self._train(state)
        self.replay_buffer.store(self._state, self._action, reward, state)
        self._state = state
        self._action = self._choose_action(state)
        return self._action

    def _choose_action(self, state):
        # If a feature ANN is provided, use it, otherwise raw state vector is used
        self._features = self.features(state) if self.features is not None else state
        deterministic_action = self.policy.eval(self._features)
        # uncomment to log the policy output
        # self.writer.add_scalar("action/det", deterministic_action)

        # Get the stochastic action by centering a Normal distribution on the policy output
        stochastic_action = self._normal(deterministic_action).sample()

        # Clip the stochastic action to the gym environment's action space
        stochastic_action = torch.max(stochastic_action, self._action_low)
        stochastic_action = torch.min(stochastic_action, self._action_high)
        return stochastic_action

    def _train(self, states):
        # only train (update weights) at the end of an episode; i.e. at a terminal state
        if states.done:
            for i in range(self.n_iter):
                _, values, targets, _ = self.generate_targets()
                # targets = rewards + self.discount_factor * values.detach()
                self.update_critic(values=values, targets=targets)
                # if not torch.isnan(targets[0]):
                #     self.writer.add_scalar("target", targets[0])

            for i in range(self.n_iter):
                features, values, targets, actions = self.generate_targets()
                greedy_actions = self.policy(features)
                self.update_actor_cacla(values=values, targets=targets, greedy_actions=greedy_actions, actions=actions)

            # Decay the exploration
            if self.sigma > self.sigma_min:
                self.sigma *= self.sigma_decay

    def update_actor_cacla(self, targets, values, greedy_actions, actions):
        # calculate TDE
        advantages = targets - values.detach()

        # Get the indexes where the TDE is positive (i.e. the action resulted in a good state transition)
        idx = torch.where(advantages > 0.0)[0]
        if len(idx) > 0:
            policy_loss = mse_loss(greedy_actions[idx], actions[idx])

            if not torch.isnan(policy_loss):
                self.policy.reinforce(policy_loss)
            else:
                print("policy loss is NaN")

    def generate_targets(self):
        # sample from replay buffer
        (states, actions, rewards, next_states, _) = self.replay_buffer.sample(self.minibatch_size)

        # forward pass
        features = self.features(states) if self.features is not None else states
        values = self.v(features)

        # compute targets
        features_next_states = self.features.target(next_states) if self.features is not None else next_states

        # compute state_{t+1} value
        next_state_value = self.v.target(features_next_states)
        targets = rewards + self.discount_factor * next_state_value
        return features, values, targets, actions

    def update_critic(self, values, targets):
        # backward pass
        value_loss = mse_loss(values, targets)
        self.v.reinforce(value_loss)

        # If feature ANN given, do a backwards pass over it too (taken from VAC agent)
        if self.features is not None:
            self.features.reinforce()
