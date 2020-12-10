import torch
from torch.nn.functional import mse_loss
from all.logging import DummyWriter
from torch.distributions.normal import Normal
from all.agents import Agent


class AC(Agent):
    '''
    Continuous Actor Critic Learning Automation (CACLA).
    CACLA is an implementation of the CACLA alogorithm found in van Hasselt and Wiering (2007).
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.75.7658&rep=rep1&type=pdf
    This implementation tweaks the algorithm slightly by only updating the Actor's weights at the end of an episode
    and using a replay buffer to collect interaction samples. The Temporal Difference Error is computed
    and saved into the experience replay buffer at each timestep, allowing it to be used to update
    the Actor at the end of the episode. At the end of the episode, the replay buffer
    gets sampled n_iter times to train the critic, and then separately another n_iter times to train actor
    A shared feature layer can also be used, but successful results did not need it

    Args:
        features (FeatureNetwork): Shared feature layers. If None, raw environment state vector is used.
        v (VNetwork): Value head which approximates the state-value function.
        policy (DeterministicPolicy): Policy head which outputs an action distribution.
        buffer (ExperienceBuffer): buffer for experience replay.
        action_space (Box): OpenAi Gym environment's action space.
        discount_factor (float): Discount factor for future rewards.
        sigma (float): initial variance of Normal distribution used for exploration.
        sigma_decay (float): decay rate for exploration.
        sigma_min (float): minimum value for exploration.
        n_iter (int): Number of times to sample the replay buffer when training.
        minibatch_size (int): Number of timesteps sampled per n_iter of training.
        writer (Writer): Used for logging.
    '''
    def __init__(self, features, v, policy, buffer, action_space,
                 discount_factor=0.99,
                 sigma=1.0,
                 sigma_decay=0.9995,
                 sigma_min=0.1,
                 n_iter=100,
                 minibatch_size=32,
                 log=True,
                 writer=DummyWriter()):
        self.features = features
        self.v = v
        self.policy = policy
        self.replay_buffer = buffer
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        self._log = log
        self.writer = writer
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.n_iter = n_iter
        self._features = None
        self._action = None
        self._state = None
        self._tde = None
        self._action_low = torch.tensor(action_space.low, device=policy.device).float()
        self._action_high = torch.tensor(action_space.high, device=policy.device).float()

    def _normal(self, output):
        if self._log:
            self.writer.add_scalar("sigma", self.sigma)
        return Normal(output, self.sigma)

    def act(self, state, reward):
        self._train(state, reward)

        if self._state is not None and self._tde is not None:
            self.replay_buffer.store(self._state, self._action, self._tde, state)
        self._state = state
        self._action = self._choose_action(state)
        return self._action

    def _choose_action(self, state):
        # If a feature ANN is provided, use it, otherwise raw state vector is used
        self._features = self.features(state) if self.features is not None else state
        deterministic_action = self.policy.eval(self._features)
        # uncomment to log the policy output
        # if self._log:
        # self.writer.add_scalar("action/det", deterministic_action)

        # Get the stochastic action by centering a Normal distribution on the policy output
        stochastic_action = self._normal(deterministic_action).sample().float()

        # Clip the stochastic action to the gym environment's action space
        stochastic_action = torch.max(stochastic_action, self._action_low)
        stochastic_action = torch.min(stochastic_action, self._action_high)
        return stochastic_action

    def _train(self, states, rewards):
        if self._state is not None:
            # forward pass
            features = self.features(self._state) if self.features is not None else self._state
            values = self.v(features)

            # compute targets
            features_next_states = self.features.target(states) if self.features is not None else states

            # compute state_{t+1} value
            next_state_value = self.v.target(features_next_states)
            targets = rewards + self.discount_factor * next_state_value
            self._tde = targets - values.detach()
            self.update_critic(values, targets)

            self._train_actor(states)

    def _train_actor(self, s):
        distribution = self.policy(s)
        # advantages = target - self.critic(s).detach()
        policy_loss = -(self._tde * self._normal(distribution).log_prob(self._action)).mean()
        self.policy.reinforce(policy_loss)

        # Only decay the exploration at the end of an episode
        if s.done:
            # Decay the exploration
            if self.sigma > self.sigma_min:
                self.sigma *= self.sigma_decay

    def generate_targets(self):
        # sample from replay buffer
        (states, actions, tde, next_states, _) = self.replay_buffer.sample(self.minibatch_size)

        # forward pass
        features = self.features(states) if self.features is not None else states
        values = self.v(features)

        # compute targets
        features_next_states = self.features.target(next_states) if self.features is not None else next_states

        # compute state_{t+1} value
        next_state_value = self.v.target(features_next_states)
        # targets = rewards + self.discount_factor * next_state_value
        return features, values, tde, actions

    def update_critic(self, values, targets):
        # backward pass
        value_loss = mse_loss(values, targets)
        self.v.reinforce(value_loss)

        # If feature ANN given, do a backwards pass over it too (taken from VAC agent)
        if self.features is not None:
            self.features.reinforce()
