import torch
from torch.nn.functional import mse_loss
from all.logging import DummyWriter
from torch.distributions.normal import Normal
from all.agents import Agent
from queue import Queue
import numpy as np

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
    def __init__(self, v, policy, buffer, action_space,
                 discount_factor=0.99,
                 trace_decay=0.9,  # lambda
                 k_max=50,
                 n=0.01,
                 sigma=1.0,
                 sigma_decay=0.9995,
                 sigma_min=0.1,
                 writer=DummyWriter()):
        self.writer = writer

        self._action = None
        self._state = None

        # Actor state
        self._replay_buffer = buffer
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self._action_low = torch.tensor(action_space.low, device=policy.device)
        self._action_high = torch.tensor(action_space.high, device=policy.device)
        self.policy = policy

        # Critic State
        self.v = v
        self.n = n
        self.K_max = k_max
        self.trace_decay = trace_decay
        self.discount_factor = discount_factor

        # Calculate K
        self.K = np.ceil(np.log(n) / np.log(self.discount_factor * self.trace_decay)) if \
            (self.discount_factor * self.trace_decay) > 0 else 1
        self.K = np.min([self.K_max, self.K])
        self._fifo = Queue(self.K)
        print("K parameter set to: {0}".format(self.K))

        self.c_final = np.power(self.discount_factor * self.trace_decay, self.K - 1)
        print("c_final parameter set to: {0}".format(self.c_final))

        # Algorithm internal variables
        self._u_sync = 0
        self._u = 0
        self._i = 0
        self._c = 1
        self._v_current = 0
        self._ready = False

    def act(self, state, reward):
        # self._train_critic(state)
        self._train_actor(state)
        print("stepping")
        self._fifo.put((self._state, self._action, reward, state))
        self._replay_buffer.store(self._state, self._action, reward, state)

        self._state = state
        self._action = self._choose_action(state)
        return self._action

    def _train_critic(self, state):
        # only train (update weights) at the end of an episode; i.e. at a terminal state
        if state.done:
            if not self._ready:
                self._u = self._u_sync
            # TODO - While F not empty (i.e. update weights whilst clearing queue

            # reset internal variables for start of next episode
            self._u_sync = 0
            self._i = 0
            self._c = 1
            self._v_current = 0
            self._ready = False

            # Decay the exploration
            if self.sigma > self.sigma_min:
                self.sigma *= self.sigma_decay
        else:
            pass

    def _train_actor(self, state):
        # only train (update weights) at the end of an episode; i.e. at a terminal state
        if state.done:
            for i in range(self.n_iter):
                features, values, targets, actions = self.generate_targets()
                greedy_actions = self.policy(features)
                self.update_actor_cacla(values=values, targets=targets, greedy_actions=greedy_actions, actions=actions)

            # Decay the exploration
            if self.sigma > self.sigma_min:
                self.sigma *= self.sigma_decay

    def generate_targets(self):
        # sample from replay buffer
        (states, actions, rewards, next_states, _) = self._replay_buffer.sample(self.minibatch_size)

        # forward pass
        values = self.v(states)

        # compute state_{t+1} value
        next_state_value = self.v.target(next_states)
        targets = rewards + self.discount_factor * next_state_value
        return states, values, targets, actions

    def update_critic(self, values, targets):
        # backward pass
        value_loss = mse_loss(values, targets)
        self.v.reinforce(value_loss)

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

    def _normal(self, output):
        self.writer.add_scalar("sigma", self.sigma)
        return Normal(output, self.sigma)

    def _choose_action(self, state):
        # If a feature ANN is provided, use it, otherwise raw state vector is used
        deterministic_action = self.policy.eval(state)
        # uncomment to log the policy output
        # self.writer.add_scalar("action/det", deterministic_action)

        # Get the stochastic action by centering a Normal distribution on the policy output
        stochastic_action = self._normal(deterministic_action).sample()

        # Clip the stochastic action to the gym environment's action space
        stochastic_action = torch.max(stochastic_action, self._action_low)
        stochastic_action = torch.min(stochastic_action, self._action_high)
        return stochastic_action