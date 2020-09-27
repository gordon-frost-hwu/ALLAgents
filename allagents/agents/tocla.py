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
    This algorithm uses the CACLA actor-update rule (of van Hasselt and Wiering (2007)) and the
    Forward TD(lambda) algorithm for critic state-value estimates.
    This implementation tweaks the CACLA actor implementation slightly by only updating weights at the end
    of an episode and using a replay buffer to collect interaction samples. At the end of the episode,
    the replay buffer gets sampled n_iter times.
    CACLA: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.75.7658&rep=rep1&type=pdf
    Forward TD(lambda): https://arxiv.org/pdf/1608.05151.pdf

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
                 log=True,
                 sigma_decay=0.9995,
                 sigma_min=0.1,
                 n_iter=100,
                 minibatch_size=32,
                 writer=DummyWriter()):
        self.writer = writer
        self._log = log
        self._action = None
        self._state = None
        self._tde = None

        # Actor state
        self._replay_buffer = buffer
        self.n_iter = n_iter
        self.minibatch_size = minibatch_size
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self._action_low = torch.tensor(action_space.low, device=policy.device).float()
        self._action_high = torch.tensor(action_space.high, device=policy.device).float()
        self.policy = policy

        # Critic state
        self.critic = v
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
        self._c = 1.0
        self._v_current = 0
        self._ready = False
        self._step = 0
        self.exploration = None

    def act(self, state, reward):
        deterministic_action = self.act_delayed(state, reward)
        self._action = self._choose_action(deterministic_action)
        self.exploration = self._action - deterministic_action
        print("EXPLORED........")
        return self._action

    def act_delayed(self, state, reward):
        print("ACT_DELAYED")
        self._train_critic(state, reward)
        self._train_actor(state)

        if self._state is not None and self._tde is not None:
            if self._log:
                # print(self.writer.file_writer.get_logdir())
                self.writer.add_scalar("state/tde", self._tde, step=self._step)
                print("tde: {0}".format(self._tde))
            self._replay_buffer.store(self._state, self._action, self._tde, state)

        self._state = state
        self._step += 1
        return self.policy.eval(state)

    @property
    def tde(self):
        return self._tde

    def _train_critic(self, state, reward):
        if self._state is None:
            return

        # compute the state value for t+1 timestep
        # detach the Torch Tensor (make leaf node?) to avoid generated graphs being shared
        # and inhibiting consectutive backward passes
        v_next = 0 if state.done else self.critic.target(state).detach()
        rho = reward + self.discount_factor * ((1.0 - self.trace_decay) * v_next)

        if not self._fifo.full():
            self._fifo.put((self._state, self._action, reward, state, rho))
        # print("v_current: {0}".format(self._v_current))
        # print("v_next: {0}".format(v_next))
        # print("agent._train_critic reward: {0}".format(reward))

        self._tde = reward + (self.discount_factor * v_next) - self._v_current
        self._v_current = v_next

        if self._i == self.K - 1:
            self._u = self._u_sync
            self._u_sync = self._v_current
            self._i = 0
            self._c = 1.0
            self._ready = True
        else:
            self._u_sync = self._u_sync + self._c * self._tde
            self._i += 1
            self._c *= self.discount_factor * self.trace_decay

        # TODO - should this actually be before ready flag is set so that another step occurs? this is as per psuedocode
        if self._ready:
            self._u = self._u + self.c_final * self._tde
            self._critic_update_weights()

        # reset stuff if we reach a terminal state
        if state.done:
            # Episode ended
            if not self._ready:
                self._u = self._u_sync

            # Empty the queue if episode ended and train on contents
            while not self._fifo.empty():
                self._critic_update_weights()

            # reset internal variables for start of next episode
            self._u_sync = 0
            # print("u sync set to 0: self._u: {0}".format(self._u))
            self._i = 0
            self._c = 1
            self._v_current = 0
            self._ready = False

    def _critic_update_weights(self):
        s, u, r, sp, rp = self._fifo.get()
        # Update critic weights
        v = self.critic(s)
        loss = mse_loss(v, self._u)
        self.critic.reinforce(loss)
        # if self._log:
        #     self.writer.add_scalar("state/value", v[0])
        #     self.writer.add_scalar("state/target", self._u[0])

        if self.K != 1:
            self._u = (self._u - rp) / (self.discount_factor * self.trace_decay)

    def _train_actor(self, state):
        # only train (update weights) at the end of an episode; i.e. at a terminal state
        # if state.done:
        if len(self._replay_buffer) == 0:
            return

        for i in range(self.n_iter):
            # features, values, targets, actions = self.generate_targets()
            features, stochastic_actions, tde, _, _ = self._replay_buffer.sample(self.minibatch_size)

            # Get the indexes where the TDE is positive (i.e. the action resulted in a good state transition)
            idx = torch.where(tde > 0.0)[0]
            if len(idx) > 0:
                greedy_actions = self.policy(features)

                policy_loss = mse_loss(greedy_actions[idx], stochastic_actions[idx])
                policy_loss = policy_loss.float()
                policy_loss = policy_loss.cpu()

                if not torch.isnan(policy_loss):
                    self.policy.reinforce(policy_loss)
                else:
                    print("policy loss is NaN")

        # Decay the exploration
        # TODO - correct location
        if self.sigma > self.sigma_min:
            self.sigma *= self.sigma_decay

    def update_critic(self, values, targets):
        # backward pass
        value_loss = mse_loss(values, targets)
        self.critic.reinforce(value_loss)

    def _normal(self, output):
        if self._log:
            self.writer.add_scalar("sigma", self.sigma)
        return Normal(output, self.sigma)

    def _choose_action(self, deterministic_action):
        # If a feature ANN is provided, use it, otherwise raw state vector is used

        # uncomment to log the policy output
        # if self._log:
        # self.writer.add_scalar("action/det", deterministic_action)

        # Get the stochastic action by centering a Normal distribution on the policy output
        stochastic_action = self._normal(deterministic_action).sample().float()

        # Clip the stochastic action to the gym environment's action space
        stochastic_action = torch.max(stochastic_action, self._action_low)
        stochastic_action = torch.min(stochastic_action, self._action_high)
        return stochastic_action
