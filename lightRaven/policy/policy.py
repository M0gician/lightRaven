from abc import ABC
import numpy as np
from typing import Union, Optional
from numba import njit, prange
from numba.typed import List as nblist


class PolicyBase(ABC):
    """
    An abstract class of the base policy for action making
    """

    def __init__(self, obs_shape: int, act_shape: int, theta=None, policy_type='discrete'):
        """
        Parameters
        ----------
        obs_shape : int
            The observation space size of the environment
        act_shape : int
            The observation space size of the environment
        theta : np.ndarray
            The weight matrix for generating actions
        policy_type : Optional[str]
            The type of the policy ('Discrete' or 'Continuous')
        """
        if policy_type not in ['discrete', 'continuous']:
            ValueError('env not supported')

        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.parameter_dim = self.obs_shape * self.act_shape
        self.type = policy_type

        if theta is not None:
            assert len(theta) == self.parameter_dim
            self.W = theta.reshape(self.obs_shape, self.act_shape)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _pi_all(theta: np.ndarray) -> np.ndarray:
        """ Get the probability (density) of all actions `a` at state `s` when using current policy

        Parameters
        ----------
        theta : np.ndarray
            The weight matrix for generating actions
        """
        exp_theta = np.exp(theta)
        return exp_theta / np.sum(exp_theta, axis=1).reshape(-1, 1)

    def load_theta(self, theta: np.ndarray):
        """ Load a weight matrix into the policy object

        Parameters
        ----------
        theta : np.ndarray
            The weight matrix for generating actions
        """
        assert len(theta) == self.parameter_dim
        self.W = theta.reshape(self.obs_shape, self.act_shape)

    def act(self, observation: np.ndarray) -> Union[int, float]:
        """ Make an `action` (a) given current state/observation

        Parameters
        ----------
        observation : np.ndarray
            A feature vector representing the current state/observation
        """
        raise NotImplementedError

    def pi(self, observation: np.ndarray, action: int) -> float:
        """ Get the probability (density) of taking action `a` in state `s` when using current policy
            ( In literature we use pi(a|s) )

        Parameters
        ----------
        observation : np.ndarray
            A feature vector representing the current state/observation
        action : int
            An int representing a specific action
        """
        raise NotImplementedError

    def pi_all(self, observations: Optional[nblist], actions: Optional[nblist]) -> np.ndarray:
        """ Get the probability (density) of all actions `a` at state `s` when using current policy
        """
        raise ValueError


class ContinuousPolicy(PolicyBase):
    def __init__(self, obs_shape: int, act_shape: int, theta: np.ndarray,
                 a_high: float, a_low: float, policy_type='continuous'):
        super(ContinuousPolicy, self).__init__(obs_shape, act_shape, theta, policy_type)
        self.action_high = a_high
        self.action_low = a_low

    def act(self, observation: np.ndarray) -> Union[int, float]:
        return float(np.clip(
            observation @ self.W,
            self.action_low,
            self.action_high
        ))

    def pi(self, observation: np.ndarray, action: int) -> float:
        raise ValueError


class TabularPolicy(PolicyBase):
    """
    An implementation of a policy for making discrete actions using tabular method
    """
    def __init__(self, obs_shape: int, act_shape: int, theta=None, policy_type='discrete'):
        """
        Parameters
        ----------
        obs_shape : int
            The observation space size of the environment
        act_shape : int
            The observation space size of the environment
        theta : Optional[np.ndarray]
            The weight matrix for generating actions
        policy_type : Optional[str]
            The type of the policy ('Discrete' or 'Continuous')
        """
        super(TabularPolicy, self).__init__(obs_shape, act_shape, theta, policy_type)

    def act(self, observation: int) -> Union[int, float]:
        """ Make an `action` (a) given current state/observation

        Parameters
        ----------
        observation : np.ndarray
            A feature vector representing the current state/observation
        """

        y = self.W[observation]
        action = np.argmax(y)
        return int(action)

    def pi(self, observation: np.ndarray, action: int) -> float:
        """ Get the probability (density) of taking action `a` in state `s` when using current policy
            ( In literature we use pi(a|s) )

        Parameters
        ----------
        observation : np.ndarray
            A feature vector representing the current state/observation
        action : int
            An int representing a specific action
        """

        assert action < self.act_shape
        y = self.W[observation]
        act_prob = np.exp(y) / np.exp(y).sum()
        return act_prob[action]

    def pi_all(self, observations=None, actions=None) -> np.ndarray:
        """ Get the probability (density) of all actions `a` at state `s` when using current policy
        """
        return self._pi_all(self.W.reshape(self.obs_shape, self.act_shape))


class FAPolicy(PolicyBase):
    """
    An implementation of a policy for making discrete actions using functional approximation
    """
    def __init__(self, obs_shape: int, act_shape: int, theta=None, policy_type='discrete'):
        """
        Parameters
        ----------
        obs_shape : int
            The observation space size of the environment
        act_shape : int
            The observation space size of the environment
        theta : np.ndarray
            The weight matrix for generating actions
        policy_type : Optional[str]
            The type of the policy ('Discrete' or 'Continuous')
        """
        super(FAPolicy, self).__init__(obs_shape, act_shape, theta, policy_type)

    # @staticmethod
    # @njit(parallel=True, fastmath=True)
    @staticmethod
    @njit(fastmath=True)
    def _pi_all(theta: np.ndarray, observations: nblist, actions: nblist, n_data: int) -> nblist:
        """ Get the probability (density) of all actions `a` at state `s` when using current policy

        Parameters
        ----------
        theta : np.ndarray
            The weight matrix for generating actions
        """
        # n_data = len(observations)
        # pi_e = nblist()
        # for i in prange(n_data):
        #     trajs = observations[i]
        #     l = trajs.shape[0]
        #     y = np.exp(trajs @ theta)
        #     act_prob = y / np.sum(y, axis=1).reshape(-1, 1)
        #     pi = np.empty(l)
        #     for j in prange(l):
        #         pi[j] = act_prob[j][actions[i][j]]
        #     pi_e.append(pi)

        pi_e = nblist()
        for i in prange(n_data):
            trajs = observations[i]
            l = trajs.shape[0]
            pi = np.empty(l)
            for j in prange(l):
                y = np.exp(trajs[j] @ theta)
                act_prob = y / np.sum(y)
                pi[j] = act_prob[actions[i][j]]
            pi_e.append(pi)

        return pi_e

    def act(self, observation: np.ndarray) -> Union[int, float]:
        """ Make an `action` (a) given current state/observation

        Parameters
        ----------
        observation : np.ndarray
            A feature vector representing the current state/observation
        """

        y = observation @ self.W
        action = np.argmax(y)
        return int(action)

    def pi(self, observation: np.ndarray, action: int) -> float:
        """ Get the probability (density) of taking action `a` in state `s` when using current policy
            ( In literature we use pi(a|s) )

        Parameters
        ----------
        observation : np.ndarray
            A feature vector representing the current state/observation
        action : int
            An int representing a specific action
        """

        assert action < self.act_shape
        y = observation @ self.W
        act_prob = np.exp(y) / np.exp(y).sum()
        return act_prob[action]

    def pi_all(self, observations: nblist, actions: nblist) -> nblist:
        """ Get the probability (density) of all actions `a` at state `s` when using current policy
        """
        return self._pi_all(self.W, observations, actions, len(observations))


class FixedPolicy(PolicyBase):
    """
    An implementation of a random policy for making discrete actions
    """
    def __init__(self, obs_shape: int, act_shape: int, theta=None, policy_type='discrete'):
        """
        Parameters
        ----------
        obs_shape : int
            The observation space size of the environment
        act_shape : int
            The observation space size of the environment
        theta : np.ndarray
            The weight matrix for generating actions
        policy_type : Optional[str]
            The type of the policy ('Discrete' or 'Continuous')
        """

        super(FixedPolicy, self).__init__(obs_shape, act_shape, theta, policy_type)

    def act(self, observation: int) -> Union[int, float]:
        """ Make a random `action` (a) given current state/observation

        Parameters
        ----------
        observation : np.ndarray
            A feature vector representing the current state/observation
        """

        y = self.W[observation]
        action = np.argmax(y)
        return int(action)

    def pi(self, observation: int, action: int) -> float:
        """ Get the probability (density) of taking action `a` in state `s` when using current policy
            ( In literature we use pi(a|s) )

        Parameters
        ----------
        observation : np.ndarray
            A feature vector representing the current state/observation
        action : int
            An int representing a specific action
        """

        return self.W[observation][action]

    def pi_all(self, observations=None, actions=None) -> np.ndarray:
        """ Get the probability (density) of all actions `a` at state `s` when using current policy
        """
        return self.W


if __name__ == "__main__":
    theta = np.array([-1.11103904,  2.65772476,  2.62261486,  1.55273624,  1.06967908,
        1.59625657,  1.0524968 , -0.55218172, -0.84406345, -3.03093374,
        0.63953049,  4.06984957,  1.77266935,  4.92305149,  0.83625335,
       -1.08213652, -1.38316531,  0.4485719 , -0.17656364,  1.10052213,
       -0.4114959 ,  3.85495898,  0.59365425,  0.3793948 ,  1.4523001 ,
        2.77265371,  2.02846595, -1.98026373,  1.87956751, -1.01344738,
        0.40333683,  0.51880066, -0.96691045,  1.26890676, -1.64650054,
        2.78428581, -2.30303155,  1.60393111,  0.25385971,  1.86596179,
       -2.35757175,  2.96227416, -5.03708822, -0.23500036, -1.27948492,
        0.38742942,  2.9907187 , -1.55446656,  1.49071407, -0.93354771,
       -0.45759545, -0.61362109,  1.74474914,  0.99892041,  1.38361903,
        2.08458883, -1.80548853, -0.0449766 ,  0.37669443,  2.98702162,
        1.94210075,  0.6154028 ,  2.31263148,  2.01465418, -0.29361901,
        0.43328224,  2.38943135,  1.26129625,  1.27986367, -1.15589531,
       -1.42575703,  5.57040527])
    policy = TabularPolicy(18, 4)
    policy.load_theta(theta)
    print(policy.pi_all())