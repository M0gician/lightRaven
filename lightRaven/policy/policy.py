from abc import ABC
import numpy as np
from typing import Union, Optional
from numba import njit, prange
from numba.typed import List as nbList


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

    def pi_all(self, observations: Optional[nbList], actions: Optional[nbList]) -> np.ndarray:
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
        theta : np.ndarray
            The weight matrix for generating actions
        policy_type : Optional[str]
            The type of the policy ('Discrete' or 'Continuous')
        """
        super(TabularPolicy, self).__init__(obs_shape, act_shape, theta, policy_type)

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
    def _pi_all(theta: np.ndarray, observations: nbList, actions: nbList, n_data: int) -> nbList:
        """ Get the probability (density) of all actions `a` at state `s` when using current policy

        Parameters
        ----------
        theta : np.ndarray
            The weight matrix for generating actions
        """
        pi_e = nbList()
        for i in range(n_data):
            y = np.exp(observations[i] @ theta)
            prob = y / np.sum(y, axis=1).reshape(-1, 1)
            l = prob.shape[0]
            pi = np.empty(l)
            for j in prange(l):
                pi[j] = prob[j][actions[i][j]]
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

    def pi_all(self, observations: nbList, actions: nbList) -> nbList:
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


# if __name__ == "__main__":

