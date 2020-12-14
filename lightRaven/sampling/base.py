import numpy as np
from typing import List, Tuple
from numba.typed import List as nblist
from lightRaven.policy import PolicyBase, FAPolicy, TabularPolicy


class SamplingBase:
    """
    The base class of a sampling method
    """
    def __init__(self, dataset: List[Tuple[np.ndarray]], gamma=0.9, n_proc=8):
        """
        Parameters
        ----------
        dataset : List[Tuple[np.ndarray]]
            A list of trajectories sampled by the behavioral policy.
            Contains (s, a, r, pi) for each timestamp
        gamma : float
            The discount factor. Mostly predefined by the MDP.
        n_proc : int
            The total number of processes that can be spawned for parallelization
        """

        self.n_proc = n_proc
        self.gamma = gamma
        self.dataset_s, self.dataset_a, self.dataset_r, self.dataset_pi = nblist(), nblist(), nblist(), nblist()

        self.n_data = len(dataset)

        for i in range(self.n_data):
            trajectory = dataset[i]
            self.dataset_s.append(trajectory[0])
            self.dataset_a.append(trajectory[1])
            self.dataset_r.append(trajectory[2])
            self.dataset_pi.append(trajectory[3])

        self.eval_policy = None
        self.eval_policy_pi = None

    def load_eval_policy(self, eval_policy: PolicyBase) -> None:
        """ Load the evaluation policy into the estimator

        Parameters
        ----------
        eval_policy : policy
            The evaluation policy object.
        """
        assert isinstance(eval_policy, PolicyBase)
        self.eval_policy = eval_policy
        if isinstance(self.eval_policy, TabularPolicy):
            self.eval_policy_pi = self.eval_policy.pi_all()
        elif isinstance(self.eval_policy, FAPolicy):
            self.eval_policy_pi = self.eval_policy.pi_all(self.dataset_s, self.dataset_a)
        else:
            raise ValueError

    def get_episodic_est(self, idx=None) -> float:
        raise NotImplementedError

    def get_importance_weights(self, t_s_a_pi_b: Tuple[int, Tuple[int, int, float]]) -> float:
        """ Computes the importance weights pi_e / pi_b

        Parameters
        ----------
        t_s_a_pi_b : Tuple[int, Tuple[int, int, int]]
            A tuple contains:
                current time `t`
                current state `s`
                current action `a`
                current reward `pi_b`
        """
        t, s_a_pi_b = t_s_a_pi_b
        s, a, pi_b = s_a_pi_b
        return self.eval_policy.pi(s, a) / pi_b

    def get_est(self) -> np.ndarray:
        raise NotImplementedError
