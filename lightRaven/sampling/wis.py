import numpy as np
from typing import List, Tuple
from numba import njit, prange
from numba.typed import List as nbList
from lightRaven.sampling import SamplingBase
from lightRaven.policy import PolicyBase, FAPolicy, TabularPolicy


class WIS(SamplingBase):
    """
    Weighted Importance Sampling
    """
    def __init__(self, dataset: List[Tuple[np.ndarray]], gamma=0.9):
        """
        Parameters
        ----------
        dataset : List[Tuple[np.ndarray]]
            A list of trajectories sampled by the behavioral policy.
            Contains (s, a, r, pi_b) for each timestamp
        gamma : float
            The discount factor. Mostly predefined by the MDP.
        """
        super(WIS, self).__init__(dataset, gamma)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _calc_tb_est(dataset_s: nbList, dataset_a: nbList, dataset_r: nbList, dataset_pi: nbList,
                     n_data: int, n_batch: int, pi_e_full: np.ndarray) -> np.ndarray:
        est_nom = np.zeros(n_batch)
        est_denom = np.zeros(n_batch)
        batch_size = n_data // n_batch

        for i in prange(n_batch):
            for j in prange(batch_size):
                trajs_idx = i * batch_size + j
                trajs = dataset_s[trajs_idx]
                compound_importance_weights = np.array(
                    [pi_e_full[trajs[k]][dataset_a[trajs_idx][k]] / dataset_pi[trajs_idx][k] for k in prange(trajs.size)]
                ).prod()
                est_nom[i] += compound_importance_weights * np.sum(dataset_r[trajs_idx])
                est_denom[i] += compound_importance_weights

        return est_nom / est_denom

    @staticmethod
    @njit(parallel=True)
    def _calc_fa_est(dataset_s: nbList, dataset_r: nbList, dataset_pi: nbList,
                     n_data: int, n_batch: int, pi_e_full: np.ndarray) -> np.ndarray:
        est_nom = np.zeros(n_batch)
        est_denom = np.zeros(n_batch)
        batch_size = n_data // n_batch

        for i in prange(n_batch):
            for j in prange(batch_size):
                trajs_idx = i * batch_size + j
                trajs_size = dataset_s[trajs_idx].shape[0]
                compound_importance_weights = np.array(
                    [pi_e_full[trajs_idx][k] / dataset_pi[trajs_idx][k] for k in prange(trajs_size)]
                ).prod()
                est_nom[i] += compound_importance_weights * np.sum(dataset_r[trajs_idx])
                est_denom[i] += compound_importance_weights

        return est_nom / est_denom

    def get_episodic_est(self, idx=None) -> float:
        raise ValueError

    def get_est(self, n_batch=100) -> np.ndarray:
        assert self.n_data >= n_batch
        if isinstance(self.eval_policy, FAPolicy):
            return self._calc_fa_est(self.dataset_s, self.dataset_r, self.dataset_pi,
                                     self.n_data, n_batch, self.eval_policy_pi)
        elif isinstance(self.eval_policy, TabularPolicy):
            return self._calc_tb_est(self.dataset_s, self.dataset_a, self.dataset_r, self.dataset_pi,
                                     self.n_data, n_batch, self.eval_policy_pi)


if __name__ == "__main__":
    import gym

    from lightRaven.utils import generate_dataset
    from lightRaven.sampling import IS, PDIS

    env_id = 'CartPole-v0'
    n_proc = 8
    test_size = 50000

    env = gym.make(env_id)
    dataset_test = generate_dataset(env_id, test_size, n_proc=n_proc)

    obs_shape = env.observation_space.shape[0]
    act_shape = env.action_space.n

    theta = np.random.normal(0, 1, obs_shape * act_shape)
    policy = FAPolicy(obs_shape, act_shape, theta)

    sampler_test_wis = WIS(dataset_test, gamma=1)
    sampler_test_is = IS(dataset_test, gamma=1)
    sampler_test_pdis = PDIS(dataset_test, gamma=1)

    sampler_test_wis.load_eval_policy(policy)
    sampler_test_is.load_eval_policy(policy)
    sampler_test_pdis.load_eval_policy(policy)

    est_wis = sampler_test_wis.get_est(n_batch=100)
    est_is = sampler_test_is.get_est()
    est_pdis = sampler_test_pdis.get_est()
    print(f"IS: {est_is.mean():.4f}, PDIS: {est_pdis.mean():.4f}, WIS: {est_wis.mean():.4f}")
