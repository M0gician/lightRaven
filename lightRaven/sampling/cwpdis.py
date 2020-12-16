import numpy as np
from typing import List, Tuple
from numba import njit, prange
from numba.typed import List as nbList
from lightRaven.sampling import SamplingBase
from lightRaven.policy import FAPolicy, TabularPolicy


class CWPDIS(SamplingBase):
    """
    Weighted Importance Sampling
    """
    def __init__(self, dataset: List[Tuple[np.ndarray]], gamma=0.9, max_l=None):
        """
        Parameters
        ----------
        dataset : List[Tuple[np.ndarray]]
            A list of trajectories sampled by the behavioral policy.
            Contains (s, a, r, pi_b) for each timestamp
        gamma : float
            The discount factor. Mostly predefined by the MDP.
        """
        super(CWPDIS, self).__init__(dataset, gamma)

        if max_l is None:
            self.max_l = np.max(np.array([dataset[i][0].shape[0] for i in range(self.n_data)]))
        else:
            self.max_l = max_l

    # @staticmethod
    # @njit(parallel=True, fastmath=True)
    @staticmethod
    def _calc_tb_est(dataset_s: nbList, dataset_a: nbList, dataset_r: nbList, dataset_pi: nbList,
                     n_data: int, n_batch: int, max_l: int, pi_e_full: np.ndarray) -> np.ndarray:
        est = np.zeros(n_batch)
        batch_size = n_data // n_batch

        for i in prange(n_batch):
            est_nom = np.zeros(max_l)
            est_denom = np.zeros(max_l)
            for j in prange(max_l):
                for k in prange(batch_size):
                    trajs_idx = i * batch_size + k
                    states = dataset_s[trajs_idx]

                    if j >= states.size:
                        est_nom[j] += 0
                        est_denom[j] += 1
                    else:
                        compound_importance_weights = np.array(
                            [pi_e_full[states[m]][dataset_a[trajs_idx][m]] / dataset_pi[trajs_idx][m] for m in
                             prange(j+1)]
                        ).prod()
                        est_nom[j] += compound_importance_weights * dataset_r[trajs_idx][j]
                        est_denom[j] += compound_importance_weights
            est[i] = np.sum(est_nom / est_denom)

        return est

    @staticmethod
    @njit(parallel=True)
    def _calc_fa_est(dataset_s: nbList, dataset_r: nbList, dataset_pi: nbList,
                     n_data: int, n_batch: int, max_l: int, pi_e_full: np.ndarray) -> np.ndarray:
        est = np.zeros(n_batch)
        batch_size = n_data // n_batch

        for i in prange(n_batch):
            est_nom = np.zeros(max_l)
            est_denom = np.zeros(max_l)
            for j in prange(max_l):
                for k in prange(batch_size):
                    trajs_idx = i * batch_size + k
                    trajs_size = dataset_s[trajs_idx].shape[0]

                    if j >= trajs_size:
                        est_nom[j] += 0
                        est_denom[j] += 1
                    else:
                        compound_importance_weights = np.array(
                            [pi_e_full[trajs_idx][m] / dataset_pi[trajs_idx][m] for m in prange(j+1)]
                        ).prod()
                        est_nom[j] += compound_importance_weights * dataset_r[trajs_idx][j]
                        est_denom[j] += compound_importance_weights
            est[i] = np.sum(est_nom / est_denom)

        return est

    def get_episodic_est(self, idx=None) -> float:
        raise ValueError

    def get_est(self, n_batch=100) -> np.ndarray:
        assert self.n_data >= n_batch
        if isinstance(self.eval_policy, FAPolicy):
            return self._calc_fa_est(self.dataset_s, self.dataset_r, self.dataset_pi,
                                     self.n_data, n_batch, self.max_l, self.eval_policy_pi)
        elif isinstance(self.eval_policy, TabularPolicy):
            return self._calc_tb_est(self.dataset_s, self.dataset_a, self.dataset_r, self.dataset_pi,
                                     self.n_data, n_batch, self.max_l, self.eval_policy_pi)


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

    sampler_test_cwpdis = CWPDIS(dataset_test, gamma=1)
    sampler_test_is = IS(dataset_test, gamma=1)
    sampler_test_pdis = PDIS(dataset_test, gamma=1)

    sampler_test_cwpdis.load_eval_policy(policy)
    sampler_test_is.load_eval_policy(policy)
    sampler_test_pdis.load_eval_policy(policy)

    est_cwpdis = sampler_test_cwpdis.get_est(n_batch=10)
    est_is = sampler_test_is.get_est()
    est_pdis = sampler_test_pdis.get_est()
    print(f"IS: {est_is.mean():.4f}, PDIS: {est_pdis.mean():.4f}, CWPDIS: {est_cwpdis.mean():.4f}")
