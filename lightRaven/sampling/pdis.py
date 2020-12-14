import numpy as np
from typing import List, Tuple
from numba import njit, prange
from numba.typed import List as nblist
from lightRaven.sampling import IS
from lightRaven.policy import PolicyBase, FAPolicy, TabularPolicy
from lightRaven.sampling import SamplingBase


class PDIS(IS):
    """
    The implementation of Importance Sampling Estimator
    """
    def __init__(self, dataset: List[Tuple[np.ndarray]], gamma=0.9, n_proc=8):
        """
        Parameters
        ----------
        dataset : List[Tuple[np.ndarray]]
            A list of trajectories sampled by the behavioral policy.
            Contains (s, a, r, s') for each timestamp
        behv_policy : policy
            The behavioral policy that generates the dataset.
        gamma : float
            The discount factor. Mostly predefined by the MDP.
        n_proc : int
            The total number of processes that can be spawned for parallelization
        """

        super(PDIS, self).__init__(dataset, gamma, n_proc)

    @staticmethod
    @njit(parallel=True)
    def _calc_tb_est(dataset_s: nblist, dataset_a: nblist, dataset_r: nblist, dataset_pi: nblist,
                     n_data: int, pi_e_full: np.ndarray) -> np.ndarray:
        est = np.empty(n_data)

        for i in prange(n_data):
            traj_size = dataset_s[i].size

            # Calculating all importance weights across the trajectory
            s = 0.0
            importance_weight = 1.0
            for j in prange(traj_size):
                s, a, pi_b = dataset_s[i][j], dataset_a[i][j], dataset_pi[i][j]
                importance_weight *= (pi_e_full[s][a] / pi_b)
                s += importance_weight * dataset_r[i][j]

            est[i] = s

        return est

    @staticmethod
    @njit(parallel=True)
    def _calc_fa_est(dataset_s: nblist, dataset_r: nblist, dataset_pi: nblist,
                     n_data: int, pi_e_full: nblist) -> np.ndarray:
        est = np.empty(n_data)

        for i in prange(n_data):
            traj_size = dataset_s[i].shape[0]

            # Calculating all importance weights across the trajectory
            s = 0.0
            importance_weight = 1.0
            for j in prange(traj_size):
                importance_weight *= pi_e_full[i][j] / dataset_pi[i][j]
                s += importance_weight * dataset_r[i][j]

            est[i] = s

        return est


if __name__ == "__main__":
    # import pickle
    # from lightRaven.policy import TabularPolicy
    #
    # n_proc = 6
    # safe_size = 10000000
    #
    # dataset = pickle.load(open('../../dataset.pickle', "rb"))
    # print("Dataset Loaded!")
    # sampler_safe = IS(dataset, gamma=1, n_proc=n_proc)
    #
    # policy_theta = np.array([-1.11036236, 2.65686747, 2.72182632, 1.55307054, 1.12477341,
    #                          1.59304398, 1.05309918, -0.5498818, -0.84351466, -2.81680243,
    #                          0.60726548, 4.09403887, 1.77653363, 4.90496071, 0.84392019,
    #                          -1.13692561, -1.4197318, 0.46920946, -0.17280731, 1.84314176,
    #                          -0.4245307, 3.85206167, 0.62269739, 0.38913152, 1.45244381,
    #                          2.77060771, 2.03162951, -1.98119988, 1.89080223, -1.00368014,
    #                          0.42235472, 0.52445275, -0.96903325, 1.23880848, -1.65574226,
    #                          2.78958083, -2.34229196, 1.61340813, 0.35849971, 1.86086199,
    #                          -2.36073352, 2.98946907, -5.0315063, -0.22035491, -1.17246592,
    #                          0.3808939, 2.99011244, -1.59308376, 1.47085365, -0.9353486,
    #                          -0.45585766, -0.61237927, 1.75028373, 0.94675292, 1.38176854,
    #                          2.08527732, -1.8067575, -0.04508603, 0.35713237, 3.03952345,
    #                          1.94232556, 0.58669684, 2.35990571, 2.05242542, -0.29257493,
    #                          0.4249482, 2.37701991, 1.22865922, 1.28674961, -1.17742659,
    #                          -1.06667484, 5.5789368])
    # test_policy = TabularPolicy(18, 4, policy_theta)
    # sampler_safe.load_eval_policy(test_policy)
    # print(sampler_safe.get_est())

    import gym

    from lightRaven.utils import generate_dataset, safety_test
    from lightRaven.sampling import IS

    env_id = 'CartPole-v0'
    n_proc = 8
    test_size = 500

    env = gym.make(env_id)
    dataset_test = generate_dataset(env_id, test_size, n_proc=n_proc)

    obs_shape = env.observation_space.shape[0]
    act_shape = env.action_space.n

    theta = np.ones(obs_shape * act_shape)
    policy = FAPolicy(obs_shape, act_shape, theta)

    sampler_test_is = IS(dataset_test, gamma=1, n_proc=n_proc)
    sampler_test_pdis = PDIS(dataset_test, gamma=1, n_proc=n_proc)

    sampler_test_is.load_eval_policy(policy)
    sampler_test_pdis.load_eval_policy(policy)

    est_is = sampler_test_is.get_est()
    est_pdis = sampler_test_pdis.get_est()

    print(est_is.mean())
    print(est_pdis.mean())
