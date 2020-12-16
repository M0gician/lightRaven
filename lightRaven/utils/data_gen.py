import gym
import numpy as np
from typing import List, Tuple
from functools import partial
from multiprocessing import Pool
from lightRaven.policy import FAPolicy, TabularPolicy, PolicyBase


def generate_episode(env_id: str, gamma: float, _) -> Tuple:
    """ A partial function for multiprocessing the dataset generation
        This function samples every `state` (s), `action` (a), `reward` (r), and `n`ext action` (s')
        using an random policy for one trajectory.

    Parameters
    ----------
    env_id : str
        !!NOTE only 'cartpole' is supported currently!!
        A string in ['cartpole', 'pendulum', 'simglucose']
    gamma : float
        The discount factor
    _: any
        A placeholder variable for multiprocessing.Pool
    """
    env = gym.make(env_id)
    act_shape = env.action_space.n
    done = False

    s = env.reset()

    s_l = []
    a_l = []
    r_l = []
    pi_l = []
    current_gamma = 1

    while not done:
        a = env.action_space.sample()
        s_prime, r, done, info = env.step(a)
        s_l.append(s)
        a_l.append(a)
        r_l.append(r * current_gamma)
        pi_l.append(1.0 / act_shape)
        s = s_prime
        current_gamma *= gamma
    return np.array(s_l), np.array(a_l), np.array(r_l), np.array(pi_l)


def generate_dataset(env_id: str, n: int, gamma: float, n_proc=6) -> List[Tuple[np.ndarray]]:
    """ Generate `n` trajectories of data of a given `env_id` using a random policy.
        [Multiprocessing enabled]

    Parameters
    ----------
    env_id : str
        !!NOTE only 'cartpole' is supported currently!!
        A string in ['cartpole', 'pendulum', 'simglucose']
    n : int
        The number of trajectories that will be generated
    gamma : float
        The discount factor
    n_proc : int
        [Default =6] The number of processes that will be created
            (#process != #thread)
    """
    with Pool(n_proc) as p:
        dataset = p.map(partial(generate_episode, env_id, gamma), range(n))
    return dataset


def generate_theta_perf(env_id: str, theta: np.ndarray, p_type: str, gamma: float, _) -> np.ndarray:
    env = gym.make(env_id)
    obs_shape = env.observation_space.shape[0]
    act_shape = env.action_space.n

    if p_type == "fa":
        policy = FAPolicy(obs_shape, act_shape, theta)
    elif p_type == "tb":
        policy = TabularPolicy(obs_shape, act_shape, theta)
    else:
        raise ValueError

    current_gamma = 1
    R = 0.0
    done = False
    s = env.reset()
    while not done:
        a = policy.act(s)
        s_prime, r, done, _ = env.step(a)
        R += r * current_gamma
        current_gamma *= gamma
        s = s_prime

    return R


def generate_policy_perf(env_id: str, policy: PolicyBase, gamma: float, _) -> np.ndarray:
    env = gym.make(env_id)

    current_gamma = 1
    R = 0.0
    done = False
    s = env.reset()
    while not done:
        a = policy.act(s)
        s_prime, r, done, _ = env.step(a)
        R += r * current_gamma
        current_gamma *= gamma
        s = s_prime

    return R


def get_theta_perf(env_id: str, n: int, theta: np.ndarray, p_type: str, gamma: float, n_proc: int) -> np.ndarray:
    with Pool(n_proc) as p:
        dataset = p.map(partial(generate_theta_perf, env_id, theta, p_type, gamma), range(n))
    return np.array(dataset)


def get_policy_perf(env_id: str, n: int, policy: PolicyBase, gamma: float, n_proc: int) -> np.ndarray:
    with Pool(n_proc) as p:
        dataset = p.map(partial(generate_policy_perf, env_id, policy, gamma), range(n))
    return np.array(dataset)
