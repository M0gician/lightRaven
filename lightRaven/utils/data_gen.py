import gym
import numpy as np
from typing import List, Tuple
from functools import partial
from multiprocessing import Pool


def generate_episode(env_id: str, _) -> Tuple:
    """ A partial function for multiprocessing the dataset generation
        This function samples every `state` (s), `action` (a), `reward` (r), and `n`ext action` (s')
        using an random policy for one trajectory.

    Parameters
    ----------
    env_id : str
        !!NOTE only 'cartpole' is supported currently!!
        A string in ['cartpole', 'pendulum', 'simglucose']
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

    while not done:
        a = env.action_space.sample()
        s_prime, r, done, info = env.step(a)
        s_l.append(s)
        a_l.append(a)
        r_l.append(r)
        pi_l.append(1.0 / act_shape)
        s = s_prime
    return np.array(s_l), np.array(a_l), np.array(r_l), np.array(pi_l)


def generate_dataset(env_id: str, n: int, n_proc=6) -> List[Tuple[np.ndarray]]:
    """ Generate `n` trajectories of data of a given `env_id` using a random policy.
        [Multiprocessing enabled]

    Parameters
    ----------
    env_id : str
        !!NOTE only 'cartpole' is supported currently!!
        A string in ['cartpole', 'pendulum', 'simglucose']
    n : int
        The number of trajectories that will be generated
    n_proc : int
        [Default =6] The number of processes that will be created
            (#process != #thread)
    """
    with Pool(n_proc) as p:
        dataset = p.map(partial(generate_episode, env_id), range(n))
    return dataset
