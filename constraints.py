import numpy as np


# Add any constraint functions here
def g0(d: np.ndarray) -> np.ndarray:
    return 20 - d


def g1(d: np.ndarray) -> np.ndarray:
    return 30 - d


def g2(d: np.ndarray) -> np.ndarray:
    return 40 - d