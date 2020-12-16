import pytest
import numpy as np
from lightRaven.func import *

sample = np.array([ 2.70128706, -7.19966469, -1.77735702,  0.40663928, -2.98029333, -3.57325518,  1.52894389, -1.41821367,  1.81137184, -0.38087356])


def test_t_lb():
    print("\nTesting Student's t lower bound...")
    assert abs(t_lb(sample) + 2.816864) <= 1e-5


def test_t_ub():
    print("\nTesting Student's t upper bound...")
    assert abs(t_ub(sample) - 0.640580) <= 1e-5


def test_hoeffding_lb():
    print("\nTesting Hoeffding's lower bound...")
    assert abs(hoeffding_lb(sample) + 4.920035) <= 1e-5


def test_hoeffding_ub():
    print("\nTesting Hoeffding's t upper bound...")
    assert abs(hoeffding_ub(sample) - 2.743752) <= 1e-5


def test_mpeb_lb():
    print("\nTesting MPeB lower bound...")
    assert abs(mpeb_lb(sample) + 13.118698) <= 1e-5


def test_mpeb_ub():
    print("\nTesting MPeB upper bound...")
    assert abs(mpeb_ub(sample) - 10.942415) <= 1e-5
    

def test_mcma_lb():
    print("\nTesting Monte Carlo m_alpha lower bound...")
    assert abs(mcma_lb(sample) + 2.291385) <= 5e-2


def test_mcma_ub():
    print("\nTesting Monte Carlo m_alpha upper bound...")
    assert abs(mcma_ub(sample) - 0.326147) <= 5e-2
