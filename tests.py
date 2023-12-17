import unittest
from amm import amm
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


class TestAmm(unittest.TestCase):

    def test_exchange(self):
        # initial reserves
        Rx0 = np.array([100, 100, 100], float)
        Ry0 = np.array([1000, 1000, 1000], float)

        # fee rates
        phi = np.array([0.003, 0.003, 0.003], float)

        # init pools
        pools = amm(Rx=Rx0, Ry=Ry0, phi=phi)

        self.assertTrue(np.allclose([316.228, 316.228, 316.228], np.round(pools.L, 3)))

        # swap test
        y = pools.swap_x_to_y([1, 0.5, 0.1], quote=False)
        self.assertTrue(np.allclose(y, np.array([9.87, 4.96, 1.]), rtol=1e-2))
        self.assertTrue(np.allclose(pools.Rx, np.array([101., 100.5, 100.1])))
        self.assertTrue(np.allclose(pools.Ry, np.array([990.13, 995.04, 999.])))

        # minting wrong amount
        x = np.array([1, 1, 1])
        self.assertRaises(AssertionError, pools.mint, x, np.random.rand(3))

        # minting correct amount
        y = x * pools.Ry / pools.Rx
        l = pools.mint(x=x, y=y)
        self.assertTrue(np.allclose(pools.l, np.array([3.13, 3.15, 3.16]), rtol=1e-2))
        self.assertTrue(np.allclose(pools.L, np.array([319.36, 319.37, 319.39]), rtol=1e-2))

        # burning correct amount
        x, y = pools.burn(l)
        self.assertTrue(np.allclose(pools.l, np.array(np.array([0., 0., 0.])), rtol=1e-2))
        self.assertTrue(np.allclose(x, np.array(np.array([1., 1., 1.])), rtol=1e-2))
        self.assertTrue(np.allclose(y, np.array([9.8, 9.9, 9.98]), rtol=1e-2))
        self.assertTrue(np.allclose(pools.L, np.array([316.23, 316.23, 316.23]), rtol=1e-2))

        # swapping 10 coin-X in each pool and minting
        l = pools.swap_and_mint([10, 10, 10])
        self.assertTrue(np.allclose(l, np.array(np.array([15.26, 15.34, 15.4])), rtol=1e-2))
        self.assertTrue(np.allclose(pools.l, np.array(np.array([15.26, 15.34, 15.4])), rtol=1e-2))
        self.assertTrue(np.allclose(pools.L, np.array(np.array([331.49, 331.56, 331.62])), rtol=1e-2))

        # burning and swapping the l LP tokens
        total_x = pools.burn_and_swap(l)
        # this output is different from the notebook, but it's incorrect in the notebook, since burn_and_swap returns an array, not a scalar
        self.assertTrue(np.allclose(total_x, np.array([9.97, 9.97, 9.97]), rtol=1e-2))
        self.assertTrue(np.allclose(pools.l, np.array(np.array([0., 0., 0.])), rtol=1e-2))
        self.assertTrue(np.allclose(pools.L, np.array(np.array([316.23, 316.23, 316.23])), rtol=1e-2))


if __name__ == '__main__':
    unittest.main()
