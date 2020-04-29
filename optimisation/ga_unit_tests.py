#! /usr/bin/python

import numpy as np
import pandas as pd
import ga
import unittest
from copy import deepcopy

def test_mutGaussian(population):
    mutated = ga.mutGaussian(population, [0, 0, 0, 0],
                                        [0.1, 0.1, 5, 0.1],
                                        [0.5, 0.3, 0.5, 0.3])
    return mutated

class GaUnitTests(unittest.TestCase):
    def generate_test_data(self):
        self.population = np.array([[-0.16300957, -0.35246093, 0.8723527, -0.82950092],
                                   [-0.97824269, -0.79152116, 1.23034209, -0.16263559],
                                   [1.36857229, 1.2070625, -0.49675109, 0.75576418],
                                   [-1.13191167, 1.99955428, -1.32043214, 0.70352007],
                                   [-0.5904881, -1.4635747, -1.74232884, -2.23014166],
                                   [-0.64454314, 0.63685633, 0.7274629, -1.15782712],
                                   [-1.87533838, -0.74503839, -0.63225075, -0.38717283],
                                   [0.40036696, -1.19759659, -0.2538508, -0.40074278]])
        self.fitness = np.array([[0.16403259],
                                [-0.61119397],
                                [0.30187458],
                                [1.6466317],
                                [0.28843214],
                                [-0.82186789],
                                [0.63385562],
                                [-1.11972866]])

    def test_get_n_best(self):
        self.generate_test_data()
        indices = ga.get_n_best(self.fitness, 3, minimise=True)
        self.assertTrue(len(indices) == 3)
        self.assertTrue(indices[0] == 7)
        self.assertTrue(indices[1] == 5)
        self.assertTrue(indices[2] == 1)

    def test_greedy_selection(self):
        self.generate_test_data()

        for i in range(10):
            parents = ga.select_mating_pool_greedy(self.population, deepcopy(self.fitness), 2)
            # Check that it returns the 2 individuals with the minimum fitness
            self.assertTrue(np.allclose(parents[0, :],  self.population[-1, :]))

    def test_random_sampling(self):
        for i in range(10):
            indices = ga.generate_indices_randomly(10, 4)
            # Check that all indices are unique, and there are the number that were asked for
            self.assertTrue(len(indices) == 4)
            self.assertTrue(len(indices) == len(set(indices)))


if __name__ == '__main__':

    # population = df.iloc[0:4, 1:5].values
    unittest.main()
    exit(0)


