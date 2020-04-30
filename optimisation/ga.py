import numpy
from copy import copy, deepcopy
import pandas as pd

def cal_pop_fitness(equation_inputs, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    fitness = numpy.sum(pop*equation_inputs, axis=1)
    return fitness


# Select the num_parents best individuals
def select_mating_pool_greedy(pop, fitness, num_parents):
    """
    Selecting the best individuals in the current generation as parents for
    producing the offspring of the next generation.
    :param pop: the population as a numpy 2D array
    :param fitness: the fitness of the population as a 1D array (same order as population)
    :param num_parents: integer of the number of parents to be returned
    :return: numpy array of the parents selected for mating
    """
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.amin(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = 99999999999
    return parents


def select_mating_pool_tournament(pop, fitness, tour,
                                  num_parents=2,
                                  minimise=True,
                                  print_debug=False):
    """
    Select the #N number of individuals using a tournament selection
    :param pop: the population as a numpy 2D array
    :param fitness: the fitness of the population as a 1D array (same order as population)
    :param tour: integer representing the number of solutions to randomly sample from the population
    :return:
    """
    entry_indices = generate_indices_randomly(pop.shape[0], tour)
    tournament_pop_entries = pop[entry_indices]
    tournament_fitness_entries = fitness[entry_indices]
    if print_debug:
        print("tournament entries:\n{0}".format(tournament_pop_entries))
        print("tournament fitness:\n{0}".format(tournament_fitness_entries))

    best_idxs = get_n_best(tournament_fitness_entries, num_parents, minimise=minimise)
    parents = tournament_pop_entries[best_idxs]
    return parents


def generate_indices_randomly(array_size, num_indices):
    """
    Generate a list of unique indexes that are randomly sampled
    :param array_size: max index
    :param num_indices: number of indices to return
    :return: list of indices that can itself be used as an index to an array
    """
    indices = []
    while len(indices) < num_indices:
        r = numpy.random.randint(0, array_size)
        if r not in indices:
            indices.append(r)
    # print(entry_indices)
    return indices


def get_n_best(array, number_of_indices, minimise=True):
    """
    :param array: 1D numpy array of floats
    :param number_of_indices: number of indexes to return
    :param minimise: whether to take the number_of_indices as minimum or maximum values
    :return: a list with the indexes of the number_of_indices minimum or maximum values
    """
    parents = []
    for i in range(number_of_indices):
        best = 99999999999 if minimise else -99999999999
        best_idx = -1
        for index, value in zip(range(len(array)), array):
            value = value[0]
            # print("evaluating {0}, {1}".format(index, value))
            is_better_condition = value < best if minimise else value > best
            if is_better_condition and index not in parents:
                best_idx = copy(index)
                best = value
        if best_idx > -1:
            parents.append(best_idx)
    return parents


def update_population_using_elitism(population,
                                    parents, parents_fitness,
                                    children, children_fitness,
                                    minimise=True):
    children_copy = deepcopy(children)
    children_fitness_copy = deepcopy(children_fitness)
    mask = numpy.ones(children_copy.shape[0], dtype=bool)
    for parent, fitness_p in zip(parents, parents_fitness):
        best_child = None
        best_child_fitness = None
        idx = 0
        for child, fitness_c in zip(children_copy[mask], children_fitness_copy[mask]):
            elitism_condition = fitness_c < fitness_p if minimise else fitness_c > fitness_p
            child_better_condition = fitness_c < best_child_fitness if minimise else fitness_c > best_child_fitness
            if elitism_condition and best_child_fitness is None or child_better_condition:
                best_child = child
                best_child_fitness = fitness_c
                # Hide the best child that has already been used
                mask[idx] = False
            idx += 1

        if best_child is not None:
            row_index = get_row_index(population, parent)
            if row_index is not None:
                population[row_index, :] = best_child
    return True


def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutation(offspring_crossover, num_mutations=1):
    print("MUTATION FUNC - START")
    print("offspring_crossover shape: {0}".format(offspring_crossover.shape))
    mutations_counter = numpy.uint8(offspring_crossover.shape[1] / num_mutations)
    print("mutations_counter: {0}".format(mutations_counter))

    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = numpy.random.normal(scale=0.3)   # numpy.random.uniform(-1.0, 1.0, 1)
            print("mutating index: {0} {1}".format(idx, gene_idx))
            offspring_crossover[idx, gene_idx] = numpy.clip(offspring_crossover[idx, gene_idx] + random_value, 0, 5)
            gene_idx = gene_idx + mutations_counter

    print("MUTATION FUNC - End")

    return offspring_crossover

def get_row_index(array, row):
    for idx, _row in zip(range(array.shape[0]), array):
        if numpy.allclose(row, _row, atol=0.01):
            return idx
    return None


def mutation_gaussian(offspring, mu, sigma, indpb):
    """This function applies a gaussian mutation of mean *mu* and standard
    deviation *sigma* on the input individual. This mutation expects a
    :term:`sequence` individual composed of real valued attributes.
    The *indpb* argument is the probability of each attribute to be mutated.
    :param individual: Individual to be mutated.
    :param mu: Mean or :term:`python:sequence` of means for the
               gaussian addition mutation.
    :param sigma: Standard deviation or :term:`python:sequence` of
                  standard deviations for the gaussian addition mutation.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    This function uses the :func:`~random.random` and :func:`~random.gauss`
    functions from the python base :mod:`random` module.
    """
    print("mutGaussian: offspring type: {0}".format(type(offspring)))
    for individual in offspring:

        size = len(individual)

        if len(mu) < size:
            raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
        if len(sigma) < size:
            raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))

        for i, m, s in zip(range(size), mu, sigma):
            if numpy.random.random() < indpb:
                individual[i] = numpy.clip(individual[i] + numpy.random.normal(scale=s), 0, 50)

    return offspring

if __name__ == '__main__':
    # Test data
    individual_length = 4
    population_size = 8
    population = pd.DataFrame(numpy.random.normal(size=(population_size, individual_length)), columns=list('ABCD'))
    fitness = pd.DataFrame(numpy.random.normal(size=(population_size)))

    print("Test population data:")
    print(population)
    print(fitness)

    print("\n######## Mating Pools #########")
    print("Greedy:")
    parents_greedy = select_mating_pool_greedy(population.values, deepcopy(fitness.values), 4)
    print(parents_greedy)

    print("\nTournament:")
    parents = select_mating_pool_tournament(population.values, fitness.values, 4, print_debug=True)
    print("result:\n{0}".format(parents))

    print("\n######## Mutation methods #########")
    # mutated = test_mutGaussian(population.values)
    # print("\nmutated:")
    # print(mutated)