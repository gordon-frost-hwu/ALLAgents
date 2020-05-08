#! /usr/bin/python
import numpy as np
import abc

if __name__ != '__main__':
    from .ga import ga
else:
    from ga import ga


class GeneticAlgorithm(object):
    def __init__(self, result_dir, solution_description,
                 population_size=8,
                 generations=2000,
                 crossover_probability=0.8,
                 mutation_probability=0.2,  # 0.05
                 elitism=True,
                 minimise_fitness=True,
                 skip_known_solutions=False):
        __metaclass__ = abc.ABCMeta

        # Make sure that we have all the properties of a solution that we need
        # Note: this does not guaranteed they are the correct/assumed shapes
        assert hasattr(solution_description, "num_genes"), "num_genes missing from solution_description"
        assert hasattr(solution_description, "gene_bounds"), "gene_bounds missing from solution_description"
        assert hasattr(solution_description, "gene_init_range"), "gene_init_range missing from solution_description"
        assert hasattr(solution_description, "gene_sigma"), "gene_sigma missing from solution_description"
        assert hasattr(solution_description, "gene_mutation_prob"), \
            "gene_mutation_prob missing from solution_description"

        self.solution_description = solution_description
        self._population_size = population_size
        self._max_generations = generations
        self._crossover_probability = crossover_probability
        self._mutation_probability = mutation_probability
        self._elitism = elitism
        self._minimise_fitness = minimise_fitness
        self._skip_known_solutions = skip_known_solutions

        self.results_dir = result_dir

        self.solution_lookup = {}

        self.f_evolution_history = open("{0}{1}".format(self.results_dir, "/evolution_history.csv"), "w", 1)
        self.f_generation_max = open("{0}{1}".format(self.results_dir, "/generation_history.csv"), "w", 1)

    def seed_population(self):
        population = None
        for gene_idx, (lower_bound, upper_bound) in zip(range(self.solution_description.num_genes),
                                                        self.solution_description.gene_init_range):
            gene_weights = np.random.uniform(low=lower_bound, high=upper_bound, size=(self._population_size, 1))
            population = np.concatenate((population, gene_weights), axis=1) if population is not None else gene_weights

        return population

    @abc.abstractmethod
    def calculate_fitness(self, solution):
        """
        [5, 7] is the optimal solution for this fitness function, equating to a fitness of 0
        :param solution:
        :return:
        """
        # return abs((solution[0] - 1e-4)) + abs(solution[1] - 1e-5)
        return abs(((2*solution[0]**2) + solution[1]) - 57)

    def run(self):
        population = self.seed_population()
        print("Initial population:")
        print(population)

        fitness = np.zeros(shape=(self._population_size, 1))
        for solution_idx, solution in zip(range(self._population_size), population):
            solution_fitness = self.calculate_fitness(solution)
            self.log_solution(1, solution, solution_fitness)

            fitness[solution_idx, 0] = solution_fitness

        generation_idx = 0
        while generation_idx < self._max_generations:
            print("Population:\n{0}".format(population))
            print("Fitness:\n{0}".format(fitness))
            parents, parents_fitness = ga.select_mating_pool_tournament(population, fitness, 4,
                                                                        minimise=self._minimise_fitness)
            # TODO - check if parents are same
            print("Parents:\n{0}".format(parents))
            child = ga.crossover_random_chromosones(parents)
            print("Child after crossover:\n{0}".format(child))
            # child = ga.mutation_gaussian(child, self.solution_description.gene_sigma,
            #                              self.solution_description.gene_mutation_prob,
            #                              self.solution_description.gene_bounds)
            child = ga.mutation_linear(child, self.solution_description.gene_mutation_prob,
                                         self.solution_description.gene_bounds)
            print("Child after mutation:\n{0}".format(child))

            call_fitness_function = True

            # Check if the individual has been run before
            if self._skip_known_solutions:
                closest = self.check_for_past_result(child)
                if closest is not None:
                    # print("Similar individual run in past")
                    # print("new  individual: {0}".format(individual))
                    # print("past individual: {0}".format(closest))
                    child_fitness = self.solution_lookup[closest]
                    call_fitness_function = False

            if call_fitness_function:
                child_fitness = self.calculate_fitness(child)
                if ga.update_population_using_elitism(population, fitness,
                                                      parents, parents_fitness,
                                                      child, child_fitness,
                                                      self.solution_description.atol,
                                                      minimise=self._minimise_fitness):
                    generation_idx += 1
                    print("CHILD REPLACED PARENT")

            self.log_solution(call_fitness_function, child, child_fitness)
            self.log_best_in_generation(population, fitness)

            print("")

    def log_best_in_generation(self, population, fitness):
        best_idx = ga.get_n_best(fitness, 1, minimise=self._minimise_fitness)
        best_fitness, best_solution = fitness[best_idx][0][0], population[best_idx, :][0]
        log_entry_individual = '\t'.join(map(str, best_solution))
        log_entry = log_entry_individual + "\t" + str(best_fitness) + "\n"
        self.f_generation_max.write(log_entry)
        self.f_generation_max.flush()

    def log_solution(self, executed, solution, fitness):
        log_entry_individual = '\t'.join(map(str, solution))
        log_entry = str(int(executed)) + "\t" + log_entry_individual + "\t" + str(fitness) + "\n"
        self.f_evolution_history.write(log_entry)
        self.solution_lookup[tuple(solution)] = fitness
        self.f_evolution_history.flush()

    def check_for_past_result(self, individual):
        closest = None
        for key in self.solution_lookup.keys():
            diff = np.array(key) - np.array(individual)
            # match = all(abs(d) < tolerance for d in diff)
            match = np.all(abs(diff) < self.solution_description.atol)
            if match:
                print("Individual {0} matched with {1}".format(individual, key))
                closest = key
        return closest


if __name__ == '__main__':
    from solution_description import SolutionDescription

    num_genes = 2
    gene_bounds = np.array([[0, 10] for gene in range(num_genes)])
    gene_init_range = np.array([[0, 10] for gene in range(num_genes)])
    gene_sigma = np.array([0.5 for gene in range(num_genes)])
    gene_mutation_probability = np.array([0.2 for gene in range(num_genes)])
    atol = np.array([0.01 for gene in range(num_genes)])


    # gene_bounds = np.array([[1e-6, 1e-1] for gene in range(num_genes)])
    # gene_init_range = np.array([[1e-6, 1e-1] for gene in range(num_genes)])
    # gene_sigma = np.array([0.1 for gene in range(num_genes)])
    # gene_mutation_probability = np.array([0.2 for gene in range(num_genes)])
    # atol = np.array([1e-6 for gene in range(num_genes)])

    solution_description = SolutionDescription(num_genes, gene_bounds,
                                               gene_init_range, gene_sigma,
                                               gene_mutation_probability,
                                               atol)

    test_ga = GeneticAlgorithm("/tmp/", solution_description, generations=40, skip_known_solutions=True)
    test_ga.run()
