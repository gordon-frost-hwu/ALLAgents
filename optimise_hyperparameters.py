# pylint: disable=unused-import
import argparse
import os
from all.environments import GymEnvironment
from all.experiments import OptimisationExperiment
from all.logging import ExperimentWriter
import presets
from pyeasyga import pyeasyga
import random
import numpy as np

class OptimisePreset(object):
    def __init__(self, args, write_loss=False):
        self.args = args
        self._write_loss = write_loss
        self.agent_name = args.agent
        self.agent = getattr(presets, self.agent_name)
        self.bounds = {
            "aa": [1e-6, 1e-2],
            "ac": [1e-6, 1e-2]
        }
        # "gamma": [0.1, 1.0]
        # }
        self.seed_data = [('aa', 1e-4), ('ac', 1e-3)]   #, ('gamma', 0.98)]

        self.result_dir = self.create_result_dir(self.agent_name, self.args.env)

        self.ga_log_file = open("{0}/GaEvolution.fso".format(self.result_dir), "w+")
        self.ind_lookup = {}
        self.individual_id = 0
        self.ga_generation_file = open("{0}/GenerationEvolution.fso".format(self.result_dir), "w+")

        num_generations = 500
        self.ga = pyeasyga.GeneticAlgorithm(self.seed_data,
                                           population_size=4,
                                           generations=num_generations,
                                           crossover_probability=0.8,
                                           mutation_probability=0.2,  # 0.05
                                           elitism=True,
                                           maximise_fitness=False)
        # assign callback methods
        self.ga.fitness_function = self.fitness
        self.ga.create_individual = self.create_individual
        self.ga.mutate_function = self.mutate
        self.ga.selection_function = self.selection
        self.ga.crossover_function = self.crossover
        self.ga.run = self.ga_run

        # TODO - add normaliser here and pass down into agent

    def run(self):

        self.ga.run()

    def ga_run(self):
        """Run (solve) the Genetic Algorithm."""
        self.ga.create_first_generation()
        self.log_best_in_generation()

        for _ in range(1, self.ga.generations):
            self.ga.create_next_generation()
            self.log_best_in_generation()

    def log_best_in_generation(self):
        best_fitness, best_genes = self.ga.best_individual()
        log_entry_individual = '\t'.join(map(str, best_genes))
        log_entry = log_entry_individual + "\t" + str(best_fitness) + "\n"
        self.ga_generation_file.write(log_entry)
        self.ga_generation_file.flush()

    def check_for_past_result(self, individual):
        closest = None
        tolerance = 1e-4
        for key in self.ind_lookup.keys():
            diff = np.array(key) - np.array(individual)
            match = all(abs(d) < tolerance for d in diff)
            if match:
                print("Individual {0} matched with {1}".format(individual, key))
                closest = key
        return closest

    def fitness(self, individual, data):
        # Check if the individual has been run before
        # closest = self.check_for_past_result(individual)
        # if closest is not None:
        #     # print("Similar individual run in past")
        #     # print("new  individual: {0}".format(individual))
        #     # print("past individual: {0}".format(closest))
        #     fitness = self.ind_lookup[closest]
        #     self.log_individual(0, individual, fitness)
        #     return fitness

        print("Running individual: {0}".format(individual))

        # create the environment and agent
        env = GymEnvironment(self.args.env, device=self.args.device)
        experiment = OptimisationExperiment(
            self.agent(device=args.device, lr_pi=individual[1], lr_v=individual[0]), env,
            episodes=args.episodes,
            frames=args.frames,
            render=args.render,
            log=True,
            quiet=True,
            write_loss=False,
            write_episode_return=True,
            writer=self._make_writer(self.agent_name, env.name, self._write_loss, self.result_dir),
        )
        returns = np.array(experiment.runner.rewards)     # returns against episodes
        solved_return_value = np.array([100.0 for x in range(len(returns))])
        fitness = sum(solved_return_value - returns)

        # Log stuff
        self.log_individual(1, individual, fitness)
        self.individual_id += 1
        return fitness

    def log_individual(self, executed, individual, fitness):
        log_entry_individual = '\t'.join(map(str, individual))
        log_entry = str(executed) + "\t" + log_entry_individual + "\t" + str(fitness) + "\n"
        self.ga_log_file.write(log_entry)
        self.ind_lookup[tuple(individual)] = fitness
        self.ga_log_file.flush()

    def selection(self, population):
        return random.choice(population)

    def crossover(self, parent_1, parent_2):
        index = random.randrange(1, len(parent_1))
        child_1 = parent_1[:index] + parent_2[index:]
        child_2 = parent_2[:index] + parent_1[index:]
        return child_1, child_2

    def mutate(self, individual):
        mutate_index = random.randrange(len(individual))
        field_key = self.seed_data[mutate_index][0]
        bounds = self.bounds[field_key]
        # individual[mutate_index] == random.uniform(bounds[0], bounds[1])
        random_value = np.random.normal(scale=0.01)
        individual[mutate_index] = np.clip(individual[mutate_index] + random_value,
                                           bounds[0], bounds[1])

    def create_individual(self, data):
        individual = [random.uniform(self.bounds[name][0], self.bounds[name][1]) for (name, value) in data]
        print("create_individual data: {0}".format(individual))
        return individual

    def create_result_dir(self, agent_name, env_name):
        idxs = [0]
        parent_folder = "{0}_{1}_optimisation".format(agent_name, env_name)
        dirs = os.listdir("runs")

        print("dirs: {0}".format(dirs))
        for s in dirs:
            if "optimisation" in s:
                idxs.append((int)(s.split("_")[-1].strip("optimisation")))
        maxDirectoryIndex = max(idxs)
        parent_folder = parent_folder + (str(maxDirectoryIndex + 1))

        result_name = os.path.join("runs", parent_folder)

        if os.path.exists(result_name):
            print("Result directory exists, aborting....")
            exit(0)

        os.mkdir(result_name)
        return result_name

    def _make_writer(self, agent_name, env_name, write_loss, parent_folder):
        return ExperimentWriter(agent_name, env_name, write_loss, parent_folder=parent_folder)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a continuous actions benchmark.")
    parser.add_argument("env", help="Name of the env (see envs)")
    parser.add_argument("agent", help="Name of the agent (e.g. cacla). See presets for available agents")

    parser.add_argument(
        "--episodes", type=int, default=2000, help="The number of training episodes"
    )
    parser.add_argument(
        "--frames", type=int, default=6e10, help="The number of training frames"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)",
    )
    parser.add_argument(
        "--render", default=False, help="Whether to render the environment."
    )
    args = parser.parse_args()

    optimiser = OptimisePreset(args)
    optimiser.run()
