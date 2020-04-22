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

        # TODO - add normaliser here and pass down into agent

    def run(self):
        ga = pyeasyga.GeneticAlgorithm(self.seed_data,
                                       population_size=8,
                                       generations=200,
                                       crossover_probability=0.8,
                                       mutation_probability=0.05,
                                       elitism=True,
                                       maximise_fitness=False)
        # assign callback methods
        ga.fitness_function = self.fitness
        ga.create_individual = self.create_individual
        ga.mutate_function = self.mutate
        ga.selection_function = self.selection
        ga.crossover_function = self.crossover
        ga.run()

    def check_for_past_fitness(self, individual):
        search_key = np.array(individual)
        closest_individual = self.ind_lookup.get(search_key) or self.ind_lookup[
            min(self.ind_lookup.keys(), key=lambda key: abs(key - search_key))]
        if closest_individual - search_key < 1e-2:
            return closest_individual
        else:
            return None

    def fitness(self, individual, data):
        print("Running individual: {0}".format(individual))
        # TODO - check for pre-done individual fitness (i.e. look up from past)
        closest = self.check_for_past_fitness(individual)
        if closest is not None:
            print("Similar individual run in past")
            print("new  individual: {0}".format(individual))
            print("past individual: {0}".format(closest))
            return self.ind_lookup[closest]

        # create the environment and agent
        env = GymEnvironment(self.args.env, device=self.args.device)
        experiment = OptimisationExperiment(
            self.agent(device=args.device, lr_v=individual[0], lr_pi=individual[1]), env,
            episodes=args.episodes,
            frames=args.frames,
            render=args.render,
            log=True,
            quiet=False,
            write_loss=False,
            write_episode_return=True,
            writer=self._make_writer(self.agent_name, env.name, self._write_loss, self.result_dir),
        )
        returns = np.array(experiment.runner.rewards)     # returns against episodes
        solved_return_value = np.array([100.0 for x in range(len(returns))])
        fitness = sum(solved_return_value - returns)
        log_entry_individual = '\t'.join(map(str, individual))
        log_entry = log_entry_individual + "\t" + str(fitness) + "\n"
        self.ga_log_file.write(log_entry)
        self.ind_lookup[np.array(individual)] = fitness
        self.ga_log_file.flush()
        self.individual_id += 1
        return fitness

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
        individual[mutate_index] == random.uniform(bounds[0], bounds[1])

    def create_individual(self, data):
        print("create_individual data: {0}".format(data))
        return [random.uniform(self.bounds[name][0], self.bounds[name][1]) for (name, value) in data]

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
