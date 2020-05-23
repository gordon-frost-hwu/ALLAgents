# pylint: disable=unused-import
import argparse
import os
from all.environments import GymEnvironment
from all.experiments import OptimisationExperiment
from all.logging import ExperimentWriter
import optimisation.pyga as pyga
from optimisation.solution_description import SolutionDescription
import presets
import random
import numpy as np
from copy import copy

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

        self.result_dir = self.create_result_dir(self.agent_name, self.args.env)

        self.ind_lookup = {}
        self.individual_id = 0
        self.run_count = 1

        num_generations = 400
        num_genes = 2
        gene_bounds = np.array([[1e-6, 1e-1] for gene in range(num_genes)])
        gene_init_range = np.array([[1e-6, 1e-1] for gene in range(num_genes)])
        gene_sigma = np.array([0.1 for gene in range(num_genes)])
        gene_mutation_probability = np.array([0.2 for gene in range(num_genes)])
        gene_mutation_type = ["log", "log"]
        atol = np.array([1e-6 for gene in range(num_genes)])

        self.f_fitness_run_map = open("{0}{1}".format(self.result_dir, "/fitness_map.csv"), "w", 1)

        # gene_bounds = np.array([[0, 10] for gene in range(num_genes)])
        # gene_init_range = np.array([[0, 10] for gene in range(num_genes)])
        # gene_sigma = np.array([0.5 for gene in range(num_genes)])
        # gene_mutation_probability = np.array([0.2 for gene in range(num_genes)])
        solution_description = SolutionDescription(num_genes, gene_bounds,
                                                   gene_init_range, gene_sigma,
                                                   gene_mutation_probability,
                                                   gene_mutation_type,
                                                   atol)
        self.ga = pyga.GeneticAlgorithm(self.result_dir,
                                        solution_description,
                                        generations=num_generations,
                                        skip_known_solutions=True)
        # assign callback methods
        self.ga.calculate_fitness = self.fitness

        # TODO - add normaliser here and pass down into agent

    def run(self):
        self.ga.run()

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

    def fitness(self, individual):
        print("Running individual: {0}".format(individual))

        returns = []
        # TODO - average past fitness values
        # TODO - loop env
        run_idxs = []
        for i in range(2):
            # create the environment and agent
            env = GymEnvironment(self.args.env, device=self.args.device)
            experiment = OptimisationExperiment(
                self.agent(device=args.device, lr_v=individual[0], lr_pi=individual[1]), env,
                episodes=args.episodes,
                frames=args.frames,
                render=args.render,
                log=True,
                quiet=True,
                write_loss=False,
                write_episode_return=True,
                writer=self._make_writer(self.agent_name, env.name, self._write_loss, self.result_dir),
            )
            episodes_returns = np.array(experiment.runner.rewards)     # returns against episodes
            solved_return_value = np.array([100.0 for x in range(len(episodes_returns))])
            fitness = sum(abs(solved_return_value - episodes_returns))
            returns.append(fitness)
            run_idxs.append(copy(self.run_count))
            self.run_count += 1
        print("runs fitnesses: {0}".format(returns))
        avg_fitness = sum(returns) / len(returns)

        for idx in run_idxs:
            self.f_fitness_run_map.write("{0}\t{1}\t{2}\n".format(self.individual_id, idx, avg_fitness))
            self.f_fitness_run_map.flush()

        self.individual_id += 1
        return avg_fitness

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
        "--repeat", type=int, default=1, help="The number of training frames"
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

    for _ in range(args.repeat):
        optimiser = OptimisePreset(args)
        optimiser.run()
