# pylint: disable=unused-import
import argparse
from all.environments import GymEnvironment
from all.experiments import OptimisationExperiment
from all.logging import ExperimentWriter
import presets
import os

def run():
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

    # create the environment
    env = GymEnvironment(args.env, device=args.device)

    agent_name = args.agent
    agent = getattr(presets, agent_name)

    # configure desired baseline (run sequentially)
    run_baseline = False
    baseline_agent_name = "cacla"
    baseline_agent = getattr(presets, baseline_agent_name)

    result_dir = create_result_dir(agent_name, args.env)

    num_repeats = 10
    for i in range(num_repeats):
        # run the experiment
        OptimisationExperiment(
            agent(device=args.device),
            env,
            episodes=args.episodes,
            frames=args.frames,
            render=args.render,
            writer=_make_writer(agent_name, env.name, True, result_dir),
            write_episode_return=True
        )

        if run_baseline:
            # run the baseline agent for comparison
            OptimisationExperiment(
                baseline_agent(device=args.device), env, episodes=args.episodes, frames=args.frames, render=args.render
            )


def create_result_dir(agent_name, env_name):
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


def _make_writer(agent_name, env_name, write_loss, parent_folder):
    return ExperimentWriter(agent_name, env_name, write_loss, parent_folder=parent_folder)


if __name__ == "__main__":
    run()
