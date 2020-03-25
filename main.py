# pylint: disable=unused-import
import argparse
from all.environments import GymEnvironment
from all.experiments import Experiment
from all.presets.continuous import ppo
from preset import cacla

def run():
    parser = argparse.ArgumentParser(description="Run a continuous actions benchmark.")
    parser.add_argument("env", help="Name of the env (see envs)")

    parser.add_argument(
        "--frames", type=int, default=6e5, help="The number of training frames"
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

    # env_id = "MountainCarContinuous-v0"
    # env_id = "Pendulum-v0"

    # create the environment
    env = GymEnvironment(args.env, device=args.device)

    # run the experiment
    Experiment(
        cacla(device=args.device), env, frames=args.frames, render=args.render
    )

    # run the baseline agent for comparison
    # Experiment(
    #     ppo(device=args.device), env, frames=args.frames, render=args.render
    # )


if __name__ == "__main__":
    run()
