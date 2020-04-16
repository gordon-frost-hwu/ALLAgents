# pylint: disable=unused-import
import argparse
from all.environments import GymEnvironment
from all.experiments import Experiment
import presets

def run():
    parser = argparse.ArgumentParser(description="Run a continuous actions benchmark.")
    parser.add_argument("env", help="Name of the env (see envs)")
    parser.add_argument("agent", help="Name of the agent (e.g. cacla). See presets for available agents")

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

    # run the experiment
    Experiment(
        agent(device=args.device), env, frames=args.frames, render=args.render
    )

    # run the baseline agent for comparison
    # Experiment(
    #     ppo(device=args.device), env, frames=args.frames, render=args.render
    # )


if __name__ == "__main__":
    run()
