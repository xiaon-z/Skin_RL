import argparse
import ast
from dataset import Dataset
from agent_DueDQN_gPSRM import Agent

def parse():
    parser = argparse.ArgumentParser()

    # Load data
    parser.add_argument('--minority_classes', type=ast.literal_eval, default=[7], help='Minority class(es)')

    # Training parameters
    parser.add_argument('--training_steps', type=int, default=20000, help='Total training steps used')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epsilon_start', type=float, default=0.5)
    parser.add_argument('--epsilon_end', type=float, default=0.01)
    parser.add_argument('--epsilon_decay_steps', type=int, default=20000, help='Steps until epsilon reaches minimum')
    parser.add_argument('--gamma', type=float, default=0.4, help='Discount rate')
    parser.add_argument('--target_update_rate', type=float, default=1., help='Rate of updating the target q network')
    parser.add_argument('--target_update_step', type=int, default=1000, help='Period to update the target q network')

    # Evaluation points
    parser.add_argument('--end_training', type=int, default=20000, help='Steps to end training and save model')
    parser.add_argument('--show_progress', type=int, default=1000, help='Steps to check training progress')

    config = parser.parse_args()
    return config

if __name__ == '__main__':

    dataset = Dataset(parse())
    agent = Agent(dataset, parse())
    agent.train()

