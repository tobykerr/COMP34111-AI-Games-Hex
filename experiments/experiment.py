import sys
import os

# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import importlib

import argparse
import importlib
import matplotlib.pyplot as plt
import numpy as np

from src.Colour import Colour
from src.Game import Game
from src.Player import Player

def run_experiment(p1, p2, p1_name, p2_name, p1_class, p2_class, num_games, board_size, verbose):
    p1_wins = 0
    p2_wins = 0
    runtimes = []
    turns = []
    p1_timeouts = 0
    p2_timeouts = 0
    for i in range(num_games):
        g = Game(
            player1=Player(
                name=p1_name,
                agent=getattr(p1, p1_class)(Colour.RED),
            ),
            player2=Player(
                name=p2_name,
                agent=getattr(p2, p2_class)(Colour.BLUE),
            ),
            board_size=board_size,
            #logDest=args.log,
            verbose=verbose,
        )
        results = g.run()
        total_turns = results['total_turns']
        winner_name = results['winner']
        total_time = results['total_game_time']
        win_method = results['win_method']

        if winner_name == p1_name:
            p1_wins += 1
            if win_method == 'TIMEOUT':
                p2_timeouts += 1
        elif winner_name == p2_name:
            p2_wins += 1
            if win_method == 'TIMEOUT':
                p1_timeouts += 1
        else:
            raise ValueError("Winner name does not match either player!")
        
        turns.append(total_turns)
        runtimes.append(total_time)

    return {
        'p1_wins': p1_wins,
        'p2_wins': p2_wins,
        'total_turns_list': turns,
        'total_runtimes_list': runtimes,
        'p1_timeouts': p1_timeouts,
        'p2_timeouts': p2_timeouts}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Experiment",
        description="""Run a series of games of Hex, with each player playing the same number of games as Red and Blue. Example use:  python experiments/experiment.py -p1 "agents.MCTSAgent.MCTSAgent MCTSAgent" -p2 "agents.TestAgents.ValidAgent ValidAgent" -p1Name "MCTSAgent" -p2Name "ValidAgent" -l "experiments/logs/test1" -n 10""",
    )
    parser.add_argument(
        "-p1",
        "--player1",
        default="agents.DefaultAgents.NaiveAgent NaiveAgent",
        type=str,
        help="Specify the player 1 agent, format: agents.GroupX.AgentFile AgentClassName .e.g. agents.Group0.NaiveAgent NaiveAgent",
    )
    parser.add_argument(
        "-p1Name",
        "--player1Name",
        default="Alice",
        type=str,
        help="Specify the player 1 name",
    )
    parser.add_argument(
        "-p2",
        "--player2",
        default="agents.DefaultAgents.NaiveAgent NaiveAgent",
        type=str,
        help="Specify the player 2 agent, format: agents.GroupX.AgentFile AgentClassName .e.g. agents.Group0.NaiveAgent NaiveAgent",
    )
    parser.add_argument(
        "-p2Name",
        "--player2Name",
        default="Bob",
        type=str,
        help="Specify the player 2 name",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "-b",
        "--board_size",
        type=int,
        default=11,
        help="Specify the board size",
    )
    parser.add_argument(
        "-l",
        "--log",
        nargs="?",
        type=str,
        default=sys.stderr,
        const="game.log",
        help=(
            "Save moves history to a log file,"
            "if the flag is present, the result will be saved to game.log."
            "If a filename is provided, the result will be saved to the provided file."
            "If the flag is not present, the result will be printed to the console, via stderr."
        ),
    )
    parser.add_argument(
        "-n",
        "--num_games",
        type=int,
        default=10,
        help="Number of games to be played as Red in the experiment. NOTE: Total games played will be twice this!"

    )

    args = parser.parse_args()
    p1_path, p1_class = args.player1.split(" ")
    p2_path, p2_class = args.player2.split(" ")
    p1 = importlib.import_module(p1_path)
    p2 = importlib.import_module(p2_path)
    num_games = args.num_games

    exp1_results = run_experiment(p1, p2, args.player1Name, args.player2Name, p1_class, p2_class, num_games, args.board_size, args.verbose)
    exp2_results = run_experiment(p2, p1, args.player2Name, args.player1Name, p2_class, p1_class, num_games, args.board_size, args.verbose)

    p1_wins = exp1_results['p1_wins'] + exp2_results['p2_wins']
    p2_wins = exp1_results['p2_wins'] + exp2_results['p1_wins']
    runtimes = exp1_results['total_runtimes_list'] + exp2_results['total_runtimes_list']
    turns = exp1_results['total_turns_list'] + exp2_results['total_turns_list']
    p1_timeouts = exp1_results['p1_timeouts'] + exp2_results['p2_timeouts']
    p2_timeouts = exp1_results['p2_timeouts'] + exp2_results['p1_timeouts']

    avg_turns_per_game = np.mean(turns)
    avg_game_time = np.mean(runtimes)
    p1_winrate = (p1_wins / (num_games*2)) * 100
    p2_winrate = (p2_wins / (num_games*2)) * 100
    with open(args.log, 'w') as results_file:
        lines = [
            "EXPERIMENT RESULTS:",
            f"\n{num_games * 2} games were played, {num_games} with {args.player1Name} as Red and {args.player2Name} as Blue, and {num_games} vice versa.",
            f"\n{args.player1Name} timed out {p1_timeouts} times.",
            f"\n{args.player2Name} timed out {p2_timeouts} times.",
            "\n",
            f"\nOVERALL RESULTS:",
            f"\n{args.player1Name} won {p1_wins} games with a winrate of {p1_winrate:.2f}%.",
            f"\n{args.player2Name} won {p2_wins} games with a winrate of {p2_winrate:.2f}%.",
            f"\nThe average total turns per game was {avg_turns_per_game:.2f} turns and the average total game time was {avg_game_time:.2f} seconds.",
            "\n",
            f"\nRESULTS WITH {args.player1Name} AS RED, {args.player2Name} AS BLUE:",
            f"\n{args.player1Name} won {exp1_results['p1_wins']} games with a winrate of {((exp1_results['p1_wins'] / num_games) * 100):.2f}%.",
            f"\n{args.player2Name} won {exp1_results['p2_wins']} games with a winrate of {((exp1_results['p2_wins'] / num_games) * 100):.2f}%.",
            f"\nThe average total turns per game was {np.mean(exp1_results['total_turns_list']):.2f} turns and the average total game time was {np.mean(exp1_results['total_runtimes_list']):.2f} seconds.",
            "\n",
            f"\nRESULTS WITH {args.player1Name} AS BLUE, {args.player2Name} AS RED:",
            f"\n{args.player1Name} won {exp2_results['p2_wins']} games with a winrate of {((exp2_results['p2_wins'] / num_games) * 100):.2f}%.",
            f"\n{args.player2Name} won {exp2_results['p1_wins']} games with a winrate of {((exp2_results['p1_wins'] / num_games) * 100):.2f}%.",
            f"\nThe average total turns per game was {np.mean(exp2_results['total_turns_list']):.2f} turns and the average total game time was {np.mean(exp2_results['total_runtimes_list']):.2f} seconds.",
        ]
        results_file.writelines(lines)
    