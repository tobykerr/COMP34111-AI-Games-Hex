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
from src.Move import Move

def run_experiment(p1, p2, p1_name, p2_name, p1_class, p2_class, num_games, board_size, verbose, initial_move):
    p1_wins = 0
    p2_wins = 0
    runtimes = []
    turns = []
    for i in range(num_games):
        agent1 = getattr(p1, p1_class)(Colour.RED)
        if hasattr(agent1, "initial_move"):
            agent1.initial_move = initial_move

        agent2 = getattr(p2, p2_class)(Colour.BLUE)
        if hasattr(agent2, "initial_move"):
            agent2.initial_move = initial_move

        g = Game(
            player1=Player(
                name=p1_name,
                agent=agent1,
            ),
            player2=Player(
                name=p2_name,
                agent=agent2,
            ),
            board_size=board_size,
            #logDest=args.log,
            verbose=verbose,
        )
        results = g.run()
        #total_turns = results['total_turns']
        winner_name = results['winner']
        #total_time = results['total_game_time']
        # win_method = results['win_method']

        match winner_name:
            case args.player1Name:
                p1_wins += 1
            case args.player2Name:
                p2_wins += 1
        
        #turns.append(total_turns)
        #runtimes.append(total_time)

    return {
        'p1_wins': p1_wins,
        'p2_wins': p2_wins
        }

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
    #command line argument doesnt work for some reason, change default value as required for now.
    parser.add_argument(
        "-n",
        "--num_games",
        type=int,
        default=1,
        help="Number of games to be played as Red in the experiment. NOTE: Total games played will be twice this!"

    )

    args = parser.parse_args()
    p1_path, p1_class = args.player1.split(" ")
    p2_path, p2_class = args.player2.split(" ")
    p1 = importlib.import_module(p1_path)
    p2 = importlib.import_module(p2_path)
    num_games = args.num_games
    
    test_range = 2 #should be changed to 11 once done
    exp_results = [[None for x in range(test_range)] for y in range(test_range)]
    p1_wins = 0
    p2_wins = 0
    p1_winrates = [[0 for x in range(test_range)] for y in range(test_range)]
    p2_winrates = [[0 for x in range(test_range)] for y in range(test_range)]

    for i in range(test_range):
        for j in range(test_range):
            if (i,j) not in [(0,0), (0,10), (10,0), (10,10)]:
                exp_results[i][j] = run_experiment(p1, p2, args.player1Name, args.player2Name, p1_class, p2_class, num_games, args.board_size, args.verbose, Move(i,j))
                p1_winrates[i][j] = (exp_results[i][j]['p1_wins'] / (num_games)) * 100
                p1_wins += exp_results[i][j]['p1_wins']
                p2_winrates[i][j] = (exp_results[i][j]['p2_wins'] / (num_games)) * 100
                p2_wins += exp_results[i][j]['p2_wins']
    
    with open(args.log, 'w') as results_file:
        lines = []
        
        for i in range(test_range):
            line = ""
            for j in range(test_range):
                line += f"{p1_winrates[i][j]},"
            line = line[:-1]
            lines.append(line)
            lines += "\n"
            
        results_file.writelines(lines)
    