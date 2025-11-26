import random
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Game import logger
import time
import math

class Node:
    """MCTS Tree Node"""
    def __init__(self, move=None, parent=None):
        self.move = move              # Move that led to this node
        self.parent = parent          # Parent node
        self.children = []            # List of child nodes
        self.visits = 0               # Number of times visited
        self.wins = 0                 # Number of wins
        self.untried_moves = []       # Moves that have not been expanded

    def uct_score(self: int, c: float = 1.4): # here, self should be the child node (i.e. node reached from s after taking action a)
        if self.visits == 0:                  # so self is like (s, a) in the UCT formula, and self.parent is like (s)
            return float('inf')
        exploitation = self.wins / self.visits
        exploration = c * ( (2*math.log(self.parent.visits) / self.visits) ** 0.5 )
        return exploitation + exploration

    def best_child(self): # TOBY
        """Possibilities:
	- Max child (default): choose the child with the highest average reward (argmax_a Q(s,a))
	- Robust child: choose the child visited most often (highest N)
	- Max-robust child: choose the child which is maximal in both rewards and visits. If none exists, run longer until one exists.
    - Secure child: choose child which maximizes a lower confidence interval"""
        return max(self.children, key=lambda n: n.visits)

class MCTSAgent(AgentBase):
    """Minimal working MCTS agent skeleton.
    Currently selects random legal moves.
    Swap logic is handled on turn 2.
    Replace `run_mcts` with a full MCTS implementation later.
    """

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.board_size = 11  # fixed for this assignment

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        # --- handle pie rule --- # MIYED IS FIGURING THIS OUT
        if turn == 2:
            # swap move for testing / placeholder
            return Move(-1, -1)

        # --- get legal moves from the board ---
        legal_moves = [
            (tile.x, tile.y)
            for row in board.tiles
            for tile in row
            if tile.colour is None
        ]

        # --- fallback: pick a random legal move ---
        if legal_moves:
            x, y = self.run_mcts(board, legal_moves)
            return Move(x, y)
        else:
            # should never happen
            print("No legal moves available!")
            return Move(0, 0)
    
    def run_mcts(self, board: Board, legal_moves: list[tuple[int, int]]) -> tuple[int, int]:
        root = Node()
        root.untried_moves = legal_moves.copy()

        start_time = time.time()
        while time.time() - start_time < self.time_limit:
            node = root
            state = self.clone_board(board)

            # 1. Selection
            while node.untried_moves == [] and node.children != []:
                node = self.uct_select(node)
                state = self.apply_move(state, node.move, self.colour)

            # 2. Expansion
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                child = Node(move=move, parent=node)
                node.children.append(child)
                node = child
                state = self.apply_move(state, move, self.colour)

            # 3. Simulation
            winner = self.simulate(state, self.colour)

            # 4. Backpropagation
            self.backpropagate(node, winner)

        # Choose the move with the most visits
        best = root.best_child()
        return best.move
    
    def uct_select(self, node: Node) -> Node: # MIYED
        return max(node.children, key=lambda child: child.uct_score())

    def clone_board(self, board: Board) -> Board: # must be very efficient # MIYED
        board_copy = Board(board._size)
        
        for i in range(board._size):
            row_original = board.tiles[i]
            row_copy = board_copy[i]
            for j in range(board._size):
                row_copy[j].colour = row_original[j].colour
                
        return board_copy

    def apply_move(self, board: Board, move: tuple[int, int], colour: Colour) -> Board: #MIYED
        pass

    def simulate(self, board: Board, colour: Colour) -> Colour: #TOBY
        pass

    def backpropagate(self, node: Node, winner: Colour): #TOBY
        pass

