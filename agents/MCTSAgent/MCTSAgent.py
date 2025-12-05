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

    def best_child(self, mode='max'): # TOBY
        """Possibilities:
	- Max child (default): choose the child with the highest average reward (argmax_a Q(s,a))
	- Robust child: choose the child visited most often (highest N)
	- Max-robust child: choose the child which is maximal in both rewards and visits. If none exists, run longer until one exists.
- Secure child: choose child which maximizes a lower confidence interval"""
        if mode == 'max':
            # Find child with highest average reward (wins / visits)
            return max(self.children, key=lambda n: n.wins / n.visits if n.visits > 0 else 0)
        if mode == 'robust':
            # Find child with highest visit count
            return max(self.children, key=lambda n: n.visits)
        if mode == 'max-robust':
            # Find children that are maximal in both wins and visits
            max_wins = max(n.wins for n in self.children)
            max_visits = max(n.visits for n in self.children)
            candidates = [n for n in self.children if n.wins == max_wins and n.visits == max_visits]
            if candidates:
                return candidates[0]  # Return the first maximal child
            else:
                raise ValueError("No maximal child found; consider running MCTS longer.")
        if mode == 'secure':
            pass  # TOBY: implement later

class MCTSAgent(AgentBase):
    """Minimal working MCTS agent skeleton.
    Currently selects random legal moves.
    Swap logic is handled on turn 2.
    Replace `run_mcts` with a full MCTS implementation later.
    """

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.board_size = 11  # fixed for this assignment
        self.time_limit = 1.8  # seconds per move
        self.swap_decided = False
        self.swap_choice = False

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        # --- handle pie rule --- # MIYED IS FIGURING THIS OUT
        if turn == 2 and opp_move is not None and not self.swap_decided:
            self.swap_decided = True
            self.swap_choice = self.decide_swap(board, opp_move)
            if self.swap_choice:
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
            
    def decide_swap(board: Board, first_move: Move):
        samples = 30 #changed depending on how much data may be needed
        
        center = self.board_size // 2
        
        bias = (abs(center - first_move.x) + abs(center - first_move.y)) / (2 * center)
        
        if (first_move.x, first_move.y) in [(0,0), (0,size-1), (size-1,0), (size-1,size-1)]:
            return False
        
        opponent = Colour.BLUE if self.colour == Colour.RED else Colour.RED
        wins = 0
        
        for i in range(samples):
            sim_board = self.clone_board(board)
            sim_board.set_tile_colour(first_move.x, first_move.y, opponent)

            winner = self.play_random_game(sim_board, opponent)
            if winner == opponent:
                wins += 1

        win_rate = wins / samples
        
        return (win_rate > 0.55 and bias < 0.65)
        
    def play_random_game(self, board: Board, current_player: Colour) -> Colour:
        size = board.size
        if current_player == Colour.RED:
            opponent = Colour.BLUE
        else:
            opponent = Colour.RED

        while True:
            empty_tiles = [(t.x, t.y) for row in board.tiles for t in row if t.colour is None]
            if not empty_tiles:
                return board.get_winner() or opponent
            
            x, y = random.choice(empty_tiles)
            board.set_tile_colour(x, y, current_player)

            if board.has_ended(Colour.RED):
                return Colour.RED
            if board.has_ended(Colour.BLUE):
                return Colour.BLUE

            current_player = opponent

    
    def run_mcts(self, board: Board, legal_moves: list[tuple[int, int]]) -> tuple[int, int]:
        root = Node()
        root.untried_moves = legal_moves.copy()
        random.shuffle(root.untried_moves)  # shuffle to avoid first-row bias

        start_time = time.time()
        while time.time() - start_time < self.time_limit:
            node = root
            state = self.clone_board(board)
            current_colour = self.colour  # start from agent's turn

            # 1. Selection
            while node.untried_moves == [] and node.children != []:
                node = self.uct_select(node)
                state = self.apply_move(state, node.move, current_colour)
                current_colour = Colour.opposite(current_colour)  # switch turn

            # 2. Expansion
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                child = Node(move=move, parent=node)
                node.children.append(child)
                node = child
                state = self.apply_move(state, move, current_colour)
                current_colour = Colour.opposite(current_colour)  # switch turn

            # 3. Simulation
            winner = self.simulate(state, current_colour)

            # 4. Backpropagation
            self.backpropagate(node, winner)

        # Choose the move with the most visits
        best = root.best_child()
        return best.move
    
    def uct_select(self, node: Node) -> Node: # MIYED
        return max(node.children, key=lambda child: child.uct_score())

    def clone_board(self, board: Board) -> Board: # must be very efficient # MIYED
        board_copy = Board(board.size)
        tiles_original = board.tiles
        tiles_copy = board_copy.tiles
        
        for i in range(board.size):
            row_original = tiles_original[i]
            row_copy = tiles_copy[i]
            for j in range(board.size):
                row_copy[j].colour = row_original[j].colour
                
        return board_copy

    def apply_move(self, board: Board, move: tuple[int, int], colour: Colour) -> Board: #MIYED # This is toby's quick implementation currently, using for testing other functions
        new_board = self.clone_board(board)  # Create a copy of the board
        x, y = move
        new_board.tiles[x][y].colour = colour  # Apply the move
        return new_board  # Return the updated board

    def simulate(self, board: Board, colour: Colour) -> Colour: #TOBY
        state = self.clone_board(board)

        while True:  # iterate until a win or draw
            legal_moves = [
                (tile.x, tile.y)
                for row in state.tiles
                for tile in row
                if tile.colour is None
            ]
            if not legal_moves:
                return None  # Draw

            move = random.choice(legal_moves)
            state = self.apply_move(state, move, colour)

            if state.has_ended(colour): 
                return state.get_winner() # return winning colour

            colour = Colour.opposite(colour) # switch turns

    def backpropagate(self, node: Node, winner: Colour): #TOBY
        '''
        Backpropagate the result of a simulation up the tree.
        Increment visits (and wins if agent won) for each node up to the root.
        Args:
            node (Node): The node to start backpropagation from.
            winner (Colour): The colour of the winning player.
        '''
        reward = 1 if winner == self.colour else 0 # only increment wins if agent's colour won

        while node is not None: # until we reach root of tree
            node.visits += 1
            node.wins += reward
            node = node.parent