import random
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Game import logger
from src.Tile import Tile
import time
import math

class Node:
    """MCTS Tree Node"""
    def __init__(self, move=None, parent=None, player: Colour = None):
        self.move = move              # Move that led to this node
        self.parent = parent          # Parent node
        self.children = {}            # Map move -> child Node
        self.visits = 0               # Number of times selected
        self.wins = 0                 # From those selections, how many wins (for the player who made the move that created this node)
        self.untried_moves = []       # Moves that have not been expanded
        self.rave_visits = 0          # RAVE visits = number of rollouts where this move was played, even if it was played later and not from this node
        self.rave_wins = 0            # RAVE wins = number of those rollouts where this move led to a win (for the player who made that move)
        self.player = player          # The player who made the move that led to this node (None for root)

    def uct_score(self: int, c: float = 1.4): # here, self should be the child node (i.e. node reached from s after taking action a)
        """Calculate the UCT score for this node. NOT CURRENTLY USED."""
        if self.visits == 0:                  # so self is like (s, a) in the UCT formula, and self.parent is like (s)
            return float('inf')
        exploitation = self.wins / self.visits
        exploration = c * ( (2*math.log(self.parent.visits) / self.visits) ** 0.5 )
        return exploitation + exploration
    
    def rave_score(self):
        """Get the RAVE score for this node. Noisy, but contains much more info in early steps."""
        if self.rave_visits == 0:
            return 0.5
        return self.rave_wins / self.rave_visits
    
    def blended_score(self, c=1.4, k=300):
        """Calculate the blended UCT + RAVE score for this node. Blends to weight RAVE more early on, and UCT more later."""
        if self.visits == 0:
            return float('inf')
        
        q = self.wins / self.visits # exploitation term in UCT
        q_rave = self.rave_score()

        beta = k / (self.visits + k)

        exploitation = beta * q_rave + (1 - beta) * q
        exploration = c * ( (2*math.log(self.parent.visits) / self.visits) ** 0.5 ) # exploration term in UCT
                    # c * math.sqrt(math.log(self.parent.visits) / self.visits)

        return exploitation + exploration

    def best_child(self, mode='robust'): # TOBY
        """Possibilities:
    - Max child (default): choose the child with the highest average reward (argmax_a Q(s,a))
    - Robust child: choose the child visited most often (highest N)
    - Max-robust child: choose the child which is maximal in both rewards and visits. If none exists, run longer until one exists.
    - Secure child: choose child which maximizes a lower confidence interval"""
        children_values = list(self.children.values())
        if not children_values:
            raise ValueError("No children to select from in best_child()")

        if mode == 'max':
            # Find child with highest average reward (wins / visits)
            return max(children_values, key=lambda n: n.wins / n.visits if n.visits > 0 else 0)
        if mode == 'robust':
            # Find child with highest visit count
            return max(children_values, key=lambda n: n.visits)
        if mode == 'max-robust':
            # Find children that are maximal in both wins and visits
            max_wins = max(n.wins for n in children_values)
            max_visits = max(n.visits for n in children_values)
            candidates = [n for n in children_values if n.wins == max_wins and n.visits == max_visits]
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
        self.time_limit = 4.5  # seconds per move, 4 seems good atm.
        self.swap_decided = False
        self.swap_choice = False

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        # --- handle pie rule --- # MIYED IS FIGURING THIS OUT
        if turn == 2 and opp_move is not None and not self.swap_decided:
            self.swap_decided = True
            self.swap_choice = self.decide_swap(board, opp_move)
            if self.swap_choice:
                return Move(-1, -1)
            
        save_move = self.check_savebridge(board, opp_move) # inefficient if it finds legal moves again later
        if save_move:
            # We found a forced move. Return it immediately.
            # (Ideally, we should inform the binary of this move if it maintains internal state,
            # but since we send the full board string every turn in 'CHANGE'/'START', 
            # the binary will sync up next turn).
            return save_move

        # --- get legal moves from the board ---
        legal_moves = [
            (tile.x, tile.y)
            for row in board.tiles
            for tile in row
            if tile.colour is None
        ]

        # --- run MCTS to select a move ---
        if legal_moves:
            x, y = self.run_mcts(board, legal_moves)
            return Move(x, y)
        else:
            # should never happen
            print("No legal moves available!")
            return Move(0, 0)
            
    def decide_swap(self, board: Board, first_move: Move):
        center = self.board_size // 2
        
        bias = (abs(center - first_move.x) + abs(center - first_move.y)) / (2 * center)
        
        if (first_move.x, first_move.y) in [(0,0), (0,board.size-1), (board.size-1,0), (board.size-1,board.size-1)]:
            return False
        
        opponent = Colour.BLUE if self.colour == Colour.RED else Colour.RED
        wins = 0
        
        #comment this once data has been obtained
        samples = 30
        for i in range(samples):
            sim_board = self.clone_board(board)
            sim_board.set_tile_colour(first_move.x, first_move.y, opponent)

            winner = self.play_random_game(sim_board, opponent)
            # winner = self.simulate(board, opponent)

            if winner == opponent:
                wins += 1
        
        win_rate = wins / samples
        
        # #uncomment this once data has been obtained, add filepath
        # '''
        # winrates = []
        # f = open()
        # for line in f:
        #     line = line.strip().split(",") #creates a node from an array from a line by seperating the numbers by ','
        #     line = [float(i) for i in line] #converts each array element from string into int 
        #     winrates.append(line) #adds new node to the graph
        # f.close()
        # '''
        
        # win_rate = winrates[first_move.x][first_move.y]
        
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
        root = Node(player=None)
        root.untried_moves = legal_moves.copy()
        random.shuffle(root.untried_moves)  # shuffle to avoid first-row bias

        start_time = time.time()
        while time.time() - start_time < self.time_limit:
            node = root
            # Clone the board once for this simulation and then apply moves inplace.
            state = self.clone_board(board)
            current_colour = self.colour  # start from agent's turn

            # 1. Selection
            while node.untried_moves == [] and node.children:
                node = self.uct_select(node)
                # apply selected child's move in-place
                state = self.apply_move(state, node.move, current_colour)
                current_colour = Colour.opposite(current_colour)  # switch turn

            # 2. Expansion
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                # create child with player = the player who plays this move
                child = Node(move=move, parent=node, player=current_colour)
                node.children[move] = child
                node = child
                state = self.apply_move(state, move, current_colour)
                current_colour = Colour.opposite(current_colour)  # switch turn

            # 3. Simulation
            winner, played_moves = self.simulate(state, current_colour)

            # 4. Backpropagation
            self.backpropagate(node, winner, played_moves)

        # Choose the move with the most visits
        best = root.best_child()
        return best.move
    
    def uct_select(self, node: Node) -> Node: # MIYED
        # select child with highest blended_score
        return max(node.children.values(), key=lambda child: child.blended_score())

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

    def apply_move(self, board: Board, move: tuple[int, int], colour: Colour) -> Board: #MIYED
        """
        Mutate the provided board in-place by applying the move, and return it.
        (Changed from previous cloning behavior to avoid repeated deep copies.)
        """
        x, y = move
        board.tiles[x][y].colour = colour  # Apply the move in-place
        return board  # Return the mutated board for convenience
    
    def get_biased_moves(self, board: Board, colour: Colour):
        """
        Scores empty tiles based on adjacency to own and opponent stones.
        Returns top 10 moves with highest scores.
        Moves are scored by:
        - +2 for each adjacent own stone
        - +1 for each adjacent opponent stone
        """
        moves = []
        for row in board.tiles:
            for t in row:
                if t.colour is None:
                    score = 0

                    # Prefer adjacency to own stones
                    for nx, ny in self.get_neighbors(t.x, t.y):
                        if board.tiles[nx][ny].colour == colour:
                            score += 2
                        elif board.tiles[nx][ny].colour == Colour.opposite(colour):
                            score += 1

                    moves.append(((t.x, t.y), score))

        moves.sort(key=lambda x: -x[1])
        return [m for m, _ in moves[:10]]  # top-k
    
    def get_neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        """Returns the coordinates of neighboring tiles for a given (x, y) position."""
        directions = [
            (-1, 0),  # up
            (1, 0),   # down
            (0, -1),  # left
            (0, 1),   # right
            (-1, 1),  # up-right
            (1, -1),  # down-left
        ]

        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                neighbors.append((nx, ny))

        return neighbors

    def _get_neighbors(self, x, y, board): # weird duplicate function? is this needed?
        """Get all valid neighbors of a position as a set of (x, y) tuples."""
        neighbors = set()
        for i in range(6):
            nx = x + Tile.I_DISPLACEMENTS[i]
            ny = y + Tile.J_DISPLACEMENTS[i]
            if self.is_valid_pos(nx, ny, board):
                neighbors.add((nx, ny))
        return neighbors

    def is_valid_pos(self, x, y, board):
        return 0 <= x < board.size and 0 <= y < board.size

    def _get_bridge_patterns(self):
        """
        Returns displacement pairs for stones that could form a bridge.
        These are all pairs of positions that are exactly 2 steps apart on the hex grid.
        """
        patterns = []
        # For each pair of neighbor indices, check if they could form a bridge
        for i in range(6):
            for j in range(i+1, 6):
                patterns.append((Tile.I_DISPLACEMENTS[i], Tile.J_DISPLACEMENTS[i],
                               Tile.I_DISPLACEMENTS[j], Tile.J_DISPLACEMENTS[j]))
        return patterns

    def check_savebridge(self, board: Board, opp_move: Move | None) -> Move | None:
        """
        Checks if the opponent's last move broke a bridge connection.
        A bridge is two of our stones that share exactly two common neighbors (carriers).
        If opponent plays on one carrier, we must play on the other.
        """
        if opp_move is None or opp_move.is_swap():
            return None

        mx, my = opp_move.x, opp_move.y
        
        # Find all pairs of our stones that have the opponent's move as a common neighbor
        # These are potential bridges being threatened
        for dx1, dy1, dx2, dy2 in self._get_bridge_patterns():
            s1_x, s1_y = mx + dx1, my + dy1
            s2_x, s2_y = mx + dx2, my + dy2
            
            # Check if both stones exist, are in bounds, and are our color
            if (self.is_valid_pos(s1_x, s1_y, board) and 
                self.is_valid_pos(s2_x, s2_y, board) and
                board.tiles[s1_x][s1_y].colour == self.colour and
                board.tiles[s2_x][s2_y].colour == self.colour):
                
                # Now find the other common neighbor (the other carrier)
                # Get neighbors of both stones
                neighbors1 = self._get_neighbors(s1_x, s1_y, board)
                neighbors2 = self._get_neighbors(s2_x, s2_y, board)
                
                # Find common neighbors
                common = neighbors1 & neighbors2
                
                # Remove the opponent's move from common neighbors
                common.discard((mx, my))
                
                # If there's exactly one other common neighbor, that's our saving move
                if len(common) == 1:
                    save_pos = list(common)[0]
                    return Move(save_pos[0], save_pos[1])
        
        return None



    def simulate(self, board: Board, colour: Colour) -> tuple[Colour, set]:
        state = board  # already cloned by caller
        played_moves = set()

        while True:  # iterate until a win or draw
            legal_moves = [
                (tile.x, tile.y)
                for row in state.tiles
                for tile in row
                if tile.colour is None
            ]
            if not legal_moves:
                return None, played_moves  # Draw (shouldn't happen in Hex, but kept just in case)
            
            if random.random() < 0.8:
                candidate_moves = self.get_biased_moves(state, colour)
                move = random.choice(candidate_moves) if candidate_moves else random.choice(legal_moves)
            else:
                move = random.choice(legal_moves)
            played_moves.add(move)
            state = self.apply_move(state, move, colour)

            # Use board-level detection of end-of-game (no dependence on passed colour)
            if state.has_ended(Colour.RED):
                return Colour.RED, played_moves
            if state.has_ended(Colour.BLUE):
                return Colour.BLUE, played_moves

            colour = Colour.opposite(colour) # switch turns

    def backpropagate(self, node: Node, winner: Colour, played_moves: set): #TOBY
        '''
        MoHex-style backpropagation with RAVE (AMAF):
        - Updates normal MCTS stats (visits, wins) per node (wins credited to node.player)
        - Updates RAVE stats for parent's children whose moves appeared in the playout
        '''
        cur = node

        while cur is not None:
            # --- normal MCTS stats ---
            cur.visits += 1
            if winner is not None and cur.player is not None and winner == cur.player:
                cur.wins += 1

            # --- RAVE updates (AMAF) for parent's children via dict lookup ---
            parent = cur.parent
            if parent is not None:
                for mv in played_moves:
                    child = parent.children.get(mv)
                    if child:
                        child.rave_visits += 1
                        # rave_wins credited to the player who would play that move (child.player)
                        if winner is not None and child.player is not None and winner == child.player:
                            child.rave_wins += 1

            # move up tree
            cur = parent
