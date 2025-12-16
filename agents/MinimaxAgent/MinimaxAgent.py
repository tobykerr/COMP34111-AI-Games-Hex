import math
import copy

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Tile import Tile

class AlphaBetaAgent(AgentBase):
    def __init__(self, colour, board_size=11):
        super().__init__(colour)
        self.board_size = board_size
        self.max_depth = 2 # this can be changed

    def make_move(self, turn, board, opp_move):
        # 1. SAVEBRIDGE CHECK
        # If the opponent's last move attacks a bridge, we must save it immediately.
        # This acts as a heuristic pruning step.
        save_move = self.check_savebridge(board, opp_move)
        if save_move:
            return save_move

        # 2. Standard Minimax (if no bridge is threatened)
        # need to handle swap rule for very first move
        if opp_move is None:
             # Playing first, center is usually best
             center = self.board_size // 2
             return Move(center, center)
        
        # choose (x, y)
        x, y = self._choose_with_alpha_beta(board, turn)
        return Move(x, y)  
    
    # ------- Savebridge Logic -------
    def check_savebridge(self, board, opp_move):
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

    def _get_bridge_patterns(self):
        """
        Returns displacement pairs for stones that could form a bridge.
        These are all pairs of positions that are exactly 2 steps apart on the hex grid.
        """
        patterns = []
        #  For each pair of neighbor indices, check if they could form a bridge
        for i in range(6):
            for j in range(i+1, 6):
                patterns.append((Tile.I_DISPLACEMENTS[i], Tile.J_DISPLACEMENTS[i],
                               Tile.I_DISPLACEMENTS[j], Tile.J_DISPLACEMENTS[j]))
        return patterns
    
    def _get_neighbors(self, x, y, board):
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

    # ------- mini max alpha beta core

    def _choose_with_alpha_beta(self, board, turn):
        best_val = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf

        legal_moves = self._generate_legal_moves(board)

        for move in legal_moves:
            new_board = copy.deepcopy(board)
            self._apply_move(new_board, move, self.colour)

            val = self._min_value(new_board, depth=self.max_depth-1,
                                  alpha=alpha, beta=beta,
                                  turn=turn+1)

            if val > best_val:
                best_val = val
                best_move = move

            alpha = max(alpha, best_val)

        return best_move # x, y

    def _max_value(self, board, depth, alpha, beta, turn):
        if self._terminal_or_cutoff(board, depth, turn):
            return self._evaluate(board)

        value = -math.inf
        for move in self._generate_legal_moves(board):
            new_board = copy.deepcopy(board)
            self._apply_move(new_board, move, self.colour)
            value = max(value, self._min_value(new_board, depth-1, alpha, beta, turn+1))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def _min_value(self, board, depth, alpha, beta, turn):
        if self._terminal_or_cutoff(board, depth, turn):
            return self._evaluate(board)

        opp_colour = Colour.RED if self.colour == Colour.BLUE else Colour.BLUE
        value = math.inf
        for move in self._generate_legal_moves(board):
            new_board = copy.deepcopy(board)
            self._apply_move(new_board, move, opp_colour)
            value = min(value, self._max_value(new_board, depth-1, alpha, beta, turn+1))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def _terminal_or_cutoff(self, board, depth, turn):
        if depth <= 0:
            return True 
        if board.has_ended(self.colour) or board.has_ended(self.opp_colour()):
            return True
        return False
    
    def _generate_legal_moves(self, board):
        moves = []
        for y, row in enumerate(board.tiles):
            for x, tile in enumerate(row):
                if tile.colour is None:
                    moves.append((x,y))
        return moves

    def _apply_move(self, board, move, colour):
        x, y = move
        board.set_tile_colour(x, y, colour)

    def _evaluate(self, board):
        # Basic placeholder evaluation: Random value
        # In a real implementation, use Dijkstra to find shortest path for Red - shortest path for Blue
        import random
        return random.randint(-10, 10)