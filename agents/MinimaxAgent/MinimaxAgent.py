import math
import copy
from heapq import heappush, heappop

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
        if turn == 1:
            return Move(1, 2)
        if turn == 2 and opp_move:
             # Simple check: If opponent played in the center 5x5 box, swap.
             if opp_move.x >= 3 and opp_move.x <= 7 and opp_move.y >= 3 and opp_move.y <= 7:
                  # no need to swap colour here, this is handled by Game.py
                  return Move(-1, -1)
        
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
            # new_board = copy.deepcopy(board)
            # self._apply_move(new_board, move, self.colour)

            # val = self._min_value(new_board, depth=self.max_depth-1,
            #                       alpha=alpha, beta=beta,
            #                       turn=turn+1)

            self._apply_move_inplace(board, move, self.colour)

            val = self._min_value(board, depth=self.max_depth-1,
                                  alpha=alpha, beta=beta,
                                  turn=turn+1)
            
            self._undo_move_inplace(board, move)

            if val > best_val:
                best_val = val
                best_move = move

            alpha = max(alpha, best_val)

        if best_move:
            return best_move # x, y
        else:
            print("best_move is None! Something went wrong.")
            print(f"Legal moves: {legal_moves}")
            print(f"Best val: {best_val}, Alpha: {alpha}, Beta: {beta}")
            raise ValueError("No best move found in Alpha-Beta search.")

    def _max_value(self, board, depth, alpha, beta, turn):
        if self._terminal_or_cutoff(board, depth, turn):
            return self._evaluate(board)

        value = -math.inf
        for move in self._generate_legal_moves(board):
            # new_board = copy.deepcopy(board)
            # self._apply_move(new_board, move, self.colour)
            # value = max(value, self._min_value(new_board, depth-1, alpha, beta, turn+1))
            self._apply_move_inplace(board, move, self.colour)
            value = max(value, self._min_value(board, depth-1, alpha, beta, turn+1))
            self._undo_move_inplace(board, move)
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
            # new_board = copy.deepcopy(board)
            # self._apply_move(new_board, move, opp_colour)
            # value = min(value, self._max_value(new_board, depth-1, alpha, beta, turn+1))
            self._apply_move_inplace(board, move, opp_colour)
            value = min(value, self._max_value(board, depth-1, alpha, beta, turn+1))
            self._undo_move_inplace(board, move)
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def _terminal_or_cutoff(self, board, depth, turn):
        if depth <= 0:
            return True
        if board.has_ended(Colour.RED) or board.has_ended(Colour.BLUE):
            return True
        return False
    
    def _generate_legal_moves(self, board):
        moves = []
        for y, row in enumerate(board.tiles):
            for x, tile in enumerate(row):
                if tile.colour is None:
                    # Return (Row, Col) => (y, x)
                    moves.append((y, x))
        return moves

    def _apply_move(self, board, move, colour):
        r, c = move # (Row, Col)
        board.set_tile_colour(r, c, colour)

    def _evaluate(self, board):
        # Calculate shortest path for self
        my_path_len = self.dijkstra_shortest_path(board, self.colour)
        # Calculate shortest path for opponent
        opp_path_len = self.dijkstra_shortest_path(board, self.opp_colour())
        
        
        # ensure no NaN results, which happen if child boards have no possible paths
        if my_path_len == math.inf and opp_path_len == math.inf:
            return 0
        if my_path_len == math.inf:
            return -10_000
        if opp_path_len == math.inf:
            return 10_000
        return opp_path_len - my_path_len

    def dijkstra_shortest_path(self, board, colour):
        size = board.size
        dists = [[math.inf for _ in range(size)] for _ in range(size)]
        pq = []
        
        # 1. Init start nodes
        if colour == Colour.RED: # Top to Bottom
            # Start nodes: Row 0
            for c in range(size):
                tile = board.tiles[0][c] # tiles[row][col]
                if tile.colour == colour:
                    dists[0][c] = 0
                    heappush(pq, (0, 0, c)) # cost, row, col
                elif tile.colour is None:
                    dists[0][c] = 1
                    heappush(pq, (1, 0, c))
        else: # Blue: Left to Right
            # Start nodes: Col 0
            for r in range(size):
                tile = board.tiles[r][0]
                if tile.colour == colour:
                    dists[r][0] = 0
                    heappush(pq, (0, r, 0)) # cost, row, col
                elif tile.colour is None:
                    dists[r][0] = 1
                    heappush(pq, (1, r, 0))
        
        processed = set()
        
        while pq:
            cost, r, c = heappop(pq)
            
            if (r, c) in processed:
                continue
            processed.add((r, c))
            
            # Check target
            if colour == Colour.RED:
                if r == size - 1:
                    return cost
            else:
                if c == size - 1:
                    return cost
            
            # Neighbors (Hex Grid: Row, Col)
            # Based on Tile.py Displacements:
            # (-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)
            neighbors = [
                (r-1, c), (r-1, c+1),
                (r, c+1), (r+1, c),
                (r+1, c-1), (r, c-1)
            ]
            
            for nr, nc in neighbors:
                if 0 <= nr < size and 0 <= nc < size:
                     tile = board.tiles[nr][nc]
                     weight = math.inf
                     if tile.colour == colour:
                         weight = 0
                     elif tile.colour is None:
                         weight = 1
                     
                     if weight != math.inf:
                         new_cost = cost + weight
                         if new_cost < dists[nr][nc]:
                             dists[nr][nc] = new_cost
                             heappush(pq, (new_cost, nr, nc))
                             
        return math.inf