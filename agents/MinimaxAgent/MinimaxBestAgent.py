import math
import copy
from heapq import heappush, heappop

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

class AlphaBetaAgent(AgentBase):
    def __init__(self, colour, board_size=11):
        super().__init__(colour)
        self.board_size = board_size
        self.max_depth = 2 # this can be changed

        # --- Transposition Table (TT)
        self.tt = {}  # key -> (stored_depth, flag, value)
        self.TT_EXACT = 0
        self.TT_LOWER = 1
        self.TT_UPPER = 2

    def make_move(self, turn, board, opp_move):
        # Swap logic for Turn 2
        # Move.x is Row, Move.y is Col (Based on Game.py)
        if turn == 1:
            return Move(1, 2)
        if turn == 2 and opp_move and self.decide_swap(opp_move):
            return Move(-1, -1)
        
        # Standard Alpha-Beta Search
        move = self._choose_with_alpha_beta(board, turn)
        # move is (Row, Col)
        return Move(move[0], move[1])  
        
    # ------- swap decider function
    
    def decide_swap(self, first_move: Move):
        no_swaps = [(0,i) for i in range(10)] + [(10,j) for j in range(1,11)] + [(1,i) for i in range(9)] + [(10,j) for j in range(1,11)]
        if (first_move.x, first_move.y) in no_swaps:
            return False
        else:
            return True
    
    # ------- helper functions for in-place move apply and undo

    def _apply_move_inplace(self, board, move, colour):
        r, c = move
        # Faster than board.set_tile_colour and avoids extra checks
        board.tiles[r][c].colour = colour

    def _undo_move_inplace(self, board, move):
        r, c = move
        board.tiles[r][c].colour = None

    def _move_score(self, board, move):
        r, c = move
        size = board.size
        center_dist = abs(r - size // 2) + abs(c - size // 2)
        
        if self.colour == Colour.RED:
            goal = min(r, size - 1 - r)
        else:
            goal = min(c, size - 1 - c)
            
        friendly = 0
        neighbors = [
            (r-1, c), (r-1, c+1),
            (r, c+1), (r+1, c),
            (r+1, c-1), (r, c-1)
        ]
        
        for nr, nc in neighbors:
            if 0 <= nr < size and 0 <= nc < size:
                if board.tiles[nr][nc].colour == self.colour:
                    friendly += 1
        
        score = (center_dist) + (goal) - (2 * friendly)
        
        return score
    
    def _order_moves(self, board, moves):
        return sorted(moves, key=lambda m: self._move_score(board, m))
        

    # ------- mini max alpha beta core

    def _choose_with_alpha_beta(self, board, turn):
        best_val = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf

        legal_moves = self._generate_legal_moves(board)
        legal_moves = self._order_moves(board, legal_moves)

        for move in legal_moves:
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
            print("best_move is None! Something went wrong. Returning first legal move.")
            print(f"Legal moves: {legal_moves}")
            print(f"Best val: {best_val}, Alpha: {alpha}, Beta: {beta}")
            return legal_moves[0]  # Fallback, should never happen

    def _max_value(self, board, depth, alpha, beta, turn):
        if self._terminal_or_cutoff(board, depth, turn):
            return self._evaluate(board)

        # TT lookup
        alpha0, beta0 = alpha, beta
        key = self._tt_key(board, self.colour)

        hit = self.tt.get(key)
        if hit is not None:
            stored_depth, flag, val = hit
            if stored_depth >= depth:
                if flag == self.TT_EXACT:
                    return val
                elif flag == self.TT_LOWER:
                    alpha = max(alpha, val)
                else:  # TT_UPPER
                    beta = min(beta, val)
                if alpha >= beta:
                    return val

        value = -math.inf
        
        legal_moves = self._generate_legal_moves(board)
        legal_moves = self._order_moves(board, legal_moves)
        
        for move in legal_moves:
            self._apply_move_inplace(board, move, self.colour)
            value = max(value, self._min_value(board, depth-1, alpha, beta, turn+1))
            self._undo_move_inplace(board, move)

            if value >= beta:
                # store as LOWER bound
                self.tt[key] = (depth, self.TT_LOWER, value)
                return value

            alpha = max(alpha, value)

        # store in TT
        if value <= alpha0:
            flag = self.TT_UPPER
        elif value >= beta0:
            flag = self.TT_LOWER
        else:
            flag = self.TT_EXACT

        self.tt[key] = (depth, flag, value)
        return value

    def _min_value(self, board, depth, alpha, beta, turn):
        if self._terminal_or_cutoff(board, depth, turn):
            return self._evaluate(board)

        opp_colour = Colour.RED if self.colour == Colour.BLUE else Colour.BLUE

        # TT lookup
        alpha0, beta0 = alpha, beta
        key = self._tt_key(board, opp_colour)

        hit = self.tt.get(key)
        if hit is not None:
            stored_depth, flag, val = hit
            if stored_depth >= depth:
                if flag == self.TT_EXACT:
                    return val
                elif flag == self.TT_LOWER:
                    alpha = max(alpha, val)
                else:  # TT_UPPER
                    beta = min(beta, val)
                if alpha >= beta:
                    return val

        value = math.inf
        
        legal_moves = self._generate_legal_moves(board)
        legal_moves = self._order_moves(board, legal_moves)
        
        for move in legal_moves:
            self._apply_move_inplace(board, move, opp_colour)
            value = min(value, self._max_value(board, depth-1, alpha, beta, turn+1))
            self._undo_move_inplace(board, move)

            if value <= alpha:
                # store as UPPER bound (fail-low)
                self.tt[key] = (depth, self.TT_UPPER, value)
                return value

            beta = min(beta, value)

        # store in TT
        if value <= alpha0:
            flag = self.TT_UPPER
        elif value >= beta0:
            flag = self.TT_LOWER
        else:
            flag = self.TT_EXACT

        self.tt[key] = (depth, flag, value)
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

        opp = Colour.RED if colour == Colour.BLUE else Colour.BLUE

        neighbors = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

        def empty_cost(r, c):
            # Cheaper if supported by own stones, pricier if contested by opponent.
            own_adj = 0
            opp_adj = 0
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    t = board.tiles[nr][nc].colour
                    if t == colour:
                        own_adj += 1
                    elif t == opp:
                        opp_adj += 1
            # Base cost of 10; adjust based on adjacency; restrict to 2 < w < 30.
            w = 10 - 2 * own_adj + 3 * opp_adj
            if w < 2:
                w = 2
            if w > 30:
                w = 30
            return w

        # Init start nodes
        if colour == Colour.RED:
            for c in range(size):
                t = board.tiles[0][c].colour
                if t == colour:
                    dists[0][c] = 0
                    heappush(pq, (0, 0, c))
                elif t is None:
                    w = empty_cost(0, c)
                    dists[0][c] = w
                    heappush(pq, (w, 0, c))
        else:
            for r in range(size):
                t = board.tiles[r][0].colour
                if t == colour:
                    dists[r][0] = 0
                    heappush(pq, (0, r, 0))
                elif t is None:
                    w = empty_cost(r, 0)
                    dists[r][0] = w
                    heappush(pq, (w, r, 0))

        processed = set()

        while pq:
            cost, r, c = heappop(pq)
            if (r, c) in processed:
                continue
            processed.add((r, c))

            if colour == Colour.RED:
                if r == size - 1:
                    return cost
            else:
                if c == size - 1:
                    return cost

            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    t = board.tiles[nr][nc].colour
                    if t == colour:
                        w = 0
                    elif t is None:
                        w = empty_cost(nr, nc)
                    else:
                        continue
                    new_cost = cost + w
                    if new_cost < dists[nr][nc]:
                        dists[nr][nc] = new_cost
                        heappush(pq, (new_cost, nr, nc))

        return math.inf

    def _tt_key(self, board, to_move_colour):
        if to_move_colour == Colour.RED:
            tm = 1
        else:
            tm = 2

        flat = []
        for row in board.tiles:
            for tile in row:
                if tile.colour is None:
                    flat.append(0)
                elif tile.colour == Colour.RED:
                    flat.append(1)
                else:
                    flat.append(2)

        return (tm, board.size, tuple(flat))