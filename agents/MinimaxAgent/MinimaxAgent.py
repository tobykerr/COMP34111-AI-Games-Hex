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

    def make_move(self, turn, board, opp_move):
        # Swap logic for Turn 2
        # Move.x is Row, Move.y is Col (Based on Game.py)
        if turn == 1:
            return self.decide_first()
        
        if turn == 2 and self.decide_swap(opp_move):
             # Simple check: If opponent played in the center 5x5 box, swap.
            self.colour = Colour.RED if self.colour == Colour.BLUE else Colour.BLUE
            return Move(-1, -1)
             '''
             if opp_move.x >= 3 and opp_move.x <= 7 and opp_move.y >= 3 and opp_move.y <= 7:
                  self.colour = Colour.RED if self.colour == Colour.BLUE else Colour.BLUE
                  return Move(-1, -1)
                '''
            
        # Standard Alpha-Beta Search
        move = self._choose_with_alpha_beta(board, turn)
        # move is (Row, Col)
        return Move(move[0], move[1])  
    
    
    def decide_first(self) -> Move:
        x,y = 0
        top = 0
        for item in [(0,1),(1,1),(2,1),(0,2),(3,1),(0,3),(1,3),(0,5),(0,6),(0,7),(0,10)]:
            if winrates[item[0]][item[1]] > top:
                top = winrates[item[0]][item[1]]
                x = item[0]
                y = item[1]
        return Move(x,y)
    
    def decide_swap(self, first_move: Move):
        if (first_move.x, first_move.y) in [(0,0),'''(0,board.size-1), (board.size-1,0),'''(board.size-1,board.size-1)]:
            return False
        elif first_move.x >= 3 and first_move.x <= 7 and first_move.y >= 3 and first_move.y <= 7:
            return True
            
        winrates = []
        
        f = open("test3")
        for line in f:
            line = line.strip().split(",")
            line = [float(i) for i in line]
            winrates.append(line)
        f.close()
        
        if winrates[first_move.x][first_move.y] < 50:
            return True
        else:
            return False

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
        
        # Simple heuristic: Opponent path length - My path length
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