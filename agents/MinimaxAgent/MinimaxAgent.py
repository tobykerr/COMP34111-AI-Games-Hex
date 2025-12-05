import math

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
        # to do 
        # need to handle swap rule for very first move



        # choose (x, y)
        x, y = self._choose_with_alpha_beta(board, turn)
        return Move(x, y)  
    


    # ------- mini max alpha beta core

    def _choose_with_alpha_beta(self, board, turn):
        best_val = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf

        legal_moves = self._generate_legal_moves(board)

        for move in legal_moves:
            new_board = board.copy()
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
            new_board = board.copy()
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
            new_board = board.copy()
            self._apply_move(new_board, move, opp_colour)
            value = min(value, self._max_value(new_board, depth-1, alpha, beta, turn+1))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def _terminal_or_cutoff(self, board, depth, turn):
        if depth <= 0:
            return True 
        
        # to do 
        # game over check (like board.winner or something like that)
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
        # to do
        pass

    def _evaluate(self, board):
        # to do 


        # ---------idea------------
        # maybe use dijsktra to find shortest path
        # shortest path for the current color. shortest path is more valuable
        # evaluate both the colors's paths
        # find the difference and try minimize / maximize
        pass