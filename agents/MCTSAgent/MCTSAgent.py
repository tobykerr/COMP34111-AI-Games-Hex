import random
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Game import logger


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
        # --- handle pie rule ---
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
            return Move(0, 0)

    def run_mcts(self, board: Board, legal_moves: list[tuple[int, int]]) -> tuple[int, int]:
        """Minimal placeholder for MCTS logic.
        Currently just picks a random legal move.
        Replace this with the full MCTS algorithm later.
        """
        return random.choice(legal_moves)
