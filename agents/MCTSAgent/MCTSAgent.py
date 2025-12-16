import subprocess

from src.Colour import Colour
from src.AgentBase import AgentBase
from src.Move import Move
from src.Board import Board
from src.Game import logger
from src.Tile import Tile

class MCTSAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)

        self.agent_process = subprocess.Popen(
            ["./agents/MCTSAgent/mcts-hex"],
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """The game engine will call this method to request a move from the agent.
        If the agent is to make the first move, opp_move will be None.
        If the opponent has made a move, opp_move will contain the opponent's move.
        If the opponent has made a swap move, opp_move will contain a Move object with x=-1 and y=-1,
        the game engine will also change your colour to the opponent colour.

        Args:
            turn (int): The current turn
            board (Board): The current board state
            opp_move (Move | None): The opponent's last move

        Returns:
            Move: The agent's move
        """
        
        # ---------------------------------------------------------
        # SAVEBRIDGE IMPLEMENTATION
        # Check if we need to save a bridge before calling the MCTS binary
        # ---------------------------------------------------------
        save_move = self.check_savebridge(board, opp_move)
        if save_move:
            # We found a forced move. Return it immediately.
            # (Ideally, we should inform the binary of this move if it maintains internal state,
            # but since we send the full board string every turn in 'CHANGE'/'START', 
            # the binary will sync up next turn).
            return save_move

        # ---------------------------------------------------------
        # EXISTING MCTS LOGIC
        # ---------------------------------------------------------

        rows = board.tiles
        board_strings = []
        for row in rows:
            row_string = ""
            for tile in row:
                colour = tile.colour
                if colour is None:
                    t = "0"
                else:
                    t = colour.get_char()
                row_string += t
            board_strings.append(row_string)
        board_string = ",".join(board_strings)

        if opp_move is None:
            command = f"START;;{board_string};{turn};"
        elif opp_move.x == -1 and opp_move.y == -1:
            command = f"SWAP;;{board_string};{turn};"
        else:
            command = f"CHANGE;{opp_move.x},{opp_move.y};{board_string};{turn};"

        self.agent_process.stdin.write(command + "\n")
        self.agent_process.stdin.flush()

        response = self.agent_process.stdout.readline().rstrip()
        # assuming the response takes the form "x,y" with -1,-1 if the agent wants to make a swap move
        x, y = map(int, response.split(","))
        return Move(x, y)

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
