import unittest
from unittest.mock import MagicMock
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from agents.MinimaxAgent.MinimaxAgent import AlphaBetaAgent
from agents.MCTSAgent.MCTSAgent import MCTSAgent

class TestSaveBridge(unittest.TestCase):
    
    def test_minimax_savebridge(self):
        print("Testing Minimax Savebridge...")
        # Setup: Red (Agent) has stones at (5,5) and (6,4).
        # These form a bridge. The carriers are (6,5) and (5,4).
        # Opponent (Blue) plays at (6,5).
        # Expected: Agent MUST play (5,4).
        
        board = Board(11)
        agent = AlphaBetaAgent(Colour.RED)
        
        # Set up the bridge
        board.set_tile_colour(5, 5, Colour.RED)
        board.set_tile_colour(6, 4, Colour.RED)
        
        # Opponent attacks one carrier
        opp_move = Move(6, 5)
        board.set_tile_colour(6, 5, Colour.BLUE)
        
        # Ask agent for move
        move = agent.make_move(10, board, opp_move)
        
        print(f"Opponent attacked at (6,5). Agent responded at: ({move.x}, {move.y})")
        self.assertEqual(move.x, 5)
        self.assertEqual(move.y, 4)
        print("Minimax Savebridge PASSED.")

    def test_mcts_savebridge(self):
        print("\nTesting MCTS Savebridge (Python Wrapper)...")
        # Same setup.
        
        board = Board(11)
        
        # Mock subprocess BEFORE creating the agent to avoid binary requirement
        from unittest.mock import patch
        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_popen.return_value = mock_process
            
            agent = MCTSAgent(Colour.RED)
            agent.agent_process = mock_process
            
            # Set up the bridge
            board.set_tile_colour(5, 5, Colour.RED)
            board.set_tile_colour(6, 4, Colour.RED)
            
            # Opponent attacks one carrier
            opp_move = Move(6, 5)
            board.set_tile_colour(6, 5, Colour.BLUE)
            
            # Ask agent for move
            move = agent.make_move(10, board, opp_move)
            
            print(f"Opponent attacked at (6,5). Agent responded at: ({move.x}, {move.y})")
            
            # If logic works, it returns (5,4) BEFORE trying to write to the (mocked) subprocess
            self.assertEqual(move.x, 5)
            self.assertEqual(move.y, 4)
            print("MCTS Savebridge PASSED.")

if __name__ == '__main__':
    unittest.main()
