import math
import copy
import random
from heapq import heappush, heappop

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class AlphaBetaAgent(AgentBase):
    def __init__(self, colour, board_size=11):
        super().__init__(colour)
        self.board_size = board_size
        self.max_depth = 2  # this can be changed

        # --- Transposition Table (TT)
        self.tt = {}  # key -> (stored_depth, flag, value)
        self.TT_EXACT = 0
        self.TT_LOWER = 1
        self.TT_UPPER = 2

    def make_move(self, turn, board, opp_move):
        if turn == 1:
            return Move(1, 2)

        if turn == 2 and opp_move and self.decide_swap(opp_move):
            return Move(-1, -1)

        # --- MoHex "savebridge" pattern (exact behavior: if opponent threatens >1 bridge, save one at random)
        # If the opponent just played a bridge carrier that attacks one of our bridges,
        # immediately reply with the other carrier to save the bridge.
        if opp_move and not (opp_move.x == -1 and opp_move.y == -1):
            sb = self._savebridge_reply(board, opp_move)
            if sb is not None:
                return Move(sb[0], sb[1])

        move = self._choose_with_alpha_beta(board, turn)
        return Move(move[0], move[1])

    def decide_swap(self, first_move: Move):
        no_swaps = [(0, i) for i in range(10)] + [(10, j) for j in range(1, 11)] + [(1, i) for i in range(9)] + [
            (10, j) for j in range(1, 11)
        ]
        if (first_move.x, first_move.y) in no_swaps:
            return False
        else:
            return True

    # -------------------- MoHex savebridge --------------------

    def _neighbors_of(self, r, c, size):
        # Hex Grid displacements: (-1,0), (-1,1), (0,1), (1,0), (1,-1), (0,-1)
        neigh = [
            (r - 1, c),
            (r - 1, c + 1),
            (r, c + 1),
            (r + 1, c),
            (r + 1, c - 1),
            (r, c - 1),
        ]
        return [(nr, nc) for (nr, nc) in neigh if 0 <= nr < size and 0 <= nc < size]

    def _savebridge_reply(self, board, opp_move: Move):
        """
        Implements MoHex's savebridge simulation pattern:

        - A "bridge" is a virtual connection between two of our stones whose two shared neighbors are empty.
        - If the opponent plays on one bridge carrier (one of the two shared neighbors),
          we deterministically reply by playing the other carrier to save the bridge.
        - If the opponent move threatens more than one bridge, we randomly save one bridge.

        Returns:
          (row, col) to play, or None if no savebridge reply exists.
        """
        size = board.size
        b1 = (opp_move.x, opp_move.y)

        # If opponent move is out of bounds or not actually on the board, ignore.
        if not (0 <= b1[0] < size and 0 <= b1[1] < size):
            return None

        # If b1 is not occupied by opponent (should be, but be robust), ignore.
        opp = Colour.RED if self.colour == Colour.BLUE else Colour.BLUE
        if board.tiles[b1[0]][b1[1]].colour != opp:
            return None

        # Any bridge carrier b1 is adjacent to both bridge endpoints.
        # So we only need to check pairs among neighbors(b1).
        nbs = self._neighbors_of(b1[0], b1[1], size)

        save_moves = []

        # Consider all unordered pairs of neighboring cells of b1 as candidate endpoints.
        L = len(nbs)
        for i in range(L):
            u = nbs[i]
            if board.tiles[u[0]][u[1]].colour != self.colour:
                continue
            u_neigh = set(self._neighbors_of(u[0], u[1], size))

            for j in range(i + 1, L):
                v = nbs[j]
                if board.tiles[v[0]][v[1]].colour != self.colour:
                    continue

                v_neigh = set(self._neighbors_of(v[0], v[1], size))

                # Bridge endpoints are at distance 2 and share exactly two common neighbors (the carriers).
                inter = u_neigh.intersection(v_neigh)
                if len(inter) != 2:
                    continue
                if b1 not in inter:
                    continue

                # The other carrier is the saving reply, if empty.
                b2 = next(iter(inter - {b1}))
                if board.tiles[b2[0]][b2[1]].colour is None:
                    save_moves.append(b2)

        if not save_moves:
            return None

        # MoHex: if more than one bridge is threatened by the opponent move, choose one at random.
        return random.choice(save_moves)

    # -------------------- helpers for in-place move apply/undo --------------------

    def _apply_move_inplace(self, board, move, colour):
        r, c = move
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
            (r - 1, c),
            (r - 1, c + 1),
            (r, c + 1),
            (r + 1, c),
            (r + 1, c - 1),
            (r, c - 1),
        ]

        for nr, nc in neighbors:
            if 0 <= nr < size and 0 <= nc < size:
                if board.tiles[nr][nc].colour == self.colour:
                    friendly += 1

        score = (center_dist) + (goal) - (2 * friendly)
        return score

    def _order_moves(self, board, moves):
        return sorted(moves, key=lambda m: self._move_score(board, m))

    def _choose_with_alpha_beta(self, board, turn):
        best_val = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf

        legal_moves = self._generate_legal_moves(board)
        legal_moves = self._order_moves(board, legal_moves)

        for move in legal_moves:
            self._apply_move_inplace(board, move, self.colour)
            val = self._min_value(board, depth=self.max_depth - 1, alpha=alpha, beta=beta, turn=turn + 1)
            self._undo_move_inplace(board, move)

            if val > best_val:
                best_val = val
                best_move = move
            alpha = max(alpha, best_val)

        if best_move:
            return best_move
        else:
            raise ValueError("No best move found in Alpha-Beta search.")

    def _max_value(self, board, depth, alpha, beta, turn):
        if self._terminal_or_cutoff(board, depth, turn):
            return self._evaluate(board)

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
                else:
                    beta = min(beta, val)
                if alpha >= beta:
                    return val

        value = -math.inf
        legal_moves = self._generate_legal_moves(board)
        legal_moves = self._order_moves(board, legal_moves)

        for move in legal_moves:
            self._apply_move_inplace(board, move, self.colour)
            value = max(value, self._min_value(board, depth - 1, alpha, beta, turn + 1))
            self._undo_move_inplace(board, move)

            if value >= beta:
                self.tt[key] = (depth, self.TT_LOWER, value)
                return value
            alpha = max(alpha, value)

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
                else:
                    beta = min(beta, val)
                if alpha >= beta:
                    return val

        value = math.inf
        legal_moves = self._generate_legal_moves(board)
        legal_moves = self._order_moves(board, legal_moves)

        for move in legal_moves:
            self._apply_move_inplace(board, move, opp_colour)
            value = min(value, self._max_value(board, depth - 1, alpha, beta, turn + 1))
            self._undo_move_inplace(board, move)

            if value <= alpha:
                self.tt[key] = (depth, self.TT_UPPER, value)
                return value
            beta = min(beta, value)

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
                    moves.append((y, x))
        return moves

    def _apply_move(self, board, move, colour):
        r, c = move
        board.set_tile_colour(r, c, colour)

    def _evaluate(self, board):
        my_path_len = self.dijkstra_shortest_path(board, self.colour)
        opp_path_len = self.dijkstra_shortest_path(board, self.opp_colour())

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
            # Base 10; adjust; clamp.
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
