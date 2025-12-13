import random
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Game import logger
import time
import math
from dataclasses import dataclass


class Node:
    """MCTS Tree Node"""
    def __init__(self, move=None, parent=None, player: Colour = None):
        self.move = move              # Move that led to this node
        self.parent = parent          # Parent node
        self.children = {}            # Map move -> child Node
        self.visits = 0               # Number of times selected
        self.wins = 0                 # wins for the player who made the move that created this node
        self.untried_moves = []       # Moves that have not been expanded
        self.rave_visits = 0          # RAVE visits (AMAF)
        self.rave_wins = 0            # RAVE wins (AMAF)
        self.player = player          # Player who made the move that led to this node (None for root)

    def uct_score(self, c: float = 1.4):
        """Calculate the UCT score for this node. NOT CURRENTLY USED."""
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration = c * ((2 * math.log(self.parent.visits) / self.visits) ** 0.5)
        return exploitation + exploration

    def rave_score(self):
        """Get the RAVE score for this node."""
        if self.rave_visits == 0:
            return 0.5
        return self.rave_wins / self.rave_visits

    def blended_score(self, c=1.4, k=300):
        """Blended UCT + RAVE score."""
        if self.visits == 0:
            return float('inf')

        q = self.wins / self.visits
        q_rave = self.rave_score()
        beta = k / (self.visits + k)

        exploitation = beta * q_rave + (1 - beta) * q
        exploration = c * ((2 * math.log(self.parent.visits) / self.visits) ** 0.5)

        return exploitation + exploration

    def best_child(self, mode='robust'):
        """Select best child."""
        children_values = list(self.children.values())
        if not children_values:
            raise ValueError("No children to select from in best_child()")

        if mode == 'max':
            return max(children_values, key=lambda n: n.wins / n.visits if n.visits > 0 else 0)
        if mode == 'robust':
            return max(children_values, key=lambda n: n.visits)
        if mode == 'max-robust':
            max_wins = max(n.wins for n in children_values)
            max_visits = max(n.visits for n in children_values)
            candidates = [n for n in children_values if n.wins == max_wins and n.visits == max_visits]
            if candidates:
                return candidates[0]
            raise ValueError("No maximal child found; consider running MCTS longer.")
        if mode == 'secure':
            pass


# ----------------------------
# Virtual Connection Engine (fast bitmask-based, MoHex-inspired)
# ----------------------------

@dataclass(slots=True)
class _VCResult:
    mustplay_mask: int            # bitmask of empty cells that must be played to stop opponent threats (0 => none found)
    opp_has_vc: bool              # opponent has a safe virtual connection between its sides
    self_has_vc: bool             # current player has a safe virtual connection between its sides


class _VirtualConnectionEngine:
    """Lightweight VC/VSC engine optimized for 11x11 using bitmasks (Python int)."""

    __slots__ = (
        "N", "SIZE", "NEI", "CELL_MASKS",
        "SIDE_MASKS", "SIDE_FRONTIER",
    )

    def __init__(self, size: int):
        self.SIZE = size
        self.N = size * size

        self.CELL_MASKS = [1 << i for i in range(self.N)]

        # Neighbor mask for each cell
        self.NEI = [0] * self.N
        for x in range(size):
            for y in range(size):
                i = x * size + y
                m = 0
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        m |= 1 << (nx * size + ny)
                self.NEI[i] = m

        # Side ids: left=0 right=1 top=2 bottom=3
        left, right, top, bot = 0, 1, 2, 3

        self.SIDE_MASKS = [0, 0, 0, 0]
        for x in range(size):
            self.SIDE_MASKS[left] |= 1 << (x * size + 0)
            self.SIDE_MASKS[right] |= 1 << (x * size + (size - 1))
        for y in range(size):
            self.SIDE_MASKS[top] |= 1 << (0 * size + y)
            self.SIDE_MASKS[bot] |= 1 << ((size - 1) * size + y)

        # Playing on border connects to that side
        self.SIDE_FRONTIER = self.SIDE_MASKS[:]

    def _flood_component(self, remaining: int) -> tuple[int, int]:
        """Extract one connected component from remaining stones mask."""
        seed_bit = remaining & -remaining
        comp = 0
        frontier = seed_bit
        remaining ^= seed_bit
        while frontier:
            b = frontier & -frontier
            frontier ^= b
            comp |= b
            idx = b.bit_length() - 1
            nb = self.NEI[idx] & remaining
            if nb:
                frontier |= nb
                remaining ^= nb
        return comp, remaining

    def _component_frontier(self, comp: int, empty: int) -> int:
        """Empty cells adjacent to component."""
        frontier = 0
        c = comp
        while c:
            b = c & -c
            c ^= b
            idx = b.bit_length() - 1
            frontier |= self.NEI[idx]
        return frontier & empty

    def _side_ids(self, colour) -> tuple[int, int]:
        # Game rule: RED connects top-bottom, BLUE connects left-right
        if str(colour).endswith("RED"):
            return 2, 3  # top, bottom
        else:
            return 0, 1  # left, right

    def analyze(self, player_mask: int, opp_mask: int, empty_mask: int, to_play_colour) -> _VCResult:
        # Opponent threats (mustplay)
        opp_mustplay, opp_has_vc = self._side_threats(opp_mask, empty_mask, to_play_colour)

        # Current player's safe VC (optional)
        self_has_vc = self._has_side_vc(player_mask, empty_mask, to_play_colour)

        return _VCResult(
            mustplay_mask=opp_mustplay,
            opp_has_vc=opp_has_vc,
            self_has_vc=self_has_vc,
        )

    def _has_side_vc(self, stones: int, empty: int, colour) -> bool:
        side0, side1 = self._side_ids(colour)
        vc, _vsc = self._compute_vc_graph(stones, empty, side0, side1)
        return vc[0][1] != 0

    def _side_threats(self, stones: int, empty: int, to_play_colour) -> tuple[int, bool]:
        # Threats for opponent of to_play_colour.
        # Rule: RED connects top-bottom (2,3), BLUE connects left-right (0,1).
        if str(to_play_colour).endswith("RED"):
            # opponent is BLUE
            side0, side1 = 0, 1  # left, right
        else:
            # opponent is RED
            side0, side1 = 2, 3  # top, bottom

        vc, vsc = self._compute_vc_graph(stones, empty, side0, side1)
        opp_has_vc = vc[0][1] != 0

        # Mustplay: union of carriers of best few VSCs between sides
        mustplay = 0
        for c in vsc[0][1]:
            mustplay |= c

        return mustplay, opp_has_vc

    def _compute_vc_graph(self, stones: int, empty: int, side0: int, side1: int):
        """
        Endpoints indexing:
          0 -> side0
          1 -> side1
          2.. -> player chains
        vc[i][j] = carrier mask (nonzero means exists; 1 is sentinel for empty carrier)
        vsc[i][j] = list of carrier masks (each nonzero)
        """
        # Build chain endpoints
        components = []
        remaining = stones
        while remaining:
            comp, remaining = self._flood_component(remaining)
            components.append(comp)

        m = 2 + len(components)

        frontier = [0] * m
        frontier[0] = self.SIDE_FRONTIER[side0] & empty
        frontier[1] = self.SIDE_FRONTIER[side1] & empty

        for idx, comp in enumerate(components, start=2):
            frontier[idx] = self._component_frontier(comp, empty)

        vc = [[0] * m for _ in range(m)]
        vsc = [[[] for _ in range(m)] for __ in range(m)]

        def add_vsc(i, j, carr):
            if i == j:
                return
            if carr == 0:
                return
            lst = vsc[i][j]
            for ex in lst:
                if ex == carr:
                    return
            lst.append(carr)
            lst.sort(key=lambda x: x.bit_count())
            if len(lst) > 6:
                del lst[6:]

            lst2 = vsc[j][i]
            for ex in lst2:
                if ex == carr:
                    return
            lst2.append(carr)
            lst2.sort(key=lambda x: x.bit_count())
            if len(lst2) > 6:
                del lst2[6:]

        def add_vc(i, j, carr):
            if i == j:
                return
            if carr == 0:
                carr = 1  # sentinel for empty carrier
            cur = vc[i][j]
            if cur == 0 or carr.bit_count() < cur.bit_count():
                vc[i][j] = carr
                vc[j][i] = carr

        # Base VC: chain touches side (stone on border)
        side0_mask = self.SIDE_MASKS[side0]
        side1_mask = self.SIDE_MASKS[side1]
        for k, comp in enumerate(components, start=2):
            if comp & side0_mask:
                add_vc(0, k, 1)
            if comp & side1_mask:
                add_vc(1, k, 1)

        # ----------------------------
        # Base VSC atoms
        # ----------------------------

        # (A) Pivot / 1-move connects:
        # If endpoints share ANY frontier empty, playing that cell connects them -> VSC with carrier {cell}.
        for i in range(m):
            fi = frontier[i]
            if fi == 0:
                continue
            for j in range(i + 1, m):
                inter1 = fi & frontier[j]
                if inter1:
                    tmp = inter1
                    added = 0
                    while tmp and added < 8:
                        b = tmp & -tmp
                        tmp ^= b
                        add_vsc(i, j, b)  # singleton carrier
                        added += 1

        # (B) Bridge-like / 2-move connects (keep from your original)
        for i in range(m):
            fi = frontier[i]
            if fi == 0:
                continue
            for j in range(i + 1, m):
                inter = fi & frontier[j]
                if inter.bit_count() >= 2:
                    bits = []
                    tmp = inter
                    while tmp and len(bits) < 8:
                        b = tmp & -tmp
                        tmp ^= b
                        bits.append(b)
                    pairs = []
                    for a in range(len(bits)):
                        for b in range(a + 1, len(bits)):
                            pairs.append(bits[a] | bits[b])
                    # all size 2 anyway; keep a few
                    added = 0
                    for carr in pairs:
                        add_vsc(i, j, carr)
                        added += 1
                        if added >= 6:
                            break

        # ----------------------------
        # Closure: AND-rule and OR-rule
        # ----------------------------
        changed = True
        iters = 0
        while changed and iters < 6:
            iters += 1
            changed = False

            # AND-rule: combine through k
            for k in range(m):
                for i in range(m):
                    if i == k:
                        continue
                    vc_ik = vc[i][k]
                    vsc_ik_list = vsc[i][k]
                    if vc_ik == 0 and not vsc_ik_list:
                        continue

                    for j in range(m):
                        if j == k or j == i:
                            continue
                        vc_kj = vc[k][j]
                        vsc_kj_list = vsc[k][j]
                        if vc_kj == 0 and not vsc_kj_list:
                            continue

                        # VC + VC => VC
                        if vc_ik and vc_kj:
                            before = vc[i][j]
                            add_vc(i, j, (vc_ik | vc_kj))
                            if before != vc[i][j]:
                                changed = True

                        # VC + VSC => VSC
                        if vc_ik and vsc_kj_list:
                            before_len = len(vsc[i][j])
                            for carr2 in vsc_kj_list:
                                add_vsc(i, j, (vc_ik | carr2))
                            if len(vsc[i][j]) != before_len:
                                changed = True

                        # VSC + VC => VSC
                        if vsc_ik_list and vc_kj:
                            before_len = len(vsc[i][j])
                            for carr1 in vsc_ik_list:
                                add_vsc(i, j, (carr1 | vc_kj))
                            if len(vsc[i][j]) != before_len:
                                changed = True

            # OR-rule: if two disjoint VSCs exist => VC
            for i in range(m):
                for j in range(i + 1, m):
                    lst = vsc[i][j]
                    if len(lst) < 2:
                        continue
                    best = vc[i][j]
                    L = min(len(lst), 6)
                    for a in range(L):
                        ca = lst[a]
                        for b in range(a + 1, L):
                            cb = lst[b]
                            if (ca & cb) == 0:
                                carr = ca | cb
                                if best == 0 or carr.bit_count() < best.bit_count():
                                    add_vc(i, j, carr)
                                    changed = True
                                    best = vc[i][j]
                                if best and best.bit_count() <= 4:
                                    break
                        else:
                            continue
                        break

        return vc, vsc


class MCTSAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.board_size = 11
        self.time_limit = 3  # seconds per move
        self.swap_decided = False
        self.swap_choice = False

        self._vc_engine = _VirtualConnectionEngine(self.board_size)
        self._vc_cache = {}
        self._use_vc_pruning = True

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        self._vc_cache.clear()

        # --- pie rule ---
        if turn == 2 and opp_move is not None and not self.swap_decided:
            self.swap_decided = True
            self.swap_choice = self.decide_swap(board, opp_move)
            if self.swap_choice:
                self.colour = Colour.RED if self.colour == Colour.BLUE else Colour.BLUE
                return Move(-1, -1)

        # --- legal moves ---
        legal_moves = [
            (tile.x, tile.y)
            for row in board.tiles
            for tile in row
            if tile.colour is None
        ]

        if not legal_moves:
            return Move(0, 0)

        # --- immediate win if available ---
        win_now = self._find_immediate_winning_move(board, legal_moves, self.colour)
        if win_now is not None:
            x, y = win_now
            return Move(x, y)

        # --- emergency: block opponent immediate win ---
        block = self._find_immediate_winning_move(board, legal_moves, Colour.opposite(self.colour))
        if block is not None:
            x, y = block
            return Move(x, y)

        # --- run MCTS ---
        x, y = self.run_mcts(board, legal_moves)
        return Move(x, y)

    def decide_swap(self, board: Board, first_move: Move):
        center = self.board_size // 2
        bias = (abs(center - first_move.x) + abs(center - first_move.y)) / (2 * center)

        if (first_move.x, first_move.y) in [(0, 0), (0, board.size - 1), (board.size - 1, 0), (board.size - 1, board.size - 1)]:
            return False

        opponent = Colour.BLUE if self.colour == Colour.RED else Colour.RED
        wins = 0
        samples = 30
        for _ in range(samples):
            sim_board = self.clone_board(board)
            sim_board.set_tile_colour(first_move.x, first_move.y, opponent)
            winner = self.play_random_game(sim_board, opponent)
            if winner == opponent:
                wins += 1

        win_rate = wins / samples
        return (win_rate > 0.55 and bias < 0.65)

    def play_random_game(self, board: Board, current_player: Colour) -> Colour:
        opponent = Colour.BLUE if current_player == Colour.RED else Colour.RED
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
        root.untried_moves = self._root_prune_moves(board, legal_moves.copy(), self.colour)
        random.shuffle(root.untried_moves)

        start_time = time.time()
        while time.time() - start_time < self.time_limit:
            node = root
            state = self.clone_board(board)
            current_colour = self.colour

            # 1) Selection
            while node.untried_moves == [] and node.children:
                node = self.uct_select(node)
                state = self.apply_move(state, node.move, current_colour)
                current_colour = Colour.opposite(current_colour)

            # 2) Expansion
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                child = Node(move=move, parent=node, player=current_colour)
                node.children[move] = child
                node = child
                state = self.apply_move(state, move, current_colour)
                current_colour = Colour.opposite(current_colour)

            # 3) Simulation
            winner, played_moves = self.simulate(state, current_colour)

            # 4) Backprop
            self.backpropagate(node, winner, played_moves)

        best = root.best_child()
        return best.move

    def uct_select(self, node: Node) -> Node:
        return max(node.children.values(), key=lambda child: child.blended_score())

    def clone_board(self, board: Board) -> Board:
        board_copy = Board(board.size)
        tiles_original = board.tiles
        tiles_copy = board_copy.tiles
        for i in range(board.size):
            row_original = tiles_original[i]
            row_copy = tiles_copy[i]
            for j in range(board.size):
                row_copy[j].colour = row_original[j].colour
        return board_copy

    def apply_move(self, board: Board, move: tuple[int, int], colour: Colour) -> Board:
        x, y = move
        board.tiles[x][y].colour = colour
        return board

    # ----------------------------
    # Immediate win / block helpers (fast, no cloning)
    # ----------------------------
    def _find_immediate_winning_move(self, board: Board, legal_moves: list[tuple[int, int]], who: Colour) -> tuple[int, int] | None:
        # Try each legal move by setting/unsetting the tile in-place.
        # This is much faster than cloning 121 boards.
        for (x, y) in legal_moves:
            tile = board.tiles[x][y]
            if tile.colour is not None:
                continue
            tile.colour = who
            try:
                if board.has_ended(who):
                    return (x, y)
            finally:
                tile.colour = None
        return None

    # ----------------------------
    # Virtual connection helpers
    # ----------------------------
    def _board_bitmasks(self, board: Board) -> tuple[int, int, int]:
        red = 0
        blue = 0
        size = self.board_size
        for x in range(size):
            row = board.tiles[x]
            base = x * size
            for y in range(size):
                c = row[y].colour
                if c is None:
                    continue
                bit = 1 << (base + y)
                if str(c).endswith('RED'):
                    red |= bit
                else:
                    blue |= bit
        empty = ((1 << (size * size)) - 1) & ~(red | blue)
        return red, blue, empty

    def _vc_analyze_cached(self, board: Board, to_play: Colour) -> _VCResult:
        red, blue, empty = self._board_bitmasks(board)
        key = (red, blue, str(to_play))
        res = self._vc_cache.get(key)
        if res is not None:
            return res

        if str(to_play).endswith('RED'):
            player_mask, opp_mask = red, blue
        else:
            player_mask, opp_mask = blue, red

        res = self._vc_engine.analyze(player_mask, opp_mask, empty, to_play)
        self._vc_cache[key] = res
        return res

    def _filter_moves_by_mask(self, moves: list[tuple[int, int]], mask: int) -> list[tuple[int, int]]:
        if mask == 0:
            return moves
        size = self.board_size
        out = []
        for (x, y) in moves:
            if mask & (1 << (x * size + y)):
                out.append((x, y))
        return out if out else moves  # never return empty

    def _root_prune_moves(self, board: Board, legal_moves: list[tuple[int, int]], to_play: Colour) -> list[tuple[int, int]]:
        if not self._use_vc_pruning:
            return legal_moves
        vcres = self._vc_analyze_cached(board, to_play)
        if vcres.mustplay_mask:
            return self._filter_moves_by_mask(legal_moves, vcres.mustplay_mask)
        return legal_moves

    # ----------------------------
    # Rollout heuristics
    # ----------------------------
    def get_biased_moves(self, board: Board, colour: Colour):
        moves = []
        for row in board.tiles:
            for t in row:
                if t.colour is None:
                    score = 0
                    for nx, ny in self.get_neighbors(t.x, t.y):
                        if board.tiles[nx][ny].colour == colour:
                            score += 2
                        elif board.tiles[nx][ny].colour == Colour.opposite(colour):
                            score += 1
                    moves.append(((t.x, t.y), score))

        moves.sort(key=lambda x: -x[1])
        return [m for m, _ in moves[:10]]

    def get_neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                neighbors.append((nx, ny))
        return neighbors

    def simulate(self, board: Board, colour: Colour) -> tuple[Colour, set]:
        state = board
        played_moves = set()

        # One-shot VC guidance: if opponent has a threat carrier, bias early rollout steps toward blocking it.
        vcres = self._vc_analyze_cached(state, colour)
        mustplay_mask = vcres.mustplay_mask

        while True:
            legal_moves = [
                (tile.x, tile.y)
                for row in state.tiles
                for tile in row
                if tile.colour is None
            ]
            if not legal_moves:
                return None, played_moves

            # bias (not hard-restrict forever)
            if mustplay_mask and random.random() < 0.7:
                legal_moves = self._filter_moves_by_mask(legal_moves, mustplay_mask)

            if random.random() < 0.8:
                candidate_moves = self.get_biased_moves(state, colour)
                move = random.choice(candidate_moves) if candidate_moves else random.choice(legal_moves)
            else:
                move = random.choice(legal_moves)

            played_moves.add(move)
            state = self.apply_move(state, move, colour)

            if state.has_ended(Colour.RED):
                return Colour.RED, played_moves
            if state.has_ended(Colour.BLUE):
                return Colour.BLUE, played_moves

            colour = Colour.opposite(colour)

    def backpropagate(self, node: Node, winner: Colour, played_moves: set):
        cur = node
        while cur is not None:
            cur.visits += 1
            if winner is not None and cur.player is not None and winner == cur.player:
                cur.wins += 1

            parent = cur.parent
            if parent is not None:
                for mv in played_moves:
                    child = parent.children.get(mv)
                    if child:
                        child.rave_visits += 1
                        if winner is not None and child.player is not None and winner == child.player:
                            child.rave_wins += 1

            cur = parent
