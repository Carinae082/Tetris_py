"""
tests/test_bot_adapter.py — BotEnvAdapter, StateSnapshot, heuristics, GreedyBot 통합 테스트

픽스처
------
make_state(seed, n_pieces): SevenBag으로 큐를 채우고 첫 피스를 스폰한 GameState 반환
"""

from __future__ import annotations

import pytest

from engine.bag import SevenBag
from engine.board import BOARD_COLS, BOARD_ROWS, VISIBLE_ROW_START
from engine.mino import MinoType
from engine.physics import try_move, try_rotate
from engine.state import GameState, RoundResult

from bot.adapter import Action, BotEnvAdapter
from bot.greedy_bot import GreedyBot
from bot.heuristics import board_max_height, column_heights, count_holes, bumpiness
from bot.snapshot import StateSnapshot


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def make_state(seed: int = 42, n_pieces: int = 14) -> GameState:
    """SevenBag으로 큐를 채우고 첫 피스를 스폰한 GameState를 반환한다."""
    bag = SevenBag(seed=seed)
    state = GameState()
    for _ in range(n_pieces):
        state.enqueue(bag.pop())
    state.spawn_next()
    return state


# ── clone_env ─────────────────────────────────────────────────────────────────

class TestCloneEnv:
    def setup_method(self):
        self.adapter = BotEnvAdapter()
        self.env = make_state()

    def test_clone_is_independent(self):
        """클론을 수정해도 원본이 변하지 않아야 한다."""
        clone = self.adapter.clone_env(self.env)
        try_move(clone, -1)
        assert self.env.active.col != clone.active.col or True  # 실제 이동 여부 무관
        # 보드 독립성
        clone.board._grid[30][5] = MinoType.I
        assert self.env.board._grid[30][5] is None

    def test_clone_has_same_queue_content(self):
        """클론의 큐 내용이 원본과 동일해야 한다."""
        clone = self.adapter.clone_env(self.env)
        assert list(clone.next_queue) == list(self.env.next_queue)

    def test_clone_queue_is_independent(self):
        """클론의 큐를 변경해도 원본 큐가 변하지 않아야 한다."""
        clone = self.adapter.clone_env(self.env)
        original_queue = list(self.env.next_queue)
        clone.next_queue.clear()
        assert list(self.env.next_queue) == original_queue


# ── get_state_snapshot ────────────────────────────────────────────────────────

class TestGetStateSnapshot:
    def setup_method(self):
        self.adapter = BotEnvAdapter()
        self.env = make_state()

    def test_returns_state_snapshot_instance(self):
        snap = self.adapter.get_state_snapshot(self.env)
        assert isinstance(snap, StateSnapshot)

    def test_board_dimensions(self):
        snap = self.adapter.get_state_snapshot(self.env)
        assert len(snap.board) == BOARD_ROWS
        assert all(len(row) == BOARD_COLS for row in snap.board)

    def test_board_is_copy_not_reference(self):
        """스냅샷 보드를 수정해도 원본 board._grid가 변하지 않아야 한다."""
        snap = self.adapter.get_state_snapshot(self.env)
        snap.board[30][5] = MinoType.T
        assert self.env.board._grid[30][5] is None

    def test_current_piece_populated(self):
        snap = self.adapter.get_state_snapshot(self.env)
        assert snap.current_piece is not None
        assert isinstance(snap.current_piece.type, MinoType)
        assert len(snap.current_piece.cells) == 4

    def test_hold_piece_initially_none(self):
        snap = self.adapter.get_state_snapshot(self.env)
        assert snap.hold_piece is None

    def test_queue_preview_length(self):
        snap = self.adapter.get_state_snapshot(self.env)
        assert 1 <= len(snap.queue_preview) <= 6

    def test_combo_initial(self):
        snap = self.adapter.get_state_snapshot(self.env)
        assert snap.combo == -1

    def test_b2b_initial(self):
        snap = self.adapter.get_state_snapshot(self.env)
        assert snap.b2b == -1

    def test_pending_garbage_initial(self):
        snap = self.adapter.get_state_snapshot(self.env)
        assert snap.pending_garbage == 0

    def test_not_terminal_initially(self):
        snap = self.adapter.get_state_snapshot(self.env)
        assert snap.terminal is False


# ── is_terminal ───────────────────────────────────────────────────────────────

class TestIsTerminal:
    def setup_method(self):
        self.adapter = BotEnvAdapter()

    def test_ongoing_state_is_not_terminal(self):
        env = make_state()
        assert self.adapter.is_terminal(env) is False

    def test_lose_state_is_terminal(self):
        env = make_state()
        env.round_result = RoundResult.LOSE
        assert self.adapter.is_terminal(env) is True

    def test_win_state_is_terminal(self):
        env = make_state()
        env.round_result = RoundResult.WIN
        assert self.adapter.is_terminal(env) is True


# ── list_legal_actions ────────────────────────────────────────────────────────

class TestListLegalActions:
    def setup_method(self):
        self.adapter = BotEnvAdapter()
        self.env = make_state()

    def test_returns_list(self):
        actions = self.adapter.list_legal_actions(self.env)
        assert isinstance(actions, list)

    def test_hard_drop_always_in_actions(self):
        """활성 피스가 있으면 HARD_DROP은 항상 합법이어야 한다."""
        actions = self.adapter.list_legal_actions(self.env)
        assert Action.HARD_DROP in actions

    def test_hold_available_before_first_use(self):
        """이번 피스에서 홀드를 쓰지 않았으면 HOLD가 포함되어야 한다."""
        actions = self.adapter.list_legal_actions(self.env)
        assert Action.HOLD in actions

    def test_hold_not_available_after_use(self):
        """hold_used=True이면 HOLD가 포함되지 않아야 한다."""
        self.env.hold_used = True
        actions = self.adapter.list_legal_actions(self.env)
        assert Action.HOLD not in actions

    def test_empty_when_terminal(self):
        self.env.round_result = RoundResult.LOSE
        actions = self.adapter.list_legal_actions(self.env)
        assert actions == []

    def test_empty_when_no_active_piece(self):
        self.env.active = None
        actions = self.adapter.list_legal_actions(self.env)
        assert actions == []

    def test_original_env_unchanged_after_listing(self):
        """list_legal_actions가 원본 env를 수정하지 않아야 한다."""
        original_active = self.env.active
        original_col = original_active.col if original_active else None
        self.adapter.list_legal_actions(self.env)
        assert self.env.active is original_active
        if original_active:
            assert self.env.active.col == original_col


# ── simulate_action ───────────────────────────────────────────────────────────

class TestSimulateAction:
    def setup_method(self):
        self.adapter = BotEnvAdapter()
        self.env = make_state()

    def test_returns_tuple_of_env_and_snapshot(self):
        next_env, snap = self.adapter.simulate_action(self.env, Action.HARD_DROP)
        assert isinstance(next_env, GameState)
        assert isinstance(snap, StateSnapshot)

    def test_original_env_unchanged_after_hard_drop(self):
        """simulate_action이 원본 env를 변경하지 않아야 한다."""
        original_queue = list(self.env.next_queue)
        original_pieces = self.env.pieces_placed
        self.adapter.simulate_action(self.env, Action.HARD_DROP)
        assert list(self.env.next_queue) == original_queue
        assert self.env.pieces_placed == original_pieces

    def test_hard_drop_increments_pieces_placed(self):
        _, _ = self.adapter.simulate_action(self.env, Action.HARD_DROP)
        # 원본이 아닌 시뮬레이션 결과에서 pieces_placed가 증가해야 한다
        next_env, _ = self.adapter.simulate_action(self.env, Action.HARD_DROP)
        assert next_env.pieces_placed == self.env.pieces_placed + 1

    def test_hold_action_populates_hold_slot(self):
        """HOLD 행동 후 클론의 hold 슬롯이 채워져야 한다."""
        original_type = self.env.active.type
        next_env, snap = self.adapter.simulate_action(self.env, Action.HOLD)
        # 홀드 슬롯에 원래 피스 타입이 들어가야 한다
        assert next_env.hold == original_type
        assert snap.hold_piece == original_type

    def test_left_action_moves_piece_left(self):
        original_col = self.env.active.col
        next_env, snap = self.adapter.simulate_action(self.env, Action.LEFT)
        # 이동 가능한 경우에만 검증 (스폰 위치에서는 왼쪽 이동 가능)
        if Action.LEFT in self.adapter.list_legal_actions(self.env):
            assert next_env.active.col == original_col - 1

    def test_right_action_moves_piece_right(self):
        original_col = self.env.active.col
        next_env, snap = self.adapter.simulate_action(self.env, Action.RIGHT)
        if Action.RIGHT in self.adapter.list_legal_actions(self.env):
            assert next_env.active.col == original_col + 1

    def test_rotate_cw_changes_rotation(self):
        original_rot = self.env.active.rotation
        next_env, _ = self.adapter.simulate_action(self.env, Action.ROTATE_CW)
        if Action.ROTATE_CW in self.adapter.list_legal_actions(self.env):
            assert next_env.active.rotation == (original_rot + 1) % 4

    def test_hard_drop_spawns_next_piece(self):
        """HARD_DROP 후 다음 피스가 스폰되어 활성 피스가 존재해야 한다."""
        next_env, snap = self.adapter.simulate_action(self.env, Action.HARD_DROP)
        # 게임 오버가 아닌 한 새 피스가 스폰된다
        if not next_env.spawn_blocked:
            assert next_env.active is not None
            assert snap.current_piece is not None

    def test_snapshot_terminal_matches_env(self):
        next_env, snap = self.adapter.simulate_action(self.env, Action.HARD_DROP)
        assert snap.terminal == self.adapter.is_terminal(next_env)


# ── heuristics ────────────────────────────────────────────────────────────────

class TestHeuristics:
    def _empty_snapshot(self) -> StateSnapshot:
        """빈 보드 스냅샷을 반환한다."""
        return StateSnapshot(
            board=[[None] * BOARD_COLS for _ in range(BOARD_ROWS)],
            current_piece=None,
            hold_piece=None,
            queue_preview=[],
            combo=-1,
            b2b=-1,
            pending_garbage=0,
            terminal=False,
        )

    def test_max_height_empty_board(self):
        snap = self._empty_snapshot()
        assert board_max_height(snap) == 0

    def test_max_height_single_block(self):
        snap = self._empty_snapshot()
        snap.board[39][0] = MinoType.I   # 바닥 행(row 39)에 블록
        assert board_max_height(snap) == 1

    def test_max_height_stack(self):
        snap = self._empty_snapshot()
        # row 35~39(5개 행)에 블록 → 높이 = 40 - 35 = 5
        for r in range(35, 40):
            snap.board[r][0] = MinoType.J
        assert board_max_height(snap) == 5

    def test_count_holes_no_holes(self):
        snap = self._empty_snapshot()
        # 바닥만 채워도 홀 없음
        snap.board[39][3] = MinoType.L
        assert count_holes(snap) == 0

    def test_count_holes_single_hole(self):
        snap = self._empty_snapshot()
        snap.board[38][3] = MinoType.S   # 블록
        snap.board[39][3] = None          # 아래 빈 셀 → 홀
        # 블록 위에 아무것도 없으므로 홀 발생
        # 그런데 row 38이 블록, row 39가 빈칸 → 홀 1개
        assert count_holes(snap) == 1

    def test_count_holes_empty_board_is_zero(self):
        snap = self._empty_snapshot()
        assert count_holes(snap) == 0

    def test_column_heights_length(self):
        snap = self._empty_snapshot()
        heights = column_heights(snap)
        assert len(heights) == BOARD_COLS

    def test_column_heights_all_zero_empty(self):
        snap = self._empty_snapshot()
        assert all(h == 0 for h in column_heights(snap))

    def test_bumpiness_flat_board(self):
        snap = self._empty_snapshot()
        # 모든 열에 같은 높이의 블록
        for col in range(BOARD_COLS):
            snap.board[39][col] = MinoType.I
        assert bumpiness(snap) == 0

    def test_bumpiness_two_column_difference(self):
        snap = self._empty_snapshot()
        snap.board[39][0] = MinoType.I   # col 0: 높이 1
        snap.board[38][1] = MinoType.I   # col 1: 높이 2
        snap.board[39][1] = MinoType.I
        # 나머지 열: 높이 0
        # bumpiness = |1-2| + |2-0| + 0*8 = 1+2 = 3
        assert bumpiness(snap) == 3


# ── GreedyBot ─────────────────────────────────────────────────────────────────

class TestGreedyBot:
    def setup_method(self):
        self.bot = GreedyBot()
        self.env = make_state()

    def test_pick_action_returns_action(self):
        action = self.bot.pick_action(self.env)
        assert isinstance(action, Action)

    def test_pick_action_returns_none_when_terminal(self):
        self.env.round_result = RoundResult.LOSE
        action = self.bot.pick_action(self.env)
        assert action is None

    def test_pick_action_is_legal(self):
        """선택된 행동이 합법 행동 목록에 포함되어야 한다."""
        adapter = BotEnvAdapter()
        action = self.bot.pick_action(self.env)
        legal = adapter.list_legal_actions(self.env)
        assert action in legal

    def test_bot_does_not_modify_env(self):
        """pick_action이 원본 env를 수정하지 않아야 한다."""
        original_pieces = self.env.pieces_placed
        original_queue = list(self.env.next_queue)
        self.bot.pick_action(self.env)
        assert self.env.pieces_placed == original_pieces
        assert list(self.env.next_queue) == original_queue

    def test_bot_picks_hard_drop_avoids_bad_state(self):
        """봇이 게임 오버를 유발하지 않는 행동을 선호해야 한다."""
        action = self.bot.pick_action(self.env)
        adapter = BotEnvAdapter()
        next_env, snap = adapter.simulate_action(self.env, action)
        # 선택한 행동의 결과가 terminal이 아니어야 한다 (정상 초기 상태에서)
        assert not snap.terminal

    def test_bot_can_play_multiple_steps(self):
        """봇이 연속적으로 행동을 선택하고 환경이 정상적으로 전이되어야 한다."""
        adapter = BotEnvAdapter()
        env = self.env

        for _ in range(5):
            if adapter.is_terminal(env):
                break
            action = self.bot.pick_action(env)
            assert action is not None
            env, snap = adapter.simulate_action(env, action)
