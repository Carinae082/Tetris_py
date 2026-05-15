"""
tests/test_tactical_eval.py — AttackConfig, OutcomeSummary, TacticalWeights,
                               build_outcome_summary, score_action, GreedyBot(v3) 테스트
"""

from __future__ import annotations

import pytest

from engine.bag import SevenBag
from engine.board import BOARD_COLS, BOARD_ROWS
from engine.mino import MinoType
from engine.state import GameState, LockResult, RoundResult
from engine.types import SpinType

from bot.adapter import Action, BotEnvAdapter
from bot.attack_model import (
    AttackConfig,
    OutcomeSummary,
    calculate_immediate_attack,
    get_base_attack,
)
from bot.greedy_bot import GreedyBot
from bot.tactical_evaluator import (
    TacticalWeights,
    build_outcome_summary,
    score_action,
)


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def make_state(seed: int = 42, n_pieces: int = 14) -> GameState:
    bag = SevenBag(seed=seed)
    state = GameState()
    for _ in range(n_pieces):
        state.enqueue(bag.pop())
    state.spawn_next()
    return state


def empty_board() -> list[list]:
    return [[None] * BOARD_COLS for _ in range(BOARD_ROWS)]


def _summary(
    lines: int = 0,
    spin: SpinType = SpinType.NONE,
    b2b: bool = False,
    combo: int = -1,
    pc: bool = False,
    terminal: bool = False,
) -> OutcomeSummary:
    return OutcomeSummary(
        lines_cleared    = lines,
        spin_type        = spin,
        did_b2b          = b2b,
        combo            = combo,
        is_perfect_clear = pc,
        terminal         = terminal,
        board            = empty_board(),
    )


# ── AttackConfig ──────────────────────────────────────────────────────────────

class TestAttackConfig:
    def test_default_matches_tetra_league(self):
        d = AttackConfig()
        tl = AttackConfig.tetra_league()
        assert d.pc_garbage         == tl.pc_garbage
        assert d.pc_b2b             == tl.pc_b2b
        assert d.b2b_chaining       == tl.b2b_chaining
        assert d.combo_table        == tl.combo_table
        assert d.garbage_multiplier == tl.garbage_multiplier

    def test_tetra_league_values(self):
        c = AttackConfig.tetra_league()
        assert c.pc_garbage   == 5
        assert c.pc_b2b       == 2
        assert c.b2b_chaining is True
        assert c.combo_table  == "multiplier"
        assert c.garbage_multiplier == 1.0

    def test_quick_play_values(self):
        c = AttackConfig.quick_play()
        assert c.pc_garbage   == 3
        assert c.pc_b2b       == 2
        assert c.b2b_chaining is False
        assert c.combo_table  == "multiplier"
        assert c.garbage_multiplier == 1.0

    def test_presets_are_different(self):
        assert AttackConfig.tetra_league().pc_garbage != AttackConfig.quick_play().pc_garbage
        assert AttackConfig.tetra_league().b2b_chaining != AttackConfig.quick_play().b2b_chaining


# ── get_base_attack ───────────────────────────────────────────────────────────

class TestGetBaseAttack:
    def test_zero_lines_any_spin(self):
        for spin in SpinType:
            assert get_base_attack(0, spin) == 0

    def test_single(self):
        assert get_base_attack(1, SpinType.NONE) == 0

    def test_double(self):
        assert get_base_attack(2, SpinType.NONE) == 1

    def test_triple(self):
        assert get_base_attack(3, SpinType.NONE) == 2

    def test_quad(self):
        assert get_base_attack(4, SpinType.NONE) == 4

    def test_tspin_single(self):
        assert get_base_attack(1, SpinType.FULL) == 2

    def test_tspin_double(self):
        assert get_base_attack(2, SpinType.FULL) == 4

    def test_tspin_triple(self):
        assert get_base_attack(3, SpinType.FULL) == 6

    def test_mini_one_line(self):
        assert get_base_attack(1, SpinType.MINI) == 0

    def test_mini_two_lines(self):
        assert get_base_attack(2, SpinType.MINI) == 1

    def test_unknown_combination_returns_zero(self):
        # 5줄 이상은 표에 없으므로 0
        assert get_base_attack(5, SpinType.NONE) == 0


# ── calculate_immediate_attack ────────────────────────────────────────────────

class TestCalculateImmediateAttack:
    def test_no_lines_is_zero(self):
        assert calculate_immediate_attack(_summary(lines=0)) == 0.0

    def test_single_no_attack(self):
        assert calculate_immediate_attack(_summary(lines=1)) == 0.0

    def test_double(self):
        assert calculate_immediate_attack(_summary(lines=2)) == pytest.approx(1.0)

    def test_triple(self):
        assert calculate_immediate_attack(_summary(lines=3)) == pytest.approx(2.0)

    def test_quad(self):
        assert calculate_immediate_attack(_summary(lines=4)) == pytest.approx(4.0)

    def test_tspin_single(self):
        assert calculate_immediate_attack(_summary(lines=1, spin=SpinType.FULL)) == pytest.approx(2.0)

    def test_tspin_double(self):
        assert calculate_immediate_attack(_summary(lines=2, spin=SpinType.FULL)) == pytest.approx(4.0)

    def test_tspin_triple(self):
        assert calculate_immediate_attack(_summary(lines=3, spin=SpinType.FULL)) == pytest.approx(6.0)

    def test_b2b_adds_one(self):
        # Quad without B2B = 4; Quad with B2B = 5
        base = calculate_immediate_attack(_summary(lines=4, b2b=False))
        with_b2b = calculate_immediate_attack(_summary(lines=4, b2b=True))
        assert with_b2b == pytest.approx(base + 1.0)

    def test_combo_multiplier_first_clear_no_bonus(self):
        # combo=0 (첫 클리어): bonus = base * 0.25 * 0 = 0
        no_combo  = calculate_immediate_attack(_summary(lines=2, combo=-1))
        first_clr = calculate_immediate_attack(_summary(lines=2, combo=0))
        assert first_clr == pytest.approx(no_combo)

    def test_combo_multiplier_second_clear(self):
        # Double (base=1), combo=1: bonus = 1 * 0.25 * 1 = 0.25 → total = 1.25
        result = calculate_immediate_attack(_summary(lines=2, combo=1))
        assert result == pytest.approx(1.25)

    def test_combo_multiplier_higher(self):
        # Double (base=1), combo=4: bonus = 1 * 0.25 * 4 = 1.0 → total = 2.0
        result = calculate_immediate_attack(_summary(lines=2, combo=4))
        assert result == pytest.approx(2.0)

    def test_pc_bonus_tetra_league(self):
        # Double + PC: base=1, pc=5 → total = 6
        result = calculate_immediate_attack(
            _summary(lines=2, pc=True), AttackConfig.tetra_league()
        )
        assert result == pytest.approx(6.0)

    def test_pc_bonus_quick_play(self):
        # Double + PC (quick_play): base=1, pc=3 → total = 4
        result = calculate_immediate_attack(
            _summary(lines=2, pc=True), AttackConfig.quick_play()
        )
        assert result == pytest.approx(4.0)

    def test_pc_b2b_tetra_league(self):
        # Quad + B2B + PC (tetra_league): base=4, b2b=1, pc=5, pc_b2b=2 → total = 12
        result = calculate_immediate_attack(
            _summary(lines=4, b2b=True, pc=True), AttackConfig.tetra_league()
        )
        assert result == pytest.approx(12.0)

    def test_pc_b2b_quick_play_no_chaining(self):
        # quick_play: b2b_chaining=False → pc_b2b not added
        # Quad + B2B + PC (quick_play): base=4, b2b=1, pc=3, no pc_b2b → total = 8
        result = calculate_immediate_attack(
            _summary(lines=4, b2b=True, pc=True), AttackConfig.quick_play()
        )
        assert result == pytest.approx(8.0)

    def test_garbage_multiplier(self):
        config = AttackConfig(garbage_multiplier=2.0)
        result = calculate_immediate_attack(_summary(lines=2), config)
        assert result == pytest.approx(2.0)  # base=1 × multiplier=2


# ── TacticalWeights ───────────────────────────────────────────────────────────

class TestTacticalWeights:
    def test_defaults(self):
        w = TacticalWeights()
        assert w.board_score_weight  == pytest.approx(1.0)
        assert w.attack_score_weight == pytest.approx(0.50)
        assert w.survival_bonus      == pytest.approx(1000.0)
        assert w.terminal_penalty    == pytest.approx(-1_000_000.0)
        assert w.line_clear_bonus    == pytest.approx(10.0)
        assert w.b2b_continue_bonus  == pytest.approx(5.0)


# ── simulate_action_full ──────────────────────────────────────────────────────

class TestSimulateActionFull:
    def setup_method(self):
        self.adapter = BotEnvAdapter()
        self.env = make_state()

    def test_returns_three_tuple(self):
        result = self.adapter.simulate_action_full(self.env, Action.HARD_DROP)
        assert len(result) == 3

    def test_hard_drop_returns_lock_result(self):
        _, _, lr = self.adapter.simulate_action_full(self.env, Action.HARD_DROP)
        assert lr is not None
        assert isinstance(lr, LockResult)

    def test_move_returns_no_lock_result(self):
        _, _, lr = self.adapter.simulate_action_full(self.env, Action.LEFT)
        assert lr is None

    def test_hold_returns_no_lock_result(self):
        _, _, lr = self.adapter.simulate_action_full(self.env, Action.HOLD)
        assert lr is None

    def test_original_env_unchanged(self):
        orig_pieces = self.env.pieces_placed
        self.adapter.simulate_action_full(self.env, Action.HARD_DROP)
        assert self.env.pieces_placed == orig_pieces

    def test_lock_result_has_spin_type(self):
        _, _, lr = self.adapter.simulate_action_full(self.env, Action.HARD_DROP)
        assert isinstance(lr.spin, SpinType)


# ── build_outcome_summary ─────────────────────────────────────────────────────

class TestBuildOutcomeSummary:
    def setup_method(self):
        self.adapter = BotEnvAdapter()
        self.env = make_state()

    def test_returns_outcome_summary(self):
        next_env, snap, lr = self.adapter.simulate_action_full(self.env, Action.HARD_DROP)
        summary = build_outcome_summary(next_env, snap, lr)
        assert isinstance(summary, OutcomeSummary)

    def test_lines_cleared_from_lock_result(self):
        next_env, snap, lr = self.adapter.simulate_action_full(self.env, Action.HARD_DROP)
        summary = build_outcome_summary(next_env, snap, lr)
        assert summary.lines_cleared == lr.lines_cleared

    def test_spin_type_from_lock_result(self):
        next_env, snap, lr = self.adapter.simulate_action_full(self.env, Action.HARD_DROP)
        summary = build_outcome_summary(next_env, snap, lr)
        assert summary.spin_type == lr.spin

    def test_combo_from_env(self):
        next_env, snap, lr = self.adapter.simulate_action_full(self.env, Action.HARD_DROP)
        summary = build_outcome_summary(next_env, snap, lr)
        assert summary.combo == next_env.combo

    def test_terminal_from_snapshot(self):
        next_env, snap, lr = self.adapter.simulate_action_full(self.env, Action.HARD_DROP)
        summary = build_outcome_summary(next_env, snap, lr)
        assert summary.terminal == snap.terminal

    def test_board_from_snapshot(self):
        next_env, snap, lr = self.adapter.simulate_action_full(self.env, Action.HARD_DROP)
        summary = build_outcome_summary(next_env, snap, lr)
        assert summary.board is snap.board

    def test_non_drop_has_zero_lines(self):
        if Action.LEFT in self.adapter.list_legal_actions(self.env):
            next_env, snap, lr = self.adapter.simulate_action_full(self.env, Action.LEFT)
            summary = build_outcome_summary(next_env, snap, lr)
            assert summary.lines_cleared == 0
            assert summary.spin_type == SpinType.NONE

    def test_not_terminal_on_fresh_state(self):
        next_env, snap, lr = self.adapter.simulate_action_full(self.env, Action.HARD_DROP)
        summary = build_outcome_summary(next_env, snap, lr)
        assert not summary.terminal

    def test_did_b2b_false_initially(self):
        # 첫 클리어에서는 B2B 없음
        next_env, snap, lr = self.adapter.simulate_action_full(self.env, Action.HARD_DROP)
        summary = build_outcome_summary(next_env, snap, lr)
        # B2B는 첫 difficult clear에서 0, 보너스는 b2b>=1부터
        assert summary.did_b2b == (next_env.b2b >= 1 and lr.lines_cleared > 0)


# ── score_action ──────────────────────────────────────────────────────────────

class TestScoreAction:
    def test_terminal_returns_penalty(self):
        s = _summary(terminal=True)
        w = TacticalWeights()
        assert score_action(s) == pytest.approx(w.terminal_penalty)

    def test_non_terminal_has_survival_bonus(self):
        s = _summary()
        w = TacticalWeights()
        # 빈 보드 → board_score = 0; attack = 0; 최소 survival_bonus
        result = score_action(s)
        assert result >= w.survival_bonus

    def test_line_clear_adds_bonus(self):
        s_no  = _summary(lines=0)
        s_yes = _summary(lines=2)
        w = TacticalWeights()
        assert score_action(s_yes) > score_action(s_no)
        assert score_action(s_yes) - score_action(s_no) >= w.line_clear_bonus

    def test_b2b_adds_bonus(self):
        s_no  = _summary(lines=4, b2b=False)
        s_yes = _summary(lines=4, b2b=True)
        w = TacticalWeights()
        diff = score_action(s_yes) - score_action(s_no)
        # B2B bonus (b2b_continue_bonus) + attack bonus (+1 attack * attack_score_weight)
        assert diff > 0

    def test_terminal_worse_than_alive(self):
        s_dead  = _summary(terminal=True)
        s_alive = _summary()
        assert score_action(s_dead) < score_action(s_alive)

    def test_custom_tactical_weights(self):
        w = TacticalWeights(survival_bonus=999.0)
        s = _summary()
        result = score_action(s, tactical_weights=w)
        assert result >= 999.0

    def test_attack_score_reflected(self):
        # attack_score_weight=0 이면 attack 기여 없음
        w0 = TacticalWeights(attack_score_weight=0.0)
        w1 = TacticalWeights(attack_score_weight=1.0)
        s  = _summary(lines=4)  # Quad = 4 attack
        diff = score_action(s, tactical_weights=w1) - score_action(s, tactical_weights=w0)
        assert diff == pytest.approx(4.0)  # +4 attack × weight_diff(1.0 - 0.0)


# ── GreedyBot (v3 통합) ───────────────────────────────────────────────────────

class TestGreedyBotV3:
    def setup_method(self):
        self.bot = GreedyBot()
        self.env = make_state()

    def test_default_construction(self):
        bot = GreedyBot()
        assert bot is not None

    def test_custom_config_construction(self):
        bot = GreedyBot(
            tactical_weights=TacticalWeights(),
            eval_weights=None,
            attack_config=AttackConfig.quick_play(),
        )
        assert bot is not None

    def test_pick_action_returns_action(self):
        action = self.bot.pick_action(self.env)
        assert isinstance(action, Action)

    def test_pick_action_is_legal(self):
        adapter = BotEnvAdapter()
        action = self.bot.pick_action(self.env)
        assert action in adapter.list_legal_actions(self.env)

    def test_pick_action_none_when_terminal(self):
        self.env.round_result = RoundResult.LOSE
        assert self.bot.pick_action(self.env) is None

    def test_does_not_modify_env(self):
        orig_pieces = self.env.pieces_placed
        orig_queue  = list(self.env.next_queue)
        self.bot.pick_action(self.env)
        assert self.env.pieces_placed == orig_pieces
        assert list(self.env.next_queue) == orig_queue

    def test_avoids_terminal_action(self):
        adapter = BotEnvAdapter()
        action = self.bot.pick_action(self.env)
        _, snap = adapter.simulate_action(self.env, action)
        assert not snap.terminal

    def test_multi_step_play(self):
        adapter = BotEnvAdapter()
        env = self.env
        for _ in range(6):
            if adapter.is_terminal(env):
                break
            action = self.bot.pick_action(env)
            assert action is not None
            env, snap = adapter.simulate_action(env, action)
