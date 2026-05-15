"""
bot/tactical_evaluator.py — 전술 평가기

구성
----
TacticalWeights        : GreedyBot용 board/attack 가중치 (하위 호환 유지)
build_outcome_summary  : simulate_action_full 결과 → OutcomeSummary
score_action           : GreedyBot용 즉시 점수 계산

CoachingState          : fusion/state.rs CoachingState 포팅
compute_coaching_state : 게임 상태에서 CoachingState 계산
coaching_context_bias  : fusion/search_expand.rs — 코칭 상태 변화량
shape_context_modifier : fusion/analysis.rs — [-1, 1] 클램프
shape_chain_value      : fusion/analysis.rs — 콤보 가치 로그 감쇠
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

from engine.board import BOARD_COLS, BOARD_ROWS
from engine.state import GameState, LockResult
from engine.types import SpinType

from .attack_model import AttackConfig, OutcomeSummary, calculate_immediate_attack
from .eval_features import column_heights
from .evaluator import EvalWeights, evaluate_board
from .snapshot import StateSnapshot


# ── GreedyBot용 전술 가중치 (하위 호환) ──────────────────────────────────────

@dataclass
class TacticalWeights:
    """
    board-shape 점수와 즉시 공격 가치를 합산하는 가중치. (GreedyBot 전용)

    final_score (non-terminal) =
        survival_bonus
        + board_score_weight  × evaluate_board(board)
        + attack_score_weight × immediate_attack
        + line_clear_bonus    × (lines_cleared > 0)
        + b2b_continue_bonus  × did_b2b
    """
    board_score_weight:  float =  1.0
    attack_score_weight: float =  0.50
    survival_bonus:      float =  1000.0
    terminal_penalty:    float = -1_000_000.0
    line_clear_bonus:    float =  10.0
    b2b_continue_bonus:  float =  5.0


_DEFAULT_TACTICAL = TacticalWeights()


# ── CoachingState ─────────────────────────────────────────────────────────────

class FatalityState(Enum):
    Safe     = auto()
    Critical = auto()
    Fatal    = auto()


class ObligationState(Enum):
    Nothing       = auto()
    MustDownstack = auto()
    MustCancel    = auto()


class SurgeState(Enum):
    Dormant  = auto()
    Building = auto()
    Active   = auto()


class PhaseState(Enum):
    Opener  = auto()
    Midgame = auto()
    Endgame = auto()


@dataclass(frozen=True)
class CoachingState:
    """
    fusion/state.rs CoachingState 포팅.
    한 배치 시점의 게임 국면 요약. SearchNode에 저장되어 delta 계산에 사용된다.
    """
    fatality:   FatalityState   = FatalityState.Safe
    obligation: ObligationState = ObligationState.Nothing
    surge:      SurgeState      = SurgeState.Dormant
    phase:      PhaseState      = PhaseState.Opener


# 각 상태 → 수치 매핑 (lookup table)
_FATALITY_W:   dict = {FatalityState.Safe: 0.0,  FatalityState.Critical: -0.35, FatalityState.Fatal: -0.70}
_OBLIGATION_W: dict = {ObligationState.Nothing: 0.0, ObligationState.MustDownstack: -0.25, ObligationState.MustCancel: -0.45}
_SURGE_W:      dict = {SurgeState.Dormant: 0.0,  SurgeState.Building: 0.20,   SurgeState.Active: 0.35}
_PHASE_W:      dict = {PhaseState.Opener: 0.10,  PhaseState.Midgame: 0.0,     PhaseState.Endgame: -0.10}


def compute_coaching_state(
    max_height:       int,
    spawn_blocked:    bool,
    b2b:              int,    # Python b2b: -1=없음, 0+=체인 횟수
    pieces_placed:    int,
    imminent_garbage: int,    # 이 배치 후 남은 가비지 수 (클리어 상쇄 후)
    lines_cleared:    int,
) -> CoachingState:
    """
    fusion/state.rs CoachingState.transition() 포팅.

    fatality   : 높이·spawn block 기반 위험도
    obligation : fatality 또는 대기 가비지 기반 의무 행동
    surge      : B2B 체인 기반 폭발 준비 상태
    phase      : pieces_placed 기반 게임 페이즈
    """
    # fatality
    if spawn_blocked or max_height >= 35:
        fatality = FatalityState.Fatal
    elif max_height >= 28:
        fatality = FatalityState.Critical
    else:
        fatality = FatalityState.Safe

    # obligation
    if fatality is FatalityState.Fatal or (imminent_garbage >= 3 and lines_cleared == 0):
        obligation = ObligationState.MustCancel
    elif max_height >= 26 or (imminent_garbage >= 1 and lines_cleared == 0):
        obligation = ObligationState.MustDownstack
    else:
        obligation = ObligationState.Nothing

    # surge (Python b2b -1=없음, 0=1번째 적격 클리어 → Rust 기준 +1 오프셋)
    rust_b2b = max(0, b2b + 1)
    if rust_b2b >= 3:
        surge = SurgeState.Active
    elif rust_b2b >= 1:
        surge = SurgeState.Building
    else:
        surge = SurgeState.Dormant

    # phase
    if pieces_placed < 8:
        phase = PhaseState.Opener
    elif pieces_placed < 28:
        phase = PhaseState.Midgame
    else:
        phase = PhaseState.Endgame

    return CoachingState(
        fatality=fatality, obligation=obligation, surge=surge, phase=phase
    )


def _coaching_score(state: CoachingState) -> float:
    return (
        _FATALITY_W[state.fatality]
        + _OBLIGATION_W[state.obligation]
        + _SURGE_W[state.surge]
        + _PHASE_W[state.phase]
    )


def coaching_context_bias(prev: CoachingState, nxt: CoachingState) -> float:
    """
    fusion/search_expand.rs coaching_context_bias 포팅.
    이전 → 현재 코칭 상태 변화량. 개선되면 양수, 악화되면 음수.
    """
    return _coaching_score(nxt) - _coaching_score(prev)


def shape_context_modifier(raw: float) -> float:
    """fusion/analysis.rs shape_context_modifier: [-1, 1] 클램프."""
    return max(-1.0, min(1.0, raw))


# ── chain 가치 ────────────────────────────────────────────────────────────────

def shape_chain_value(combo: float) -> float:
    """
    fusion/analysis.rs shape_chain_value 포팅.
    콤보 가치를 [0, 1] 범위의 로그 감쇠 값으로 변환한다.

    combo=0 → 0.0
    combo=1 → ~0.22
    combo=4 → ~0.63
    combo→∞ → 1.0
    """
    if combo <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - math.exp(-0.25 * combo)))


# ── OutcomeSummary 생성 ───────────────────────────────────────────────────────

def build_outcome_summary(
    next_env: GameState,
    next_snapshot: StateSnapshot,
    lock_result: LockResult | None,
) -> OutcomeSummary:
    """
    simulate_action_full의 반환값에서 OutcomeSummary를 생성한다.
    """
    if lock_result is not None:
        lines_cleared = lock_result.lines_cleared
        spin_type     = lock_result.spin
    else:
        lines_cleared = 0
        spin_type     = SpinType.NONE

    # fusion 스케일: 첫 번째 difficult clear(Python b2b=0)부터 보너스 적용
    did_b2b   = lines_cleared > 0 and next_env.b2b >= 0
    b2b_count = next_env.b2b + 1   # Rust 스케일로 변환
    combo     = next_env.combo + 1  # Fusion 스케일: Python -1(없음)→0, Python 0(첫 클리어)→1

    is_pc = False
    if lines_cleared > 0:
        is_pc = all(
            next_snapshot.board[r][c] is None
            for r in range(BOARD_ROWS)
            for c in range(BOARD_COLS)
        )

    clears_garbage = lock_result.clears_garbage if lock_result is not None else False

    return OutcomeSummary(
        lines_cleared    = lines_cleared,
        spin_type        = spin_type,
        did_b2b          = did_b2b,
        b2b_count        = b2b_count,
        combo            = combo,
        is_perfect_clear = is_pc,
        terminal         = next_snapshot.terminal,
        board            = next_snapshot.board,
        clears_garbage   = clears_garbage,
    )


# ── GreedyBot용 즉시 행동 점수 (하위 호환) ───────────────────────────────────

def score_action(
    summary: OutcomeSummary,
    tactical_weights: TacticalWeights | None = None,
    eval_weights: EvalWeights | None = None,
    attack_config: AttackConfig | None = None,
) -> float:
    """
    OutcomeSummary를 입력으로 즉시 행동 점수를 계산한다. (GreedyBot 전용)
    """
    if tactical_weights is None:
        tactical_weights = _DEFAULT_TACTICAL

    if summary.terminal:
        return tactical_weights.terminal_penalty

    board_score  = evaluate_board(summary.board, eval_weights)
    attack_score = calculate_immediate_attack(summary, attack_config)
    line_bonus   = tactical_weights.line_clear_bonus   if summary.lines_cleared > 0 else 0.0
    b2b_bonus    = tactical_weights.b2b_continue_bonus if summary.did_b2b          else 0.0

    return (
        tactical_weights.survival_bonus
        + tactical_weights.board_score_weight  * board_score
        + tactical_weights.attack_score_weight * attack_score
        + line_bonus
        + b2b_bonus
    )
