"""
bot/attack_model.py — 즉시 공격력 계산 모델 (TETR.IO Season 2 / fusion 스타일)

OutcomeSummary  : 행동 시뮬레이션 결과 요약 (봇 평가 공유 구조)
AttackConfig    : 게임 설정별 공격 파라미터 (tetra_league / quick_play)
get_base_attack : (lines, spin) → 기본 공격력 (5+ 라인 지원)
calculate_immediate_attack : fusion/attack.rs calculate_attack_full 포팅
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from engine.board import BOARD_COLS, BOARD_ROWS
from engine.types import SpinType


# ── 행동 결과 요약 ─────────────────────────────────────────────────────────────

@dataclass
class OutcomeSummary:
    """
    봇이 simulate_action_full 결과를 평가할 때 사용하는 읽기 전용 요약.
    공격 계산과 전술 평가에 공통으로 사용된다.

    b2b_count는 하위 호환을 위해 기본값 0을 가지며 필드 목록 끝에 위치한다.
    build_summary_from_env / build_outcome_summary 는 항상 올바른 값을 설정한다.
    """
    lines_cleared:    int         # 이번 행동으로 지워진 줄 수 (0 = 클리어 없음)
    spin_type:        SpinType    # lock 시 감지된 스핀 타입
    did_b2b:          bool        # 이 클리어에서 B2B 보너스가 적용됐는가
    combo:            int         # 클리어 후 combo 값 (-1 = 없음, 0+ = 연속 횟수)
    is_perfect_clear: bool        # 클리어 후 보드가 완전히 빈 상태인가
    terminal:         bool        # 게임 종료 여부
    board:            list[list]  # StateSnapshot.board (40×10, None 또는 MinoType)
    b2b_count:        int  = 0    # B2B 체인 횟수 (Python: 1=첫 보너스, 2+=체인 유지)
    b2b_broken_from:  int | None = None  # non-difficult 클리어가 B2B 체인을 끊었을 때 이전 체인 수 (Rust 스케일: python+1)
    clears_garbage:   bool = False       # difficult 클리어가 가비지 행을 포함했는가


# ── 공격 설정 ─────────────────────────────────────────────────────────────────

@dataclass
class AttackConfig:
    """
    Season 2 공격 파라미터.
    기본 preset은 tetra_league()이다.
    """
    pc_garbage:          int   = 5
    pc_b2b:              int   = 2
    b2b_chaining:        bool  = True
    combo_table:         str   = "multiplier"
    garbage_multiplier:  float = 1.0

    @classmethod
    def tetra_league(cls) -> AttackConfig:
        return cls(
            pc_garbage=5,
            pc_b2b=2,
            b2b_chaining=True,
            combo_table="multiplier",
            garbage_multiplier=1.0,
        )

    @classmethod
    def quick_play(cls) -> AttackConfig:
        return cls(
            pc_garbage=3,
            pc_b2b=2,
            b2b_chaining=False,
            combo_table="multiplier",
            garbage_multiplier=1.0,
        )


_DEFAULT_CONFIG = AttackConfig.tetra_league()


# ── 기본 공격력 ───────────────────────────────────────────────────────────────

def get_base_attack(lines: int, spin: SpinType) -> float:
    """
    줄 수와 스핀 타입으로 기본 공격력을 반환한다. fusion/attack.rs base_attack 포팅.
    5줄 이상(PENTA+)도 지원한다.
    """
    if spin == SpinType.NONE:
        if lines <= 1: return 0.0
        if lines == 2: return 1.0
        if lines == 3: return 2.0
        if lines == 4: return 4.0
        if lines == 5: return 5.0
        return 5.0 + (lines - 5)         # 6줄+: PENTA + 초과분
    elif spin == SpinType.MINI:
        if lines <= 1: return 0.0
        if lines == 2: return 1.0
        if lines == 3: return 2.0
        if lines == 4: return 10.0
        return 10.0 + 2.0 * (lines - 4)  # 5줄+
    else:  # FULL spin
        if lines == 0: return 0.0
        if lines == 1: return 2.0
        if lines == 2: return 4.0
        if lines == 3: return 6.0
        if lines == 4: return 10.0
        if lines == 5: return 12.0
        return 12.0 + 2.0 * (lines - 5)  # 6줄+


# ── B2B 체이닝 보너스 (fusion/attack.rs b2b_chaining_bonus 포팅) ──────────────

def _b2b_chaining_bonus(b2b: int) -> float:
    """
    B2B 체인 수에 따른 로그 스케일 보너스.

    b2b=1 → 1.0 (첫 번째 체인)
    b2b=2 → ~1.32
    b2b=3 → ~2.08
    b2b=5 → ~2.20
    b2b=10 → ~3.07
    """
    if b2b <= 1:
        return 1.0
    log_part  = math.log(1.0 + b2b * 0.8)
    floored   = math.floor(1.0 + log_part)
    remainder = (1.0 + log_part) - floored
    return floored + remainder / 3.0


# ── 콤보 배율 (fusion/attack.rs apply_combo Multiplier 포팅) ─────────────────

def _apply_combo_multiplier(base: float, combo: int) -> float:
    """
    콤보 배율 적용. log floor로 최소 공격량을 보장한다.

    combo=0 : 보너스 없음 (base 그대로)
    combo=1 : base * 1.25
    combo=2+: max(base * (1 + 0.25*combo), ln(1 + combo*1.25))
    """
    if combo <= 0:
        return base
    multiplied = base * (1.0 + 0.25 * combo)
    if combo > 1:
        log_floor = math.log(1.0 + combo * 1.25)
        return max(multiplied, log_floor)
    return multiplied


# ── 즉시 공격력 계산 ──────────────────────────────────────────────────────────

def calculate_immediate_attack(
    summary: OutcomeSummary,
    config: AttackConfig | None = None,
) -> float:
    """
    행동 결과에서 즉시 공격력(가비지 줄 수)을 계산한다.
    줄 클리어가 없으면 0.0을 반환한다.

    계산 순서 (fusion/attack.rs calculate_attack_full 포팅)
    --------------------------------------------------------
    1. base            = get_base_attack(lines, spin)
    2. pc_bonus        = pc_garbage                        [PC일 때]
    3. b2b_bonus       = b2b_chaining_bonus 또는 flat 1.0  [B2B일 때]
    4. pc_b2b_bonus    = pc_b2b                            [PC + B2B일 때]
    5. surge_release   = b2b_broken_from                   [non-difficult 클리어가 긴 체인 끊을 때]
    6. garbage_boost   = +1.0                              [difficult 클리어가 가비지 제거 시]
    7. attack          = _apply_combo_multiplier(합산, combo)
    8. total           = attack * garbage_multiplier
    """
    if config is None:
        config = _DEFAULT_CONFIG

    if summary.lines_cleared == 0:
        return 0.0

    attack = get_base_attack(summary.lines_cleared, summary.spin_type)
    is_b2b_eligible = (
        summary.spin_type != SpinType.NONE or summary.lines_cleared >= 4
    )

    # 퍼펙트 클리어 보너스 (B2B 보너스 이전에 추가 — Rust 순서와 동일)
    if summary.is_perfect_clear:
        attack += config.pc_garbage

    # B2B 보너스
    if summary.did_b2b:
        attack += (
            _b2b_chaining_bonus(summary.b2b_count)
            if config.b2b_chaining
            else 1.0
        )

    # 퍼펙트 클리어 + B2B 추가 보너스
    if summary.is_perfect_clear and summary.did_b2b and config.b2b_chaining:
        attack += config.pc_b2b

    # Surge release: non-difficult 클리어가 긴 B2B 체인(4+)을 끊을 때
    # b2b_broken_from 은 Rust 스케일 (= python_prev_b2b + 1)
    if (
        summary.b2b_broken_from is not None
        and summary.b2b_broken_from >= 4
        and not is_b2b_eligible
    ):
        attack += float(summary.b2b_broken_from)

    # Garbage clear boost: difficult 클리어가 가비지 행을 포함했을 때 +1
    if summary.clears_garbage and is_b2b_eligible:
        attack += 1.0

    # 콤보 배율 (log floor 보장)
    attack = _apply_combo_multiplier(attack, summary.combo)

    return attack * config.garbage_multiplier
