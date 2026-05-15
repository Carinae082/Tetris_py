"""
bot/adapter.py — BotEnvAdapter: 봇과 GameState 사이의 최소 인터페이스

설계 원칙
----------
* 게임 규칙을 직접 구현하지 않는다.
  이동/회전 합법 여부 판정, 줄 클리어, 콤보·B2B 갱신은 engine 함수에 완전히 위임한다.
* clone_env / _fast_clone은 deepcopy를 사용한다.
  GameState가 별도의 clone API를 제공하지 않으므로 가장 안전한 방법을 선택한다.
  향후 GameState가 copy() 메서드를 제공하면 _fast_clone에서 우선 사용한다.
* simulate_action은 원본 env를 절대 변경하지 않는다.
* simulate_action_lean은 배치 완료까지 한 번의 복제로 처리해 성능을 높인다.
"""

from __future__ import annotations

import copy
from enum import Enum, auto
from typing import Optional

from engine.board import BOARD_COLS, BOARD_ROWS
from engine.mino import MinoType
from engine.board import BOARD_ROWS as _BOARD_ROWS, BOARD_COLS as _BOARD_COLS
from engine.state import GameState, LockResult, RoundResult
from engine.physics import try_move, try_rotate, hard_drop
from engine.clear import process_clear
from engine.types import SpinType

from .attack_model import OutcomeSummary
from .snapshot import StateSnapshot, PieceSnapshot


# ── 행동 정의 ─────────────────────────────────────────────────────────────────

class Action(Enum):
    """봇이 선택할 수 있는 원자적 행동."""
    LEFT       = auto()   # 좌 이동
    RIGHT      = auto()   # 우 이동
    ROTATE_CW  = auto()   # 시계 방향 회전
    ROTATE_CCW = auto()   # 반시계 방향 회전
    ROTATE_180 = auto()   # 180° 회전
    HOLD       = auto()   # 홀드 교환
    HARD_DROP  = auto()   # 즉시 바닥 낙하 + 고정
    SOFT_DROP  = auto()   # 한 칸 아래 (input_sequence 재생용)


# ── 모듈 레벨 함수: lean OutcomeSummary 생성 ──────────────────────────────────

def build_summary_from_env(
    env: GameState,
    lock_result: LockResult | None,
) -> OutcomeSummary:
    """
    시뮬레이션 완료된 env에서 직접 OutcomeSummary를 생성한다.
    StateSnapshot을 거치지 않아 보드 복사를 생략한다.
    env.board._grid를 직접 참조하므로 env가 살아 있는 동안만 유효하다.

    lean 경로: board 복사 없음, 읽기만 수행.
    """
    if lock_result is not None:
        lines_cleared = lock_result.lines_cleared
        spin_type     = lock_result.spin
    else:
        lines_cleared = 0
        spin_type     = SpinType.NONE

    # fusion 스케일: Rust b2b=1 이 Python b2b=0 에 해당 → 첫 번째 difficult clear 부터 보너스
    did_b2b   = lines_cleared > 0 and env.b2b >= 0
    b2b_count = env.b2b + 1   # Rust 스케일로 변환 (Python b2b + 1 = Rust b2b)
    combo     = env.combo + 1  # Fusion 스케일: Python -1(없음)→0, Python 0(첫 클리어)→1

    is_pc = False
    if lines_cleared > 0:
        # col_heights 는 캐시됨 (O(10) vs O(400) 전체 스캔)
        is_pc = max(env.board.col_heights) == 0

    terminal = env.round_result != RoundResult.ONGOING
    clears_garbage = lock_result.clears_garbage if lock_result is not None else False

    return OutcomeSummary(
        lines_cleared    = lines_cleared,
        spin_type        = spin_type,
        did_b2b          = did_b2b,
        b2b_count        = b2b_count,
        combo            = combo,
        is_perfect_clear = is_pc,
        terminal         = terminal,
        board            = env.board._grid,  # 직접 참조 (복사 없음)
        clears_garbage   = clears_garbage,
    )


# ── BotEnvAdapter ─────────────────────────────────────────────────────────────

class BotEnvAdapter:
    """
    봇과 기존 GameState 환경 사이의 얇은 인터페이스 레이어.

    모든 메서드는 engine의 공개 함수/메서드를 최대한 그대로 사용하며,
    게임 규칙을 중복 구현하지 않는다.
    """

    # ── 상태 스냅샷 ───────────────────────────────────────────────────────────

    def get_state_snapshot(self, env: GameState) -> StateSnapshot:
        """
        봇이 읽을 수 있는 읽기 전용 상태 스냅샷을 반환한다.
        환경 원본 객체를 노출하지 않으며, 각 컨테이너를 얕게 복사한다.
        """
        board: list[list[Optional[MinoType]]] = [
            row[:] for row in env.board._grid
        ]

        current_piece: Optional[PieceSnapshot] = None
        if env.active is not None:
            p = env.active
            current_piece = PieceSnapshot(
                type=p.type,
                row=p.row,
                col=p.col,
                rotation=p.rotation,
                cells=p.cells(),
            )

        queue_preview: list[MinoType] = list(env.next_queue)[:6]

        return StateSnapshot(
            board=board,
            current_piece=current_piece,
            hold_piece=env.hold,
            queue_preview=queue_preview,
            combo=env.combo,
            b2b=env.b2b,
            pending_garbage=env.incoming.total_lines(),
            terminal=self.is_terminal(env),
        )

    # ── 합법 행동 목록 ────────────────────────────────────────────────────────

    def list_legal_actions(self, env: GameState) -> list[Action]:
        """
        현재 상태에서 가능한 행동 목록을 반환한다.
        각 행동의 합법 여부는 engine 함수가 판정한다.
        """
        if env.active is None or self.is_terminal(env):
            return []

        legal: list[Action] = []

        if try_move(self._fast_clone(env), -1):
            legal.append(Action.LEFT)
        if try_move(self._fast_clone(env), +1):
            legal.append(Action.RIGHT)

        if try_rotate(self._fast_clone(env), 1):
            legal.append(Action.ROTATE_CW)
        if try_rotate(self._fast_clone(env), -1):
            legal.append(Action.ROTATE_CCW)
        if try_rotate(self._fast_clone(env), 2):
            legal.append(Action.ROTATE_180)

        if not env.hold_used:
            legal.append(Action.HOLD)

        legal.append(Action.HARD_DROP)

        return legal

    # ── 행동 시뮬레이션 (기존 API 유지) ───────────────────────────────────────

    def simulate_action(
        self, env: GameState, action: Action
    ) -> tuple[GameState, StateSnapshot]:
        """
        env 복사본에 action을 적용하고 (다음 환경, 스냅샷)을 반환한다.
        원본 env는 변경되지 않는다.
        """
        clone = self._fast_clone(env)

        if action == Action.LEFT:
            try_move(clone, -1)
        elif action == Action.RIGHT:
            try_move(clone, +1)
        elif action == Action.ROTATE_CW:
            try_rotate(clone, 1)
        elif action == Action.ROTATE_CCW:
            try_rotate(clone, -1)
        elif action == Action.ROTATE_180:
            try_rotate(clone, 2)
        elif action == Action.HOLD:
            clone.try_hold()
        elif action == Action.HARD_DROP:
            lock_result = hard_drop(clone)
            process_clear(clone, lock_result.lines_cleared, lock_result.spin)
            if clone.next_queue:
                clone.spawn_next()

        return clone, self.get_state_snapshot(clone)

    def simulate_action_full(
        self, env: GameState, action: Action
    ) -> tuple[GameState, StateSnapshot, LockResult | None]:
        """
        simulate_action과 동일하되 HARD_DROP일 때 LockResult를 추가로 반환한다.
        세 번째 요소: HARD_DROP → LockResult, 그 외 → None.
        """
        clone = self._fast_clone(env)
        lock_result: LockResult | None = None

        if action == Action.LEFT:
            try_move(clone, -1)
        elif action == Action.RIGHT:
            try_move(clone, +1)
        elif action == Action.ROTATE_CW:
            try_rotate(clone, 1)
        elif action == Action.ROTATE_CCW:
            try_rotate(clone, -1)
        elif action == Action.ROTATE_180:
            try_rotate(clone, 2)
        elif action == Action.HOLD:
            clone.try_hold()
        elif action == Action.HARD_DROP:
            lock_result = hard_drop(clone)
            process_clear(clone, lock_result.lines_cleared, lock_result.spin)
            if clone.next_queue:
                clone.spawn_next()

        return clone, self.get_state_snapshot(clone), lock_result

    # ── Lean 시뮬레이션 (빔 서치 전용 최적화 경로) ───────────────────────────

    def simulate_action_lean(
        self, env: GameState, action: Action
    ) -> tuple[GameState, LockResult | None]:
        """
        배치 완료까지 단 한 번의 env 복제로 처리하는 최적화 시뮬레이션.
        StateSnapshot을 생성하지 않는다.

        HARD_DROP / HOLD  : 기존과 동일 동작.
        LEFT / RIGHT / ROTATE : 이동 후 즉시 HARD_DROP까지 수행 (복제 1회).
            → 반환 env는 배치 완료 후 다음 피스가 스폰된 상태.
            → 기존 simulate_action_full(move) + simulate_action_full(HARD_DROP)과
               최종 결과 동일, 복제 횟수 절반.

        반환: (배치 완료된 env, LockResult | None)
        """
        clone: GameState = self._fast_clone(env)
        lock_result: LockResult | None = None

        if action == Action.LEFT:
            try_move(clone, -1)
            lock_result = hard_drop(clone)
            process_clear(clone, lock_result.lines_cleared, lock_result.spin)
            if clone.next_queue:
                clone.spawn_next()

        elif action == Action.RIGHT:
            try_move(clone, +1)
            lock_result = hard_drop(clone)
            process_clear(clone, lock_result.lines_cleared, lock_result.spin)
            if clone.next_queue:
                clone.spawn_next()

        elif action == Action.ROTATE_CW:
            try_rotate(clone, 1)
            lock_result = hard_drop(clone)
            process_clear(clone, lock_result.lines_cleared, lock_result.spin)
            if clone.next_queue:
                clone.spawn_next()

        elif action == Action.ROTATE_CCW:
            try_rotate(clone, -1)
            lock_result = hard_drop(clone)
            process_clear(clone, lock_result.lines_cleared, lock_result.spin)
            if clone.next_queue:
                clone.spawn_next()

        elif action == Action.ROTATE_180:
            try_rotate(clone, 2)
            lock_result = hard_drop(clone)
            process_clear(clone, lock_result.lines_cleared, lock_result.spin)
            if clone.next_queue:
                clone.spawn_next()

        elif action == Action.HOLD:
            clone.try_hold()
            # lock_result = None (홀드는 배치 없음)

        elif action == Action.HARD_DROP:
            lock_result = hard_drop(clone)
            process_clear(clone, lock_result.lines_cleared, lock_result.spin)
            if clone.next_queue:
                clone.spawn_next()

        return clone, lock_result

    @staticmethod
    def build_summary_from_env(
        env: GameState,
        lock_result: LockResult | None,
    ) -> OutcomeSummary:
        """
        시뮬레이션 완료된 env에서 직접 OutcomeSummary를 생성한다.
        모듈 레벨 build_summary_from_env와 동일하다 (메서드로도 접근 가능하도록 노출).
        """
        return build_summary_from_env(env, lock_result)

    # ── 배치 시뮬레이션 ───────────────────────────────────────────────────────

    def simulate_placement(
        self,
        env: GameState,
        placement,   # FinalPlacement — 순환 임포트 방지를 위해 타입 힌트 생략
    ) -> tuple[GameState, "LockResult"]:
        """
        FinalPlacement 를 env 복사본에 직접 적용한다.
        원본 env 는 변경하지 않는다.

        구현 전략:
          1. use_hold 이면 clone.try_hold() 로 피스를 교환한다.
          2. 활성 피스를 배치의 최종 위치로 덮어쓴다.
          3. 스핀 판정 컨텍스트(last_was_rotation 등)를 배치 값으로 설정한다.
          4. hard_drop 호출 → _drop_to_floor 는 이미 착지 상태라 no-op,
             detect_spin → lock_active 순서로 처리된다.
          5. process_clear + spawn_next 로 다음 피스를 스폰한다.

        반환: (배치 완료된 GameState, LockResult)
        """
        from engine.mino import ActivePiece as _AP

        clone = self._fast_clone(env)

        if placement.use_hold:
            clone.try_hold()

        # 최종 배치 위치로 덮어쓰기
        clone.active = _AP(
            placement.piece_type,
            placement.row,
            placement.col,
            placement.rotation,
        )
        # 스핀 판정 컨텍스트 복원
        clone.last_was_rotation  = placement.last_was_rotation
        clone.last_kick_index    = placement.last_kick_index
        clone.last_kick_upgrades = placement.last_kick_upgrades

        # hard_drop: _drop_to_floor (no-op) → detect_spin → lock_active
        lock_result = hard_drop(clone)
        process_clear(clone, lock_result.lines_cleared, lock_result.spin)
        if clone.next_queue:
            clone.spawn_next()

        return clone, lock_result

    # ── 종료 판정 ─────────────────────────────────────────────────────────────

    def is_terminal(self, env: GameState) -> bool:
        """게임이 끝난 상태(WIN 또는 LOSE)이면 True를 반환한다."""
        return env.round_result != RoundResult.ONGOING

    # ── 환경 복제 ─────────────────────────────────────────────────────────────

    def _fast_clone(self, env: GameState) -> GameState:
        """
        빠른 복제 전략.
        GameState가 자체 copy() 메서드를 제공하면 그것을 우선 사용한다.
        현재는 deepcopy를 사용한다 (향후 GameState.copy() 구현 시 여기서만 교체).
        """
        copy_fn = getattr(env, "copy", None)
        if copy_fn is not None and callable(copy_fn):
            return copy_fn()
        return copy.deepcopy(env)

    def clone_env(self, env: GameState) -> GameState:
        """공개 API. _fast_clone의 alias."""
        return self._fast_clone(env)
