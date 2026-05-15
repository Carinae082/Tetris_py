"""
bot/snapshot.py — 봇이 읽는 읽기 전용 상태 스냅샷 정의

StateSnapshot은 환경 원본 객체(GameState)를 노출하지 않으며,
평가 함수·디버깅에만 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from engine.mino import MinoType


@dataclass
class PieceSnapshot:
    """활성 피스의 읽기 전용 표현."""
    type: MinoType
    row: int               # 바운딩 박스 좌상단 row (보드 절대 좌표)
    col: int               # 바운딩 박스 좌상단 col (보드 절대 좌표)
    rotation: int          # 0=스폰, 1=CW, 2=180, 3=CCW
    cells: list[tuple[int, int]]  # 점유하는 셀의 절대 (row, col) 목록


@dataclass
class StateSnapshot:
    """
    봇이 읽는 읽기 전용 상태 스냅샷.

    board는 환경의 _grid를 얕게 복사한 40×10 리스트이다.
    각 셀은 None(빈칸) 또는 MinoType(고정 블록/가비지).
    """

    # ── 보드 ──────────────────────────────────────────────────────────────
    # 40 rows × 10 cols.  row 0~19 = 버퍼(숨김), row 20~39 = 표시 영역.
    board: list[list[Optional[MinoType]]]

    # ── 피스 상태 ──────────────────────────────────────────────────────────
    current_piece: Optional[PieceSnapshot]  # 현재 활성 피스 (없으면 None)
    hold_piece: Optional[MinoType]          # 홀드 슬롯 (비어 있으면 None)

    # ── 넥스트 큐 ─────────────────────────────────────────────────────────
    queue_preview: list[MinoType]           # 최대 6개 미리보기

    # ── 콤보 / B2B ────────────────────────────────────────────────────────
    combo: int   # -1 = 없음, 0 이상 = 연속 클리어 횟수
    b2b: int     # -1 = 없음, 0 이상 = 연속 difficult 클리어 횟수

    # ── 가비지 ────────────────────────────────────────────────────────────
    pending_garbage: int  # 수신 대기 중인 가비지 총 줄 수

    # ── 종료 여부 ─────────────────────────────────────────────────────────
    terminal: bool
