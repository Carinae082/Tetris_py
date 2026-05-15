"""
bot/transposition_table.py — 보드 점수 캐시 (Transposition Table)

Board.zobrist_hash 를 키로 board_score(float)를 저장한다.
탐색 시작 시 clear() 를 호출해 이전 탐색 결과를 초기화한다.
"""

from __future__ import annotations


class TranspositionTable:
    """
    dict 기반 board_score 캐시.

    동일 보드 상태가 다른 경로로 여러 번 탐색될 때,
    evaluate_board() 의 중복 호출을 제거한다.
    """

    __slots__ = ("_cache",)

    def __init__(self) -> None:
        self._cache: dict[int, float] = {}

    def get(self, board_hash: int) -> float | None:
        """캐시 히트 → float 반환. 미스 → None."""
        return self._cache.get(board_hash)

    def set(self, board_hash: int, score: float) -> None:
        self._cache[board_hash] = score

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
