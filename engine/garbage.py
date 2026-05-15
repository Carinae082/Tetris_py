"""
engine/garbage.py -- incoming garbage queue

TETR.IO Season 2 / Tetra League 규칙
--------------------------------------
change-on-attack
  가비지는 "attack 단위"로 큐에 쌓인다.
  각 attack은 큐에 들어오는 시점에 hole column이 결정된다.
  다음 attack이 도착하면 새 hole column이 배정되므로, 연속 attack이라도
  hole이 달라질 수 있다 (= change-on-attack).

full blocking (passthrough off)
  내 클리어가 만든 공격력은 즉시 내 큐를 상쇄할 수 있다.
  "전송 중(in-transit)"이라 상쇄 불가한 시간 구간은 없다.
  상쇄 로직은 attack.py의 compute_attack()에서 처리한다.

큐 상쇄 방향
  앞(먼저 쌓인) attack부터 상쇄한다.
  한 attack 내에서 부분 상쇄가 가능하다.
  부분 상쇄 시 해당 attack의 hole column은 유지된다.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class GarbageLine:
    """
    큐에 들어 있는 attack 단위 하나.

    lines    : 이 attack으로 올라올 가비지 줄 수
    hole_col : 가비지 줄의 구멍 열 위치 (0-indexed, 0~9)
               attack이 큐에 삽입된 시점에 결정된다 (change-on-attack).
    """
    lines: int
    hole_col: int


class GarbageQueue:
    """
    한 플레이어의 incoming garbage 큐.

    add(lines)  : 상대 attack을 받아 큐에 넣는다. hole은 자동 배정.
    cancel(n)   : 자신의 공격력 n만큼 큐 앞에서 상쇄한다.
    pop_all()   : piece lock 전에 호출 — 큐를 비우고 보드에 적용할 목록을 반환.
    total_lines : 현재 대기 중인 가비지 총 줄 수 (attack.py의 opener 판정용).
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        """
        rng : 외부에서 주입하는 Random 인스턴스.
              None이면 독립적인 새 인스턴스를 생성한다.
              테스트 시 seed를 고정하려면 random.Random(seed) 를 넘긴다.
        """
        self._queue: deque[GarbageLine] = deque()
        self._rng: random.Random = rng if rng is not None else random.Random()

    # ── 가비지 추가 (상대 attack 수신) ───────────────────────────────────────

    def add(self, lines: int, hole_col: int | None = None) -> None:
        """
        attack 1회 = GarbageLine 1개.
        hole_col 미지정 시 change-on-attack 규칙에 따라 랜덤 배정.
        lines == 0 이면 아무것도 추가하지 않는다.
        """
        if lines <= 0:
            return
        if hole_col is None:
            hole_col = self._rng.randrange(10)
        self._queue.append(GarbageLine(lines=lines, hole_col=hole_col))

    # ── 가비지 상쇄 (내 공격으로 blocking) ───────────────────────────────────

    def cancel(self, lines: int) -> int:
        """
        공격력 lines만큼 큐 앞에서 상쇄한다 (full blocking).

        반환값: 상쇄 후 남은 공격력 (0 = 전부 차단됨, >0 = 상대에게 보낼 양).

        부분 상쇄: attack 1개를 다 지우지 못하면 남은 줄 수만큼 새 GarbageLine으로
                   교체한다. hole_col은 그대로 유지된다.
        """
        remaining = lines
        while remaining > 0 and self._queue:
            front = self._queue[0]
            if front.lines <= remaining:
                remaining -= front.lines
                self._queue.popleft()
            else:
                # 부분 상쇄 — hole_col 유지, 줄 수만 줄임
                self._queue[0] = GarbageLine(
                    lines=front.lines - remaining,
                    hole_col=front.hole_col,
                )
                remaining = 0
        return remaining

    # ── 가비지 적용 (piece lock 직전) ────────────────────────────────────────

    def pop_all(self) -> list[GarbageLine]:
        """
        대기 중인 모든 GarbageLine을 꺼낸다.
        board.add_garbage() 에 순서대로 넘겨 보드에 올린다.
        """
        result = list(self._queue)
        self._queue.clear()
        return result

    # ── 조회 ─────────────────────────────────────────────────────────────────

    def total_lines(self) -> int:
        """현재 큐에 대기 중인 가비지 총 줄 수."""
        return sum(g.lines for g in self._queue)

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)          # attack 단위 개수 (줄 수 아님)

    def __repr__(self) -> str:
        return (
            f"GarbageQueue(attacks={len(self._queue)}, "
            f"total_lines={self.total_lines()}, "
            f"entries={list(self._queue)})"
        )
