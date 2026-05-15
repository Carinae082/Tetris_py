"""
bot/greedy_bot.py — 전술 평가 기반 단일 스텝 탐욕 봇 (v3)

pick_action() 호출 흐름
-----------------------
1. list_legal_actions로 가능한 행동을 모두 가져온다.
2. 각 행동을 simulate_action_full로 복사 환경에 적용한다.
   - HARD_DROP / HOLD: 결과에서 OutcomeSummary를 바로 만든다.
   - 이동·회전: 추가로 HARD_DROP을 한 번 더 시뮬레이션한 결과를 평가한다.
3. score_action으로 점수를 계산한다.
4. 가장 점수가 높은 행동을 반환한다.

중요
----
* 1-ply greedy. search/beam search 없음.
* spin 판정·line clear·combo/b2b 갱신은 엔진 결과를 그대로 사용한다.
* attack, combo, b2b는 score_action 내 즉시 가치로만 반영한다.
"""

from __future__ import annotations

from engine.state import GameState

from .adapter import BotEnvAdapter, Action
from .attack_model import AttackConfig
from .evaluator import EvalWeights
from .tactical_evaluator import TacticalWeights, build_outcome_summary, score_action

_NEG_INF = float("-inf")


class GreedyBot:
    """
    전술 평가 기반 단일 스텝 탐욕 봇.
    weights / config를 주입해 평가 기준을 변경할 수 있다.
    """

    def __init__(
        self,
        tactical_weights: TacticalWeights | None = None,
        eval_weights:     EvalWeights     | None = None,
        attack_config:    AttackConfig    | None = None,
    ) -> None:
        self._adapter  = BotEnvAdapter()
        self._tactical = tactical_weights   # None → score_action 내부 기본값 사용
        self._eval     = eval_weights
        self._attack   = attack_config

    def pick_action(self, env: GameState) -> Action | None:
        """
        현재 환경에서 가장 높은 점수의 합법 행동을 반환한다.
        합법 행동이 없거나 게임이 종료된 상태면 None을 반환한다.
        """
        legal_actions = self._adapter.list_legal_actions(env)
        if not legal_actions:
            return None

        best_action: Action | None = None
        best_score = _NEG_INF

        for action in legal_actions:
            next_env, next_snap, lock_result = self._adapter.simulate_action_full(
                env, action
            )

            if action in (Action.HARD_DROP, Action.HOLD):
                summary = build_outcome_summary(next_env, next_snap, lock_result)
            else:
                # 이동·회전: 보드가 바뀌지 않으므로 이 위치에서 HARD_DROP한 결과를 평가
                if (
                    not self._adapter.is_terminal(next_env)
                    and next_env.active is not None
                ):
                    drop_env, drop_snap, drop_lock = self._adapter.simulate_action_full(
                        next_env, Action.HARD_DROP
                    )
                    summary = build_outcome_summary(drop_env, drop_snap, drop_lock)
                else:
                    summary = build_outcome_summary(next_env, next_snap, lock_result)

            sc = score_action(summary, self._tactical, self._eval, self._attack)
            if sc > best_score:
                best_score  = sc
                best_action = action

        return best_action
