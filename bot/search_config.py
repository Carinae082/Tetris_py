"""
bot/search_config.py — 빔 서치 설정 dataclass

fusion/search_config.rs 에 대응하는 Python 구현.
합성 점수 가중치(board/attack/chain/context)와 깊이 스케일링 상한을 포함한다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchConfig:
    """
    BeamSearchBot의 탐색 파라미터.

    탐색 파라미터
    -------------
    beam_width                : 매 depth마다 유지할 후보 노드 수
    depth                     : 최대 탐색 깊이 (배치 단위)
    futility_delta            : 즉시 점수가 최고 후보보다 이 값 이상 낮으면 가지치기
    time_budget_ms            : 탐색 시간 예산 (ms). None 이면 depth 기준으로만 탐색
    use_hold                  : 홀드 행동 허용 여부
    root_top_k                : 루트 깊이에서 유지할 최대 후보 수 (None → beam_width 그대로)
    quiescence_max_extensions : 정적 탐색 최대 추가 depth
    quiescence_beam_fraction  : 정적 탐색에 참여할 beam 상위 비율
    track_path                : True 이면 path_actions를 전체 저장.
                                False(기본값) 이면 root_action만 저장한다.

    합성 점수 가중치 (fusion/search_config.rs 대응)
    -----------------------------------------------
    board_weight     : 보드 형태 점수 가중치
    attack_weight    : 공격력(누적, 깊이 스케일) 가중치
    chain_weight     : 체인 가치(누적, 깊이 스케일) 가중치
    context_weight   : coaching context 점수 가중치
    max_depth_factor : sqrt(depth) 스케일링 상한 (= 2.45 in fusion)
    """

    # ── 탐색 파라미터 (fusion/search_config.rs default 기준) ─────────────────
    beam_width:                int          = 800
    depth:                     int          = 14
    futility_delta:            float        = 15.0
    time_budget_ms:            float | None = None
    use_hold:                  bool         = True
    root_top_k:                int | None   = None
    quiescence_max_extensions: int          = 3
    quiescence_beam_fraction:  float        = 0.15
    track_path:                bool         = False

    # ── 합성 점수 가중치 (fusion 기본값) ─────────────────────────────────────
    board_weight:     float = 1.0
    attack_weight:    float = 0.50
    chain_weight:     float = 0.15
    context_weight:   float = 0.10
    max_depth_factor: float = 2.45

    # ── 미리 정의된 preset ────────────────────────────────────────────────────

    @classmethod
    def safe_default(cls) -> SearchConfig:
        """
        실시간 플레이용 기본 설정.
        fusion 기본값(beam=800, depth=14)에 35 ms 시간 예산을 적용해
        Python 성능 한계 내에서 최대한 깊이 탐색한다.
        """
        return cls(
            beam_width                = 800,
            depth                     = 14,
            futility_delta            = 15.0,
            time_budget_ms            = 35.0,
            use_hold                  = True,
            root_top_k                = None,
            quiescence_max_extensions = 3,
            quiescence_beam_fraction  = 0.15,
        )

    @classmethod
    def fast_play(cls) -> SearchConfig:
        """
        빠른 플레이용 설정 (20 ms 예산, 빔 200).
        시간 압박이 큰 환경에서 beam_width를 줄여 반응 속도를 높인다.
        """
        return cls(
            beam_width                = 200,
            depth                     = 14,
            futility_delta            = 20.0,
            time_budget_ms            = 20.0,
            use_hold                  = True,
            root_top_k                = None,
            quiescence_max_extensions = 2,
            quiescence_beam_fraction  = 0.15,
        )

    @classmethod
    def deeper_play(cls) -> SearchConfig:
        """
        품질 우선 설정 (100 ms 예산, fusion 기본값 그대로).
        대국 분석이나 시간 여유가 있는 환경에 적합하다.
        """
        return cls(
            beam_width                = 800,
            depth                     = 14,
            futility_delta            = 12.0,
            time_budget_ms            = 100.0,
            use_hold                  = True,
            root_top_k                = None,
            quiescence_max_extensions = 3,
            quiescence_beam_fraction  = 0.15,
        )
