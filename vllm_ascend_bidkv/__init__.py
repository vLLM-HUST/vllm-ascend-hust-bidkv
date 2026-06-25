# SPDX-License-Identifier: Apache-2.0
"""
BidKV: Utility-based victim selection plugin for vLLM Ascend.

Provides ``BidkvVictimSelector`` that implements the vLLM Ascend
victim selector protocol with utility-based ranking:

    U = r / (delta + epsilon)

where delta = 1 + w_c * completion + w_p * preemptions.

Install this plugin alongside vllm-ascend-hust, and it will be
auto-discovered via the ``vllm_ascend.victim_selector`` entry-point.
"""

from vllm_ascend_bidkv.selector import (
    BidkvVictimSelector,
    UtilityCandidateScore,
    UtilityVictimSelectorConfig,
)

__all__ = [
    "BidkvVictimSelector",
    "UtilityCandidateScore",
    "UtilityVictimSelectorConfig",
]
