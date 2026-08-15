"""CLI: replay a past gameweek's transfer/captaincy recommendation. Logic lives in fpl_agent.pipeline.weekly_pipeline."""

from __future__ import annotations

import sys

from fpl_agent.pipeline.weekly_pipeline import print_replay_report, run_replay

if __name__ == "__main__":
    target_season = sys.argv[1] if len(sys.argv) > 1 else "2025-26"
    as_of_gw = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    print_replay_report(run_replay(target_season, as_of_gw))
