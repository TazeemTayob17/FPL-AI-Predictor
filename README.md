# FPL Agent

Personal FPL AI predictor: squad selection, weekly transfer/captaincy/chip recommendations,
running as a local Streamlit dashboard. Full design and phase-by-phase build plan live in
[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

## Setup

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\pip.exe install -e ".[dev]"
copy .env.example .env   # then fill in FPL_TEAM_ID
```

## Status

Phase 0 (environment + skeleton), Phase 1 (data layer), Phase 2 (rules engine + CP-SAT
optimizer with a naive last-season-points predictor), Phase 3 (live team sync), and Phase 4
(real prediction model) complete.

Run `python -m fpl_agent.pipeline.sync_team` to sync your real squad, bank, free transfers,
and chips once the FPL API has them (i.e. after that gameweek's deadline passes) - before
then, it automatically falls back to a recommended squad from
`python -m fpl_agent.pipeline.build_initial_squad`, for you to enter yourself.

Run `python -m fpl_agent.models.train` to (re)train the 4 per-position LightGBM models, and
`python scripts/backtest_season.py <season>` to walk-forward backtest them against a completed
season. `models/predict.py` auto-switches from cold-start priors to the trained model once
enough current-season gameweeks exist (`cold_start_gw_threshold` in `config/settings.yaml`).

See `docs/IMPLEMENTATION_PLAN.md` for the full phase-by-phase plan and current architecture
decisions.
