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
optimizer with a naive last-season-points predictor), Phase 3 (live team sync), Phase 4
(real prediction model), Phase 5 (weekly transfer + captaincy logic), and Phase 6
(injury/news layer) complete.

Run `python -m fpl_agent.pipeline.sync_team` to sync your real squad, bank, free transfers,
and chips once the FPL API has them (i.e. after that gameweek's deadline passes) - before
then, it automatically falls back to a recommended squad from
`python -m fpl_agent.pipeline.build_initial_squad`, for you to enter yourself.

Run `python -m fpl_agent.models.train` to (re)train the 4 per-position LightGBM models, and
`python scripts/backtest_season.py <season>` to walk-forward backtest them against a completed
season. `models/predict.py` auto-switches from cold-start priors to the trained model once
enough current-season gameweeks exist (`cold_start_gw_threshold` in `config/settings.yaml`).

`optimizer/transfer_optimizer.py` searches 0-2 transfer moves per gameweek, folding the -4
hit cost into the objective and explaining every recommendation in plain language (including
"hit not worth it" when it isn't). `optimizer/captaincy.py` now also attaches a risk score
(injury doubt + minutes volatility) alongside the predicted-points ranking. Run
`python scripts/replay_gameweek.py <season> <gameweek>` to see both in action against a real
past gameweek, with the actual outcome shown alongside for sanity-checking.

`ingestion/news_rss.py` pulls BBC Sport's per-team feeds (all 20 clubs) plus Sky Sports'
general football feed, matching player mentions by name. `ingestion/news_scrape.py` is a
rate-limited (3h), cached fallback scraper for Fantasy Football Scout's injury table.
`overrides/manager.py` provides manual override CRUD (injury flags, "don't sell") that always
takes precedence over automated data, soft-deleted for a full audit trail, and visibly badged
"(manual)" on the Player Explorer page - automated refreshes never touch the overrides table.

See `docs/IMPLEMENTATION_PLAN.md` for the full phase-by-phase plan and current architecture
decisions.
