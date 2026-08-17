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

## Using it

**Daily update** (run this once a day, or whenever you want fresh predictions before making a
decision): `python scripts/update_now.py` - fetches fresh data, computes the full weekly
recommendation, caches it, and prints a terminal summary. Per-player match-history fetches
reuse the same day's data instead of re-pulling everything on every run.

**Dashboard**: double-click `run_dashboard.bat`, or run `streamlit run src/fpl_agent/ui/app.py`
yourself, then open http://localhost:8501. Pages:
Squad Planner, Transfers (with reasoning), Captaincy, Chip Strategy, Player Explorer (predicted
points, ownership%, template vs. differential), Overrides, and Settings. The dashboard only
ever reads the cache `update_now.py` populates - it never calls the live FPL API on page open -
and shows a "data last refreshed: X ago" indicator with its own manual "Refresh now" button.

Before your squad exists on FPL's live API (i.e. before that gameweek's deadline has passed),
both of the above fall back to a recommended squad for you to enter yourself.

## Status

Phases 0-9 complete: environment/skeleton, data layer, rules engine + CP-SAT optimizer, live
team sync, real LightGBM prediction model (cold-start-aware), weekly transfer/captaincy logic,
injury/news layer, season-long planning + chip strategy + mini-league awareness, the full
Streamlit dashboard, and a manual daily-update entrypoint (scheduled automation was deliberately
descoped in favor of this - see the plan doc for why). Phase 10 (season-long validation): the
replay-mode half is done; the live-squad smoke test half is pending the first gameweek deadline.

## Other tools

Run `python -m fpl_agent.models.train` to (re)train the 4 per-position LightGBM models, and
`python scripts/backtest_season.py <season>` to walk-forward backtest them against a completed
season. `models/predict.py` auto-switches from cold-start priors to the trained model once
enough current-season gameweeks exist (`cold_start_gw_threshold` in `config/settings.yaml`).

`optimizer/transfer_optimizer.py` searches 0-2 transfer moves per gameweek, folding the -4
hit cost into the objective and explaining every recommendation in plain language (including
"hit not worth it" when it isn't). `optimizer/captaincy.py` also attaches a risk score
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
