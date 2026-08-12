# FPL AI Predictor — Full Build Plan

## Context

The user wants a personal, always-on decision-support tool for Fantasy Premier League: pick an optimal 15-man squad before the 2026/27 season starts, then every gameweek get data-driven transfer/captaincy/chip recommendations, running as a local Streamlit dashboard. Two hard requirements shape every choice below: it must be **entirely free** to build and operate (no paid APIs, no paid hosting), and it must be **accurate and strategically sound**, not a toy — it needs to reflect real FPL strategy (chip timing, fixture swings, differential vs template, transfer-hit tradeoffs) and needs to actually be trustworthy enough to use for real decisions all season.

The working directory is empty — this is a from-scratch build. Research during planning corrected one assumption in the original brief: the free FPL API *does* expose structured injury signals (`news`, `chance_of_playing_this_round/next_round`, `status`), so injury handling doesn't start from zero — it's layered on top of that baseline.

Decisions locked in with the user:
- **Live team sync**: the app reads the user's actual FPL squad via their public team ID (bank, free transfers, chip status), not just abstract recommendations.
- **News/injury sourcing**: RSS feeds first (zero ToS ambiguity), FPL API flags as baseline, light rate-limited/cached scraping of sites like Fantasy Football Scout / AllAboutFPL as a fallback for speed, plus a manual override box in the dashboard for anything automated sources miss. Scraping is kept low-frequency, cached, and never redistributes content — private single-user use only.
- **Automation**: Windows Task Scheduler runs the refresh pipeline locally on a schedule (not GitHub Actions) — user accepted the tradeoff that freshness is coupled to laptop uptime, mitigated by a "data last refreshed: X ago" indicator plus a manual refresh button.
- **User is comfortable with Python** — plan favors a real project structure with tests over maximal hand-holding.
- **Success metric is season-end performance, not single-GW accuracy**: the user's actual goal is winning (or being highly competitive in) their private mini-leagues with friends over the full season. Every recommendation layer must balance "what's best for the very next gameweek" against "what keeps the squad well-positioned to maximize points across the rest of the season" — greedy single-GW optimization alone is explicitly insufficient and is treated as a bug, not a simplification.

## Confirmed 2026/27 Season Rules (encode as config, not hardcoded literals — verify each season)

- Squad: 15 players, 2 GK / 5 DEF / 5 MID / 3 FWD. Budget £100.0m. Starting XI: 1 GK + valid formation (min 3 DEF, min 2 MID, min 1 FWD). Max 3 players per real-world club.
- Transfers: 1 free transfer/week, banks up to a max of **5** saved (raised from 2 in recent seasons; occasionally topped up further by mid-season rule changes, e.g. the AFCON top-up in 2025/26 — the pipeline should treat this as a config value re-checked each season, not an assumption baked into logic). Extra transfers beyond free ones cost **-4 pts** each.
- Chips: **two full sets** of Wildcard / Free Hit / Bench Boost / Triple Captain — one set per half-season. First-half chips must be used before the **GW19 deadline (13:30 GMT, Sat 2 Jan 2027)** or are lost; second-half set then unlocks. Only **one chip per gameweek**.
- Captain = 2x points (3x under Triple Captain); Vice-captain = 2x if captain doesn't play any minutes.

## Architecture

```
FPL-Agent/
  pyproject.toml
  README.md
  .env.example                     # FPL_TEAM_ID (public, not a secret)
  config/settings.yaml             # season, team_id, mini_league_ids, chip windows, free-transfer cap, cold-start GW threshold, refresh cadence

  data/
    raw/                           # immutable snapshots: bootstrap/, fixtures/, element_summary/{id}/, event_live/{gw}/
    external/                      # vaastav CSV mirror, understat/FBref scrape dumps
    processed/                     # cleaned parquet feature tables (model-ready, rebuilt idempotently)
    news/                          # RSS + scraped injury/news, raw + parsed
    db/fpl.db                      # SQLite: overrides, snapshots, run history

  models/
    artifacts/                     # LightGBM models, versioned by season/gw/position
    registry.json                  # active model per position + metadata

  src/fpl_agent/
    ingestion/
      fpl_api.py                   # bootstrap-static, fixtures, element-summary, event/live, entry/picks, leagues-classic standings
      vaastav_sync.py              # historical training data
      scrape_understat.py / scrape_fbref.py   # xG/xA underlying stats
      news_rss.py                  # RSS feeds (primary)
      news_scrape.py               # rate-limited, cached fallback scraping
      cache.py
    storage/
      db.py, schema.sql, repository.py, parquet_store.py
    features/
      build_features.py, rolling_stats.py, fixtures_features.py   # FDR, DGW/BGW detection
    models/
      train.py, predict.py, evaluate.py, cold_start.py            # pre-season prior model
    optimizer/
      constraints.py               # budget, 2-5-5-3, club cap, formation
      squad_optimizer.py           # CP-SAT: initial/wildcard squad
      transfer_optimizer.py        # weekly transfers, near-term horizon, hit tradeoff, guided by season_planner targets
      captaincy.py                 # argmax over chosen XI
      chip_strategy.py             # DGW/BGW + fixture-swing based chip timing signals
      season_planner.py            # long-horizon squad-shape/value/chip-window plan the weekly optimizer is guided by
    overrides/manager.py           # CRUD for manual injury/do-not-sell/lock flags (SQLite, precedence over automated data)
    pipeline/
      refresh_data.py              # entrypoint for Task Scheduler
      weekly_pipeline.py           # refresh -> features -> predict -> optimize -> cache; supports "replay mode" for a past GW
    ui/
      app.py
      pages/  (squad planner, transfers, captaincy, chip strategy, player explorer, overrides, settings)
    utils/logging.py, dates.py     # GW deadlines, chip-window logic

  scripts/
    run_refresh.ps1                # Task Scheduler entrypoint
    backtest_season.py

  tests/
    test_ingestion/ test_features/ test_optimizer/ test_models/
    fixtures/                      # frozen sample API responses / historical GW data
```

### Key technical decisions (validated during planning)
- **Environment**: create the venv from **Python 3.13**, not the globally installed 3.14 — ML packages (LightGBM, OR-Tools) and Streamlit lag behind brand-new CPython releases on Windows wheels; 3.14 risks missing prebuilt wheels and a forced C++ build toolchain. (Originally planned as 3.12, but 3.12 has moved to security-fixes-only maintenance and no longer ships Windows installers as of this build — 3.13 is now the mature, fully-supported prior-stable release, installed side-by-side via the official installer, registered with the `py` launcher as `py -3.13`, without touching the existing 3.14/Anaconda installs or system PATH.)
- **Storage — SQLite + Parquet hybrid, not one store**: SQLite (`data/db/fpl.db`) is the system of record for small, mutable, relational state — overrides, run history, point-in-time snapshots. Parquet is the system of record for large, append-mostly historical/feature data — native to pandas/LightGBM, faster and smaller than CSV. Raw API/scrape pulls stay as flat files (JSON/HTML) under `data/raw/` — immutable, diffable, no query engine needed.
- **Scheduling — Windows Task Scheduler**: runs `scripts/run_refresh.ps1` on a cadence (e.g. 3x/day baseline, denser hourly in the 24h window before a GW deadline via `utils/dates.py`). "Run task as soon as possible after a scheduled start is missed" stays on so a laptop that was off catches up on next boot. The Streamlit app never calls live APIs on open — it only reads the local store, with a "data last refreshed: X ago" indicator and a manual "Refresh now" button as the safety net.
- **Optimizer — Google OR-Tools CP-SAT over PuLP**: FPL squad/transfer selection is dominated by boolean/logical structure (selected/captain/starting-XI/club-cap/formation/chip-conditional scoring). CP-SAT's reification (`OnlyEnforceIf`) maps directly onto these rules and scales better for the multi-GW transfer-horizon planning the user wants than manually linearizing the same logic in PuLP. Both are free/pip-installable on Windows with no separate solver setup; this is a modeling-expressiveness choice, not a cost or install-friction one. Captaincy is a simple post-optimization argmax over the optimizer's chosen XI, not a separate ILP.
- **ML — LightGBM, 4 separate per-position models (GK/DEF/MID/FWD), single-GW-ahead target**: scoring rules differ enough by position (saves vs. clean sheets vs. goals) to warrant separate models. LightGBM chosen over XGBoost for native categorical handling (team/opponent/position) and faster iterative retraining through a season — a mild, not decisive, preference; both are free.
  - Features: bootstrap-static (cost, ownership%, form, `chance_of_playing_next_round`, ICT), element-summary rolling 3/5/10-GW windows (minutes, goals, assists, BPS, clean sheets, saves), fixtures/FDR + opponent strength + DGW/BGW flags, vaastav multi-season history, understat/FBref xG90/xA90/npxG.
  - Train/validate: strictly time-ordered splits (never shuffle gameweeks — leaks future rolling stats), walk-forward across seasons, retrain weekly using only data available strictly before the target GW.
  - **Cold start** (pre-season / early GWs, no current-season data yet): a separate, simpler `cold_start.py` baseline blending prior-season points-per-90 (transfer/team-change adjusted), underlying-stats regression-to-mean, opening-fixture difficulty, and a league-average prior for promoted clubs. Auto-switches to the full LightGBM model once a configurable number of current-season GWs exist, with an explicit "using pre-season priors until GW N" indicator in the UI.
- **Injury/news layer**: `news_rss.py` pulls official RSS feeds where available (BBC Sport, Sky Sports team news) as the zero-risk primary source; `fpl_api.py`'s own status fields are the always-available baseline; `news_scrape.py` is a slow, cached, low-frequency (capped, e.g. once per few hours), identifying-User-Agent fallback scraper for FFScout/AllAboutFPL injury pages, used only to fill gaps RSS doesn't cover. All three feed into a single `player_status` table; the manual overrides table always takes precedence and is visibly badged "manual" in the UI, and automated refreshes never silently overwrite a user override.
- **Live team sync**: `fpl_api.py` reads `entry/{team_id}/` and `entry/{team_id}/event/{gw}/picks/` (public endpoints, no auth) to pull the user's actual 15, bank, free-transfer count, and chip-used status each refresh, so `transfer_optimizer.py` operates on the real squad/budget rather than a hypothetical one.
- **Two-layer planning, not one greedy solver**: a long-horizon `season_planner.py` runs less frequently (e.g. after every GW result, or whenever fixtures/chip-window data changes materially) and produces a *strategic* target — a rolling squad-shape/value trajectory and planned chip windows for the rest of the season (or at minimum the rest of the current chip half, since chips are lost if unused by the GW19 cutoff). `transfer_optimizer.py` then runs every week and makes the *tactical* call, but its objective is a weighted blend of near-term predicted points **and** alignment with the season_planner's current target — e.g. it won't recommend selling a player the season plan is holding for a known upcoming Bench Boost DGW just because they look marginally suboptimal for the very next GW alone. This mirrors how strong human FPL managers actually play: a season-long shape with in-week tactical adjustments, not a fresh from-scratch optimization every Friday.
- **Mini-league / rival awareness**: since the user's actual goal is beating specific friends in their private mini-leagues (not just an abstract overall-rank score), `fpl_api.py` also pulls `leagues-classic/{league_id}/standings/` (public, no auth) for any mini-league IDs the user configures. This feeds a risk-posture signal into `season_planner.py`/`chip_strategy.py`: play safer/template picks when comfortably leading a mini-league late in the season (protect the lead, minimize variance), lean into differentials and more aggressive chip timing when chasing rivals with limited gameweeks left (need variance to close a points gap). Rival squads themselves aren't reliably fetchable in detail via the free API beyond what `entry/{team_id}` exposes for a given manager, so this stays rank/points-gap-driven rather than pretending to model rivals' exact transfers.

## FPL Strategy Logic to Encode (this is what makes it "solid," not just a squad-picker)

- **Season-long shape planning (strategic layer)**: `season_planner.py` maintains a rolling plan for the rest of the season (or rest of the current chip half): a target squad-value trajectory, which chip goes in which future window, and which "core" players the plan is building around vs. which are short-term/rotational. This is the layer that stops the agent from being a pure next-GW maximizer — every weekly tactical decision is checked against it, and the plan itself is recomputed after every GW's results and whenever new fixture/chip-window information arrives.
- **Transfer-hit tradeoff (tactical layer)**: `transfer_optimizer.py` objective folds in `-4 × num_hits_beyond_free`; only recommend a hit when the predicted point gain (summed over a near-term horizon, e.g. next 3-5 GWs, discounted for prediction uncertainty further out) **plus** its contribution to the season_planner's target shape clearly exceeds the hit cost — a transfer that helps this week but actively fights the long-term plan (e.g. sells a player earmarked as Triple Captain core for an upcoming DGW) is penalized even if it looks marginally positive in isolation.
- **Team value growth as a long-term asset**: squad value (not just bank) compounds a manager's options all season — selling a player right before a price rise or holding one through a price fall both cost future flexibility. `season_planner.py` tracks squad value trajectory and factors likely near-term price rises/falls (from ownership/transfer-momentum trends in the bootstrap-static data) into transfer timing, not just raw predicted points.
- **DGW/BGW detection**: `fixtures_features.py` flags gameweeks where a team has 0 or 2+ fixtures directly from the fixtures API — feeds both the predictor (summed/zeroed expected points), `chip_strategy.py`, and the season_planner's forward chip-window targets (Bench Boost/Triple Captain candidates need a strong DGW; Free Hit candidates need a personally-bad BGW).
- **Chip timing signals**: `chip_strategy.py` surfaces (not auto-plays) suggested windows, sourced from `season_planner.py`'s forward plan rather than decided week-by-week in isolation — Wildcard when squad value/fixture swings drift far from optimal or ahead of a fixture swing, Free Hit for a personal blank-heavy GW, Bench Boost when the bench itself has a strong DGW, Triple Captain on a nailed premium player with a DGW or standout fixture — always respecting the "one set per half, must-use-by-GW19" 2026/27 structure.
- **Fixture-swing awareness**: rolling N-GW forward FDR per team (not just next-GW) informs both transfer timing and wildcard-window suggestions.
- **Mini-league rank-aware risk posture**: using the pulled mini-league standings, the season_planner shifts template-vs-differential and chip-aggression posture based on the user's actual position relative to friends — e.g. more conservative/template-heavy when protecting a lead late in the season, more differential/variance-seeking when chasing with few gameweeks left. This is exposed as a visible signal in the UI ("currently 2nd in [league], X points behind 1st with Y GWs left — leaning differential"), not a hidden black-box adjustment.
- **Template vs. differential**: surface ownership% alongside predicted points so the user can see both "safe/template" and "differential" options per position, with a config toggle for how aggressively to lean into differentials (rank-chasing) vs. nailed-on template picks (rank-protecting) — this toggle is the manual override on top of the automatic mini-league-driven posture above, not a replacement for it.
- **Set-piece/nailed-on minutes risk**: fold `chance_of_playing_next_round` and minutes-trend volatility into a confidence/risk indicator per player alongside the raw point prediction, so two players with equal predicted points but different rotation risk are visibly distinguishable.
- Price-change prediction and detailed set-piece-taker tagging are explicitly **out of scope for v1** (no reliable free structured source without heavier scraping) — flagged as a v2 stretch, not silently dropped. The season_planner's price-trend awareness above uses ownership/transfer momentum already present in bootstrap-static, not a dedicated price-prediction model.

## Build Order — Full Phase-by-Phase Plan

Each phase lists concrete steps, its deliverable, and how to know it's actually done before moving on. Phases are sequential; later phases depend on earlier ones being real and working, not stubbed.

### Phase 0 — Environment + Skeleton
- Install Python 3.13 side-by-side with the existing global 3.14 (python.org installer, per-user, registered with the `py` launcher); create the project venv from 3.13 specifically via `py -3.13`. (3.12 was the original target but no longer ships Windows installers — see Key Technical Decisions above.)
- Scaffold the full directory tree from the Architecture section (`src/fpl_agent/...`, `data/`, `models/`, `scripts/`, `tests/`).
- `pyproject.toml` with dependencies pinned: `requests`, `pandas`, `pyarrow`, `lightgbm`, `ortools`, `streamlit`, `feedparser` (RSS), `beautifulsoup4`/`lxml` (scraping), `pytest`.
- `config/settings.yaml`: season string, `team_id` placeholder, chip-window dates, free-transfer cap, cold-start GW threshold, refresh cadence.
- `data/db/fpl.db` created from `storage/schema.sql` (empty tables: `player_status`, `overrides`, `snapshots`, `run_history`).
- **Deliverable / done when**: `python -m fpl_agent.storage.db --init` creates the DB with no errors; `pytest` runs (even with zero tests) inside the venv.
 
### Phase 1 — Data Layer (Live API + Historical)
- `ingestion/fpl_api.py`: typed fetchers for bootstrap-static, fixtures, `entry/{team_id}`, `entry/{team_id}/event/{gw}/picks`, `element-summary/{id}`, `event/{gw}/live`.
- `ingestion/cache.py`: writes every raw pull to `data/raw/<endpoint>/<timestamp>.json`, never overwritten.
- `storage/repository.py`: normalizes raw JSON into SQLite snapshot tables and parquet feature tables.
- `ingestion/vaastav_sync.py`: clones/pulls the vaastav historical CSVs into `data/external/`, converts to `data/processed/all_seasons.parquet`.
- Bare Streamlit page (`ui/pages/player_explorer.py` v0) that just tables the current players/prices from the local store, to visually confirm the pipe end to end.
- **Deliverable / done when**: running the refresh once populates `data/raw`, `data/processed`, and SQLite; the Streamlit table shows real current player data pulled from local storage, not the live API directly.

### Phase 2 — Rules Engine + Optimizer (Naive Predictor)
- `optimizer/constraints.py`: hardcode confirmed 2026/27 rules (budget £100m, 2-5-5-3, max 3/club, valid formation) as named constants sourced from `config/settings.yaml`, not magic numbers.
- Naive predictor: use last completed season's total points (or the cold-start heuristic if before the season starts) as the point estimate — no ML yet.
- `optimizer/squad_optimizer.py`: CP-SAT model that selects 15 players maximizing predicted points under all constraints, then a starting-XI/formation sub-solve.
- `optimizer/captaincy.py`: argmax over the chosen XI.
- Unit tests: small synthetic player pools (10-20 fake players) with a hand-calculable optimum; explicit tests that over-budget, >3-per-club, and invalid-formation inputs are correctly rejected as infeasible.
- **Deliverable / done when**: running the optimizer against real current player data produces a valid, rules-compliant 15-man squad + starting XI + captain, verified against the constraint tests.

### Phase 3 — Live Team Sync
- Extend `fpl_api.py`/`repository.py` to pull the user's actual squad, bank balance, free-transfer count, and chips-used status from their `team_id` each refresh.
- Wire the optimizer to operate on the real current squad + real bank instead of a from-scratch selection, for anything beyond the initial pre-season pick.
- **Deliverable / done when**: the dashboard can display the user's actual 15 players, correct bank, and correct free-transfer count, matching what the official FPL app shows.

### Phase 4 — Real Prediction Model
- `features/build_features.py` + `rolling_stats.py` + `fixtures_features.py`: build the full feature set (rolling form windows, FDR, DGW/BGW flags, ownership, underlying stats).
- `ingestion/scrape_understat.py` / `scrape_fbref.py`: pull xG/xA/npxG into `data/external/`.
- `models/train.py`, `predict.py`, `cold_start.py`: train 4 per-position LightGBM models with strict time-ordered splits; implement the cold-start blend for pre-season/early GWs with the "using pre-season priors until GW N" UI flag.
- `scripts/backtest_season.py`: retrain-and-predict walk-forward across a completed past season; report MAE/RMSE + rank correlation, and cumulative points vs. baselines (always-captain-most-owned, never-transfer, FPL average-manager score).
- **Deliverable / done when**: backtest results are reviewed and judged good enough to trust (beats the naive baselines on rank correlation and cumulative points) before this model replaces the naive one in the optimizer.

### Phase 5 — Weekly Transfer + Captaincy Logic
- `optimizer/transfer_optimizer.py`: given current squad + multi-GW-ahead predictions, search 0/1/2-transfer moves, fold `-4 × hits_beyond_free` into the objective, only recommend a hit when point gain over the configured horizon clearly exceeds it.
- Extend `captaincy.py` with a confidence/risk indicator (from `chance_of_playing_next_round` + minutes-trend volatility) alongside the raw predicted-points ranking.
- **Deliverable / done when**: replay mode (see Phase 8) against a past GW produces a transfer recommendation with visible reasoning ("captain X over Y because...", "hit not worth it because...") that a human can sanity-check against what actually happened.

### Phase 6 — Injury/News Layer
- `ingestion/news_rss.py`: pull official team-news RSS feeds (BBC Sport, Sky Sports) as the primary, zero-risk source.
- `ingestion/news_scrape.py`: slow, cached, rate-limited (e.g. once per few hours), identifying-User-Agent fallback scraper for FFScout/AllAboutFPL injury pages, filling gaps RSS misses.
- `overrides/manager.py`: SQLite-backed manual override CRUD (flag doubtful/out, "don't sell"), always taking precedence over automated `player_status`, visibly badged "manual" in the UI with an audit trail.
- **Deliverable / done when**: `player_status` reflects a blend of API flags + RSS + scrape fallback, manual overrides visibly override it, and automated refreshes never silently clobber a manual entry.

### Phase 7 — Season-Long Planning + Chip Strategy
- `optimizer/season_planner.py`: builds the rolling long-horizon plan — target squad-value trajectory, "core vs. rotational" player classification, and forward chip-window targets for the rest of the season/chip-half. Recomputes after each GW's results and on material fixture/chip-window changes.
- Extend `fpl_api.py` with `leagues-classic/{league_id}/standings/`; add `mini_league_ids` to `config/settings.yaml`; season_planner uses standings to set the mini-league rank-aware risk posture (template-heavy when leading, differential-heavy when chasing).
- `optimizer/chip_strategy.py`: DGW/BGW detection from fixtures data, rolling forward-FDR fixture-swing scoring, and rule-aware suggestion windows (respecting the 2026/27 "two sets, must-use-by-GW19" structure) for Wildcard/Free Hit/Bench Boost/Triple Captain, sourced from the season_planner's forward plan — surfaced as suggestions, never auto-played.
- Re-wire `transfer_optimizer.py` (Phase 5) so its objective also scores alignment with the season_planner's current target shape, not near-term predicted points alone.
- **Deliverable / done when**: replay mode against a past season's known good/bad chip weeks (e.g. a real double gameweek) produces a sensible chip suggestion for that week, **and** a side-by-side backtest shows the two-layer (season_planner + transfer_optimizer) approach beating a pure greedy-single-GW ablation on cumulative season points — proving the long-term layer earns its complexity rather than just asserting it helps.

### Phase 8 — Dashboard
- Full Streamlit pages: squad planner, transfers (with reasoning), captaincy, chip strategy, player explorer/comparison (predicted points + ownership% + risk, template vs. differential view), overrides, settings (team ID, differential-aggressiveness toggle).
- "Data last refreshed: X ago" staleness indicator + manual "Refresh now" button wired to `pipeline/refresh_data.py`.
- **Deliverable / done when**: the user can open the dashboard, see their real squad, and get a full weekly recommendation (transfers, captain, chip signal) with visible reasoning, without touching a terminal.

### Phase 9 — Automation
- `scripts/run_refresh.ps1` wrapping `pipeline/refresh_data.py`; register as a Windows Task Scheduler task (e.g. 8am/1pm/8pm daily, plus denser hourly triggers in the 24h window before a GW deadline computed from `utils/dates.py`); enable "run as soon as possible after a missed start."
- **Deliverable / done when**: the task runs unattended on schedule and the dashboard's staleness indicator confirms refreshes are landing without manual intervention.

### Phase 10 — Season-Long Validation
- Run `weekly_pipeline.py` in replay mode across several past completed gameweeks as a final end-to-end confidence check before relying on it for a live decision.
- Do a live smoke test once a real gameweek is upcoming: confirm the dashboard's live recommendation looks sane against the user's own judgment before the deadline, and periodically re-run this smoke test through the season as an API-schema-drift check.

## Verification

- **Unit tests with frozen fixtures** (`tests/fixtures/`: saved real API/CSV samples) — no live network calls in the regular suite. Cover ingestion parsing, feature leakage (GW *k* features must never see GW *k+1*+ data), and optimizer constraints (budget/club-cap/formation violations correctly rejected; small synthetic pools with a hand-calculable optimum).
- **Backtesting is the primary accuracy check**: `scripts/backtest_season.py` retrains using only data available up to GW *k*, predicts GW *k+1*, and compares against actual results (MAE/RMSE + rank correlation, since relative ranking drives decisions more than exact point values). Simulate what the full pipeline would have recommended across a completed season and compare cumulative points to simple baselines (always captain most-owned, never transfer, FPL's own average-manager score) — the most concrete evidence of real accuracy available without waiting on live deadlines.
- **Long-term-vs-greedy ablation**: because the whole point of the season_planner layer is that it should beat pure short-horizon optimization over a full season, `backtest_season.py` runs two variants across the same historical season — (1) `transfer_optimizer.py` alone, greedy next-GW/short-horizon only, and (2) the full two-layer pipeline with `season_planner.py` guiding it — and reports cumulative season-end points for both. The long-term layer only stays in the default recommendation path if it demonstrably beats the greedy-only baseline; if a backtest ever shows it doesn't, that's a signal to revisit the season_planner's weighting, not to ignore the result.
- **End-to-end replay mode**: `weekly_pipeline.py` accepts a historical GW instead of "now," running refresh→feature→predict→optimize→UI against frozen historical data — doubles as a test harness and a working demo before the live season data is trustworthy.
- **Live smoke test**: run the real pipeline against the live API manually once implemented, confirm the Streamlit dashboard renders the user's actual squad and a sane recommendation, and re-run periodically as a schema-drift check (FPL has changed field names before).

### Critical files
- `src/fpl_agent/ingestion/fpl_api.py`
- `src/fpl_agent/storage/db.py` (+ `schema.sql`)
- `src/fpl_agent/features/build_features.py`
- `src/fpl_agent/optimizer/squad_optimizer.py`, `transfer_optimizer.py`, `chip_strategy.py`, `season_planner.py`
- `src/fpl_agent/models/train.py`, `cold_start.py`
- `src/fpl_agent/pipeline/weekly_pipeline.py`
- `scripts/run_refresh.ps1`
