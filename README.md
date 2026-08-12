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

Phase 0 (environment + skeleton), Phase 1 (data layer), and Phase 2 (rules engine + CP-SAT
optimizer with a naive last-season-points predictor) complete. Run
`python -m fpl_agent.pipeline.build_initial_squad` to see a rules-compliant 15-man squad,
starting XI, and captain picked from real local player data. See `docs/IMPLEMENTATION_PLAN.md`
for the full phase-by-phase plan and current architecture decisions.
