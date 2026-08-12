-- FPL Agent SQLite schema.
-- SQLite is the system of record for small, mutable, relational state only -
-- bulk historical/feature data lives in parquet under data/processed/, not here.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL,
    applied_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Latest merged automated view of each player's availability, one row per player_id.
-- Populated by the FPL API baseline + RSS + scrape fallback (Phase 6). Never written to by the UI.
CREATE TABLE IF NOT EXISTS player_status (
    player_id                      INTEGER PRIMARY KEY,
    status                         TEXT,       -- FPL's own code: 'a' available, 'd' doubtful, 'i' injured, 's' suspended, 'u' unavailable
    news                           TEXT,
    news_added                     TEXT,
    chance_of_playing_this_round   INTEGER,
    chance_of_playing_next_round   INTEGER,
    source                         TEXT NOT NULL,   -- 'fpl_api' | 'rss' | 'scrape'
    updated_at                     TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Manual overrides always take precedence over player_status at feature-build/optimizer time.
-- Kept as its own table (not merged into player_status) so automated refreshes can never
-- silently clobber a user-entered flag - the UI badges these visibly as "manual".
CREATE TABLE IF NOT EXISTS overrides (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id   INTEGER NOT NULL,
    field       TEXT NOT NULL,      -- e.g. 'availability', 'do_not_sell', 'note'
    value       TEXT NOT NULL,
    reason      TEXT,
    gw_scope    INTEGER,            -- NULL = applies until removed; otherwise a single gameweek
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Point-in-time snapshot of key per-player fields on every refresh pull - the input for
-- price-trend/ownership-momentum awareness (season_planner.py) and a debuggable history
-- of what the pipeline actually saw at each point in the season.
CREATE TABLE IF NOT EXISTS snapshots (
    id                              INTEGER PRIMARY KEY AUTOINCREMENT,
    pulled_at                       TEXT NOT NULL DEFAULT (datetime('now')),
    gameweek                        INTEGER,
    player_id                       INTEGER NOT NULL,
    now_cost                        INTEGER,   -- tenths of a million, as returned by the FPL API
    selected_by_percent             REAL,
    form                            REAL,
    total_points                    INTEGER,
    transfers_in_event               INTEGER,
    transfers_out_event              INTEGER
);
CREATE INDEX IF NOT EXISTS idx_snapshots_player_time ON snapshots (player_id, pulled_at);

-- Live team sync (Phase 3): the user's actual squad/bank/chip state as of each refresh.
CREATE TABLE IF NOT EXISTS team_snapshot (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    pulled_at           TEXT NOT NULL DEFAULT (datetime('now')),
    gameweek            INTEGER,
    bank                REAL,
    squad_value         REAL,
    free_transfers      INTEGER,
    chips_used_json      TEXT,   -- JSON list of {chip, gameweek} already played this season
    picks_json          TEXT    -- JSON list of {player_id, is_captain, is_vice_captain, multiplier, position}
);

-- Log of every pipeline run (refresh / train / predict / optimize), for debugging and the
-- dashboard's "data last refreshed: X ago" staleness indicator.
CREATE TABLE IF NOT EXISTS run_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type    TEXT NOT NULL,   -- 'refresh' | 'train' | 'predict' | 'optimize'
    started_at  TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,
    status      TEXT,            -- 'success' | 'failed'
    details     TEXT
);
