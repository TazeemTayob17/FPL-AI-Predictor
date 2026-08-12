PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL,
    applied_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Populated automatically; never written to by the UI (see overrides table for user input)
CREATE TABLE IF NOT EXISTS player_status (
    player_id                      INTEGER PRIMARY KEY,
    status                         TEXT,       -- FPL codes: a=available, d=doubtful, i=injured, s=suspended, u=unavailable
    news                           TEXT,
    news_added                     TEXT,
    chance_of_playing_this_round   INTEGER,
    chance_of_playing_next_round   INTEGER,
    source                         TEXT NOT NULL,   -- fpl_api | rss | scrape
    updated_at                     TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Always takes precedence over player_status; automated refreshes must never overwrite these
CREATE TABLE IF NOT EXISTS overrides (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id   INTEGER NOT NULL,
    field       TEXT NOT NULL,
    value       TEXT NOT NULL,
    reason      TEXT,
    gw_scope    INTEGER,            -- null = applies until removed
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

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

CREATE TABLE IF NOT EXISTS team_snapshot (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    pulled_at           TEXT NOT NULL DEFAULT (datetime('now')),
    gameweek            INTEGER,
    bank                REAL,
    squad_value         REAL,
    free_transfers      INTEGER,
    chips_used_json      TEXT,   -- JSON: [{chip, gameweek}]
    picks_json          TEXT    -- JSON: [{player_id, is_captain, is_vice_captain, multiplier, position}]
);

CREATE TABLE IF NOT EXISTS run_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type    TEXT NOT NULL,   -- refresh | train | predict | optimize
    started_at  TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,
    status      TEXT,            -- success | failed
    details     TEXT
);
