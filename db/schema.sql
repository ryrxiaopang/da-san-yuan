-- da-san-yuan self-play database (PostgreSQL 13+).
-- Holds hand-level and decision-level summaries for analysis.
-- The full observation arrays used for training stay in the .npy shards;
-- runs.name / hands.hand_id link the two.
--
-- Apply with:  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q -f db/schema.sql
-- Safe to re-run: tables use IF NOT EXISTS and views are replaced.

CREATE TABLE IF NOT EXISTS runs (
    run_id         serial PRIMARY KEY,
    name           text NOT NULL UNIQUE,          -- usually the output folder name
    seed           bigint NOT NULL,
    hands          integer NOT NULL,
    random_lineup  boolean NOT NULL,
    obs_version    integer NOT NULL,
    engine_commit  text,                          -- git commit the data was made with
    shard_path     text,                          -- where the .npy shards live
    bots           jsonb NOT NULL,                -- [{bot_id, name, speed, value, ...}]
    notes          text,
    loaded_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS hands (
    run_id      integer NOT NULL REFERENCES runs ON DELETE CASCADE,
    hand_id     integer NOT NULL,                 -- unique within a run, matches meta.npy
    seed        numeric(20,0) NOT NULL,           -- u64 deal seed: Game::new(cfg, seed) replays it
    dealer      smallint NOT NULL,                -- seat 0-3 that sits East
    prevailing  smallint NOT NULL,                -- 0=E 1=S 2=W 3=N
    winner      smallint,                         -- NULL = draw
    discarder   smallint,                         -- NULL = self-draw or draw
    self_draw   boolean NOT NULL,
    tai         smallint NOT NULL,                -- after the 5-tai cap (0 on a draw)
    raw_tai     smallint NOT NULL,                -- before the cap
    turns       integer NOT NULL,
    PRIMARY KEY (run_id, hand_id)
);

CREATE TABLE IF NOT EXISTS hand_seats (
    run_id     integer NOT NULL,
    hand_id    integer NOT NULL,
    seat       smallint NOT NULL,                 -- absolute seat 0-3
    seat_wind  smallint NOT NULL,                 -- 0=E 1=S 2=W 3=N
    bot        text NOT NULL,                     -- style name
    points     integer NOT NULL,                  -- tai-point payments only
    won        boolean NOT NULL,
    dealt_in   boolean NOT NULL,
    PRIMARY KEY (run_id, hand_id, seat),
    FOREIGN KEY (run_id, hand_id) REFERENCES hands ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS hand_patterns (
    run_id   integer NOT NULL,
    hand_id  integer NOT NULL,
    pattern  text NOT NULL,                       -- e.g. 'Dragon pong'
    tai      smallint NOT NULL,
    FOREIGN KEY (run_id, hand_id) REFERENCES hands ON DELETE CASCADE
);

-- One row per decision a bot made. Optional (loader --no-decisions); about 60 rows per hand.
CREATE TABLE IF NOT EXISTS decisions (
    run_id           integer NOT NULL,
    hand_id          integer NOT NULL,
    idx              smallint NOT NULL,           -- decision number within the hand
    seat             smallint NOT NULL,
    bot              text NOT NULL,
    phase            text NOT NULL,               -- own_turn | claim | rob_kong
    action           smallint NOT NULL,           -- 0-75, see README
    action_type      text NOT NULL,               -- discard | kong | tsumo | ron | pong | exposed_kong | chow | pass
    tile             text,                        -- '5p', '7z' ... for discards / kongs, else NULL
    n_legal          smallint NOT NULL,           -- number of legal actions
    draws_left       smallint NOT NULL,
    own_shanten      smallint NOT NULL,           -- before acting; 0 = ready
    opp_ready        smallint NOT NULL,           -- opponents that are ready (exact, hidden info)
    unsafe_options   smallint,                    -- legal discards that would deal in (discard decisions only)
    safe_options     smallint,                    -- legal discards that would not
    deal_in_tai      smallint,                    -- tai the chosen discard would give away (0 = safe)
    PRIMARY KEY (run_id, hand_id, idx),
    FOREIGN KEY (run_id, hand_id) REFERENCES hands ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS hand_seats_bot_idx    ON hand_seats (run_id, bot);
CREATE INDEX IF NOT EXISTS hand_patterns_idx     ON hand_patterns (run_id, pattern);
CREATE INDEX IF NOT EXISTS decisions_bot_idx     ON decisions (run_id, bot, action_type);

-- ------------------------------------------------------------------ views

-- Results per bot style. Seats are drawn at random, so compare styles with
-- `dsy tournament` (duplicate format) before drawing strong conclusions.
CREATE OR REPLACE VIEW v_style_summary AS
SELECT s.run_id,
       s.bot,
       count(*)                                         AS seat_hands,
       round(avg(s.points)::numeric, 3)                 AS points_per_hand,
       round((stddev_samp(s.points) / sqrt(count(*)))::numeric, 3) AS points_se,
       round(100.0 * avg(s.won::int), 1)                AS win_pct,
       round(100.0 * avg((s.won AND h.self_draw)::int), 1) AS self_draw_pct,
       round(100.0 * avg(s.dealt_in::int), 1)           AS deal_in_pct,
       round(avg(h.tai) FILTER (WHERE s.won), 2)        AS avg_tai_when_winning
FROM hand_seats s
JOIN hands h USING (run_id, hand_id)
GROUP BY s.run_id, s.bot;

-- How often each tai pattern appears in winning hands.
CREATE OR REPLACE VIEW v_pattern_frequency AS
SELECT p.run_id,
       p.pattern,
       count(DISTINCT p.hand_id)                        AS winning_hands,
       round(100.0 * count(DISTINCT p.hand_id)
             / (SELECT count(*) FROM hands h WHERE h.run_id = p.run_id AND h.winner IS NOT NULL), 2)
                                                        AS pct_of_wins
FROM hand_patterns p
GROUP BY p.run_id, p.pattern;

-- Defence: discards made while at least one legal discard could deal in.
-- random_pct = chance a uniformly random legal discard would have dealt in.
CREATE OR REPLACE VIEW v_defence AS
SELECT run_id,
       bot,
       count(*)                                                     AS risky_discards,
       round(100.0 * avg((deal_in_tai > 0)::int), 2)                AS chose_deal_in_pct,
       round(100.0 * avg(unsafe_options::numeric / (unsafe_options + safe_options)), 2) AS random_pct
FROM decisions
WHERE action_type = 'discard' AND unsafe_options > 0
GROUP BY run_id, bot;
