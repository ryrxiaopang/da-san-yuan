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

-- Position of each hand in a full game (empty for older independent-hand runs).
ALTER TABLE hands ADD COLUMN IF NOT EXISTS game         integer;   -- 1, 2, 3 ...
ALTER TABLE hands ADD COLUMN IF NOT EXISTS dealer_no    smallint;  -- 1-4: the round's first to fourth dealer
ALTER TABLE hands ADD COLUMN IF NOT EXISTS repeat_no    smallint;  -- 0 = dealer's first hand, 1 = first repeat ...
ALTER TABLE hands ADD COLUMN IF NOT EXISTS hand_in_game smallint;  -- 1, 2, 3 ... counting every hand of the game

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

-- Claims spelled out (added Oct 2026; reload a run with --replace to fill them).
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS claim_tile text;      -- tile on offer when reacting to a discard / added kong
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS claim_from smallint;  -- seat that threw it
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS meld       text;      -- set formed: '777z' pong, '345m' chow, '2222s' kong
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS options    text;      -- claims that were possible: 'win, pong, chow'
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS move       text;      -- plain language, e.g. 'Chow 345m using 4m thrown by Seat 0'

-- Expected points of each move by Monte Carlo rollouts (`dsy evaluate`, db/load_move_values.py).
CREATE TABLE IF NOT EXISTS move_values (
    run_id       integer NOT NULL,
    hand_id      integer NOT NULL,
    idx          smallint NOT NULL,           -- same numbering as decisions.idx
    worlds       smallint NOT NULL,           -- sampled worlds per move
    oracle       boolean NOT NULL,            -- true: real hidden tiles; false: re-dealt (fair)
    chosen_ev    real NOT NULL,               -- expected points of the move played
    best_action  smallint NOT NULL,
    best_move    text NOT NULL,
    best_ev      real NOT NULL,
    regret       real NOT NULL,               -- best_ev - chosen_ev (0 = played the best move)
    regret_se    real NOT NULL,               -- standard error of the regret
    hand_points  integer NOT NULL,            -- what the player actually scored in the hand
    all_moves    text NOT NULL,               -- every legal move with its expected points
    PRIMARY KEY (run_id, hand_id, idx, oracle)
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

-- One easy-to-read row per hand: where it sits in the game (Game 1, East round,
-- Dealer 1, Repeat 0), players named by seat wind and play style, e.g. "South (Balanced)".
-- Same layout as tools/readable_hands.py. Filter with WHERE run_id = ...
DROP VIEW IF EXISTS v_hands_readable;
CREATE VIEW v_hands_readable AS
WITH seats AS (
    SELECT run_id, hand_id, seat, seat_wind, points,
           (ARRAY['East','South','West','North'])[seat_wind + 1] AS wind_name,
           CASE bot WHEN 'fast' THEN 'Fast' WHEN 'high_tai' THEN 'High-tai'
                    WHEN 'defensive' THEN 'Defensive' WHEN 'balanced' THEN 'Balanced'
                    ELSE bot END AS style
    FROM hand_seats
),
pats AS (
    SELECT run_id, hand_id,
           string_agg(CASE WHEN n = 1 THEN pattern || ' +' || t
                           ELSE pattern || ' x' || n || ' +' || t END,
                      ', ' ORDER BY t DESC, pattern) AS how
    FROM (SELECT run_id, hand_id, pattern, count(*) AS n, sum(tai) AS t
          FROM hand_patterns GROUP BY run_id, hand_id, pattern) x
    GROUP BY run_id, hand_id
),
pos AS (
    -- Older runs (dealer passed every hand) have no labels: derive them.
    SELECT run_id, hand_id,
           coalesce(game, hand_id / 16 + 1)                 AS game,
           coalesce(dealer_no, (hand_id % 4) + 1)           AS dealer_no,
           coalesce(repeat_no, 0)                           AS repeat_no,
           coalesce(hand_in_game, (hand_id % 16) + 1)       AS hand_in_game
    FROM hands
)
SELECT h.run_id,
       pos.game                                                          AS "Game",
       (ARRAY['East','South','West','North'])[h.prevailing + 1] || ' round' AS "Round",
       pos.dealer_no                                                     AS "Dealer no.",
       pos.repeat_no                                                     AS "Repeat",
       pos.hand_in_game                                                  AS "Hand in game",
       'Game ' || pos.game || ', ' || (ARRAY['East','South','West','North'])[h.prevailing + 1]
           || ' round, Dealer ' || pos.dealer_no
           || CASE WHEN pos.repeat_no > 0 THEN ', Repeat ' || pos.repeat_no ELSE '' END AS "Hand",
       max(s.style) FILTER (WHERE s.seat_wind = 0)                       AS "East player (dealer)",
       max(s.style) FILTER (WHERE s.seat_wind = 1)                       AS "South player",
       max(s.style) FILTER (WHERE s.seat_wind = 2)                       AS "West player",
       max(s.style) FILTER (WHERE s.seat_wind = 3)                       AS "North player",
       CASE WHEN h.winner IS NULL THEN 'Draw (no winner)'
            WHEN h.self_draw THEN 'Self-drawn win'
            ELSE 'Won on a discard' END                                  AS "Result",
       coalesce(max(s.wind_name || ' (' || s.style || ')')
                FILTER (WHERE s.seat = h.winner), '-')                   AS "Winner",
       CASE WHEN h.winner IS NULL THEN '-'
            WHEN h.self_draw THEN 'Nobody (self-drawn)'
            ELSE max(s.wind_name || ' (' || s.style || ')')
                 FILTER (WHERE s.seat = h.discarder) END                 AS "Threw the winning tile",
       h.tai                                                             AS "Tai",
       h.raw_tai                                                         AS "Tai before 5-tai cap",
       coalesce(p.how, '')                                               AS "Where the tai came from",
       max(s.points) FILTER (WHERE s.seat_wind = 0)                      AS "East points",
       max(s.points) FILTER (WHERE s.seat_wind = 1)                      AS "South points",
       max(s.points) FILTER (WHERE s.seat_wind = 2)                      AS "West points",
       max(s.points) FILTER (WHERE s.seat_wind = 3)                      AS "North points",
       h.turns                                                           AS "Turns",
       h.hand_id                                                         AS hand_id
FROM hands h
JOIN pos USING (run_id, hand_id)
JOIN seats s USING (run_id, hand_id)
LEFT JOIN pats p USING (run_id, hand_id)
GROUP BY h.run_id, h.hand_id, h.prevailing, h.winner, h.discarder, h.self_draw,
         h.tai, h.raw_tai, h.turns, p.how, pos.game, pos.dealer_no, pos.repeat_no, pos.hand_in_game;


-- Every move in plain language, with the expected value of the move when it has been evaluated.
DROP VIEW IF EXISTS v_moves_readable;
CREATE VIEW v_moves_readable AS
SELECT d.run_id,
       h.game                                                          AS "Game",
       h.hand_in_game                                                  AS "Hand in game",
       (ARRAY['East','South','West','North'])[h.prevailing + 1] || ' round' AS "Round",
       d.idx + 1                                                       AS "Move no.",
       'Seat ' || d.seat || ' (' ||
         (ARRAY['East','South','West','North'])[((d.seat - h.dealer + 4) % 4) + 1] || ')' AS "Player",
       d.bot                                                           AS "Style",
       CASE d.phase WHEN 'own_turn' THEN 'Own turn'
                    WHEN 'claim'    THEN 'Reacting to a discard'
                    ELSE 'Reacting to an added kong' END               AS "Situation",
       d.move                                                          AS "Move",
       d.action_type                                                   AS "Move type",
       d.claim_tile                                                    AS "Tile on offer",
       CASE WHEN d.claim_from IS NULL THEN NULL ELSE 'Seat ' || d.claim_from || ' (' ||
         (ARRAY['East','South','West','North'])[((d.claim_from - h.dealer + 4) % 4) + 1] || ')' END AS "Thrown by",
       d.meld                                                          AS "Set formed",
       d.options                                                       AS "Could have",
       d.draws_left                                                    AS "Tiles left to draw",
       d.own_shanten                                                   AS "Tiles from ready",
       d.opp_ready                                                     AS "Opponents ready",
       d.deal_in_tai                                                   AS "Tai given away",
       round(v.chosen_ev::numeric, 2)                                  AS "Expected points of move",
       v.best_move                                                     AS "Best move",
       round(v.best_ev::numeric, 2)                                    AS "Expected points of best",
       round(v.regret::numeric, 2)                                     AS "Points lost vs best",
       v.hand_points                                                   AS "Points scored in hand",
       d.hand_id
FROM decisions d
JOIN hands h USING (run_id, hand_id)
LEFT JOIN move_values v ON v.run_id = d.run_id AND v.hand_id = d.hand_id AND v.idx = d.idx AND NOT v.oracle;

-- Move quality by style and situation, from evaluated moves (fair mode).
-- "Best or near-best" = within two standard errors of the best move's expected points.
DROP VIEW IF EXISTS v_move_quality;
CREATE VIEW v_move_quality AS
SELECT d.run_id,
       d.bot AS style,
       CASE WHEN d.phase = 'own_turn' AND d.action < 34 AND d.opp_ready > 0 THEN 'discard, an opponent is ready'
            WHEN d.phase = 'own_turn' AND d.action < 34                     THEN 'discard, nobody ready'
            WHEN d.phase = 'own_turn' AND d.action = 68                     THEN 'self-drawn win'
            WHEN d.phase = 'own_turn'                                        THEN 'kong'
            WHEN d.options LIKE '%win%'                                      THEN 'could win on a discard'
            WHEN d.options LIKE '%pong%'                                     THEN 'could pong'
            WHEN d.options LIKE '%chow%'                                     THEN 'could chow'
            ELSE 'other claim' END                                          AS situation,
       count(*)                                                             AS moves,
       round(avg(v.chosen_ev)::numeric, 2)                                  AS avg_expected_points,
       round(avg(v.regret)::numeric, 2)                                     AS avg_points_lost,
       round(100.0 * avg((v.regret <= 2 * v.regret_se)::int), 1)            AS pct_best_or_near,
       round(100.0 * avg((v.regret > 2 * v.regret_se AND v.regret >= 4)::int), 1) AS pct_clear_mistakes
FROM decisions d
JOIN move_values v ON v.run_id = d.run_id AND v.hand_id = d.hand_id AND v.idx = d.idx AND NOT v.oracle
GROUP BY 1, 2, 3;
