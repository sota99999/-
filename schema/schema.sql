-- =============================================================================
-- 競馬予想データベース スキーマ (SQLite)
--
-- netkeiba 等からスクレイピングしたデータを格納する前提の設計。
-- 中心となるのは「出走馬1頭1レース = 1行」の results テーブル（ファクト表）で、
-- races / horses / jockeys / trainers などのマスタ表を外部キーで参照する。
--
-- 使い方:
--   sqlite3 keiba.db < schema/schema.sql
-- =============================================================================

PRAGMA foreign_keys = ON;      -- 外部キー制約を有効化（接続ごとに毎回必要）
PRAGMA journal_mode = WAL;     -- 書き込み中の読み取り性能を改善

-- -----------------------------------------------------------------------------
-- マスタ: 競馬場
--   netkeiba のレースIDに含まれる場コード(01=札幌 .. 10=小倉)を想定。
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS venues (
    venue_id   TEXT PRIMARY KEY,            -- 場コード ("05"=東京 など)
    name       TEXT NOT NULL,               -- 競馬場名 (東京, 中山, ...)
    category   TEXT NOT NULL DEFAULT 'JRA'  -- 'JRA' / 'NAR'(地方)
        CHECK (category IN ('JRA', 'NAR'))
);

-- -----------------------------------------------------------------------------
-- マスタ: 騎手
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS jockeys (
    jockey_id  TEXT PRIMARY KEY,   -- netkeiba の騎手ID
    name       TEXT NOT NULL,
    affiliation TEXT               -- 所属 (美浦/栗東/地方/外国 など)
);

-- -----------------------------------------------------------------------------
-- マスタ: 調教師
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trainers (
    trainer_id TEXT PRIMARY KEY,   -- netkeiba の調教師ID
    name       TEXT NOT NULL,
    affiliation TEXT               -- 所属 (美浦/栗東 など)
);

-- -----------------------------------------------------------------------------
-- マスタ: 馬
--   血統(父/母/母父)は予想で重要なので保持。
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS horses (
    horse_id      TEXT PRIMARY KEY,  -- netkeiba の馬ID
    name          TEXT NOT NULL,
    sex           TEXT CHECK (sex IN ('牡', '牝', 'セ')),  -- 性別
    birth_date    TEXT,              -- 生年月日 (YYYY-MM-DD)
    coat_color    TEXT,              -- 毛色
    sire          TEXT,              -- 父
    dam           TEXT,              -- 母
    broodmare_sire TEXT,             -- 母父 (BMS)
    owner         TEXT,              -- 馬主
    breeder       TEXT,              -- 生産者
    trainer_id    TEXT REFERENCES trainers(trainer_id)
);

-- -----------------------------------------------------------------------------
-- レース
--   race_id は netkeiba の12桁ID (例: 202405021211)。
--   年(4) + 場コード(2) + 開催回(2) + 開催日(2) + レース番号(2) を想定。
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS races (
    race_id        TEXT PRIMARY KEY,
    race_date      TEXT NOT NULL,                 -- 開催日 (YYYY-MM-DD)
    venue_id       TEXT REFERENCES venues(venue_id),
    race_number    INTEGER,                       -- 第何レース
    race_name      TEXT,                          -- レース名
    grade          TEXT,                          -- G1/G2/G3/L/OP/3勝/2勝/1勝/未勝利/新馬 等
    surface        TEXT CHECK (surface IN ('芝', 'ダート', '障害')),  -- 馬場種別
    distance       INTEGER,                       -- 距離(m)
    direction      TEXT CHECK (direction IN ('右', '左', '直線', '障害')), -- 回り
    weather        TEXT,                          -- 天候
    track_condition TEXT CHECK (track_condition IN ('良', '稍重', '重', '不良')), -- 馬場状態
    field_size     INTEGER,                       -- 出走頭数
    prize_1st      INTEGER                        -- 1着賞金(円)。クラス推定に有用
);

-- -----------------------------------------------------------------------------
-- 出走・成績 (ファクト表)
--   1頭の1レース出走を1行で表す。予想分析の中心となるテーブル。
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS results (
    race_id        TEXT NOT NULL REFERENCES races(race_id),
    horse_id       TEXT NOT NULL REFERENCES horses(horse_id),
    jockey_id      TEXT REFERENCES jockeys(jockey_id),
    trainer_id     TEXT REFERENCES trainers(trainer_id),

    post_position  INTEGER,        -- 枠番
    horse_number   INTEGER,        -- 馬番
    finish_position INTEGER,       -- 着順 (NULL=中止/除外/失格など)
    finish_status  TEXT,           -- 着順が付かない場合の状態 (中止/除外/取消/失格)

    time_seconds   REAL,           -- 走破タイム(秒)
    margin         TEXT,           -- 着差 (クビ, ハナ, 1.1/2 など。テキスト保持)
    passing        TEXT,           -- 通過順位 ("3-3-2-1" など)
    last_3f        REAL,           -- 上がり3ハロン(秒)

    weight_carried REAL,           -- 斤量(kg)
    horse_weight   INTEGER,        -- 馬体重(kg)
    weight_change  INTEGER,        -- 馬体重増減(kg, 前走比)

    odds           REAL,           -- 単勝オッズ(確定)
    popularity     INTEGER,        -- 単勝人気
    prize_won      INTEGER,        -- 獲得賞金(円)

    PRIMARY KEY (race_id, horse_number),
    UNIQUE (race_id, horse_id)
);

-- -----------------------------------------------------------------------------
-- 払戻
--   bet_type ごとに組番と払戻金・人気を格納。三連単など組番が長いものもTEXTで保持。
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS payouts (
    race_id      TEXT NOT NULL REFERENCES races(race_id),
    bet_type     TEXT NOT NULL,    -- 単勝/複勝/枠連/馬連/ワイド/馬単/三連複/三連単
    combination  TEXT NOT NULL,    -- 組番 ("7" や "3-7" や "3-7-12" など)
    payout       INTEGER,          -- 払戻金(100円あたり, 円)
    popularity   INTEGER,          -- 人気
    PRIMARY KEY (race_id, bet_type, combination)
);

-- =============================================================================
-- インデックス: 分析クエリ・JOINを高速化
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_results_horse    ON results(horse_id);
CREATE INDEX IF NOT EXISTS idx_results_jockey   ON results(jockey_id);
CREATE INDEX IF NOT EXISTS idx_results_trainer  ON results(trainer_id);
CREATE INDEX IF NOT EXISTS idx_results_finish   ON results(finish_position);
CREATE INDEX IF NOT EXISTS idx_races_date       ON races(race_date);
CREATE INDEX IF NOT EXISTS idx_races_venue      ON races(venue_id);
CREATE INDEX IF NOT EXISTS idx_races_surf_dist  ON races(surface, distance);
CREATE INDEX IF NOT EXISTS idx_horses_sire      ON horses(sire);

-- =============================================================================
-- ビュー: 分析でよく使う結合済みデータ
-- =============================================================================

-- 出走成績にレース情報・馬名・騎手名を結合したフラットなビュー。
-- 多くの分析クエリはまずこのビューを起点にすると書きやすい。
CREATE VIEW IF NOT EXISTS v_results_full AS
SELECT
    r.race_id,
    ra.race_date,
    v.name           AS venue_name,
    ra.race_number,
    ra.race_name,
    ra.grade,
    ra.surface,
    ra.distance,
    ra.direction,
    ra.track_condition,
    ra.field_size,
    r.horse_id,
    h.name           AS horse_name,
    h.sex,
    h.sire,
    h.broodmare_sire,
    r.jockey_id,
    j.name           AS jockey_name,
    r.trainer_id,
    t.name           AS trainer_name,
    r.post_position,
    r.horse_number,
    r.finish_position,
    r.finish_status,
    r.time_seconds,
    r.last_3f,
    r.weight_carried,
    r.horse_weight,
    r.weight_change,
    r.odds,
    r.popularity,
    r.prize_won
FROM results r
JOIN races    ra ON r.race_id   = ra.race_id
LEFT JOIN venues   v ON ra.venue_id  = v.venue_id
LEFT JOIN horses   h ON r.horse_id   = h.horse_id
LEFT JOIN jockeys  j ON r.jockey_id  = j.jockey_id
LEFT JOIN trainers t ON r.trainer_id = t.trainer_id;
