-- =============================================================================
-- 機械学習向け 特徴量ビュー
--
-- 各「出走（race_id × horse_id）」の時点で、その馬の "過去走" だけを集計した
-- 特徴量を作る。集計系ウィンドウのフレームを
--   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
-- とすることで当該レース自身の結果を含めず（=未来情報のリークを防ぎ）、
-- 「レース直前に分かっている情報」だけで特徴量を構成する。
--
-- 構成:
--   v_speed     ... 1走ごとの「スピード指数」（同コース・同馬場内の相対タイム）
--   v_features  ... 学習用の特徴量＋ターゲット。v_speed を参照して過去スピードも集計
--
-- 前提: schema.sql 適用済み。
-- 使い方: sqlite3 keiba.db < schema/features.sql
-- =============================================================================

-- -----------------------------------------------------------------------------
-- v_speed: スピード指数
--   同じ (馬場種別, 距離, 馬場状態) のレース群の中での相対タイムを指数化。
--   指数 = 50 + (グループ平均タイム - 自分のタイム) / グループ平均 * 1000
--   → 平均ちょうどで 50、平均より 1% 速いと +10。値が大きいほど速い。
--   標準偏差(sqrt)を使わず移植性を確保している。母数が少ないバケットは NULL。
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS v_features;   -- v_speed に依存するため先に落とす
DROP VIEW IF EXISTS v_speed;

CREATE VIEW v_speed AS
WITH base AS (
    SELECT
        r.race_id,
        r.horse_id,
        ra.race_date,
        ra.venue_id,
        ra.surface,
        ra.distance,
        r.time_seconds,
        AVG(r.time_seconds) OVER grp AS mean_t,   -- 同条件の平均タイム
        COUNT(*)            OVER grp AS grp_n      -- 同条件のサンプル数
    FROM results r
    JOIN races ra ON r.race_id = ra.race_id
    WHERE r.time_seconds IS NOT NULL
    WINDOW grp AS (PARTITION BY ra.surface, ra.distance, ra.track_condition)
)
SELECT
    race_id,
    horse_id,
    race_date,
    venue_id,
    surface,
    distance,
    time_seconds,
    CASE
        WHEN grp_n >= 20 AND mean_t > 0
        THEN ROUND(50 + (mean_t - time_seconds) / mean_t * 1000, 1)
        ELSE NULL
    END AS speed_index
FROM base;


-- -----------------------------------------------------------------------------
-- v_features: 学習用 特徴量 + ターゲット
-- -----------------------------------------------------------------------------
CREATE VIEW v_features AS
SELECT
    r.race_id,
    r.horse_id,
    ra.race_date,
    h.name              AS horse_name,
    ra.venue_id,
    ra.surface,
    ra.distance,
    r.post_position,
    r.horse_number,
    r.weight_carried,
    r.odds,
    r.popularity,

    -- ---- 過去走の通算成績（当該レースを除く） -------------------------------
    COUNT(*)            OVER w_hist                         AS runs_prior,
    SUM(r.finish_position = 1)  OVER w_hist                 AS wins_prior,
    ROUND(1.0 * SUM(r.finish_position = 1)  OVER w_hist
              / COUNT(*) OVER w_hist, 3)                    AS win_rate_prior,
    ROUND(1.0 * SUM(r.finish_position <= 3) OVER w_hist
              / COUNT(*) OVER w_hist, 3)                    AS show_rate_prior,
    ROUND(AVG(r.finish_position) OVER w_hist, 2)            AS avg_finish_prior,
    MIN(r.last_3f)      OVER w_hist                         AS best_last3f_prior,

    -- ---- 過去走のスピード指数（当該レースを除く） --------------------------
    ROUND(AVG(s.speed_index) OVER w_hist, 1)               AS avg_speed_prior,   -- 平均
    MAX(s.speed_index)       OVER w_hist                   AS best_speed_prior,  -- 自己最高
    LAG(s.speed_index)       OVER w_ord                    AS prev_speed,        -- 前走

    -- ---- 同コース実績（同 競馬場×馬場種別、当該レースを除く） --------------
    COUNT(*) OVER w_course                                 AS course_runs_prior,
    ROUND(1.0 * SUM(r.finish_position <= 3) OVER w_course
              / COUNT(*) OVER w_course, 3)                 AS course_show_rate_prior,

    -- ---- 前走の情報 ---------------------------------------------------------
    LAG(r.finish_position) OVER w_ord                      AS prev_finish,
    LAG(r.popularity)      OVER w_ord                      AS prev_popularity,
    LAG(ra.surface)        OVER w_ord                      AS prev_surface,
    LAG(ra.distance)       OVER w_ord                      AS prev_distance,
    CAST(julianday(ra.race_date)
         - julianday(LAG(ra.race_date) OVER w_ord) AS INTEGER) AS days_since_last,
    ra.distance - LAG(ra.distance) OVER w_ord              AS distance_change,

    -- ---- 予測ターゲット（学習の正解ラベル） --------------------------------
    CASE WHEN r.finish_position = 1  THEN 1 ELSE 0 END     AS target_win,
    CASE WHEN r.finish_position <= 3 THEN 1 ELSE 0 END     AS target_show
FROM results r
JOIN races  ra ON r.race_id  = ra.race_id
LEFT JOIN horses  h ON r.horse_id = h.horse_id
LEFT JOIN v_speed s ON s.race_id = r.race_id AND s.horse_id = r.horse_id
WINDOW
    -- 過去走のみ（馬ごと・日付順、現在行を除く）
    w_hist AS (PARTITION BY r.horse_id ORDER BY ra.race_date
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
    -- 全行順序付け（LAG=前走参照用）
    w_ord  AS (PARTITION BY r.horse_id ORDER BY ra.race_date),
    -- 同コース過去走のみ（馬×競馬場×馬場種別、現在行を除く）
    w_course AS (PARTITION BY r.horse_id, ra.venue_id, ra.surface ORDER BY ra.race_date
                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING);
