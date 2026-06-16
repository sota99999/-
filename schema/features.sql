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

-- Eloレーティング表が無くても v_features を作れるよう、空でも用意しておく
-- （実データは scripts/compute_ratings.py が投入する）
CREATE TABLE IF NOT EXISTS horse_ratings (
    race_id TEXT NOT NULL, horse_id TEXT NOT NULL,
    elo_before REAL, PRIMARY KEY (race_id, horse_id)
);

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
--   raw（1行=1出走の素データ＋クラス格・Elo）→ base（過去走の窓集計）→
--   外側（レース単位の展開ペース）の3段構成。集計は当該レースを除く窓でリーク防止。
--   ※ 当該レースの結果（着順・タイム・スピード指数等）は最終列に出さない。
-- -----------------------------------------------------------------------------
CREATE VIEW v_features AS
WITH raw AS (
    SELECT
        r.race_id, r.horse_id, r.jockey_id, ra.race_date,
        h.name AS horse_name,
        ra.venue_id, ra.surface, ra.distance, ra.field_size,
        ra.track_condition, ra.direction, ra.weather,
        r.post_position, r.horse_number, r.weight_carried,
        r.horse_weight, r.weight_change, r.odds, r.popularity,
        r.finish_position, r.last_3f,
        s.speed_index,
        hr.elo_before,                                     -- Eloレーティング（レース直前）
        -- 内外位置（0=最内, 1=最外）
        CASE WHEN ra.field_size > 0
             THEN ROUND(1.0 * r.post_position / ra.field_size, 3) END AS draw_ratio,
        -- 脚質の素: 最初のコーナー通過順 ÷ 頭数（0=前, 1=後）
        CASE WHEN r.passing IS NOT NULL AND r.passing <> '' AND ra.field_size > 0
             THEN 1.0 * CAST(substr(r.passing, 1, instr(r.passing || '-', '-') - 1) AS REAL)
                  / ra.field_size END AS early_pos_ratio,
        -- クラスの格（0〜9）。グレード優先、無ければレース名から推定
        CASE
            WHEN ra.grade IN ('G1','GⅠ') THEN 9
            WHEN ra.grade IN ('G2','GⅡ') THEN 8
            WHEN ra.grade IN ('G3','GⅢ') THEN 7
            WHEN ra.race_name LIKE '%オープン%' OR ra.race_name LIKE '%(L)%' THEN 6
            WHEN ra.race_name LIKE '%3勝%' OR ra.race_name LIKE '%1600万%' THEN 5
            WHEN ra.race_name LIKE '%2勝%' OR ra.race_name LIKE '%1000万%' THEN 4
            WHEN ra.race_name LIKE '%1勝%' OR ra.race_name LIKE '%500万%'  THEN 3
            WHEN ra.race_name LIKE '%未勝利%' THEN 2
            WHEN ra.race_name LIKE '%新馬%'   THEN 1
            WHEN ra.race_name LIKE '%ステークス%' OR ra.race_name LIKE '%特別%'
              OR ra.race_name LIKE '%記念%'   OR ra.race_name LIKE '%賞%' THEN 6
            ELSE 0
        END AS class_level
    FROM results r
    JOIN races  ra ON r.race_id  = ra.race_id
    LEFT JOIN horses h ON r.horse_id = h.horse_id
    LEFT JOIN v_speed s ON s.race_id = r.race_id AND s.horse_id = r.horse_id
    LEFT JOIN horse_ratings hr ON hr.race_id = r.race_id AND hr.horse_id = r.horse_id
),
base AS (
    SELECT
        raw.*,
        -- 過去走の通算成績（当該レースを除く）
        COUNT(*) OVER w_hist                              AS runs_prior,
        SUM(finish_position = 1) OVER w_hist              AS wins_prior,
        ROUND(1.0 * SUM(finish_position = 1)  OVER w_hist / COUNT(*) OVER w_hist, 3) AS win_rate_prior,
        ROUND(1.0 * SUM(finish_position <= 3) OVER w_hist / COUNT(*) OVER w_hist, 3) AS show_rate_prior,
        ROUND(AVG(finish_position) OVER w_hist, 2)        AS avg_finish_prior,
        MIN(last_3f) OVER w_hist                          AS best_last3f_prior,
        -- スピード指数（過去走のみ）
        ROUND(AVG(speed_index) OVER w_hist, 1)            AS avg_speed_prior,
        MAX(speed_index) OVER w_hist                      AS best_speed_prior,
        LAG(speed_index) OVER w_ord                       AS prev_speed,
        -- 脚質（過去の早い段階の位置取り平均）
        ROUND(AVG(early_pos_ratio) OVER w_hist, 3)        AS run_style_prior,
        -- クラス補正: 過去の平均クラス格 と「スピード指数＋3×クラス格」の自己最高
        ROUND(AVG(class_level) OVER w_hist, 2)            AS avg_class_level_prior,
        MAX(speed_index + 3 * class_level) OVER w_hist    AS class_adj_speed_prior,
        -- 騎手の過去成績（騎手ごと・当該レースを除く）
        COUNT(*) OVER w_jockey                            AS jockey_rides_prior,
        ROUND(1.0 * SUM(finish_position = 1)  OVER w_jockey / COUNT(*) OVER w_jockey, 3) AS jockey_win_rate_prior,
        ROUND(1.0 * SUM(finish_position <= 3) OVER w_jockey / COUNT(*) OVER w_jockey, 3) AS jockey_show_rate_prior,
        -- 同コース実績（馬×競馬場×馬場種別、当該レースを除く）
        COUNT(*) OVER w_course                            AS course_runs_prior,
        ROUND(1.0 * SUM(finish_position <= 3) OVER w_course / COUNT(*) OVER w_course, 3) AS course_show_rate_prior,
        -- 前走情報
        LAG(finish_position) OVER w_ord                   AS prev_finish,
        LAG(popularity)      OVER w_ord                   AS prev_popularity,
        LAG(surface)         OVER w_ord                   AS prev_surface,
        LAG(distance)        OVER w_ord                   AS prev_distance,
        CAST(julianday(race_date) - julianday(LAG(race_date) OVER w_ord) AS INTEGER) AS days_since_last,
        distance - LAG(distance) OVER w_ord               AS distance_change,
        -- ターゲット（正解ラベル）
        CASE WHEN finish_position = 1  THEN 1 ELSE 0 END  AS target_win,
        CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END  AS target_show
    FROM raw
    WINDOW
        w_hist   AS (PARTITION BY horse_id ORDER BY race_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        w_ord    AS (PARTITION BY horse_id ORDER BY race_date),
        w_jockey AS (PARTITION BY jockey_id ORDER BY race_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        w_course AS (PARTITION BY horse_id, venue_id, surface ORDER BY race_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
-- 最終出力: 当該レースの結果列（finish_position/speed_index/last_3f/passing 等）は出さない
SELECT
    race_id, horse_id, race_date, horse_name,
    venue_id, surface, distance, field_size,
    track_condition, direction, weather,
    post_position, horse_number, weight_carried, horse_weight, weight_change,
    odds, popularity, draw_ratio,
    class_level,                                          -- 当該レースのクラス格（出走前に既知）
    elo_before,                                           -- Eloレーティング（レース直前）
    runs_prior, wins_prior, win_rate_prior, show_rate_prior, avg_finish_prior, best_last3f_prior,
    avg_speed_prior, best_speed_prior, prev_speed,
    run_style_prior, avg_class_level_prior, class_adj_speed_prior,
    jockey_rides_prior, jockey_win_rate_prior, jockey_show_rate_prior,
    course_runs_prior, course_show_rate_prior,
    prev_finish, prev_popularity, prev_surface, prev_distance, days_since_last, distance_change,
    -- 展開（ペース）推定: 出走各馬の脚質の平均（低い=前残り少なめ=ハイペース傾向）
    ROUND(AVG(run_style_prior) OVER (PARTITION BY race_id), 3) AS race_pace_estimate,
    target_win, target_show
FROM base;
