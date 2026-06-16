-- =============================================================================
-- 機械学習向け 特徴量ビュー
--
-- 各「出走（race_id × horse_id）」の時点で、その馬の "過去走" だけを集計した
-- 特徴量を作る。ウィンドウフレームを
--   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
-- とすることで、当該レース自身の結果を含めず（=未来情報のリークを防ぎ）、
-- 「レース直前に分かっている情報」だけで特徴量を構成する。
--
-- 予測ターゲット（target_win / target_show）も同居させているので、
-- そのまま学習用テーブルとして CSV エクスポートできる。
--
-- 前提: schema.sql 適用済み。
-- 使い方: sqlite3 keiba.db < schema/features.sql
-- =============================================================================

DROP VIEW IF EXISTS v_features;

CREATE VIEW v_features AS
SELECT
    r.race_id,
    r.horse_id,
    ra.race_date,
    h.name              AS horse_name,
    ra.surface,
    ra.distance,
    r.post_position,
    r.horse_number,
    r.weight_carried,
    r.odds,
    r.popularity,

    -- ---- 過去走の通算成績（当該レースを除く） -------------------------------
    COUNT(*)            OVER w_hist                         AS runs_prior,        -- 過去出走数
    SUM(r.finish_position = 1)  OVER w_hist                 AS wins_prior,        -- 過去勝利数
    -- 勝率/複勝率: 初出走(runs_prior=0)では分母0→SQLiteではNULLになる
    ROUND(1.0 * SUM(r.finish_position = 1)  OVER w_hist
              / COUNT(*) OVER w_hist, 3)                    AS win_rate_prior,
    ROUND(1.0 * SUM(r.finish_position <= 3) OVER w_hist
              / COUNT(*) OVER w_hist, 3)                    AS show_rate_prior,
    ROUND(AVG(r.finish_position) OVER w_hist, 2)            AS avg_finish_prior,  -- 平均着順
    MIN(r.last_3f)      OVER w_hist                         AS best_last3f_prior, -- 自己ベスト上がり

    -- ---- 前走の情報 ---------------------------------------------------------
    LAG(r.finish_position) OVER w_ord                       AS prev_finish,       -- 前走着順
    LAG(r.popularity)      OVER w_ord                       AS prev_popularity,   -- 前走人気
    LAG(ra.surface)        OVER w_ord                       AS prev_surface,      -- 前走馬場種別
    LAG(ra.distance)       OVER w_ord                       AS prev_distance,     -- 前走距離
    -- 前走からの間隔(日)。ローテーション（叩き/詰め込み）の指標
    CAST(julianday(ra.race_date)
         - julianday(LAG(ra.race_date) OVER w_ord) AS INTEGER) AS days_since_last,
    -- 距離変更幅（プラス=距離延長, マイナス=短縮）
    ra.distance - LAG(ra.distance) OVER w_ord               AS distance_change,

    -- ---- 予測ターゲット（学習の正解ラベル） --------------------------------
    CASE WHEN r.finish_position = 1  THEN 1 ELSE 0 END      AS target_win,   -- 1着か
    CASE WHEN r.finish_position <= 3 THEN 1 ELSE 0 END      AS target_show   -- 3着以内か
FROM results r
JOIN races  ra ON r.race_id  = ra.race_id
LEFT JOIN horses h ON r.horse_id = h.horse_id
-- 同名ウィンドウ定義（馬ごと・日付順）
WINDOW
    -- 過去走のみ（現在行を除く）。集計系で使用
    w_hist AS (PARTITION BY r.horse_id ORDER BY ra.race_date
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
    -- 全行順序付け。LAG（前走参照）で使用
    w_ord  AS (PARTITION BY r.horse_id ORDER BY ra.race_date);
