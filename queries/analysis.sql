-- =============================================================================
-- 競馬予想 分析クエリ集
--
-- v_results_full ビュー（schema.sql で定義）を起点にした、予想に役立つ
-- 集計クエリのサンプル。:param のプレースホルダ部分は実際の値に置き換えるか、
-- sqlite3 のパラメータバインドで利用する。
--
-- 「勝率」=1着率, 「連対率」=2着以内率, 「複勝率」=3着以内率。
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. 騎手成績ランキング（直近データ）
--    勝率・連対率・複勝率と単勝回収率を集計。
--    回収率 = 払戻総額 / 賭け金総額。単勝100円を全頭に賭けた想定。
-- -----------------------------------------------------------------------------
SELECT
    jockey_name,
    COUNT(*)                                                   AS rides,        -- 騎乗数
    SUM(finish_position = 1)                                   AS wins,
    ROUND(100.0 * SUM(finish_position = 1) / COUNT(*), 1)      AS win_pct,      -- 勝率%
    ROUND(100.0 * SUM(finish_position <= 2) / COUNT(*), 1)     AS quinella_pct, -- 連対率%
    ROUND(100.0 * SUM(finish_position <= 3) / COUNT(*), 1)     AS show_pct,     -- 複勝率%
    -- 単勝回収率%: 1着なら (odds*100) が払い戻し、賭け金は 100*騎乗数
    ROUND(100.0 * SUM(CASE WHEN finish_position = 1 THEN odds * 100 ELSE 0 END)
          / (COUNT(*) * 100), 1)                               AS win_roi
FROM v_results_full
WHERE finish_position IS NOT NULL
GROUP BY jockey_id
HAVING rides >= 50          -- サンプル数が少ない騎手は除外
ORDER BY win_pct DESC;


-- -----------------------------------------------------------------------------
-- 2. 種牡馬（父）のコース適性
--    父ごとに「芝/ダート × 距離帯」での複勝率を見る。
--    指定の条件に強い血統を探すのに使う。
-- -----------------------------------------------------------------------------
SELECT
    sire,
    surface,
    CASE
        WHEN distance <  1400 THEN '短距離(~1399)'
        WHEN distance <  1800 THEN 'マイル(1400-1799)'
        WHEN distance <  2200 THEN '中距離(1800-2199)'
        ELSE '長距離(2200~)'
    END                                                     AS dist_band,
    COUNT(*)                                                AS runs,
    ROUND(100.0 * SUM(finish_position = 1)  / COUNT(*), 1) AS win_pct,
    ROUND(100.0 * SUM(finish_position <= 3) / COUNT(*), 1) AS show_pct
FROM v_results_full
WHERE finish_position IS NOT NULL AND sire IS NOT NULL
GROUP BY sire, surface, dist_band
HAVING runs >= 20
ORDER BY sire, surface, dist_band;


-- -----------------------------------------------------------------------------
-- 3. 枠番別の有利不利（特定コース）
--    同一競馬場・馬場種別・距離での枠番ごとの複勝率。内枠/外枠の傾向を把握。
--    :venue, :surface, :distance を指定して使う。
-- -----------------------------------------------------------------------------
SELECT
    post_position                                          AS waku,
    COUNT(*)                                               AS runs,
    ROUND(100.0 * SUM(finish_position = 1)  / COUNT(*), 1) AS win_pct,
    ROUND(100.0 * SUM(finish_position <= 3) / COUNT(*), 1) AS show_pct
FROM v_results_full
WHERE venue_name = :venue
  AND surface    = :surface
  AND distance   = :distance
  AND finish_position IS NOT NULL
GROUP BY post_position
ORDER BY post_position;


-- -----------------------------------------------------------------------------
-- 4. 人気別の信頼度と回収率
--    単勝人気ごとに、実際の勝率・複勝率と単勝回収率を見る。
--    「○番人気は買い時か」を判断する基礎データ。
-- -----------------------------------------------------------------------------
SELECT
    popularity,
    COUNT(*)                                               AS runs,
    ROUND(100.0 * SUM(finish_position = 1)  / COUNT(*), 1) AS win_pct,
    ROUND(100.0 * SUM(finish_position <= 3) / COUNT(*), 1) AS show_pct,
    ROUND(AVG(odds), 1)                                    AS avg_odds,
    ROUND(100.0 * SUM(CASE WHEN finish_position = 1 THEN odds * 100 ELSE 0 END)
          / (COUNT(*) * 100), 1)                           AS win_roi
FROM v_results_full
WHERE finish_position IS NOT NULL AND popularity IS NOT NULL
GROUP BY popularity
ORDER BY popularity;


-- -----------------------------------------------------------------------------
-- 5. 馬1頭の過去走サマリ（出走前の検討用）
--    特定馬の全成績を新しい順に。コース替わり・距離変更の影響を確認。
--    :horse_name を指定して使う。
-- -----------------------------------------------------------------------------
SELECT
    race_date,
    venue_name,
    race_name,
    grade,
    surface,
    distance,
    track_condition,
    finish_position || '/' || field_size AS rank,
    popularity,
    jockey_name,
    weight_carried,
    horse_weight,
    last_3f
FROM v_results_full
WHERE horse_name = :horse_name
ORDER BY race_date DESC;


-- -----------------------------------------------------------------------------
-- 6. 騎手 × 調教師 のコンビ成績
--    「鉄板コンビ」を探す。連対率の高い組み合わせを抽出。
-- -----------------------------------------------------------------------------
SELECT
    jockey_name,
    trainer_name,
    COUNT(*)                                               AS rides,
    ROUND(100.0 * SUM(finish_position = 1)  / COUNT(*), 1) AS win_pct,
    ROUND(100.0 * SUM(finish_position <= 2) / COUNT(*), 1) AS quinella_pct
FROM v_results_full
WHERE finish_position IS NOT NULL
  AND jockey_id IS NOT NULL AND trainer_id IS NOT NULL
GROUP BY jockey_id, trainer_id
HAVING rides >= 20
ORDER BY win_pct DESC;


-- -----------------------------------------------------------------------------
-- 7. 馬体重増減と成績の関係
--    増減区分ごとの複勝率。大幅な増減が好走/凡走につながるか確認。
-- -----------------------------------------------------------------------------
SELECT
    CASE
        WHEN weight_change <= -10 THEN '-10kg以上減'
        WHEN weight_change <   0  THEN '減'
        WHEN weight_change =   0  THEN '増減なし'
        WHEN weight_change <= 10  THEN '増'
        ELSE '+10kg以上増'
    END                                                   AS wc_band,
    COUNT(*)                                              AS runs,
    ROUND(100.0 * SUM(finish_position <= 3) / COUNT(*), 1) AS show_pct
FROM v_results_full
WHERE finish_position IS NOT NULL AND weight_change IS NOT NULL
GROUP BY wc_band
ORDER BY MIN(weight_change);


-- -----------------------------------------------------------------------------
-- 8. 上がり最速馬は次走で買えるか（上がり3F順位の指標化）
--    各レースで上がり3F最速だった馬を抽出。ローテ次第で次走の妙味につながる。
-- -----------------------------------------------------------------------------
WITH ranked AS (
    SELECT
        race_id, horse_id, horse_name, race_date, finish_position, last_3f,
        RANK() OVER (PARTITION BY race_id ORDER BY last_3f ASC) AS l3f_rank
    FROM v_results_full
    WHERE last_3f IS NOT NULL
)
SELECT race_date, horse_name, finish_position, last_3f
FROM ranked
WHERE l3f_rank = 1            -- そのレースで上がり最速
ORDER BY race_date DESC;
