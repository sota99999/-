-- =============================================================================
-- v_run_adj: 1走ごとの「展開・斤量を補正した実力値」(quality_sp) と能力素点
--   詳細設計は docs/ability_design.md を参照。
--   前提: schema.sql / features.sql（v_speed, v_pace_ref）/ course_master.sql（v_course）適用済み。
--   使い方: sqlite3 keiba.db < schema/ability.sql
-- =============================================================================
DROP VIEW IF EXISTS v_run_adj;

CREATE VIEW v_run_adj AS
WITH base AS (
    SELECT
        r.race_id, r.horse_id, ra.race_date, ra.venue_id, ra.surface, ra.distance,
        ra.field_size, r.finish_position AS fin, r.last_3f, r.weight_carried,
        s.speed_index AS sp,
        -- 距離帯
        CASE WHEN ra.distance < 1400 THEN 'sprint'
             WHEN ra.distance < 1800 THEN 'mile'
             WHEN ra.distance < 2200 THEN 'mid' ELSE 'long' END AS dist_band,
        -- 道悪フラグ（良以外）
        CASE WHEN ra.track_condition IS NOT NULL AND ra.track_condition <> '良'
             THEN 1 ELSE 0 END AS is_off,
        -- その日の隊列: 最初のコーナー通過順 / 頭数（0=前, 1=後）
        CASE WHEN r.passing IS NOT NULL AND r.passing <> '' AND ra.field_size > 0
             THEN 1.0 * CAST(substr(r.passing, 1, instr(r.passing || '-', '-') - 1) AS REAL)
                  / ra.field_size END AS pos,
        rl.race_first3f, pr.avg_first3f,
        cm.hill_grade, cm.straight_m, cm.turn_size, cm.pace_bias
    FROM results r
    JOIN races ra ON ra.race_id = r.race_id
    LEFT JOIN v_speed s ON s.race_id = r.race_id AND s.horse_id = r.horse_id
    LEFT JOIN race_laps rl ON rl.race_id = r.race_id
    LEFT JOIN v_pace_ref pr ON pr.surface = ra.surface
         AND pr.dist_band = CASE WHEN ra.distance < 1400 THEN 'sprint'
                                 WHEN ra.distance < 1800 THEN 'mile'
                                 WHEN ra.distance < 2200 THEN 'mid' ELSE 'long' END
    LEFT JOIN v_course cm ON cm.race_id = r.race_id
    WHERE s.speed_index IS NOT NULL
),
adj AS (
    SELECT base.*,
        ROUND(AVG(weight_carried) OVER (PARTITION BY race_id), 1) AS race_avg_weight,
        -- 前半3Fが基準より速い=ハイ(+)、遅い=スロー(-)。単位は秒
        CASE WHEN race_first3f IS NULL OR avg_first3f IS NULL THEN 0.0
             ELSE ROUND(avg_first3f - race_first3f, 2) END AS pace_fast
    FROM base
)
SELECT
    race_id, horse_id, race_date, venue_id, surface, distance, dist_band,
    field_size, fin, last_3f, weight_carried, race_avg_weight, sp,
    is_off, pos, pace_fast, hill_grade, straight_m, turn_size, pace_bias,
    -- 斤量補正: 重い斤量で出した時計は価値が高い（0.1秒≒0.8指数/kg）
    ROUND(sp + (weight_carried - race_avg_weight) * 0.8, 1) AS sp_wadj,
    -- 展開補正: (0.5-pos)*pace_fast を ±3 にクリップ
    --   前×ハイ=加点 / 前×スロー=減点 / 後×ハイ=減点（恵まれ差し）
    MAX(-3.0, MIN(3.0, ROUND(1.2 * (0.5 - COALESCE(pos, 0.5)) * pace_fast, 2))) AS trip_adj,
    -- 展開・斤量ニュートラルな1走の実力値
    ROUND(sp + (weight_carried - race_avg_weight) * 0.8
          + MAX(-3.0, MIN(3.0, 1.2 * (0.5 - COALESCE(pos, 0.5)) * pace_fast)), 1) AS quality_sp,
    -- 好走フラグ（3着内。能力集約では凡走を無視する）
    CASE WHEN fin <= 3 THEN 1 ELSE 0 END AS good
FROM adj;


-- =============================================================================
-- v_ability: 各出走時点で「その馬の過去走だけ」を集約した6能力（STEP2）
--   窓は features.sql と同じ ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
--   （＝当該レースを含めない＝リーク防止）。凡走は無視し好走(3着内)中心に集約。
--
--   ● power_top … 総合能力の主軸。好走時 quality_sp を「全走平均」へ縮約(shrinkage)。
--       power_top = (n好走 * 好走平均q + K * 全走平均q) / (n好走 + K),  K=2
--       → 好走1回だけの馬（例: 好走平均⭐の罠）は全走平均へ引き戻され吊り上げを防ぐ。
--         天井(MAX)ではなく安定実力ベースなので"まぐれ図"にも汚染されにくい。
--   ● shunpatsu … 瞬発力。好走時の上がり3F平均（小さいほど鋭い＝direction: 低いほど良）。
--   ● jizoku   … 持続力。前傾(ハイ前半)＝ロングスパート戦で好走した時の quality_sp。
--   ● toppspeed… トップスピード。良馬場好走時の quality_sp。
--   ● power    … パワー。急坂 or 道悪 好走時の quality_sp。
--   ● stamina  … スタミナ。中長距離(1800m〜)好走時の quality_sp と複勝率。
--   ● start    … 先行力。平均隊列 avg_pos（小さいほど前）。
--   ＋道悪/距離の複勝率とサンプル数（STEP3の適性判定・小サンプルガード用）。
--   使い方: sqlite3 keiba.db < schema/ability.sql
-- =============================================================================
DROP VIEW IF EXISTS v_ability;

CREATE VIEW v_ability AS
WITH agg AS (
    SELECT
        race_id, horse_id, race_date, surface, distance, dist_band,
        COUNT(*)        OVER w                                        AS runs_prior,
        SUM(good)       OVER w                                        AS good_prior,
        AVG(quality_sp) OVER w                                        AS all_avg_q,
        AVG(CASE WHEN good=1 THEN quality_sp END) OVER w              AS good_avg_q,
        -- 瞬発力: 好走時 上がり3F 平均（低いほど鋭い）
        AVG(CASE WHEN good=1 THEN last_3f END) OVER w                 AS good_last3f,
        -- 持続力: 前傾(ハイ前半)好走時の quality
        AVG(CASE WHEN good=1 AND pace_fast > 0.5 THEN quality_sp END) OVER w AS jizoku_q,
        SUM(CASE WHEN good=1 AND pace_fast > 0.5 THEN 1 ELSE 0 END) OVER w   AS n_jizoku_good,
        -- トップスピード: 良馬場好走時の quality
        AVG(CASE WHEN good=1 AND is_off=0 THEN quality_sp END) OVER w AS top_q,
        -- パワー: 急坂 or 道悪 好走時の quality
        AVG(CASE WHEN good=1 AND (hill_grade='steep' OR is_off=1) THEN quality_sp END) OVER w AS power_q,
        -- スタミナ: 中長距離(mid/long=1800m〜)好走時の quality と複勝率
        AVG(CASE WHEN good=1 AND dist_band IN ('mid','long') THEN quality_sp END) OVER w AS stamina_q,
        SUM(CASE WHEN dist_band IN ('mid','long') THEN 1 ELSE 0 END) OVER w AS n_longish,
        SUM(CASE WHEN dist_band IN ('mid','long') AND good=1 THEN 1 ELSE 0 END) OVER w AS n_long_good,
        -- 道悪: 経験数と好走数（道悪適性・良/道悪2択の判定に使う）
        SUM(CASE WHEN is_off=1 THEN 1 ELSE 0 END) OVER w             AS n_off,
        SUM(CASE WHEN is_off=1 AND good=1 THEN 1 ELSE 0 END) OVER w  AS n_off_good,
        -- 先行力: 平均隊列（小さいほど前）
        ROUND(AVG(pos) OVER w, 3)                                    AS avg_pos
    FROM v_run_adj
    WINDOW w AS (PARTITION BY horse_id ORDER BY race_date
                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
),
-- 全体の好走平均 quality（縮約の母集団プライア μ）。データから算出＝マジック定数を避ける
mu AS (SELECT AVG(quality_sp) AS m FROM v_run_adj WHERE good = 1)
SELECT
    race_id, horse_id, race_date, surface, distance, dist_band,
    runs_prior, good_prior,
    -- 総合能力（経験的ベイズ縮約）:
    --   power_top = (好走qの合計 + K×μ) / (好走数 + K)
    --   K は好走数で可変: 好走2回以上=3、1回以下=8（単発フリーク図を強く母集団へ引き戻す）
    --   → 「好走1回だけ超高値」の吊り上げ（ビザンチン118.3→15着の罠）を抑制。
    --   初出走(runs_prior=0)は情報ゼロで NULL。
    CASE WHEN runs_prior = 0 THEN NULL ELSE
        ROUND((COALESCE(good_prior * good_avg_q, 0.0)
               + (CASE WHEN good_prior >= 2 THEN 3.0 ELSE 8.0 END) * mu.m)
              / (COALESCE(good_prior, 0) + CASE WHEN good_prior >= 2 THEN 3.0 ELSE 8.0 END),
              1) END                                                 AS power_top,
    ROUND(good_last3f, 2)                                            AS shunpatsu,   -- 低いほど良
    ROUND(jizoku_q, 1)                                              AS jizoku,
    n_jizoku_good,
    ROUND(top_q, 1)                                                AS toppspeed,
    ROUND(power_q, 1)                                              AS power,
    ROUND(stamina_q, 1)                                            AS stamina,
    n_longish, n_long_good,
    -- スタミナ複勝率（中長距離）
    CASE WHEN n_longish > 0 THEN ROUND(1.0*n_long_good/n_longish, 3) END AS stamina_show,
    n_off,
    -- 道悪複勝率（経験2走以上で信頼）
    CASE WHEN n_off > 0 THEN ROUND(1.0*n_off_good/n_off, 3) END      AS off_show,
    avg_pos AS start
FROM agg CROSS JOIN mu;
