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
