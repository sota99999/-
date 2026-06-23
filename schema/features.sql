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
DROP VIEW IF EXISTS v_features;   -- v_speed/v_course_bias に依存するため先に落とす
DROP VIEW IF EXISTS v_pace_ref;
DROP VIEW IF EXISTS v_course_bias;
DROP VIEW IF EXISTS v_trainer_region;
DROP VIEW IF EXISTS v_speed;

-- Eloレーティング表が無くても v_features を作れるよう、空でも用意しておく
-- （実データは scripts/compute_ratings.py が投入する）
CREATE TABLE IF NOT EXISTS horse_ratings (
    race_id TEXT NOT NULL, horse_id TEXT NOT NULL,
    elo_before REAL, PRIMARY KEY (race_id, horse_id)
);

-- ラップ表が無くても v_features を作れるよう、空でも用意しておく
-- （実データは scripts/collect_laps.py が投入する）
CREATE TABLE IF NOT EXISTS race_laps (
    race_id TEXT PRIMARY KEY, lap_seq TEXT, n_laps INTEGER,
    first_half REAL, second_half REAL, race_first3f REAL,
    race_last3f REAL, pace_diff REAL
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
-- v_course_bias: コース別の内外枠バイアス
--   (競馬場×馬場種別×距離帯) ごとに、内枠半分と外枠半分の複勝率の差を集計。
--   inner_bias > 0 = 内枠有利, < 0 = 外枠有利。長期の統計なのでリークは無視できる。
-- -----------------------------------------------------------------------------
CREATE VIEW v_course_bias AS
WITH d AS (
    SELECT
        ra.venue_id, ra.surface,
        CASE WHEN ra.distance < 1400 THEN 'sprint'
             WHEN ra.distance < 1800 THEN 'mile'
             WHEN ra.distance < 2200 THEN 'mid'
             ELSE 'long' END AS dist_band,
        1.0 * r.post_position / ra.field_size AS dr,
        CASE WHEN r.finish_position <= 3 THEN 1.0 ELSE 0.0 END AS placed
    FROM results r JOIN races ra ON r.race_id = ra.race_id
    WHERE r.finish_position IS NOT NULL AND ra.field_size > 0 AND r.post_position IS NOT NULL
)
SELECT
    venue_id, surface, dist_band,
    COUNT(*) AS n,
    ROUND(AVG(CASE WHEN dr <  0.5 THEN placed END)
        - AVG(CASE WHEN dr >= 0.5 THEN placed END), 4) AS inner_bias
FROM d
GROUP BY venue_id, surface, dist_band;


-- -----------------------------------------------------------------------------
-- v_pace_ref: 距離帯×馬場種別ごとの平均 pace_diff（前後傾の基準）
--   pace_diff は短距離ほど前傾に偏るため、レースの「後傾(瞬発)/前傾(持続)」は
--   この基準より上か下かで相対判定する（距離由来の偏りを除く）。
-- -----------------------------------------------------------------------------
CREATE VIEW v_pace_ref AS
SELECT
    ra.surface,
    CASE WHEN ra.distance < 1400 THEN 'sprint'
         WHEN ra.distance < 1800 THEN 'mile'
         WHEN ra.distance < 2200 THEN 'mid'
         ELSE 'long' END AS dist_band,
    AVG(rl.pace_diff) AS avg_pace_diff,
    AVG(rl.race_first3f) AS avg_first3f,
    COUNT(*) AS n
FROM race_laps rl
JOIN races ra ON rl.race_id = ra.race_id
WHERE rl.pace_diff IS NOT NULL
GROUP BY ra.surface, dist_band;


-- -----------------------------------------------------------------------------
-- v_trainer_region: 調教師の東西所属を出走分布から推定（西=中京07/京都08/阪神09/小倉10）
--   trainers.affiliation(美浦/栗東)が未取得のため、その厩舎の全出走の東西多数決で代用。
--   栗東所属は西開催に集中するので実用上ほぼ正確。輸送(遠征)判定に使う。
-- -----------------------------------------------------------------------------
CREATE VIEW v_trainer_region AS
SELECT r.trainer_id,
       CASE WHEN SUM(CASE WHEN ra.venue_id IN ('07','08','09','10') THEN 1 ELSE 0 END)
                 > SUM(CASE WHEN ra.venue_id IN ('01','02','03','04','05','06') THEN 1 ELSE 0 END)
            THEN 'West' ELSE 'East' END AS region
FROM results r
JOIN races ra ON ra.race_id = r.race_id
WHERE r.trainer_id IS NOT NULL
GROUP BY r.trainer_id;

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
        END AS class_level,
        -- 距離帯（距離適性の集計キー）
        CASE WHEN ra.distance < 1400 THEN 'sprint'
             WHEN ra.distance < 1800 THEN 'mile'
             WHEN ra.distance < 2200 THEN 'mid'
             ELSE 'long' END AS dist_band,
        -- 道悪フラグ（良以外）
        CASE WHEN ra.track_condition IS NOT NULL AND ra.track_condition <> '良'
             THEN 1 ELSE 0 END AS is_offtrack,
        -- コース別枠バイアス適合: 内枠有利コースで内枠なら正, 外枠なら負（n>=100のみ）
        CASE WHEN cb.inner_bias IS NOT NULL AND cb.n >= 100 AND ra.field_size > 0
             THEN ROUND(cb.inner_bias * (0.5 - 1.0 * r.post_position / ra.field_size) * 2, 4)
             END AS draw_bias_fit,
        -- 直線の長さ区分（②似た条件 sim_ のキー）。京都外/阪神外も実際は長いが、
        --   検証で旧定義(新潟04・東京05・中京07のみ)の方が◎軸が良かったため旧定義を採用。
        CASE WHEN ra.venue_id IN ('04','05','07') THEN 'long' ELSE 'short' END AS straight_cat,
        -- 坂区分: ゴール前の坂（パワー適性）。中山06・阪神09・中京07=急坂、
        --   東京05・福島03=直線に坂（緩い）、他=平坦（京都08・新潟04・小倉10・函館02・札幌01）
        CASE WHEN ra.venue_id IN ('06','09','07') THEN 'steep'
             WHEN ra.venue_id IN ('05','03') THEN 'mild'
             ELSE 'flat' END AS hill_type,
        -- 内回り/外回り区分（芝のみ。確実な距離だけ内/外を判定し、不確実は other=通常）
        --   ダートは内外区別なしで dirt。誤ラベルを避けるため曖昧な距離は other に倒す
        CASE WHEN ra.surface <> '芝' THEN 'dirt'
             WHEN ra.venue_id='06' AND ra.distance IN (1800,2000,2500,3600) THEN 'inner'  -- 中山
             WHEN ra.venue_id='06' AND ra.distance IN (1200,1600,2200,4000) THEN 'outer'
             WHEN ra.venue_id='08' AND ra.distance IN (1100,1200,2000,3000) THEN 'inner'  -- 京都
             WHEN ra.venue_id='08' AND ra.distance IN (1800,2200,2400,3200) THEN 'outer'
             WHEN ra.venue_id='09' AND ra.distance IN (1200,1400,2000,2200,2400,3000) THEN 'inner' -- 阪神
             WHEN ra.venue_id='09' AND ra.distance IN (1600,1800) THEN 'outer'
             WHEN ra.venue_id='04' AND ra.distance = 1200 THEN 'inner'                    -- 新潟
             WHEN ra.venue_id='04' AND ra.distance IN (1400,1600,1800) THEN 'outer'
             ELSE 'other' END AS io_type,
        -- 小回り/広いコース区分（コーナーがタイト＝先行/器用さ有利）。
        --   小回り: 札幌01・函館02・福島03・小倉10（全体が小回り）＋主要4場の内回り
        CASE WHEN ra.venue_id IN ('01','02','03','10') THEN 'tight'
             WHEN ra.venue_id='06' AND ra.distance IN (1800,2000,2500,3600) THEN 'tight'  -- 中山内
             WHEN ra.venue_id='08' AND ra.distance IN (1100,1200,2000,3000) THEN 'tight'  -- 京都内
             WHEN ra.venue_id='09' AND ra.distance IN (1200,1400,2000,2200,2400,3000) THEN 'tight' -- 阪神内
             WHEN ra.venue_id='04' AND ra.distance = 1200 THEN 'tight'                    -- 新潟内
             ELSE 'wide' END AS turn_type,
        -- 輸送（遠征）フラグ: レースの東西と調教師の所属東西が違えば1（遠征＝輸送あり）。
        --   西開催=中京07/京都08/阪神09/小倉10、他=東。所属不明(NULL)の厩舎は0扱い。
        CASE WHEN tr.region IS NULL THEN 0
             WHEN tr.region = (CASE WHEN ra.venue_id IN ('07','08','09','10') THEN 'West' ELSE 'East' END)
                  THEN 0 ELSE 1 END AS is_transport,
        -- そのレースのラップ性質: 同距離帯平均より後傾(=瞬発/上がり勝負)なら1, 前傾(=持続)なら0
        CASE WHEN rl.pace_diff IS NULL OR pr.avg_pace_diff IS NULL THEN NULL
             WHEN rl.pace_diff > pr.avg_pace_diff THEN 1 ELSE 0 END AS lap_back,
        -- ペースの速さ区分: 前半3F(600m)が同距離帯平均より速ければハイ、遅ければスロー。
        --   「流れ(瞬発/持続=lap_back)」とは独立の軸（スロー×持続=ロングスパートも有り得る）
        CASE WHEN rl.race_first3f IS NULL OR pr.avg_first3f IS NULL THEN NULL
             WHEN rl.race_first3f <= pr.avg_first3f - 0.6 THEN 'high'
             WHEN rl.race_first3f >= pr.avg_first3f + 0.6 THEN 'slow'
             ELSE 'mid' END AS pace_level
    FROM results r
    JOIN races  ra ON r.race_id  = ra.race_id
    LEFT JOIN horses h ON r.horse_id = h.horse_id
    LEFT JOIN v_trainer_region tr ON tr.trainer_id = r.trainer_id
    LEFT JOIN v_speed s ON s.race_id = r.race_id AND s.horse_id = r.horse_id
    LEFT JOIN horse_ratings hr ON hr.race_id = r.race_id AND hr.horse_id = r.horse_id
    LEFT JOIN race_laps rl ON rl.race_id = r.race_id
    LEFT JOIN v_pace_ref pr
           ON pr.surface = ra.surface
          AND pr.dist_band = CASE WHEN ra.distance < 1400 THEN 'sprint'
                                  WHEN ra.distance < 1800 THEN 'mile'
                                  WHEN ra.distance < 2200 THEN 'mid'
                                  ELSE 'long' END
    LEFT JOIN v_course_bias cb
           ON cb.venue_id = ra.venue_id AND cb.surface = ra.surface
          AND cb.dist_band = CASE WHEN ra.distance < 1400 THEN 'sprint'
                                  WHEN ra.distance < 1800 THEN 'mile'
                                  WHEN ra.distance < 2200 THEN 'mid'
                                  ELSE 'long' END
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
        -- 適性: 道悪（過去の道悪レースだけの複勝率）。道悪未経験なら NULL
        SUM(is_offtrack) OVER w_hist                      AS off_runs_prior,
        ROUND(1.0 * SUM(CASE WHEN is_offtrack = 1 AND finish_position <= 3 THEN 1 ELSE 0 END) OVER w_hist
              / NULLIF(SUM(is_offtrack) OVER w_hist, 0), 3) AS off_show_rate_prior,
        -- 適性: 距離帯（同じ距離帯での過去複勝率）
        COUNT(*) OVER w_dist                              AS dist_runs_prior,
        ROUND(1.0 * SUM(finish_position <= 3) OVER w_dist / COUNT(*) OVER w_dist, 3) AS dist_show_rate_prior,
        -- 適性: 回り（同じ右/左回りでの過去複勝率）
        COUNT(*) OVER w_dir                               AS dir_runs_prior,
        ROUND(1.0 * SUM(finish_position <= 3) OVER w_dir / COUNT(*) OVER w_dir, 3) AS dir_show_rate_prior,
        -- ① 同条件（競馬場×馬場種別×距離帯×道悪フラグ）の過去能力・成績
        COUNT(*) OVER w_same                              AS same_runs_prior,
        SUM(finish_position = 1) OVER w_same              AS same_wins_prior,
        ROUND(1.0 * SUM(finish_position <= 3) OVER w_same / COUNT(*) OVER w_same, 3) AS same_show_rate_prior,
        ROUND(AVG(speed_index) OVER w_same, 1)            AS same_avg_speed_prior,
        MAX(speed_index) OVER w_same                      AS same_best_speed_prior,
        ROUND(AVG(CASE WHEN finish_position <= 3 THEN speed_index END) OVER w_same, 1) AS same_good_avg_speed_prior,
        ROUND(AVG(finish_position) OVER w_same, 2)        AS same_avg_finish_prior,
        MIN(last_3f) OVER w_same                          AS same_best_last3f_prior,
        -- ① 同条件（緩め1: 競馬場×馬場種別×距離帯、道悪不問）。勝利数=コース実績の鋭い指標
        COUNT(*) OVER w_samed                             AS samed_runs_prior,
        SUM(finish_position = 1) OVER w_samed             AS samed_wins_prior,
        ROUND(1.0 * SUM(finish_position <= 3) OVER w_samed / COUNT(*) OVER w_samed, 3) AS samed_show_rate_prior,
        ROUND(AVG(speed_index) OVER w_samed, 1)           AS samed_avg_speed_prior,
        MAX(speed_index) OVER w_samed                     AS samed_best_speed_prior,
        ROUND(AVG(CASE WHEN finish_position <= 3 THEN speed_index END) OVER w_samed, 1) AS samed_good_avg_speed_prior,
        -- ① 正確距離スペシャリスト（馬場種別×ちょうど同じ距離、競馬場不問）
        COUNT(*) OVER w_xd                               AS xd_runs_prior,
        SUM(finish_position = 1) OVER w_xd               AS xd_wins_prior,
        ROUND(1.0 * SUM(finish_position <= 3) OVER w_xd / COUNT(*) OVER w_xd, 3) AS xd_show_rate_prior,
        ROUND(AVG(speed_index) OVER w_xd, 1)             AS xd_avg_speed_prior,
        MAX(speed_index) OVER w_xd                       AS xd_best_speed_prior,
        ROUND(AVG(CASE WHEN finish_position <= 3 THEN speed_index END) OVER w_xd, 1) AS xd_good_avg_speed_prior,
        -- ① 坂適性（馬場種別×坂区分。急坂/緩坂/平坦が同じ過去走＝パワー適性）
        COUNT(*) OVER w_hill                              AS hill_runs_prior,
        SUM(finish_position = 1) OVER w_hill             AS hill_wins_prior,
        ROUND(1.0 * SUM(finish_position <= 3) OVER w_hill / COUNT(*) OVER w_hill, 3) AS hill_show_rate_prior,
        MAX(speed_index) OVER w_hill                     AS hill_best_speed_prior,
        ROUND(AVG(CASE WHEN finish_position <= 3 THEN speed_index END) OVER w_hill, 1) AS hill_good_avg_speed_prior,
        -- ① 内外適性（馬場種別×内外区分。内/外が同じ過去走＝コース形態適性）
        COUNT(*) OVER w_io                               AS io_runs_prior,
        SUM(finish_position = 1) OVER w_io               AS io_wins_prior,
        ROUND(1.0 * SUM(finish_position <= 3) OVER w_io / COUNT(*) OVER w_io, 3) AS io_show_rate_prior,
        MAX(speed_index) OVER w_io                       AS io_best_speed_prior,
        ROUND(AVG(CASE WHEN finish_position <= 3 THEN speed_index END) OVER w_io, 1) AS io_good_avg_speed_prior,
        -- ① 小回り適性（馬場種別×小回り/広い。器用さ・先行力）
        COUNT(*) OVER w_turn                             AS turn_runs_prior,
        SUM(finish_position = 1) OVER w_turn            AS turn_wins_prior,
        ROUND(1.0 * SUM(finish_position <= 3) OVER w_turn / COUNT(*) OVER w_turn, 3) AS turn_show_rate_prior,
        MAX(speed_index) OVER w_turn                     AS turn_best_speed_prior,
        ROUND(AVG(CASE WHEN finish_position <= 3 THEN speed_index END) OVER w_turn, 1) AS turn_good_avg_speed_prior,
        -- ① 輸送適性（輸送有無が同じ過去走＝遠征時にどれだけ走れるか）
        COUNT(*) OVER w_trans                            AS trans_runs_prior,
        ROUND(1.0 * SUM(finish_position <= 3) OVER w_trans / COUNT(*) OVER w_trans, 3) AS trans_show_rate_prior,
        MAX(speed_index) OVER w_trans                    AS trans_best_speed_prior,
        -- ① 同条件（緩め2: 競馬場×馬場種別、距離不問＝コース適性）
        COUNT(*) OVER w_vs                                AS vs_runs_prior,
        ROUND(1.0 * SUM(finish_position <= 3) OVER w_vs / COUNT(*) OVER w_vs, 3) AS vs_show_rate_prior,
        ROUND(AVG(speed_index) OVER w_vs, 1)              AS vs_avg_speed_prior,
        MAX(speed_index) OVER w_vs                        AS vs_best_speed_prior,
        ROUND(AVG(CASE WHEN finish_position <= 3 THEN speed_index END) OVER w_vs, 1) AS vs_good_avg_speed_prior,
        -- ② 似た条件（回り×直線長×馬場種別×距離帯）の過去能力・成績
        COUNT(*) OVER w_sim                               AS sim_runs_prior,
        ROUND(1.0 * SUM(finish_position <= 3) OVER w_sim / COUNT(*) OVER w_sim, 3) AS sim_show_rate_prior,
        ROUND(AVG(speed_index) OVER w_sim, 1)             AS sim_avg_speed_prior,
        MAX(speed_index) OVER w_sim                       AS sim_best_speed_prior,
        ROUND(AVG(CASE WHEN finish_position <= 3 THEN speed_index END) OVER w_sim, 1) AS sim_good_avg_speed_prior,
        -- ② 馬場種別×距離帯（控えのベース能力。同/似条件が無い馬の下支え）
        COUNT(*) OVER w_sd                                AS sd_runs_prior,
        ROUND(AVG(speed_index) OVER w_sd, 1)              AS sd_avg_speed_prior,
        MAX(speed_index) OVER w_sd                        AS sd_best_speed_prior,
        ROUND(AVG(CASE WHEN finish_position <= 3 THEN speed_index END) OVER w_sd, 1) AS sd_good_avg_speed_prior,
        -- ③ 瞬発力適性（後傾ラップ=上がり勝負だった過去走での成績・時計）
        SUM(CASE WHEN lap_back = 1 THEN 1 ELSE 0 END) OVER w_hist AS shun_runs_prior,
        ROUND(1.0 * SUM(CASE WHEN lap_back = 1 AND finish_position <= 3 THEN 1 ELSE 0 END) OVER w_hist
              / NULLIF(SUM(CASE WHEN lap_back = 1 THEN 1 ELSE 0 END) OVER w_hist, 0), 3) AS shun_show_rate_prior,
        ROUND(AVG(CASE WHEN lap_back = 1 THEN speed_index END) OVER w_hist, 1) AS shun_avg_speed_prior,
        -- ③ 持続力適性（前傾ラップ=ロングスパートだった過去走での成績・時計）
        SUM(CASE WHEN lap_back = 0 THEN 1 ELSE 0 END) OVER w_hist AS mochi_runs_prior,
        ROUND(1.0 * SUM(CASE WHEN lap_back = 0 AND finish_position <= 3 THEN 1 ELSE 0 END) OVER w_hist
              / NULLIF(SUM(CASE WHEN lap_back = 0 THEN 1 ELSE 0 END) OVER w_hist, 0), 3) AS mochi_show_rate_prior,
        ROUND(AVG(CASE WHEN lap_back = 0 THEN speed_index END) OVER w_hist, 1) AS mochi_avg_speed_prior,
        -- ③ ペースの速さ適性（過去のスロー/ハイ ペース戦での好走率＝流れの速さへの強さ）
        SUM(CASE WHEN pace_level = 'slow' THEN 1 ELSE 0 END) OVER w_hist AS slowp_runs_prior,
        ROUND(1.0 * SUM(CASE WHEN pace_level = 'slow' AND finish_position <= 3 THEN 1 ELSE 0 END) OVER w_hist
              / NULLIF(SUM(CASE WHEN pace_level = 'slow' THEN 1 ELSE 0 END) OVER w_hist, 0), 3) AS slowp_show_rate_prior,
        SUM(CASE WHEN pace_level = 'high' THEN 1 ELSE 0 END) OVER w_hist AS highp_runs_prior,
        ROUND(1.0 * SUM(CASE WHEN pace_level = 'high' AND finish_position <= 3 THEN 1 ELSE 0 END) OVER w_hist
              / NULLIF(SUM(CASE WHEN pace_level = 'high' THEN 1 ELSE 0 END) OVER w_hist, 0), 3) AS highp_show_rate_prior,
        -- 直近フォーム（直近3走の複勝率・平均着順、当該レースを除く）
        ROUND(1.0 * SUM(finish_position <= 3) OVER w_recent / COUNT(*) OVER w_recent, 3) AS recent3_show_rate,
        ROUND(AVG(finish_position) OVER w_recent, 2)      AS recent3_avg_finish,
        -- 前走情報・ローテーション
        LAG(finish_position) OVER w_ord                   AS prev_finish,
        LAG(popularity)      OVER w_ord                   AS prev_popularity,
        LAG(surface)         OVER w_ord                   AS prev_surface,
        LAG(distance)        OVER w_ord                   AS prev_distance,
        LAG(class_level)     OVER w_ord                   AS prev_class_level,   -- 前走クラス格
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
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        -- 距離帯別・回り別の過去走のみ（現在行を除く）
        w_dist   AS (PARTITION BY horse_id, dist_band ORDER BY race_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        w_dir    AS (PARTITION BY horse_id, direction ORDER BY race_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        -- ① 同条件: 競馬場×馬場種別×距離帯×道悪フラグ が一致する過去走のみ
        w_same   AS (PARTITION BY horse_id, venue_id, surface, dist_band, is_offtrack
                     ORDER BY race_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        -- ① 同条件（緩め1）: 道悪を問わない 競馬場×馬場種別×距離帯（サンプル増）
        w_samed  AS (PARTITION BY horse_id, venue_id, surface, dist_band
                     ORDER BY race_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        -- ① 同条件（緩め2）: 距離も問わない 競馬場×馬場種別（コース適性。最もサンプルが多い）
        w_vs     AS (PARTITION BY horse_id, venue_id, surface
                     ORDER BY race_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        -- ② 似た条件: 回り×直線長×馬場種別×距離帯 が一致（例 東京⇔新潟=左・長直線）
        w_sim    AS (PARTITION BY horse_id, direction, straight_cat, surface, dist_band
                     ORDER BY race_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        -- ② 馬場種別×距離帯（控えのベース能力）
        w_sd     AS (PARTITION BY horse_id, surface, dist_band
                     ORDER BY race_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        -- ① 正確距離: 馬場種別×「ちょうど同じ距離」（距離帯では薄まる専門性を取り出す）
        w_xd     AS (PARTITION BY horse_id, surface, distance
                     ORDER BY race_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        -- ① 坂適性: 馬場種別×坂区分（急坂/緩坂/平坦）が一致する過去走（パワー適性）
        w_hill   AS (PARTITION BY horse_id, surface, hill_type
                     ORDER BY race_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        -- ① 内外適性: 馬場種別×内外区分（内/外）が一致する過去走（コース形態適性）
        w_io     AS (PARTITION BY horse_id, surface, io_type
                     ORDER BY race_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        -- ① 小回り適性: 馬場種別×小回り/広い が一致する過去走（器用さ・先行力）
        w_turn   AS (PARTITION BY horse_id, surface, turn_type
                     ORDER BY race_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        -- ① 輸送適性: 輸送有無が一致する過去走（遠征時の好走耐性）
        w_trans  AS (PARTITION BY horse_id, is_transport
                     ORDER BY race_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        -- 直近3走（現在行の直前3走）
        w_recent AS (PARTITION BY horse_id ORDER BY race_date
                     ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING)
),
rot AS (
    -- ローテ: 休み明け(70日超 or 初出走)で区切り番号を増やし「叩き何戦目」を数える
    SELECT base.*,
        SUM(CASE WHEN days_since_last IS NULL OR days_since_last > 70 THEN 1 ELSE 0 END)
            OVER (PARTITION BY horse_id ORDER BY race_date
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS layoff_group
    FROM base
),
withpace AS (
    -- 展開（ペース）推定をレース単位で先に付与（pace_fit 計算のため）
    SELECT rot.*,
        ROUND(AVG(run_style_prior) OVER (PARTITION BY race_id), 3) AS race_pace_estimate
    FROM rot
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
    -- 適性・直近フォーム（再収集不要・既存データから導出）
    off_runs_prior, off_show_rate_prior,                 -- 道悪適性
    dist_runs_prior, dist_show_rate_prior,               -- 距離適性
    dir_runs_prior, dir_show_rate_prior,                 -- 回り適性
    -- ① 同条件の能力（競馬場×馬場種別×距離帯×道悪）。*_wins_prior=同条件での勝利数
    same_runs_prior, same_wins_prior, same_show_rate_prior, same_avg_speed_prior, same_best_speed_prior,
    same_good_avg_speed_prior,
    samed_runs_prior, samed_wins_prior, samed_show_rate_prior, samed_avg_speed_prior, samed_best_speed_prior,
    samed_good_avg_speed_prior,
    xd_runs_prior, xd_wins_prior, xd_show_rate_prior, xd_avg_speed_prior, xd_best_speed_prior,
    xd_good_avg_speed_prior,
    hill_runs_prior, hill_wins_prior, hill_show_rate_prior, hill_best_speed_prior, hill_good_avg_speed_prior,
    io_runs_prior, io_wins_prior, io_show_rate_prior, io_best_speed_prior, io_good_avg_speed_prior,
    turn_runs_prior, turn_wins_prior, turn_show_rate_prior, turn_best_speed_prior, turn_good_avg_speed_prior,
    is_transport, trans_runs_prior, trans_show_rate_prior, trans_best_speed_prior,
    vs_runs_prior, vs_show_rate_prior, vs_avg_speed_prior, vs_best_speed_prior, vs_good_avg_speed_prior,
    same_avg_finish_prior, same_best_last3f_prior,
    -- ② 似た条件の能力（回り×直線長×馬場種別×距離帯 / 馬場種別×距離帯）
    sim_runs_prior, sim_show_rate_prior, sim_avg_speed_prior, sim_best_speed_prior, sim_good_avg_speed_prior,
    sd_runs_prior, sd_avg_speed_prior, sd_best_speed_prior, sd_good_avg_speed_prior,
    -- ③ 展開・ラップ適性（瞬発力＝後傾実績 / 持続力＝前傾実績）
    shun_runs_prior, shun_show_rate_prior, shun_avg_speed_prior,
    mochi_runs_prior, mochi_show_rate_prior, mochi_avg_speed_prior,
    slowp_runs_prior, slowp_show_rate_prior, highp_runs_prior, highp_show_rate_prior,  -- ペース速さ適性(参照)
    recent3_show_rate, recent3_avg_finish,               -- 直近3走フォーム
    prev_finish, prev_popularity, prev_surface, prev_distance, days_since_last, distance_change,
    -- ---- ローテーション（再収集不要・既存データから導出） ------------------
    prev_class_level,                                     -- 前走のクラス格
    class_level - prev_class_level                        AS class_change,        -- 正=昇級, 負=降級
    CASE WHEN days_since_last IS NULL OR days_since_last > 70 THEN 1 ELSE 0 END AS is_layoff,       -- 休み明け
    CASE WHEN days_since_last <= 8 THEN 1 ELSE 0 END      AS is_back_to_back,     -- 連闘
    CASE WHEN prev_surface IS NOT NULL AND prev_surface <> surface THEN 1 ELSE 0 END AS surface_change, -- 芝⇄ダ替わり
    ROW_NUMBER() OVER (PARTITION BY horse_id, layoff_group ORDER BY race_date)  AS races_since_layoff, -- 叩き何戦目
    -- 展開（ペース）推定: 出走各馬の脚質の平均（低い=前残り少なめ=ハイペース傾向）
    race_pace_estimate,
    -- 展開×脚質適合: 差し馬×ハイペース / 逃げ馬×スローペースで正（展開に恵まれる）
    CASE WHEN run_style_prior IS NOT NULL AND race_pace_estimate IS NOT NULL
         THEN ROUND(-(run_style_prior - 0.5) * (race_pace_estimate - 0.5) * 4, 3)
         END AS pace_fit,
    draw_bias_fit,                                        -- コース別枠バイアス適合
    target_win, target_show
FROM withpace;
