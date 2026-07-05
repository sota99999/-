-- =============================================================================
-- course_master: 中央10競馬場のコース形態マスタ（静的リファレンス）
--
-- 直線長・ゴール前坂の高低差・1周距離・回り・小回り/広い・洋芝・想定ペースを
-- 競馬場×馬場種別×コース区分(内/外/直/単一) の粒度で持つ。数値はJRA公表の
-- 標準値に基づく代表値（一部概算）。改修で変わり得るため代表値として扱う。
--
-- ※この段階では v_features には JOIN しない（距離→内外の対応付けは誤りやすく、
--   検証してから繋ぐ方針）。まずは正確な地形データを DB に載せることが目的。
--
-- 使い方: sqlite3 keiba.db < schema/course_master.sql
-- =============================================================================

DROP VIEW IF EXISTS v_course;
DROP TABLE IF EXISTS course_master;

CREATE TABLE course_master (
    venue_id    TEXT NOT NULL,          -- 01札幌 02函館 03福島 04新潟 05東京 06中山 07中京 08京都 09阪神 10小倉
    venue_name  TEXT NOT NULL,
    surface     TEXT NOT NULL,          -- '芝' / 'ダ'
    course_type TEXT NOT NULL,          -- '内' '外' '直' '芝' 'ダ'（内外の無い場は '芝'/'ダ'）
    turn_dir    TEXT,                   -- '右' / '左'
    straight_m  REAL,                   -- ゴール前の直線長(m)
    hill_m      REAL,                   -- ゴール前坂の高低差(m)。平坦=0、ダートは未確定でNULL
    hill_grade  TEXT,                   -- 'steep'(急坂) / 'mild'(緩坂) / 'flat'(平坦)
    lap_m       REAL,                   -- 1周距離(m)。直線コース等はNULL
    turn_size   TEXT,                   -- 'tight'(小回り) / 'wide'(広い) / 'none'(直線)
    turf_type   TEXT,                   -- 'normal'(野芝系) / 'noshiba'(洋芝) / 'dirt'
    pace_bias   TEXT,                   -- 'high' / 'mid' / 'slow'（そのコースで流れやすい地合い）
    note        TEXT,
    PRIMARY KEY (venue_id, surface, course_type)
);

-- ---- 芝 ----------------------------------------------------------------------
INSERT INTO course_master
(venue_id,venue_name,surface,course_type,turn_dir,straight_m,hill_m,hill_grade,lap_m,turn_size,turf_type,pace_bias,note) VALUES
('05','東京','芝','芝','左',525.9,2.0,'mild',2083.1,'wide','normal','mid','長い直線＋緩く長い上り坂。2000mは1角まで約126mで内先行有利'),
('06','中山','芝','内','右',310.0,2.2,'steep',1667.1,'tight','normal','mid','JRA最急のゴール前坂。小回り先行＋坂を上るパワー'),
('06','中山','芝','外','右',310.0,2.2,'steep',1839.7,'tight','normal','mid','外回りも直線310m。1200は下りで速い流れ'),
('09','阪神','芝','内','右',356.5,1.8,'steep',1689.0,'tight','normal','mid','小回り先行＋急坂。2000/2200など'),
('09','阪神','芝','外','右',476.3,1.8,'steep',2089.0,'wide','normal','mid','直線476mで瞬発力・差しが届く。1600/1800'),
('08','京都','芝','内','右',328.4,0.0,'flat',1782.8,'tight','normal','mid','直線は平坦。3角に淀の坂（上って下る）'),
('08','京都','芝','外','右',403.7,0.0,'flat',1894.3,'wide','normal','mid','直線平坦・長め。キレ味の瞬発力勝負'),
('07','中京','芝','芝','左',412.5,2.0,'steep',1705.9,'wide','normal','mid','急坂＋3-4角スパイラルCで差し台頭・タフ'),
('04','新潟','芝','外','左',658.7,0.0,'flat',2223.0,'wide','normal','slow','JRA最長直線658m・平坦。超スロー→瞬発力'),
('04','新潟','芝','内','左',358.7,0.0,'flat',1623.0,'tight','normal','mid','内回りは小回りで先行・立ち回り'),
('04','新潟','芝','直','左',1000.0,0.0,'flat',NULL,'none','normal','high','芝1000m直線競走。外枠一発のスプリント'),
('03','福島','芝','芝','右',292.0,1.9,'mild',1600.0,'tight','normal','mid','小回り＋直線に緩坂。荒れると外差し'),
('10','小倉','芝','芝','右',293.0,0.0,'flat',1615.1,'tight','normal','high','平坦・下り＋スパイラルCで高速決着'),
('01','札幌','芝','芝','右',266.1,0.0,'flat',1640.9,'wide','noshiba','mid','大回り円形・平坦・洋芝。前残り＋スタミナ'),
('02','函館','芝','芝','右',262.1,0.0,'flat',1626.6,'tight','noshiba','mid','JRA最短直線262m・小回り・洋芝。徹底先行');

-- ---- ダート（坂の高低差は未確定のためNULL。hill_gradeは各場の傾向）----------
INSERT INTO course_master
(venue_id,venue_name,surface,course_type,turn_dir,straight_m,hill_m,hill_grade,lap_m,turn_size,turf_type,pace_bias,note) VALUES
('05','東京','ダ','ダ','左',501.6,NULL,'mild',NULL,'wide','dirt','mid','ダート直線501m。芝スタートの距離は流れが速い'),
('06','中山','ダ','ダ','右',308.0,NULL,'steep',NULL,'tight','dirt','mid','ゴール前急坂。先行＋パワー'),
('09','阪神','ダ','ダ','右',352.7,NULL,'steep',NULL,'tight','dirt','mid','急坂。先行＋パワー'),
('08','京都','ダ','ダ','右',329.1,NULL,'flat',NULL,'tight','dirt','mid','平坦'),
('07','中京','ダ','ダ','左',410.7,NULL,'steep',NULL,'wide','dirt','mid','直線＋坂で差しも届く'),
('04','新潟','ダ','ダ','左',353.9,NULL,'flat',NULL,'tight','dirt','mid','平坦・速い時計'),
('03','福島','ダ','ダ','右',296.0,NULL,'mild',NULL,'tight','dirt','mid','小回り先行'),
('10','小倉','ダ','ダ','右',291.3,NULL,'flat',NULL,'tight','dirt','mid','小回り先行'),
('01','札幌','ダ','ダ','右',264.9,NULL,'flat',NULL,'wide','dirt','mid','洋芝開催のダート'),
('02','函館','ダ','ダ','右',260.3,NULL,'flat',NULL,'tight','dirt','mid','小回り先行');

-- -----------------------------------------------------------------------------
-- v_course: 各レース(races)にコース形態を紐づけるビュー。
--   距離→コース区分(内/外/直)を解決して course_master をJOINする。
--   内外のある芝は 中山06/京都08/阪神09/新潟04 のみ。他の芝は単一'芝'、ダートは'ダ'。
--   ※新潟2000は内外どちらも存在するが距離だけでは判別不能のため既定=外。
--     （新潟記念など内回り2000は個別に要修正）。想定外距離は未一致(NULL)。
-- -----------------------------------------------------------------------------
CREATE VIEW v_course AS
WITH r AS (
    SELECT
        race_id, venue_id, distance,
        CASE WHEN surface = '芝' THEN '芝' ELSE 'ダ' END AS surf,
        CASE
            WHEN surface <> '芝' THEN 'ダ'                                   -- ダートは内外なし
            WHEN venue_id IN ('05','07','03','10','01','02') THEN '芝'        -- 内外の無い単一芝
            -- 中山
            WHEN venue_id='06' AND distance IN (1800,2000,2500,3600) THEN '内'
            WHEN venue_id='06' AND distance IN (1200,1600,2200,4000) THEN '外'
            -- 京都
            WHEN venue_id='08' AND distance IN (1100,1200,2000,3000) THEN '内'
            WHEN venue_id='08' AND distance IN (1400,1600,1800,2200,2400,3200) THEN '外'
            -- 阪神
            WHEN venue_id='09' AND distance IN (1200,1400,2000,2200,2400,3000) THEN '内'
            WHEN venue_id='09' AND distance IN (1600,1800) THEN '外'
            -- 新潟（1000=直線コース、2000は既定で外）
            WHEN venue_id='04' AND distance=1000 THEN '直'
            WHEN venue_id='04' AND distance IN (1200,1400,2200,2400) THEN '内'
            WHEN venue_id='04' AND distance IN (1600,1800,2000) THEN '外'
            ELSE '芝'                                                        -- 想定外は単一扱い(未一致になり得る)
        END AS course_type
    FROM races
)
SELECT r.race_id, r.venue_id, r.surf AS surface, r.distance, r.course_type,
       cm.turn_dir, cm.straight_m, cm.hill_m, cm.hill_grade, cm.lap_m,
       cm.turn_size, cm.turf_type, cm.pace_bias
FROM r
LEFT JOIN course_master cm
  ON cm.venue_id = r.venue_id AND cm.surface = r.surf AND cm.course_type = r.course_type;
