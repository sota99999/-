-- =============================================================================
-- 動作確認用サンプルデータ（ダミー）
--   実在データではなく、スキーマとクエリの動作確認のための最小データ。
--   sqlite3 keiba.db < data/sample.sql で投入。
-- =============================================================================

-- 騎手
INSERT OR IGNORE INTO jockeys (jockey_id, name, affiliation) VALUES
    ('J001', '川田将雅', '栗東'),
    ('J002', 'ルメール', '美浦'),
    ('J003', '武豊',     '栗東');

-- 調教師
INSERT OR IGNORE INTO trainers (trainer_id, name, affiliation) VALUES
    ('T001', '友道康夫', '栗東'),
    ('T002', '国枝栄',   '美浦');

-- 馬
INSERT OR IGNORE INTO horses (horse_id, name, sex, birth_date, sire, broodmare_sire, trainer_id) VALUES
    ('H001', 'サンプルホース', '牡', '2020-03-15', 'ディープインパクト', 'キングカメハメハ', 'T001'),
    ('H002', 'テストランナー',  '牝', '2021-04-02', 'キズナ',           'クロフネ',         'T002'),
    ('H003', 'ダミーゴールド',  'セ', '2020-02-20', 'エピファネイア',   'サンデーサイレンス', 'T001');

-- レース（東京 芝2000m を2レース）
INSERT OR IGNORE INTO races
    (race_id, race_date, venue_id, race_number, race_name, grade, surface, distance, direction, weather, track_condition, field_size, prize_1st) VALUES
    ('202405021011', '2024-05-12', '05', 11, 'サンプルステークス', 'OP', '芝', 2000, '左', '晴', '良', 3, 35000000),
    ('202405021211', '2024-06-02', '05', 11, 'テスト記念',         'G3', '芝', 2000, '左', '曇', '稍重', 3, 43000000);

-- 出走・成績
INSERT OR IGNORE INTO results
    (race_id, horse_id, jockey_id, trainer_id, post_position, horse_number, finish_position, time_seconds, last_3f, weight_carried, horse_weight, weight_change, odds, popularity, prize_won) VALUES
    -- レース1
    ('202405021011', 'H001', 'J001', 'T001', 1, 1, 1, 118.5, 33.8, 57.0, 480,  4, 2.1, 1, 35000000),
    ('202405021011', 'H002', 'J002', 'T002', 2, 2, 2, 118.7, 33.5, 55.0, 446, -2, 3.4, 2, 14000000),
    ('202405021011', 'H003', 'J003', 'T001', 3, 3, 3, 119.0, 34.2, 57.0, 502, 10, 8.9, 3, 8800000),
    -- レース2
    ('202405021211', 'H002', 'J002', 'T002', 1, 1, 1, 120.1, 34.0, 55.0, 444, -2, 1.8, 1, 43000000),
    ('202405021211', 'H001', 'J001', 'T001', 2, 2, 2, 120.3, 34.1, 58.0, 484,  4, 2.5, 2, 17000000),
    ('202405021211', 'H003', 'J003', 'T001', 3, 3, 3, 120.8, 34.5, 57.0, 498, -4, 12.0, 3, 11000000);

-- 払戻（レース1の例）
INSERT OR IGNORE INTO payouts (race_id, bet_type, combination, payout, popularity) VALUES
    ('202405021011', '単勝', '1',     210,  1),
    ('202405021011', '複勝', '1',     110,  1),
    ('202405021011', '馬連', '1-2',   380,  2),
    ('202405021011', '三連複', '1-2-3', 1200, 3);
