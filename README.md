# 競馬予想データベース

netkeiba 等からスクレイピングしたデータを格納し、予想に活用するための
**SQLite** データベースのスキーマと分析クエリ集です。

## 特徴

- ファイル1つで完結する SQLite。セットアップ不要ですぐ使える
- 「出走馬1頭×1レース = 1行」の `results` ファクト表を中心とした設計
- 血統（父・母父）、騎手、調教師、馬体重、上がり3F など予想に重要な項目を網羅
- すぐ使える予想用分析クエリ（騎手成績・血統適性・枠順・人気別回収率 など）

## ディレクトリ構成

```
.
├── schema/
│   ├── schema.sql        # テーブル定義(DDL)・インデックス・ビュー
│   ├── seed_venues.sql   # 競馬場マスタ初期データ
│   └── features.sql      # 機械学習向け特徴量ビュー v_features
├── queries/
│   └── analysis.sql      # 予想用の分析クエリ集
├── data/
│   └── sample.sql        # 動作確認用ダミーデータ
├── scripts/
│   ├── init_db.sh         # DB初期化スクリプト
│   ├── scraper.py         # netkeiba 単一レース取り込み
│   ├── crawler.py         # 未取得レースだけを巡回する差分クローラ
│   ├── export_features.py # 学習用 特徴量CSVエクスポート
│   └── requirements.txt   # 依存パッケージ
└── README.md
```

## セットアップ

### `sqlite3` CLI がある場合

```bash
# スキーマ + 競馬場マスタを投入して keiba.db を作成
./scripts/init_db.sh

# サンプルデータも入れて動作確認したい場合
./scripts/init_db.sh keiba.db --sample
```

### Python だけで使う場合（CLI が無い環境）

```bash
python3 - <<'PY'
import sqlite3, pathlib
db = sqlite3.connect("keiba.db")
for f in ["schema/schema.sql", "schema/seed_venues.sql"]:
    db.executescript(pathlib.Path(f).read_text())
db.commit()
print("created keiba.db")
PY
```

## データモデル

| テーブル | 役割 |
|----------|------|
| `venues`   | 競馬場マスタ（場コード→競馬場名） |
| `horses`   | 馬マスタ（血統・性別・所属厩舎など） |
| `jockeys`  | 騎手マスタ |
| `trainers` | 調教師マスタ |
| `races`    | レース情報（日付・コース・距離・馬場状態など） |
| `results`  | **出走・成績（中心となるファクト表）** |
| `payouts`  | 払戻（券種ごとの組番・払戻金） |

分析時は、これらを結合済みの **`v_results_full` ビュー** を起点にすると
クエリが書きやすくなっています。

### `race_id` について

netkeiba の12桁レースID（例 `202405021211`）をそのまま主キーに使う想定です。

```
2024  05    02   01    11
年    場ｺｰﾄﾞ 開催回 開催日 R番号
```

## 分析クエリの例

`queries/analysis.sql` に以下を収録しています。

1. 騎手成績ランキング（勝率・連対率・複勝率・単勝回収率）
2. 種牡馬（父）のコース適性（芝/ダート × 距離帯）
3. 枠番別の有利不利（特定コース）
4. 人気別の信頼度と回収率
5. 馬1頭の過去走サマリ（出走前の検討用）
6. 騎手 × 調教師 コンビ成績
7. 馬体重増減と成績の関係
8. 各レースの上がり3F最速馬の抽出

`:venue` `:horse_name` などの `:param` はパラメータです。実行時に値を
バインドするか、リテラルに置き換えて使ってください。

```bash
# 例: 人気別回収率（クエリ#4）を実行
sqlite3 keiba.db < queries/analysis.sql
```

## データの投入（スクレイピング）

`scripts/scraper.py` が netkeiba（db.netkeiba.com）のレース結果ページを取得・
パースし、各テーブルへ UPSERT します。

```bash
pip install -r scripts/requirements.txt

# レースID(12桁)を指定して取り込み（複数可）
python scripts/scraper.py --db keiba.db 202405021211 202405021212
```

- リクエスト間に必ずウェイト（`REQUEST_INTERVAL=1.5秒`）を入れています。短縮しないでください。
- UPSERT なので同じレースを再取得しても重複しません。
- netkeiba の HTML 構造は変わりうるため、パースに失敗する場合は
  `parse_race()` 内のセレクタを調整してください。

### 差分クローラ（未取得レースだけ巡回）

`scripts/crawler.py` は、指定日（範囲）の開催レース一覧を netkeiba から取得し、
**まだ DB に無いレースだけ**を取り込みます。何度回しても無駄打ちがありません。

```bash
# 単一日
python scripts/crawler.py --db keiba.db --date 2024-05-12
# 期間（両端含む）。週末をまとめて取り込み
python scripts/crawler.py --db keiba.db --from 2024-05-01 --to 2024-05-31
# 取得対象の確認だけ（DBに書き込まない）
python scripts/crawler.py --date 2024-05-12 --dry-run
```

> netkeiba をスクレイピングする際は、対象サイトの利用規約・robots.txt を
> 確認し、アクセス間隔を空けるなど節度を持って行ってください。個人利用の
> 範囲にとどめることを推奨します。

## 機械学習用の特徴量

`schema/features.sql` は2つのビューを作ります。

- **`v_speed`**: 1走ごとの「スピード指数」。同じ (馬場種別×距離×馬場状態) の
  レース群の中での相対タイムを `平均=50, 速いほど高い` に指数化（母数20走未満は NULL）。
- **`v_features`**: 各出走時点で「その馬の過去走だけ」を集計した学習用特徴量。
  ウィンドウフレームを過去行に限定することで **当該レースの結果が混入しない
  （リークしない）** よう設計しています。

主な特徴量:

| 区分 | 特徴量 |
|------|--------|
| 通算成績 | 過去出走数 / 勝率・複勝率 / 平均着順 / 自己ベスト上がり |
| スピード | 過去平均スピード指数 / 自己最高スピード指数 / 前走スピード指数 |
| コース実績 | 同 競馬場×馬場種別 での過去出走数・複勝率 |
| 前走情報 | 前走着順・人気・馬場・距離 / 前走間隔(日) / 距離変更幅 |
| ターゲット | `target_win`（1着か）, `target_show`（3着以内か） |

```bash
sqlite3 keiba.db < schema/features.sql          # ビュー作成

# 学習用データを CSV 出力（scikit-learn / LightGBM 等の入力に）
python scripts/export_features.py --db keiba.db -o train.csv --min-runs 3
```

## 今後の拡張アイデア

- 調教（追い切り）データ、コーナー通過順位の構造化
- オッズの時系列（前日→締切）テーブル
- 騎手・血統の同条件成績を結合した特徴量
- 学習〜予測のサンプル（LightGBM 等）の追加
