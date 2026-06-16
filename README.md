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
│   ├── compute_ratings.py # 馬Eloレーティング算出（horse_ratings へ保存）
│   ├── mlcommon.py        # 学習・予測の共通処理（特徴量整形/モデル保存読込）
│   ├── train_predict.py   # 学習スクリプト（LightGBM/sklearn・較正・モデル保存）
│   ├── predict.py         # 保存モデルで出走前レースを予測（単勝・ケリー資金配分）
│   ├── bet_optimizer.py   # 馬連/ワイド/三連複/三連単の期待値最適化/バックテスト
│   ├── requirements.txt   # スクレイピングの依存
│   └── requirements-ml.txt# 学習・予測の依存
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
- 一時エラー（タイムアウト/429/5xx）は**自動でリトライ**（指数バックオフ 2→4→8→16秒＋ジッタ）。
  恒久エラー（403/404）は即座にあきらめます。UA はブラウザ相当に設定済みです。
- netkeiba の HTML 構造は変わりうるため、パースに失敗する場合は
  `parse_race()` / `parse_shutuba()` 内のセレクタを調整してください。

> **注意:** Claude Code on the web の実行環境では外部ネットワークが遮断されている
> 場合があり、その場合スクレイピングは 403 になります。データ収集は
> **ネットワークの通る環境（手元のPC等）で実行**してください。

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

- 一覧取得・各レース取得とも**リトライ＋指数バックオフ**付き。途中で止まっても、
  再実行すれば**取得済みは自動スキップ**して続きから収集します（中断・再開に強い）。
- 進捗は `[3/365] 2024-01-03: 開催36R / 未取得12R（累計 取込…）` の形で表示されます。

### まとまった学習データを集める例

予想モデル用に、まず1〜2年分をまとめて収集します（手元のPCで実行）。

```bash
# 例: 2024-01-01〜2026-06-15 を収集（数時間かかる。途中停止しても再実行で再開可）
python scripts/crawler.py --db keiba.db --from 2024-01-01 --to 2026-06-15
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
| レース条件 | 競馬場 / 芝ダート / 距離 / 馬場状態 / 回り / 天候 / 枠番・馬番 / 斤量 / 内外位置(draw_ratio) |
| 馬体 | 馬体重 / 馬体重増減 |
| 通算成績 | 過去出走数 / 勝率・複勝率 / 平均着順 / 自己ベスト上がり |
| スピード | 過去平均スピード指数 / 自己最高スピード指数 / 前走スピード指数 |
| 能力(相対) | Eloレーティング(elo_before) / クラス格(class_level) / クラス補正スピード(class_adj_speed_prior) / 平均クラス格 |
| 脚質・展開 | 脚質(run_style_prior, 通過順位から) / 展開ペース推定(race_pace_estimate) |
| 騎手 | 騎手の過去騎乗数・勝率・複勝率（リーク防止の集計） |
| コース実績 | 同 競馬場×馬場種別 での過去出走数・複勝率 |
| 前走情報 | 前走着順・人気・馬場・距離 / 前走間隔(日) / 距離変更幅 |
| ターゲット | `target_win`（1着か）, `target_show`（3着以内か） |

Eloレーティング(`elo_before`)は順序依存のため、SQLではなく専用スクリプトで
算出して `horse_ratings` に保存します。**特徴量ビューを作る前に実行**してください。

```bash
python scripts/compute_ratings.py --db keiba.db   # Elo算出（数秒〜）
sqlite3 keiba.db < schema/features.sql            # 特徴量ビュー作成

# 学習用データを CSV 出力（scikit-learn / LightGBM 等の入力に）
python scripts/export_features.py --db keiba.db -o train.csv --min-runs 3
```

## 学習・予測

`scripts/train_predict.py` が `v_features` を読み、単勝(1着)/複勝(3着以内)を
予測するモデルを学習し、**各馬の的中確率と期待値(EV)** を出力します。

```bash
pip install -r scripts/requirements-ml.txt

# DBから直接（既定: 単勝を予測）
python scripts/train_predict.py --db keiba.db
# CSVから / 複勝を予測 / 購入EV閾値や表示頭数を指定
python scripts/train_predict.py --csv train.csv --target target_show
python scripts/train_predict.py --db keiba.db --ev-threshold 1.2 --topk 3
# モデルを保存（予測で再利用）
python scripts/train_predict.py --db keiba.db --save-model model.pkl
# 確率較正（オッズと整合する確率に補正。Brierスコアで良し悪しを確認）
python scripts/train_predict.py --db keiba.db --calibrate isotonic --save-model model.pkl
```

検証スコアには **AUC・LogLoss** に加え **Brier スコア**（確率較正の良さ。小さいほど
予測確率が実際の的中率に近い）を表示します。`--calibrate sigmoid|isotonic` を付けると
学習データ内の交差検証で較正器も学習し、確率の偏りを補正します。

- **EV(単勝) = P(1着) × 単勝オッズ**。1.0 を超えると理論上の期待値プラス。
- 学習/検証は `race_date` による**時系列分割**（古い→学習・新しい→検証）で行い、
  リークを避けます。出力には AUC・LogLoss と、検証データでの**単勝回収率
  バックテスト**（EV閾値以上の馬に100円ずつ賭けた場合の的中率・回収率）を含みます。
- **LightGBM** があればそれを、無ければ scikit-learn の
  `HistGradientBoostingClassifier` を自動で使います。

> 出力はサンプル実装です。実運用には十分なデータ量・特徴量の追加・期間を分けた
> 厳密な検証が必要で、回収率の保証はありません。馬券は自己責任で。

## これから走るレースを予想する

保存したモデルで、**結果がまだ出ていないレース**を予想できます。出馬表を取り込むと
`results` に `finish_position=NULL` で登録され、`v_features` は過去走のみから特徴量を
作るため、結果未確定でも予測できます。

```bash
# 1) 出馬表（出走前）を取り込む（複数レース可）
python scripts/scraper.py --shutuba --db keiba.db 202406010111

# 2) 特徴量ビューを最新化
sqlite3 keiba.db < schema/features.sql

# 3) 保存済みモデルで予測（既定では結果未確定レースを自動で対象に）
python scripts/predict.py --db keiba.db --model model.pkl
# 特定レースだけ / EV閾値で買い目候補に絞る
python scripts/predict.py --db keiba.db --model model.pkl --race-id 202406010111
python scripts/predict.py --db keiba.db --model model.pkl --ev-threshold 1.2
```

`predict.py` は各馬の的中確率 `p` と期待値 `ev = p × 単勝オッズ` を高い順に表示します。
学習時は出走前レース（正解ラベルが無い）を自動で除外します。

### ケリー基準による資金配分

`--bankroll` を指定すると、各馬の推奨賭け金 `stake` をケリー基準で算出します。

```bash
# 資金1万円・ハーフケリー(既定)で推奨賭け金を表示
python scripts/predict.py --db keiba.db --model model.pkl --bankroll 10000
# フルケリー（分散大）にする場合
python scripts/predict.py --db keiba.db --model model.pkl --bankroll 10000 --kelly-fraction 1.0
```

- ケリー比率 `f* = (p×オッズ − 1) / (オッズ − 1)`。期待値マイナスの馬は 0（賭けない）。
- 既定は**ハーフケリー**（`--kelly-fraction 0.5`）。フルケリーは理論上の成長率は最大
  ですが分散が大きいため、実務では 1/2〜1/4 が無難です。

## 連勝式馬券の期待値最適化（馬連/ワイド/三連複/三連単）

`scripts/bet_optimizer.py` は、モデルの単勝確率を **Harville モデル**で組み合わせ確率に
変換し、買い目を評価します。

- **馬連(i,j)** = P(i,j が1・2着, 順不同)
- **ワイド(i,j)** = P(i,j がともに3着以内)
- **三連複(i,j,k)** = P(i,j,k が上位3着, 順不同)
- **三連単(i,j,k)** = P(i,j,k がこの着順) = `p_i · p_j/(1-p_i) · p_k/(1-p_i-p_j)`

`--bet` に `quinella`(馬連) / `wide`(ワイド) / `trio`(三連複) / `trifecta`(三連単) を指定します。

```bash
# 過去レースで回収率をバックテスト（実際の払戻 payouts と突き合わせ）
python scripts/bet_optimizer.py --db keiba.db --model model.pkl --backtest --bet quinella
python scripts/bet_optimizer.py --db keiba.db --model model.pkl --backtest --bet trio --topn 2

# 出走前レースの推奨買い目（確率順とフェアオッズ=1/確率）
python scripts/bet_optimizer.py --db keiba.db --model model.pkl --predict --bet trifecta --topn 5
```

- 単勝確率はレース内で**合計1に正規化**してから組み合わせ確率に変換します。
- バックテストは `payouts` テーブルの実配当で**的中率・回収率**を算出します
  （払戻は結果ページ取り込み `scraper.py` で登録されます）。
- predict では **フェアオッズ(=1/確率)** を表示。実際の馬連/ワイドオッズがこれより
  高ければ期待値プラスの目安になります。

## 今後の拡張アイデア

- 調教（追い切り）データ、コーナー通過順位の構造化
- オッズの時系列（前日→締切）テーブル
- 騎手・血統の同条件成績を結合した特徴量
- 連勝式の実オッズ取得による事前EV算出、複数券種のポートフォリオ最適化
