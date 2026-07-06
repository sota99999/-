# JRA競馬予想システム — プロジェクトメモ

## 実行環境（最重要）
- **コードは全て Google Colab で実行**。このリポジトリ内に DB・ネットワークは無い。
  Claude はコードを書いてコミットし、**ユーザーが Colab で実行して出力を貼り返す**往復で進める。
- DB は `/content/keiba.db`。永続化は **Google Drive** (`/content/drive/MyDrive/keiba.db`)。
  Colab 環境は頻繁にリセットされるため、セルは必ず**自己完結**（リカバリ込み）で書く:
  Drive mount → `git clone -b claude/upbeat-ride-x477hz`（既存なら fetch+reset --hard）→
  DB を Drive から復元 → `schema/features.sql` `course_master.sql` `ability.sql` を executescript。
- 開発ブランチ: `claude/upbeat-ride-x477hz`。コミットしたら必ず push（Colab が pull するため）。

## 確定版モデル（2026-07 クローズ。再設計しない）
**予想値 = 相対R（r_adj）一本**。カードは `scripts/card.py --ability-only`。
1. **相対R** (`scripts/compute_relative_r.py` → `horse_relative_r`):
   反復 strength-of-schedule レーティング。perf = 出走全馬平均R + 40×着順スコア。
   リーク防止は月次スナップショット（当月より前の結果のみ）。r_before/perf/pstd/pmax を保存。
2. **ムラ馬の条件補正** (`scripts/cond_rating.py`, THRESH=14):
   pstd(perfのばらつき)≥14 の馬**だけ**、今回条件（芝ダ・坂・回り・距離±300m）に
   合致した過去走 perf の平均で置換（印「条」）。堅実馬は触らない。
3. **参考列（スコア非加算）**: pmax(最高=天井)、power_top(能力=時計)、レース条件に応じた
   6能力（東京→ﾄｯﾌﾟ/瞬発、急坂→ﾊﾟﾜｰ、小回り→先行、京都外→持続、長距離→ｽﾀﾐﾅ/距離、道悪→道悪）。
4. **印**: ⭐=特出(z≥1.5)が最上位枠、◎○▲△はその後ろから（⭐1頭→○から、2頭→▲から）。
   ✅=オッズ妙味。回顧では印つき馬の的中のみ「◎単/○複」表示。

### 実証成績（リーク防止・実オッズ）
- 直近3週(216R): 本命 単25%/複49%、回収84%/75%。
- ⭐=買われすぎ（複勝向き）。**◎（⭐不在の混戦筆頭）が単勝妙味**（単回収86%）。
- 得意: クラス戦(1勝以上)・マイルG1・中長距離の実力決着。苦手: 新馬(予想不能)・短距離G1・3歳混戦。

### 検証済みで**不採用**の案（再提案・再実験しない）
- 予想値への加算: 斤量(単勝を破壊)・展開trip・コース傾き → STEP1のquality_spで織込済み、二重計上。
- 一本化の試み: 時計を相手強度で補正 / 相対Rの着差版 / z線形ブレンド(U字で有害) →
  **順位ベース相対評価が時計・着差に常に勝つ**。
- 条件評価の全馬適用（加重平均・best3/6・trim・max）→ ムラでない馬まで動き集計悪化。
  ゲート付き（pstd≥14のみ）だけが有効。
- 格上挑戦ディスカウント → 全クラス悪化、主目的（デビットバローズ函館11着）は格上ですらなかった。
- 教訓: **個別の取りこぼしをルールで潰すと過学習**。ベースは変えず実戦追跡で検証する。

## パイプライン（週次運用）
```
python scripts/weekly.py --db keiba.db --raceday YYYY-MM-DD
  = 結果収集→出馬表→最新オッズ→ラップ→Elo→相対R→スキーマ適用→確定版カード
python scripts/site_single.py --db keiba.db --out /content/drive/MyDrive/keiba.html
  = 自分専用の閲覧ページ(1ファイル: 馬検索/カード/ランキング)を Drive に生成
最後に必ず: shutil.copy(DB, '/content/drive/MyDrive/keiba.db')
```
個別ツール: `horse.py 馬名`（プロファイル）/ `ranking.py`（上位100）/ `doctor.py`（健全性診断）。

## データの既知の仕様（バグではない）
- 相対Rの立ち上げ期（最初の90日）と新馬戦（全馬デビュー）には r_before が付かない。
- power_top はレース内で「61.8」が並ぶことがある（好走0の馬は母集団平均μに縮約されるため）。
- v_run_adj は speed_index −50〜150 の範囲外を除外（異常タイム対策）。
- 新潟芝2000 の内外は距離で判別不能 → 新潟記念のみ内回り扱い（course_master.sql 参照）。

## 旧系統（残置・確定版では未使用）
Elo(`horse_ratings`)はフォールバック用。v_features/LightGBM(model_*.pkl)/predict.py/
bet_optimizer.py は別系統のMLモード（weekly --ml でのみ使用）。v_predict の
dist_adj/off_adj はビューに残るがカードは読まない。

## 出力の流儀
- ユーザーへの Colab セルは**コピペで動く完全形**で提示（変数未定義エラーを出さない）。
- 検証は必ず「in-memory スタブでロジック確認 → コミット → 実データはユーザーが実行」の順。
- バックテストは全期間・クラス別（未勝利/1勝/2勝/OP重賞）で頭合わせし、
  「下級を壊さず上位で改善」を確認してから採用する。
- 設計判断・実験結果は `docs/ability_design.md` に記録してある（詳細はそちら）。
