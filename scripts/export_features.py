#!/usr/bin/env python3
# =============================================================================
# 学習用特徴量 CSV エクスポート
#   v_features ビュー（schema/features.sql）の内容を CSV に書き出す。
#   機械学習（scikit-learn / LightGBM 等）の入力データとして使う。
#
# 使い方:
#   python scripts/export_features.py                       # features.csv へ出力
#   python scripts/export_features.py --db keiba.db -o train.csv
#   python scripts/export_features.py --min-runs 3          # 過去3走以上の馬に限定
# =============================================================================
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="v_features を CSV 出力")
    parser.add_argument("--db", default="keiba.db", help="SQLite DBファイル")
    parser.add_argument("-o", "--out", default="features.csv", help="出力CSVパス")
    parser.add_argument("--min-runs", type=int, default=0,
                        help="過去出走数(runs_prior)がこの値以上の行のみ出力")
    args = parser.parse_args(argv)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "SELECT * FROM v_features WHERE COALESCE(runs_prior, 0) >= ? ORDER BY race_date",
            (args.min_runs,),
        )
    except sqlite3.OperationalError as e:
        print(f"[NG] v_features を参照できません。schema/features.sql を適用しましたか? ({e})",
              file=sys.stderr)
        return 1

    rows = cur.fetchall()
    if not rows:
        print("[警告] 出力対象の行がありません。", file=sys.stderr)
        return 0

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(rows[0].keys())
        writer.writerows([tuple(r) for r in rows])

    print(f"[OK] {len(rows)} 行を {args.out} に出力しました。")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
