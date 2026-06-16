#!/usr/bin/env python3
# =============================================================================
# 競馬予想 予測スクリプト（出走前レースへの適用）
#
# train_predict.py --save-model で保存したモデルを読み込み、
# 「まだ結果の出ていないレース（results.finish_position が NULL）」の各馬に対して
# 的中確率と期待値(EV)を算出してランキング表示する。
#
# 出馬表は scraper.py --shutuba で取り込める（results に finish=NULL で登録）。
# v_features は過去走のみから特徴量を作るため、結果未確定でも特徴量が成立する。
#
# 使い方:
#   # 1) 出馬表を取り込む（結果未確定レースを登録）
#   python scripts/scraper.py --shutuba --db keiba.db 202406010111
#   # 2) 特徴量ビューを最新化して予測
#   sqlite3 keiba.db < schema/features.sql
#   python scripts/predict.py --db keiba.db --model model.pkl
#   # 特定レースだけ / EV閾値で買い目を絞る
#   python scripts/predict.py --db keiba.db --model model.pkl --race-id 202406010111
#   python scripts/predict.py --db keiba.db --model model.pkl --ev-threshold 1.2
# =============================================================================
from __future__ import annotations

import argparse
import sys

import pandas as pd

import mlcommon


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="保存済みモデルで出走前レースを予測")
    p.add_argument("--db", default="keiba.db", help="SQLite DB（v_features を参照）")
    p.add_argument("--model", default="model.pkl", help="保存済みモデル（train_predict.py の出力）")
    p.add_argument("--race-id", help="このレースだけ予測（未指定なら結果未確定レース全件）")
    p.add_argument("--ev-threshold", type=float, default=None,
                   help="指定すると EV がこの値以上の馬（買い目候補）だけ表示")
    p.add_argument("--topk", type=int, default=5, help="各レースで表示する上位頭数")
    args = p.parse_args(argv)

    bundle = mlcommon.load_model(args.model)
    model = bundle["model"]
    print(f"モデル: {bundle.get('model_name')} / 学習ターゲット: {bundle.get('target')}")

    df = mlcommon.load_data(args.db, None)

    # 対象レースの決定
    if args.race_id:
        target_ids = [args.race_id]
    else:
        target_ids = mlcommon.upcoming_race_ids(args.db)
        if not target_ids:
            print("結果未確定のレースが見つかりません。"
                  "出馬表を scraper.py --shutuba で取り込んでください。", file=sys.stderr)
            return 0

    df = df[df["race_id"].isin(target_ids)].reset_index(drop=True)
    if df.empty:
        print(f"[NG] 対象レースの特徴量がありません（race_id={target_ids}）。"
              "features.sql を適用しましたか?", file=sys.stderr)
        return 1

    # 学習時と同じ特徴量列に揃えて予測
    x = mlcommon.build_features(df, feature_columns=bundle["feature_columns"])
    df = df.assign(p=model.predict_proba(x)[:, 1])
    df["ev"] = df["p"] * pd.to_numeric(df["odds"], errors="coerce")

    cols = [c for c in ["horse_number", "horse_name", "odds", "popularity", "p", "ev"]
            if c in df.columns]
    for rid, g in df.groupby("race_id"):
        g = g.sort_values("p", ascending=False)
        if args.ev_threshold is not None:
            g = g[g["ev"] >= args.ev_threshold]
        g = g.head(args.topk)
        print(f"\n=== race_id={rid} ===")
        if g.empty:
            print("（条件を満たす馬なし）")
            continue
        out = g[cols].assign(p=g["p"].round(3), ev=g["ev"].round(2))
        print(out.to_string(index=False))

    print("\n※ EV>1.0 は理論上の期待値プラス。馬券は自己責任で。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
