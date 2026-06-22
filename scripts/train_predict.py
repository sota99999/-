#!/usr/bin/env python3
# =============================================================================
# 競馬予想 学習サンプル
#
# v_features（schema/features.sql）の特徴量で「単勝(1着) or 複勝(3着以内)」を
# 予測するモデルを学習し、検証スコアと単勝回収率バックテストを表示する。
# --save-model でモデルを保存すると、predict.py で出走前レースに適用できる。
#
#   EV(単勝) = P(1着) × 単勝オッズ      … 1.0 を超えれば期待値プラス
#
# リーク防止のため、学習/検証は race_date による時系列分割（古い→学習,
# 新しい→検証）で行う。v_features 自体も過去走限定で作られている。
#
# 使い方:
#   pip install -r scripts/requirements-ml.txt
#   sqlite3 keiba.db < schema/features.sql        # ビュー作成
#   python scripts/train_predict.py --db keiba.db
#   python scripts/train_predict.py --csv train.csv --target target_show
#   python scripts/train_predict.py --db keiba.db --save-model model.pkl
# =============================================================================
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import mlcommon


# -----------------------------------------------------------------------------
# 時系列分割
# -----------------------------------------------------------------------------
def time_split(df: pd.DataFrame, valid_frac: float):
    """race_date の新しい方 valid_frac 割を検証に回す。境界はレース単位で揃える。"""
    dates = np.sort(df["race_date"].unique())
    cut = dates[int(len(dates) * (1 - valid_frac))]
    train_mask = df["race_date"] < cut
    return train_mask, ~train_mask, cut


# -----------------------------------------------------------------------------
# 回収率バックテスト（検証データで単勝EV>閾値の馬に100円ずつ賭ける）
# -----------------------------------------------------------------------------
def backtest_win(df_valid: pd.DataFrame, proba: np.ndarray, ev_threshold: float):
    odds = pd.to_numeric(df_valid["odds"], errors="coerce").to_numpy()
    win = df_valid["target_win"].to_numpy()
    ev = proba * odds
    pick = (ev >= ev_threshold) & np.isfinite(odds)
    n = int(pick.sum())
    if n == 0:
        return {"bets": 0, "hits": 0, "roi": None, "hit_rate": None}
    stake = n * 100
    payout = float(np.sum(np.where(pick & (win == 1), odds * 100, 0)))
    return {
        "bets": n,
        "hits": int(np.sum(pick & (win == 1))),
        "hit_rate": round(100.0 * np.sum(pick & (win == 1)) / n, 1),
        "roi": round(100.0 * payout / stake, 1),
    }


# -----------------------------------------------------------------------------
# メイン
# -----------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="競馬予想 学習サンプル")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--db", default="keiba.db", help="SQLite DB（v_features を参照）")
    src.add_argument("--csv", help="features CSV（export_features.py の出力）")
    p.add_argument("--target", default="target_win",
                   choices=["target_win", "target_show"], help="予測ターゲット")
    p.add_argument("--valid-frac", type=float, default=0.2, help="検証に回す割合（新しい側）")
    p.add_argument("--ev-threshold", type=float, default=1.1, help="購入する単勝EVの閾値")
    p.add_argument("--topk", type=int, default=3, help="各レースで表示する上位頭数")
    p.add_argument("--min-runs", type=int, default=1, help="過去出走数の下限（新馬を除外）")
    p.add_argument("--calibrate", default="none", choices=["none", "sigmoid", "isotonic"],
                   help="確率較正の方法（既定none）。sigmoid=Platt, isotonic=単調回帰")
    p.add_argument("--no-odds", action="store_true",
                   help="オッズ・人気を特徴量から除外（出馬表段階でオッズ未取得でも予想できるモデル）")
    p.add_argument("--feature-set", choices=["full", "condition"], default="full",
                   help="full=全特徴量, condition=①同条件②似た条件③展開のみ（条件特化）")
    p.add_argument("--save-model", help="学習後にモデルを保存するパス（例: model.pkl）")
    args = p.parse_args(argv)

    df = mlcommon.load_data(None if args.csv else args.db, args.csv, args.min_runs)
    if not args.csv:
        # 結果未確定（出走前）レースは正解ラベルが無いので学習から除外
        upcoming = set(mlcommon.upcoming_race_ids(args.db))
        if upcoming:
            df = df[~df["race_id"].isin(upcoming)].reset_index(drop=True)
    if len(df) < 50:
        print(f"[警告] データが少なすぎます（{len(df)}行）。結果は参考程度です。", file=sys.stderr)

    train_mask, valid_mask, cut = time_split(df, args.valid_frac)
    extra_drop = ["odds", "popularity"] if args.no_odds else None
    keep = mlcommon.CONDITION_FEATURES if args.feature_set == "condition" else None
    x = mlcommon.build_features(df, extra_drop=extra_drop, keep=keep)
    feature_columns = list(x.columns)
    if keep is not None:
        print(f"特徴量セット: condition（{len(feature_columns)}列）→ {feature_columns}")
    y = df[args.target].astype(int)
    x_tr, y_tr = x[train_mask], y[train_mask]
    x_va, y_va = x[valid_mask], y[valid_mask]
    print(f"分割: 学習 {len(x_tr)}行 / 検証 {len(x_va)}行（{cut} 以降を検証）")
    if len(x_tr) == 0 or len(x_va) == 0 or y_tr.nunique() < 2:
        sys.exit("[NG] 学習・検証に十分なデータ/クラスがありません。")

    base_model, model_name = mlcommon.make_model()
    if args.calibrate != "none":
        # CalibratedClassifierCV が交差検証で較正器も学習する（学習データ内で完結）
        from sklearn.calibration import CalibratedClassifierCV
        model = CalibratedClassifierCV(base_model, method=args.calibrate, cv=3)
        model_name = f"{model_name}+calib({args.calibrate})"
    else:
        model = base_model
    print(f"モデル: {model_name} / ターゲット: {args.target}")
    model.fit(x_tr, y_tr)
    proba = model.predict_proba(x_va)[:, 1]

    # ---- 評価指標 ----
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    try:
        auc = roc_auc_score(y_va, proba)
    except ValueError:
        auc = float("nan")
    ll = log_loss(y_va, proba, labels=[0, 1])
    brier = brier_score_loss(y_va, proba)
    print("\n=== 検証スコア ===")
    print(f"AUC      : {auc:.3f}")
    print(f"LogLoss  : {ll:.3f}")
    print(f"Brier    : {brier:.4f}  （較正の良さ。小さいほど良い）")

    # ---- 単勝回収率バックテスト ----
    df_va = df[valid_mask].reset_index(drop=True)
    bt = backtest_win(df_va, proba, args.ev_threshold)
    print(f"\n=== 単勝バックテスト（EV>={args.ev_threshold} の馬に100円） ===")
    if bt["bets"] == 0:
        print("対象ベットなし（閾値が高すぎる可能性）")
    else:
        print(f"購入: {bt['bets']}点 / 的中: {bt['hits']}点 "
              f"/ 的中率: {bt['hit_rate']}% / 回収率: {bt['roi']}%")

    # ---- 直近レースの予測例 ----
    df_va = df_va.assign(p=proba)
    df_va["ev"] = df_va["p"] * pd.to_numeric(df_va["odds"], errors="coerce")
    latest = df_va[df_va["race_id"] == df_va["race_id"].iloc[-1]]
    show_cols = [c for c in ["horse_name", "odds", "popularity", "p", "ev"] if c in latest.columns]
    print(f"\n=== 予測例: race_id={latest['race_id'].iloc[0]} 上位{args.topk}頭 ===")
    out = latest.sort_values("p", ascending=False).head(args.topk)[show_cols]
    out = out.assign(p=out["p"].round(3), ev=out["ev"].round(2))
    print(out.to_string(index=False))

    # ---- モデル保存 ----
    if args.save_model:
        mlcommon.save_model(args.save_model, model, feature_columns, args.target, model_name)
        print(f"\n[OK] モデルを保存しました: {args.save_model}")

    print("\n※ EV>1.0 は理論上の期待値プラス。実運用は十分なデータ量と検証が前提です。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
