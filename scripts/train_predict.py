#!/usr/bin/env python3
# =============================================================================
# 競馬予想 学習・予測サンプル
#
# v_features（schema/features.sql）の特徴量で「単勝(1着) or 複勝(3着以内)」を
# 予測するモデルを学習し、各馬の的中確率と期待値(EV)を出力する。
#
#   EV(単勝) = P(1着) × 単勝オッズ      … 1.0 を超えれば期待値プラス
#   EV(複勝) は複勝オッズが必要だが本DBには無いため、単勝EVを既定とする。
#
# リーク防止のため、学習/検証は race_date による時系列分割（古い→学習,
# 新しい→検証）で行う。v_features 自体も過去走限定で作られている。
#
# 使い方:
#   pip install -r scripts/requirements-ml.txt
#   sqlite3 keiba.db < schema/features.sql        # ビュー作成
#   python scripts/train_predict.py --db keiba.db
#   python scripts/train_predict.py --csv train.csv --target target_show
#   python scripts/train_predict.py --db keiba.db --ev-threshold 1.2 --topk 3
#
# LightGBM が入っていれば LightGBM を、無ければ scikit-learn の
# HistGradientBoostingClassifier を自動的に使う。
# =============================================================================
from __future__ import annotations

import argparse
import sqlite3
import sys

import numpy as np
import pandas as pd

# 特徴量に使わない列（識別子・正解ラベル・分割キー）
DROP_COLS = ["race_id", "horse_id", "horse_name", "race_date",
             "target_win", "target_show"]
# カテゴリとして扱う列
CATEGORICAL = ["venue_id", "surface", "prev_surface"]


# -----------------------------------------------------------------------------
# データ読み込み
# -----------------------------------------------------------------------------
def load_data(db: str | None, csv: str | None) -> pd.DataFrame:
    if csv:
        df = pd.read_csv(csv)
    else:
        conn = sqlite3.connect(db)
        try:
            df = pd.read_sql_query("SELECT * FROM v_features ORDER BY race_date", conn)
        finally:
            conn.close()
    if "race_date" not in df.columns:
        sys.exit("[NG] race_date 列がありません。v_features を参照していますか?")
    return df


def build_xy(df: pd.DataFrame, target: str):
    """特徴量行列 X とラベル y を作る。カテゴリは one-hot 化する。"""
    y = df[target].astype(int)
    x = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    cat_cols = [c for c in CATEGORICAL if c in x.columns]
    x = pd.get_dummies(x, columns=cat_cols, dummy_na=True)
    # 残りは数値化（変換不能は NaN に。決定木系は NaN をそのまま扱える）
    x = x.apply(pd.to_numeric, errors="coerce")
    return x, y


# -----------------------------------------------------------------------------
# モデル（LightGBM 優先、無ければ sklearn）
# -----------------------------------------------------------------------------
def make_model():
    try:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
        )
        return model, "LightGBM"
    except Exception:  # noqa: BLE001  未導入/ビルド不可ならフォールバック
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, random_state=42,
        )
        return model, "sklearn.HistGradientBoosting"


# -----------------------------------------------------------------------------
# 時系列分割
# -----------------------------------------------------------------------------
def time_split(df: pd.DataFrame, valid_frac: float):
    """race_date の新しい方 valid_frac 割を検証に回す。境界はレース単位で揃える。"""
    dates = np.sort(df["race_date"].unique())
    cut = dates[int(len(dates) * (1 - valid_frac))]
    train_mask = df["race_date"] < cut
    valid_mask = ~train_mask
    return train_mask, valid_mask, cut


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
    p = argparse.ArgumentParser(description="競馬予想 学習・予測サンプル")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--db", default="keiba.db", help="SQLite DB（v_features を参照）")
    src.add_argument("--csv", help="features CSV（export_features.py の出力）")
    p.add_argument("--target", default="target_win",
                   choices=["target_win", "target_show"], help="予測ターゲット")
    p.add_argument("--valid-frac", type=float, default=0.2, help="検証に回す割合（新しい側）")
    p.add_argument("--ev-threshold", type=float, default=1.1, help="購入する単勝EVの閾値")
    p.add_argument("--topk", type=int, default=3, help="各レースで表示する上位頭数")
    p.add_argument("--min-runs", type=int, default=1, help="過去出走数の下限（新馬を除外）")
    args = p.parse_args(argv)

    df = load_data(None if args.csv else args.db, args.csv)
    if "runs_prior" in df.columns:
        df = df[df["runs_prior"].fillna(0) >= args.min_runs].reset_index(drop=True)
    if len(df) < 50:
        print(f"[警告] データが少なすぎます（{len(df)}行）。結果は参考程度です。", file=sys.stderr)

    train_mask, valid_mask, cut = time_split(df, args.valid_frac)
    x, y = build_xy(df, args.target)
    x_tr, y_tr = x[train_mask], y[train_mask]
    x_va, y_va = x[valid_mask], y[valid_mask]
    print(f"分割: 学習 {len(x_tr)}行 / 検証 {len(x_va)}行（{cut} 以降を検証）")
    if len(x_tr) == 0 or len(x_va) == 0 or y_tr.nunique() < 2:
        sys.exit("[NG] 学習・検証に十分なデータ/クラスがありません。")

    model, model_name = make_model()
    print(f"モデル: {model_name} / ターゲット: {args.target}")
    model.fit(x_tr, y_tr)
    proba = model.predict_proba(x_va)[:, 1]

    # ---- 評価指標 ----
    from sklearn.metrics import log_loss, roc_auc_score
    try:
        auc = roc_auc_score(y_va, proba)
    except ValueError:
        auc = float("nan")
    ll = log_loss(y_va, proba, labels=[0, 1])
    print(f"\n=== 検証スコア ===")
    print(f"AUC      : {auc:.3f}")
    print(f"LogLoss  : {ll:.3f}")

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
    show_cols = ["horse_name", "odds", "popularity", "p", "ev"]
    show_cols = [c for c in show_cols if c in latest.columns]
    print(f"\n=== 予測例: race_id={latest['race_id'].iloc[0]} 上位{args.topk}頭 ===")
    out = latest.sort_values("p", ascending=False).head(args.topk)[show_cols]
    out = out.assign(p=out["p"].round(3), ev=out["ev"].round(2))
    print(out.to_string(index=False))
    print("\n※ EV>1.0 は理論上の期待値プラス。実運用は十分なデータ量と検証が前提です。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
