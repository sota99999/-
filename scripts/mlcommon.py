#!/usr/bin/env python3
# =============================================================================
# 学習・予測の共通ユーティリティ
#   train_predict.py（学習）と predict.py（予測）で共有する。
#   - データ読み込み
#   - 特徴量行列の構築（学習時と同じ列に揃える reindex 付き）
#   - モデル生成（LightGBM 優先・sklearn フォールバック）
#   - モデルの保存/読込（特徴量列・ターゲット等のメタも一緒に保存）
# =============================================================================
from __future__ import annotations

import pickle
import sqlite3
import sys

import pandas as pd

# 特徴量に使わない列（識別子・正解ラベル・分割キー）
DROP_COLS = ["race_id", "horse_id", "horse_name", "race_date",
             "target_win", "target_show"]
# カテゴリとして扱う列（one-hot 化）
CATEGORICAL = ["venue_id", "surface", "prev_surface",
               "track_condition", "direction", "weather"]


# -----------------------------------------------------------------------------
# データ読み込み
# -----------------------------------------------------------------------------
def load_data(db: str | None, csv: str | None, min_runs: int = 0) -> pd.DataFrame:
    """v_features を DataFrame で返す。min_runs で新馬等を除外できる。"""
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
    if min_runs and "runs_prior" in df.columns:
        df = df[df["runs_prior"].fillna(0) >= min_runs].reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# 特徴量行列
# -----------------------------------------------------------------------------
def build_features(df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    """特徴量行列 X を作る。

    feature_columns を渡すと、その列順・列集合に reindex して揃える
    （学習時と予測時で one-hot 後の列を一致させるために必須）。
    """
    x = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    cat = [c for c in CATEGORICAL if c in x.columns]
    x = pd.get_dummies(x, columns=cat, dummy_na=True)
    x = x.apply(pd.to_numeric, errors="coerce")
    if feature_columns is not None:
        x = x.reindex(columns=feature_columns, fill_value=0)
    return x


# -----------------------------------------------------------------------------
# モデル生成
# -----------------------------------------------------------------------------
def make_model():
    """LightGBM が使えればそれを、無ければ sklearn を返す。"""
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
# モデルの保存 / 読込
# -----------------------------------------------------------------------------
def save_model(path: str, model, feature_columns: list[str], target: str,
               model_name: str) -> None:
    """モデル本体と、予測時に必要なメタ情報を一緒に保存する。"""
    bundle = {
        "model": model,
        "feature_columns": list(feature_columns),
        "target": target,
        "model_name": model_name,
        "format": 1,
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_model(path: str) -> dict:
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if "model" not in bundle or "feature_columns" not in bundle:
        sys.exit(f"[NG] {path} は不正なモデルファイルです。")
    return bundle


# -----------------------------------------------------------------------------
# 結果未確定（出走前）レース
# -----------------------------------------------------------------------------
def normalize_by_race(df: pd.DataFrame, prob_col: str = "p",
                      race_col: str = "race_id", out_col: str = "p_norm") -> pd.DataFrame:
    """各レース内で勝率を合計1に正規化した列を付与して返す。

    モデルの単勝確率は1頭ずつ独立に出るためレース内合計が1にならない。
    Harville モデルや出馬間の比較では合計1の確率が前提になるので正規化する。
    """
    df = df.copy()
    s = df.groupby(race_col)[prob_col].transform("sum")
    df[out_col] = df[prob_col] / s.where(s > 0, other=1.0)
    return df


def kelly_fraction(p: float, odds: float) -> float:
    """単勝のケリー基準による最適賭け率（資金に対する割合）を返す。

    odds は単勝オッズ（払戻倍率, 例 3.0 なら的中で元本含め3倍）。
    純オッズ b = odds - 1 として f* = (p*odds - 1) / b。
    期待値マイナス（p*odds <= 1）や不正な値は 0 にクリップする。
    """
    if not (odds and odds > 1) or p is None or not (0 <= p <= 1):
        return 0.0
    b = odds - 1.0
    f = (p * odds - 1.0) / b
    return max(0.0, min(1.0, f))


def upcoming_race_ids(db: str) -> list[str]:
    """結果が1頭も確定していない（finish_position が全 NULL）レースID一覧。"""
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            """SELECT race_id FROM results
               GROUP BY race_id
               HAVING COUNT(finish_position) = 0
               ORDER BY race_id"""
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]
