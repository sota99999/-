#!/usr/bin/env python3
# =============================================================================
# モデルが馬の評価に使っている情報の「重要度」を % で表示する
#
# 学習済みモデル（model_*.pkl）から特徴量の重要度を取り出し、評価項目
# （能力・コース適性・距離適性・道悪適性・騎手・展開・枠 …）ごとに合算して
# 全体に占める割合(%)を出す。「今どの情報をどれだけ重視しているか」が分かる。
#
# 使い方:
#   python scripts/importance.py --model model_win_noodds.pkl
#   python scripts/importance.py --model model_show_noodds.pkl   # 複勝モデル
#
# 注意: 重要度は LightGBM の gain（分岐での損失改善量）ベース。値は相対的な
#       目安で、因果関係や「効き方の向き」を示すものではない。
# =============================================================================
from __future__ import annotations

import argparse

import numpy as np

import mlcommon

# 評価項目（カテゴリ）への分類。先頭一致/完全一致で振り分ける。
ONEHOT_PREFIX = {
    "venue_id": "コース(競馬場)",
    "surface": "馬場種別(芝/ダ)",
    "track_condition": "馬場状態(道悪)",
    "direction": "回り(右/左)",
    "weather": "天候",
    "prev_surface": "ローテ(前走馬場)",
}
COL_CATEGORY = {
    # 能力（基礎）
    "elo_before": "能力(Elo/実績)", "win_rate_prior": "能力(Elo/実績)",
    "show_rate_prior": "能力(Elo/実績)", "avg_finish_prior": "能力(Elo/実績)",
    "wins_prior": "能力(Elo/実績)", "runs_prior": "能力(Elo/実績)",
    # 能力（時計）
    "avg_speed_prior": "能力(時計/SP)", "best_speed_prior": "能力(時計/SP)",
    "prev_speed": "能力(時計/SP)", "best_last3f_prior": "能力(時計/SP)",
    "class_adj_speed_prior": "能力(時計/SP)",
    # クラス
    "class_level": "クラス(格)", "avg_class_level_prior": "クラス(格)",
    "prev_class_level": "クラス(格)", "class_change": "クラス(格)",
    # 近走フォーム
    "recent3_show_rate": "近走フォーム", "recent3_avg_finish": "近走フォーム",
    "prev_finish": "近走フォーム", "prev_popularity": "近走フォーム",
    # 適性
    "course_show_rate_prior": "コース適性", "course_runs_prior": "コース適性",
    "dist_show_rate_prior": "距離適性", "dist_runs_prior": "距離適性",
    "distance": "距離適性", "distance_change": "距離適性", "prev_distance": "距離適性",
    "off_show_rate_prior": "道悪適性", "off_runs_prior": "道悪適性",
    "dir_show_rate_prior": "回り適性", "dir_runs_prior": "回り適性",
    # 騎手
    "jockey_win_rate_prior": "騎手", "jockey_show_rate_prior": "騎手",
    "jockey_rides_prior": "騎手",
    # 展開
    "race_pace_estimate": "展開(ペース)", "pace_fit": "展開(ペース)",
    "run_style_prior": "展開(脚質)",
    # 枠・バイアス
    "draw_ratio": "枠/バイアス", "draw_bias_fit": "枠/バイアス",
    "post_position": "枠/バイアス", "horse_number": "枠/バイアス",
    # ローテーション
    "days_since_last": "ローテーション", "is_layoff": "ローテーション",
    "is_back_to_back": "ローテーション", "races_since_layoff": "ローテーション",
    "layoff_group": "ローテーション", "surface_change": "ローテーション",
    # 馬体・斤量・条件
    "horse_weight": "馬体重", "weight_change": "馬体重",
    "weight_carried": "斤量", "field_size": "出走頭数",
    "odds": "オッズ/人気", "popularity": "オッズ/人気",
}


def categorize(col: str) -> str:
    for pref, cat in ONEHOT_PREFIX.items():
        if col == pref or col.startswith(pref + "_"):
            return cat
    return COL_CATEGORY.get(col, "その他")


def _raw_fi(est):
    """1モデルから特徴量重要度ベクトルを取り出す（LightGBM gain 優先）。"""
    if est is None:
        return None
    try:
        if hasattr(est, "booster_"):
            return np.asarray(est.booster_.feature_importance(importance_type="gain"),
                              dtype=float)
    except Exception:  # noqa: BLE001
        pass
    if hasattr(est, "feature_importances_"):
        return np.asarray(est.feature_importances_, dtype=float)
    return None


def importances(model):
    """CalibratedClassifierCV 等のラッパーを剥がして重要度を平均で取り出す。"""
    if hasattr(model, "calibrated_classifiers_"):
        arr = []
        for cc in model.calibrated_classifiers_:
            est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            fi = _raw_fi(est)
            if fi is not None:
                arr.append(fi)
        if arr:
            return np.mean(arr, axis=0)
    return _raw_fi(model)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="特徴量の重要度を評価項目ごとに%表示")
    p.add_argument("--model", default="model_win_noodds.pkl")
    p.add_argument("--detail", action="store_true", help="特徴量1本ごとの内訳も表示")
    args = p.parse_args(argv)

    bundle = mlcommon.load_model(args.model)
    cols = bundle["feature_columns"]
    fi = importances(bundle["model"])
    if fi is None:
        print("このモデルからは重要度を取得できません（LightGBM以外の可能性）。"
              "train_predict.py を LightGBM で学習し直すと取得できます。")
        return 0
    if len(fi) != len(cols):
        n = min(len(fi), len(cols))
        fi, cols = fi[:n], cols[:n]

    total = float(fi.sum()) or 1.0
    # カテゴリ集計
    cat_sum: dict[str, float] = {}
    detail: dict[str, list] = {}
    for c, v in zip(cols, fi):
        cat = categorize(c)
        cat_sum[cat] = cat_sum.get(cat, 0.0) + float(v)
        detail.setdefault(cat, []).append((c, float(v)))

    print(f"\n=== 評価項目の重要度（{args.model}） ===")
    print("モデルが馬を評価する際に各情報をどれだけ使っているか（gain比率, 合計100%）\n")
    print(f"{'評価項目':<16}{'重要度':>8}")
    for cat, v in sorted(cat_sum.items(), key=lambda kv: kv[1], reverse=True):
        bar = "█" * int(round(v / total * 50))
        print(f"{cat:<16}{v/total*100:>6.1f}%  {bar}")

    if args.detail:
        print("\n--- 特徴量ごとの内訳 ---")
        for cat, v in sorted(cat_sum.items(), key=lambda kv: kv[1], reverse=True):
            print(f"\n[{cat}] {v/total*100:.1f}%")
            for c, fv in sorted(detail[cat], key=lambda x: x[1], reverse=True):
                if fv > 0:
                    print(f"   {c:<26}{fv/total*100:>5.1f}%")

    print("\n※gain（分岐での損失改善量）ベースの相対重要度。"
          "効き方の向き(プラス/マイナス)ではなく『どれだけ判断に使うか』を示す。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
