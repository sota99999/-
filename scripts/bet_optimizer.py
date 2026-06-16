#!/usr/bin/env python3
# =============================================================================
# 組み合わせ馬券（馬連・ワイド・三連複・三連単）の期待値最適化 / バックテスト
#
# 保存済みモデルの単勝確率を Harville モデルで組み合わせ確率に変換し、
#   - backtest: 過去レースで実際の払戻(payouts)に対する的中率・回収率を検証
#   - predict : 出走前レースの推奨買い目（確率の高い組）とフェアオッズを提示
# を行う。
#
# Harville モデル（単勝確率 p_i から着順を逐次サンプリングする近似）:
#   P(i1着,j2着,k3着) = p_i * p_j/(1-p_i) * p_k/(1-p_i-p_j)
#   馬連(i,j)   = P(i,j が1・2着, 順不同)
#   ワイド(i,j) = P(i,j がともに3着以内)
#   三連複(i,j,k) = P(i,j,k が上位3着, 順不同)
#   三連単(i,j,k) = P(i,j,k がこの着順)
#
# 使い方:
#   # バックテスト（検証期間の確定レースで検証）
#   python scripts/bet_optimizer.py --db keiba.db --model model.pkl --backtest --bet trio
#   # 出走前レースの買い目提示
#   python scripts/bet_optimizer.py --db keiba.db --model model.pkl --predict --bet trifecta
#   # 組数や賭け方を調整
#   python scripts/bet_optimizer.py --db keiba.db --model model.pkl --backtest \
#       --bet wide --topn 2 --valid-frac 0.2
# =============================================================================
from __future__ import annotations

import argparse
import itertools
import sqlite3
import sys

import numpy as np
import pandas as pd

import mlcommon


# -----------------------------------------------------------------------------
# Harville 組み合わせ確率（いずれも combo=馬番のタプルを受ける）
# -----------------------------------------------------------------------------
def _exacta(p: dict[int, float], a: int, b: int) -> float:
    """P(a=1着, b=2着)。"""
    d = 1 - p[a]
    return p[a] * p[b] / d if d > 0 else 0.0


def _trifecta_ordered(p: dict[int, float], a: int, b: int, c: int) -> float:
    """P(a=1着, b=2着, c=3着)。"""
    d1 = 1 - p[a]
    if d1 <= 0:
        return 0.0
    d2 = 1 - p[a] - p[b]
    if d2 <= 0:
        return 0.0
    return p[a] * (p[b] / d1) * (p[c] / d2)


def quinella_prob(p, combo) -> float:
    """馬連（i,j が1・2着、順不同）。"""
    i, j = combo
    return _exacta(p, i, j) + _exacta(p, j, i)


def wide_prob(p, combo) -> float:
    """ワイド（i,j がともに3着以内）。"""
    i, j = combo
    others = [k for k in p if k not in (i, j)]
    return sum(_trifecta_ordered(p, *perm)
               for k in others
               for perm in itertools.permutations((i, j, k)))


def trio_prob(p, combo) -> float:
    """三連複（i,j,k が上位3着、順不同）。"""
    return sum(_trifecta_ordered(p, *perm) for perm in itertools.permutations(combo))


def trifecta_prob(p, combo) -> float:
    """三連単（i,j,k がこの着順）。"""
    return _trifecta_ordered(p, *combo)


# 馬券種ごとの設定: ラベル, 組の頭数, 着順を区別するか, 確率関数
BETS = {
    "quinella": {"label": "馬連",   "size": 2, "ordered": False, "prob": quinella_prob},
    "wide":     {"label": "ワイド", "size": 2, "ordered": False, "prob": wide_prob},
    "trio":     {"label": "三連複", "size": 3, "ordered": False, "prob": trio_prob},
    "trifecta": {"label": "三連単", "size": 3, "ordered": True,  "prob": trifecta_prob},
}


# -----------------------------------------------------------------------------
# 払戻参照
# -----------------------------------------------------------------------------
def combo_key(combo, ordered: bool) -> str:
    """払戻参照用のキー文字列。順不同なら馬番昇順、着順ありならそのまま。"""
    nums = combo if ordered else sorted(combo)
    return "-".join(str(n) for n in nums)


def load_payouts(db: str, cfg: dict) -> dict[tuple[str, str], int]:
    """{(race_id, combo_key): payout} の辞書。"""
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT race_id, combination, payout FROM payouts WHERE bet_type = ?",
            (cfg["label"],),
        ).fetchall()
    finally:
        conn.close()
    out = {}
    for rid, combo, payout in rows:
        nums = [int(x) for x in combo.replace("=", "-").split("-") if x.strip().isdigit()]
        if len(nums) >= cfg["size"]:
            nums = nums[:cfg["size"]]
            out[(rid, combo_key(tuple(nums), cfg["ordered"]))] = payout
    return out


# -----------------------------------------------------------------------------
# レースごとの推奨組を算出
# -----------------------------------------------------------------------------
def ranked_combos(df_race: pd.DataFrame, bet: str, topn: int):
    """1レース分から確率の高い上位 topn 組を [(combo, prob), ...] で返す。"""
    cfg = BETS[bet]
    p = dict(zip(df_race["horse_number"].astype(int), df_race["p_norm"]))
    nums = sorted(n for n in p if pd.notna(n))
    gen = (itertools.permutations(nums, cfg["size"]) if cfg["ordered"]
           else itertools.combinations(nums, cfg["size"]))
    scored = [(c, cfg["prob"](p, c)) for c in gen]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:topn]


# -----------------------------------------------------------------------------
# バックテスト
# -----------------------------------------------------------------------------
def backtest(df: pd.DataFrame, db: str, bet: str, topn: int, min_prob: float):
    cfg = BETS[bet]
    payouts = load_payouts(db, cfg)
    bets = hits = stake = ret = 0
    for rid, g in df.groupby("race_id"):
        for combo, prob in ranked_combos(g, bet, topn):
            if prob < min_prob:
                continue
            bets += 1
            stake += 100
            payout = payouts.get((rid, combo_key(combo, cfg["ordered"])))
            if payout:
                hits += 1
                ret += payout
    roi = round(100.0 * ret / stake, 1) if stake else None
    hit_rate = round(100.0 * hits / bets, 1) if bets else None
    print(f"\n=== {cfg['label']} バックテスト（各レース上位{topn}組 / 確率>={min_prob}） ===")
    if not bets:
        print("対象ベットなし。")
    else:
        print(f"購入: {bets}点 / 的中: {hits}点 / 的中率: {hit_rate}% / 回収率: {roi}%")
        if not payouts:
            print("※ payouts テーブルにデータが無いため回収率は0です。"
                  "結果ページ取り込み(scraper.py)で払戻も登録されます。")


# -----------------------------------------------------------------------------
# 出走前レースの買い目提示
# -----------------------------------------------------------------------------
def predict(df: pd.DataFrame, bet: str, topn: int):
    cfg = BETS[bet]
    for rid, g in df.groupby("race_id"):
        print(f"\n=== race_id={rid} / {cfg['label']} 推奨買い目 上位{topn} ===")
        rows = ranked_combos(g, bet, topn)
        if not rows:
            print("（候補なし）")
            continue
        print(f"{'買い目':<12}{'確率':>9}{'フェアオッズ':>12}")
        for combo, prob in rows:
            fair = round(1.0 / prob, 1) if prob > 0 else float("inf")
            print(f"{combo_key(combo, cfg['ordered']):<12}{prob:>9.4f}{fair:>12}")
    print("\n※ フェアオッズ(=1/確率)より実際のオッズが高ければ期待値プラスの目安。")


# -----------------------------------------------------------------------------
# メイン
# -----------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="馬連・ワイドの期待値最適化/バックテスト")
    p.add_argument("--db", default="keiba.db")
    p.add_argument("--model", default="model.pkl", help="保存済みモデル（単勝予測推奨）")
    p.add_argument("--bet", default="quinella",
                   choices=["quinella", "wide", "trio", "trifecta"],
                   help="馬券種（quinella=馬連 / wide=ワイド / trio=三連複 / trifecta=三連単）")
    p.add_argument("--topn", type=int, default=1, help="各レースで買う組数")
    p.add_argument("--min-prob", type=float, default=0.0, help="購入する最小的中確率")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--backtest", action="store_true", help="過去レースで回収率検証")
    mode.add_argument("--predict", action="store_true", help="出走前レースの買い目提示")
    p.add_argument("--valid-frac", type=float, default=0.2, help="backtest時に検証へ回す割合")
    p.add_argument("--race-id", help="predict時、特定レースだけ対象にする")
    args = p.parse_args(argv)

    bundle = mlcommon.load_model(args.model)
    model = bundle["model"]
    df = mlcommon.load_data(args.db, None)

    if df.empty:
        sys.exit("[NG] v_features にデータがありません。")

    # 対象レースの絞り込み
    if args.predict:
        ids = [args.race_id] if args.race_id else mlcommon.upcoming_race_ids(args.db)
        if not ids:
            print("結果未確定レースがありません。--shutuba で出馬表を取り込んでください。",
                  file=sys.stderr)
            return 0
        df = df[df["race_id"].isin(ids)].reset_index(drop=True)
    else:  # backtest: 結果未確定レースを除き、新しい側 valid-frac を検証に
        upcoming = set(mlcommon.upcoming_race_ids(args.db))
        df = df[~df["race_id"].isin(upcoming)].reset_index(drop=True)
        dates = np.sort(df["race_date"].unique())
        if len(dates) == 0:
            sys.exit("[NG] 対象データがありません。")
        cut = dates[int(len(dates) * (1 - args.valid_frac))]
        df = df[df["race_date"] >= cut].reset_index(drop=True)

    # 単勝確率を推定 → レース内で正規化
    x = mlcommon.build_features(df, feature_columns=bundle["feature_columns"])
    df = df.assign(p=model.predict_proba(x)[:, 1])
    df = mlcommon.normalize_by_race(df, prob_col="p")

    print(f"モデル: {bundle.get('model_name')} / 馬券: {BETS[args.bet]['label']}")
    if args.backtest:
        backtest(df, args.db, args.bet, args.topn, args.min_prob)
    else:
        predict(df, args.bet, args.topn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
