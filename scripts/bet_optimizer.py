#!/usr/bin/env python3
# =============================================================================
# 組み合わせ馬券（馬連・ワイド）の期待値最適化 / バックテスト
#
# 保存済みモデルの単勝確率を Harville モデルで組み合わせ確率に変換し、
#   - backtest: 過去レースで実際の払戻(payouts)に対する的中率・回収率を検証
#   - predict : 出走前レースの推奨買い目（確率の高いペア）とフェアオッズを提示
# を行う。
#
# Harville モデル（単勝確率 p_i から着順を逐次サンプリングする近似）:
#   P(i=1着, j=2着) = p_i * p_j/(1-p_i)
#   馬連(i,j) = P(i1,j2) + P(j1,i2)
#   ワイド(i,j) = P(i,j がともに3着以内)
#
# 使い方:
#   # バックテスト（検証期間の確定レースで馬連・ワイドを検証）
#   python scripts/bet_optimizer.py --db keiba.db --model model.pkl --backtest
#   # 出走前レースの買い目提示
#   python scripts/bet_optimizer.py --db keiba.db --model model.pkl --predict
#   # ペア/レース数や賭け方を調整
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
# Harville 組み合わせ確率
# -----------------------------------------------------------------------------
def quinella_prob(p: dict[int, float], i: int, j: int) -> float:
    """馬連（i,j が1・2着、順不同）の的中確率。p は {馬番: 勝率(正規化済)}。"""
    pi, pj = p[i], p[j]
    term = 0.0
    if pi < 1:
        term += pi * pj / (1 - pi)      # i→j
    if pj < 1:
        term += pj * pi / (1 - pj)      # j→i
    return term


def wide_prob(p: dict[int, float], i: int, j: int) -> float:
    """ワイド（i,j がともに3着以内）の的中確率（Harville 近似）。"""
    others = [k for k in p if k not in (i, j)]
    total = 0.0
    # i,j と第三の馬 k が podium(上位3) を占める全順列を合算
    for k in others:
        for a, b, c in itertools.permutations((i, j, k)):
            pa = p[a]
            denom1 = 1 - pa
            if denom1 <= 0:
                continue
            denom2 = 1 - pa - p[b]
            if denom2 <= 0:
                continue
            total += pa * (p[b] / denom1) * (p[c] / denom2)
    # i,j のみで上位を占め、3着が誰でも良いケースは上のkループに含まれる
    return total


PROB_FUNC = {"quinella": quinella_prob, "wide": wide_prob}
BET_LABEL = {"quinella": "馬連", "wide": "ワイド"}


# -----------------------------------------------------------------------------
# 払戻参照
# -----------------------------------------------------------------------------
def load_payouts(db: str, bet_label: str) -> dict[tuple[str, str], int]:
    """{(race_id, "min-max"): payout} の辞書。combination は馬番昇順に正規化。"""
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT race_id, combination, payout FROM payouts WHERE bet_type = ?",
            (bet_label,),
        ).fetchall()
    finally:
        conn.close()
    out = {}
    for rid, combo, payout in rows:
        nums = [int(x) for x in combo.replace("=", "-").split("-") if x.strip().isdigit()]
        if len(nums) >= 2:
            key = (rid, "-".join(str(n) for n in sorted(nums[:2])))
            out[key] = payout
    return out


def pair_key(a: int, b: int) -> str:
    return "-".join(str(n) for n in sorted((a, b)))


# -----------------------------------------------------------------------------
# レースごとの推奨ペアを算出
# -----------------------------------------------------------------------------
def ranked_pairs(df_race: pd.DataFrame, bet: str, topn: int) -> list[tuple[int, int, float]]:
    """1レース分の DataFrame から、確率の高い上位 topn ペアを返す。"""
    p = dict(zip(df_race["horse_number"].astype(int), df_race["p_norm"]))
    nums = [n for n in p if pd.notna(n)]
    func = PROB_FUNC[bet]
    pairs = [(i, j, func(p, i, j)) for i, j in itertools.combinations(sorted(nums), 2)]
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:topn]


# -----------------------------------------------------------------------------
# バックテスト
# -----------------------------------------------------------------------------
def backtest(df: pd.DataFrame, db: str, bet: str, topn: int, min_prob: float):
    label = BET_LABEL[bet]
    payouts = load_payouts(db, label)
    bets = hits = stake = ret = 0
    for rid, g in df.groupby("race_id"):
        for a, b, prob in ranked_pairs(g, bet, topn):
            if prob < min_prob:
                continue
            bets += 1
            stake += 100
            payout = payouts.get((rid, pair_key(a, b)))
            if payout:
                hits += 1
                ret += payout
    roi = round(100.0 * ret / stake, 1) if stake else None
    hit_rate = round(100.0 * hits / bets, 1) if bets else None
    print(f"\n=== {label} バックテスト（各レース上位{topn}ペア / 確率>={min_prob}） ===")
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
    label = BET_LABEL[bet]
    name = dict(zip(df["horse_number"].astype("Int64"), df.get("horse_name", df["horse_number"])))
    for rid, g in df.groupby("race_id"):
        print(f"\n=== race_id={rid} / {label} 推奨買い目 上位{topn} ===")
        rows = ranked_pairs(g, bet, topn)
        if not rows:
            print("（候補なし）")
            continue
        print(f"{'買い目':<10}{'確率':>8}{'フェアオッズ':>12}")
        for a, b, prob in rows:
            fair = round(1.0 / prob, 1) if prob > 0 else float("inf")
            print(f"{pair_key(a, b):<10}{prob:>8.3f}{fair:>12}")
    print("\n※ フェアオッズ(=1/確率)より実際のオッズが高ければ期待値プラスの目安。")


# -----------------------------------------------------------------------------
# メイン
# -----------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="馬連・ワイドの期待値最適化/バックテスト")
    p.add_argument("--db", default="keiba.db")
    p.add_argument("--model", default="model.pkl", help="保存済みモデル（単勝予測推奨）")
    p.add_argument("--bet", default="quinella", choices=["quinella", "wide"],
                   help="馬券種（quinella=馬連 / wide=ワイド）")
    p.add_argument("--topn", type=int, default=1, help="各レースで買うペア数")
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

    print(f"モデル: {bundle.get('model_name')} / 馬券: {BET_LABEL[args.bet]}")
    if args.backtest:
        backtest(df, args.db, args.bet, args.topn, args.min_prob)
    else:
        predict(df, args.bet, args.topn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
