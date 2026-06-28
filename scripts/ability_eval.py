#!/usr/bin/env python3
# =============================================================================
# 能力2指標(最高SP・Elo)の印の回顧
#
# card._ability_marks と同じ印付け（各指標で ⭐◎○▲△、オッズ妙味に ✅）を
# 結果確定レースに適用し、印ごとに
#   単勝的中率 / 複勝的中率 / 単勝回収率 / 複勝回収率
# を集計する。最高SP・Elo それぞれの印別に出し、さらに
#   「両指標で◎/⭐が一致した馬（=二冠軸）」と「✅妙味馬」も別枠で出す。
#
# 使い方:
#   python scripts/ability_eval.py --db keiba.db --from 2026-06-27 --to 2026-06-28
#
# 注意: 単勝回収=最終単勝オッズ、複勝回収=複勝の確定払戻(payouts)。
#       払戻未取得のレースは0円扱い。各印を100円ずつ買った前提。
# =============================================================================
from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

import mlcommon
import card


def _summ(b: pd.DataFrame) -> dict:
    """的中率・回収率（単勝オッズと複勝確定払戻から）。"""
    n = len(b)
    if n == 0:
        return {"n": 0}
    win = b["finish"] == 1
    place = b["finish"] <= 3
    payout = pd.to_numeric(b.loc[win, "odds"], errors="coerce").fillna(0).sum() * 100
    pp = pd.to_numeric(b.get("place_payout"), errors="coerce").fillna(0)
    return {"n": n, "win": win.mean() * 100, "place": place.mean() * 100,
            "roi": payout / (100 * n) * 100, "proi": pp.sum() / (100 * n) * 100}


def _row(label: str, s: dict) -> str:
    if not s.get("n"):
        return f"{label:<10}{0:>5}      該当なし"
    return (f"{label:<10}{s['n']:>5}{s['win']:>8.1f}%{s['place']:>8.1f}%"
            f"{s['roi']:>8.1f}%{s['proi']:>8.1f}%")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="能力2指標の印の回顧")
    p.add_argument("--db", default="keiba.db")
    p.add_argument("--from", dest="date_from")
    p.add_argument("--to", dest="date_to")
    args = p.parse_args(argv)

    df = mlcommon.load_data(args.db, None)
    conn = sqlite3.connect(args.db)
    q = ["""SELECT race_id FROM races WHERE race_id IN
            (SELECT race_id FROM results GROUP BY race_id
             HAVING COUNT(finish_position) > 0)"""]
    params: list = []
    if args.date_from:
        q.append("AND race_date >= ?"); params.append(args.date_from)
    if args.date_to:
        q.append("AND race_date <= ?"); params.append(args.date_to)
    ids = [r[0] for r in conn.execute(" ".join(q), params)]
    actual = pd.read_sql_query(
        "SELECT race_id, horse_id, finish_position AS finish FROM results "
        "WHERE finish_position IS NOT NULL", conn)
    place_pay = pd.read_sql_query(
        "SELECT race_id, combination, payout FROM payouts WHERE bet_type = '複勝'", conn)
    conn.close()
    place_pay["hn"] = pd.to_numeric(place_pay["combination"], errors="coerce")
    pmap = {(r, h): p for r, h, p in
            zip(place_pay["race_id"], place_pay["hn"], place_pay["payout"])}

    sub = df[df["race_id"].isin(ids)].copy()
    if sub.empty:
        print("対象レースがありません（結果確定・期間指定を確認）"); return 0
    sub = sub.merge(actual, on=["race_id", "horse_id"], how="left")
    sub["odds"] = pd.to_numeric(sub.get("odds"), errors="coerce")
    hn = pd.to_numeric(sub["horse_number"], errors="coerce")
    sub["place_payout"] = [pmap.get((r, h)) for r, h in zip(sub["race_id"], hn)]

    marked = [card._ability_marks(g) for _, g in sub.groupby("race_id")]
    sub = pd.concat(marked, ignore_index=True)
    ran = sub[sub["finish"].notna()].copy()

    n_races = sub["race_id"].nunique()
    print(f"\n=== 能力2指標の印別 回顧（{args.date_from or '最初'}〜{args.date_to or '最後'}） ===")
    print(f"対象レース数: {n_races}（結果確定）")
    head = f"{'印':<10}{'本数':>5}{'単的中':>9}{'複的中':>9}{'単回収':>9}{'複回収':>9}"

    for col, label, _ in card.ABILITY_COLS:
        print(f"\n― {label} の印 ―")
        print(head)
        for mk in ["⭐", "◎", "○", "▲", "△"]:
            print(_row(mk, _summ(ran[ran[col + "_mark"] == mk])))

    # 2指標の印の組み合わせ（指定の7パターン）
    sp_m = ran["best_speed_prior_mark"]; el_m = ran["elo_before_mark"]
    sp_v = ran["best_speed_prior_val"]; el_v = ran["elo_before_val"]
    combos = [
        ("Elo⭐ SP⭐", (el_m == "⭐") & (sp_m == "⭐")),
        ("Elo⭐ SP◎", (el_m == "⭐") & (sp_m == "◎")),
        ("Elo◎ SP⭐", (el_m == "◎") & (sp_m == "⭐")),
        ("Elo◎ SP◎", (el_m == "◎") & (sp_m == "◎")),
        ("Elo✅ SP✅", (el_v == "✅") & (sp_v == "✅")),
        ("Elo⭐ SP✅", (el_m == "⭐") & (sp_v == "✅")),
        ("Elo✅ SP⭐", (el_v == "✅") & (sp_m == "⭐")),
        ("Elo◎ SP✅", (el_m == "◎") & (sp_v == "✅")),
        ("Elo✅ SP◎", (el_v == "✅") & (sp_m == "◎")),
    ]
    print("\n― 2指標の組み合わせ ―")
    print(head)
    for label, mask in combos:
        print(_row(label, _summ(ran[mask])))

    print("\n※単回収=最終単勝オッズ、複回収=複勝確定払戻。各印を100円ずつ買った前提。"
          "払戻未取得レースは0円扱い。100%超で利益。✅は指標ごと。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
