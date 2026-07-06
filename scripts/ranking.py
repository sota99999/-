#!/usr/bin/env python3
# =============================================================================
# 指標ランキング: 全競走馬の上位N頭を表で出す
#
#   --by r      : 相対R（現在値=最新出走時点）… 既定
#   --by pmax   : 最高perf（天井。条件が合えばどこまでやれるか）
#   --by power  : 能力 power_top（時計ベースの総合実力）
#   --active N  : 直近N日以内に出走した馬に限定（既定365。0で無制限）
#   --min-runs  : 最低出走数（既定3。まぐれ上位を除く）
#   --surface 芝/ダ : 直近走の馬場種別で絞る
#
# 使い方:
#   python scripts/ranking.py --db keiba.db                  # 相対R 上位100
#   python scripts/ranking.py --db keiba.db --by pmax --top 50
# =============================================================================
from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

VENUE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
         "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="指標ランキング")
    ap.add_argument("--db", default="keiba.db")
    ap.add_argument("--by", choices=["r", "pmax", "power"], default="r")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--active", type=int, default=365, help="直近N日以内に出走(0=無制限)")
    ap.add_argument("--min-runs", type=int, default=3)
    ap.add_argument("--surface", choices=["芝", "ダ"], help="直近走の馬場種別で絞る")
    args = ap.parse_args(argv)

    con = sqlite3.connect(args.db)
    d = pd.read_sql("""
        SELECT hr.horse_id, h.name, ra.race_date, ra.race_name, ra.venue_id,
               ra.surface, ra.distance, hr.r_before, hr.perf, hr.pmax, hr.pstd,
               r.finish_position fin
        FROM horse_relative_r hr
        JOIN races ra ON ra.race_id = hr.race_id
        JOIN horses h ON h.horse_id = hr.horse_id
        LEFT JOIN results r ON r.race_id = hr.race_id AND r.horse_id = hr.horse_id""", con)
    pt = pd.read_sql(
        "SELECT horse_id, race_date, power_top FROM v_ability WHERE power_top IS NOT NULL", con)
    mx = con.execute("SELECT MAX(race_date) FROM races").fetchone()[0]
    con.close()

    d = d.sort_values("race_date")
    runs = d[d["fin"].notna()].groupby("horse_id").size().rename("runs")
    last = d.groupby("horse_id").tail(1).set_index("horse_id")   # 最新出走(または出走予定)
    last = last.join(runs)
    last["pmax_all"] = d.groupby("horse_id")["perf"].max()
    ptl = pt.sort_values("race_date").groupby("horse_id").tail(1).set_index("horse_id")
    last["power_top"] = ptl["power_top"]

    if args.active:
        cutoff = (pd.to_datetime(mx) - pd.Timedelta(days=args.active)).strftime("%Y-%m-%d")
        last = last[last["race_date"] >= cutoff]
    last = last[last["runs"].fillna(0) >= args.min_runs]
    if args.surface:
        last = last[last["surface"] == args.surface]

    col = {"r": "r_before", "pmax": "pmax_all", "power": "power_top"}[args.by]
    lab = {"r": "相対R", "pmax": "最高perf", "power": "能力"}[args.by]
    top = last.dropna(subset=[col]).sort_values(col, ascending=False).head(args.top)

    print(f"\n=== {lab} ランキング 上位{len(top)}頭 "
          f"（直近{args.active}日以内に出走・{args.min_runs}走以上"
          + (f"・{args.surface}" if args.surface else "") + "） ===")
    print(f"{'順':>3} {'馬名':<14}{'相対R':>7}{'最高':>7}{'能力':>7}{'ムラ':>6}  直近走")
    for i, (hid, r) in enumerate(top.iterrows(), 1):
        v = VENUE.get(str(r["venue_id"]).zfill(2), "")
        fin = f"{int(r['fin'])}着" if pd.notna(r["fin"]) else "予定"
        def f(x, w="6.1f"):
            return format(x, w) if pd.notna(x) else "     -"
        print(f"{i:>3} {str(r['name'])[:14]:<14}{f(r['r_before']):>7}{f(r['pmax_all']):>7}"
              f"{f(r['power_top']):>7}{f(r['pstd'],'5.1f'):>6}  "
              f"{r['race_date']} {v}{r['surface']}{int(r['distance'])} "
              f"{str(r['race_name'])[:12]} {fin}")
    print("\n※相対R=現在の格 / 最高=キャリア天井perf / 能力=時計ベースpower_top / "
          "ムラ=pstd(14以上で条件補正対象)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
