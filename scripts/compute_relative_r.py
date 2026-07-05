#!/usr/bin/env python3
# =============================================================================
# 馬「相対レーティング(相対R)」の計算
#
# Elo(compute_ratings.py)がオンライン更新で格差が圧縮されるのに対し、こちらは
# シーズン全体の勝敗ネットワークを一括で収束させる「反復 strength-of-schedule」
# レーティング。各レースで
#     perf_i = (出走全馬の平均R) + SPREAD * (0.5 - (着順-1)/(頭数-1))
#     R_i    = 自分の全出走の perf_i 平均         を収束するまで反復
# これにより「強い相手を負かした馬」が上位へ大きく開く（G1好走の格を捉える）。
#
# リーク防止: レースの相対R(r_before)は「そのレースが属する月より前」の結果だけで
# 計算した月次スナップショットを用いる（当月内のレース同士は互いを見ない）。
# 一度も走っていない/データが薄い時期の馬は NULL。
#
# 使い方:
#   python scripts/compute_relative_r.py --db keiba.db
#   （collect後・features.sql 適用前あたりで実行。compute_ratings.py と同様）
# =============================================================================
from __future__ import annotations

import argparse
import sqlite3

SPREAD = 40.0        # 着順による1レースあたりの加減点幅（格差の開き具合）
ITERS = 25           # 反復回数（warm-start するので十分収束する）
MIN_PRIOR = 1000     # スナップショットを作るのに必要な「それ以前の出走数」


def compute_snapshot(race_arrays, warm, iters=ITERS, spread=SPREAD):
    """race_arrays: [[(horse_id, pos, n_ranked), ...], ...] からRを反復収束。"""
    r = dict(warm)
    for arr in race_arrays:
        for h, _p, _n in arr:
            r.setdefault(h, 0.0)
    if not r:
        return {}
    for _ in range(iters):
        acc: dict[str, list[float]] = {}
        for arr in race_arrays:
            fm = sum(r[h] for h, _p, _n in arr) / len(arr)
            for h, pos, n in arr:
                place = 0.5 - (pos - 1) / (n - 1) if n > 1 else 0.0
                a = acc.get(h)
                if a is None:
                    acc[h] = [fm + spread * place, 1]
                else:
                    a[0] += fm + spread * place
                    a[1] += 1
        r = {h: acc[h][0] / acc[h][1] for h in acc}
        mu = sum(r.values()) / len(r)
        r = {h: v - mu for h, v in r.items()}
    return r


def compute(conn: sqlite3.Connection) -> int:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS horse_relative_r (
               race_id TEXT NOT NULL, horse_id TEXT NOT NULL,
               r_before REAL, PRIMARY KEY (race_id, horse_id))"""
    )

    rows = conn.execute(
        """SELECT ra.race_date, r.race_id, r.horse_id, r.finish_position
           FROM results r JOIN races ra ON r.race_id = ra.race_id
           ORDER BY ra.race_date, r.race_id"""
    ).fetchall()

    # 月 -> [race_id, ...]、race_id -> 出走(全頭), race_id -> ranked配列
    months: list[str] = []
    month_races: dict[str, list[str]] = {}
    race_runners: dict[str, list[str]] = {}
    race_ranked: dict[str, list[tuple[str, int, int]]] = {}
    for race_date, race_id, horse_id, fp in rows:
        ym = (race_date or "")[:7]
        if race_id not in race_runners:
            race_runners[race_id] = []
            race_ranked[race_id] = []
            month_races.setdefault(ym, []).append(race_id)
            if ym not in months:
                months.append(ym)
        race_runners[race_id].append(horse_id)
        if fp is not None:
            race_ranked[race_id].append((horse_id, int(fp), 0))
    months.sort()
    # ranked配列の頭数(n)を埋める
    for rid, ranked in race_ranked.items():
        n = len(ranked)
        race_ranked[rid] = [(h, p, n) for (h, p, _z) in ranked]

    out: list[tuple[str, str, float]] = []
    prior_arrays: list[list[tuple[str, int, int]]] = []  # これまでの月の ranked 配列
    prior_count = 0
    snapshot: dict[str, float] = {}

    for ym in months:
        # ① 当月レースには「前月まで」で作った snapshot を付与（リーク防止）
        if prior_count >= MIN_PRIOR and snapshot:
            for rid in month_races[ym]:
                for hid in race_runners[rid]:
                    if hid in snapshot:
                        out.append((rid, hid, round(snapshot[hid], 2)))

        # ② 当月を prior に加え、次月に向けて snapshot を更新（warm-start）
        for rid in month_races[ym]:
            arr = race_ranked[rid]
            if len(arr) >= 2:
                prior_arrays.append(arr)
                prior_count += len(arr)
        if prior_count >= MIN_PRIOR:
            snapshot = compute_snapshot(prior_arrays, snapshot)

    conn.execute("DELETE FROM horse_relative_r")
    conn.executemany(
        "INSERT OR REPLACE INTO horse_relative_r (race_id, horse_id, r_before) VALUES (?,?,?)",
        out,
    )
    conn.commit()
    return len(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="反復 strength-of-schedule 相対レーティングを計算")
    ap.add_argument("--db", default="keiba.db")
    args = ap.parse_args()
    conn = sqlite3.connect(args.db)
    try:
        n = compute(conn)
        print(f"horse_relative_r: {n} 行を書き込みました。")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
