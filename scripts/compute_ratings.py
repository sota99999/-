#!/usr/bin/env python3
# =============================================================================
# 馬 Eloレーティングの計算
#
# 全レースを時系列順に処理し、各馬が各レースに「入る直前」のレーティング
# (elo_before) を horse_ratings テーブルに保存する。
#
# Elo は本来「対戦ゲームの強さ指標」で、ここでは1レース内の着順を
# 「総当たり戦（上位馬が下位馬に勝った）」とみなして更新する。
#   期待勝率 E_i = 1 / (1 + 10^((R_j - R_i)/400))
#   更新     R_i += K * (実際の勝敗 - E_i)   （K はレース頭数で割って正規化）
#
# elo_before は当該レースより前の結果だけで決まるので、特徴量に使っても
# 未来情報のリークにならない。一度も走っていない馬は初期値 1500。
#
# 使い方:
#   python scripts/compute_ratings.py --db keiba.db
#   （収集後・features.sql 適用前に実行するのがおすすめ）
# =============================================================================
from __future__ import annotations

import argparse
import itertools
import sqlite3

INITIAL_ELO = 1500.0
K_FACTOR = 32.0   # 1レースあたりの更新の強さ（頭数-1 で割って各ペアに配分）


def compute(conn: sqlite3.Connection) -> int:
    # 念のためテーブルを用意（既存DBにも対応）
    conn.execute(
        """CREATE TABLE IF NOT EXISTS horse_ratings (
               race_id TEXT NOT NULL, horse_id TEXT NOT NULL,
               elo_before REAL, PRIMARY KEY (race_id, horse_id))"""
    )

    # 時系列順（同レースは連続）に全出走を取得。着順NULLは後ろに。
    rows = conn.execute(
        """SELECT r.race_id, r.horse_id, r.finish_position
           FROM results r JOIN races ra ON r.race_id = ra.race_id
           ORDER BY ra.race_date, r.race_id,
                    (r.finish_position IS NULL), r.finish_position"""
    ).fetchall()

    elo: dict[str, float] = {}
    out: list[tuple[str, str, float]] = []

    for race_id, grp in itertools.groupby(rows, key=lambda x: x[0]):
        grp = list(grp)
        # ① レース直前のレーティングを記録（出走全頭）
        for rid, hid, _fp in grp:
            out.append((rid, hid, round(elo.get(hid, INITIAL_ELO), 1)))

        # ② 着順のある馬だけで総当たり更新
        ranked = [(hid, fp) for _rid, hid, fp in grp if fp is not None]
        if len(ranked) < 2:
            continue
        k = K_FACTOR / (len(ranked) - 1)
        cur = {hid: elo.get(hid, INITIAL_ELO) for hid, _ in ranked}
        delta = {hid: 0.0 for hid, _ in ranked}
        for (hi, fi), (hj, fj) in itertools.combinations(ranked, 2):
            si = 1.0 if fi < fj else (0.0 if fi > fj else 0.5)  # 着順が小さい=勝ち
            ei = 1.0 / (1.0 + 10 ** ((cur[hj] - cur[hi]) / 400.0))
            delta[hi] += k * (si - ei)
            delta[hj] += k * ((1.0 - si) - (1.0 - ei))
        for hid, _ in ranked:
            elo[hid] = cur[hid] + delta[hid]

    conn.execute("DELETE FROM horse_ratings")  # 全再計算（順序依存のため）
    conn.executemany(
        "INSERT INTO horse_ratings(race_id, horse_id, elo_before) VALUES (?, ?, ?)", out
    )
    conn.commit()
    return len(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="馬Eloレーティングの計算")
    p.add_argument("--db", default="keiba.db", help="SQLite DBファイル")
    args = p.parse_args(argv)

    conn = sqlite3.connect(args.db)
    try:
        n = compute(conn)
    finally:
        conn.close()
    print(f"[OK] {n} 出走分の elo_before を計算・保存しました（horse_ratings）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
