#!/usr/bin/env python3
# =============================================================================
# races.grade を取り直して埋める（G1/G2/G3）。
#   取り込み時にグレードが拾えず NULL のままのレースが多いので、結果ページを
#   再取得して grade を更新する。条件戦（未勝利・新馬・勝クラス等）は重賞でないので
#   名前で除外し、フェッチ回数を抑える。結果が確定したレースのみ対象。
#
# 使い方:
#   python scripts/backfill_grade.py --db keiba.db --year 2026
#   python scripts/backfill_grade.py --db keiba.db            # 全年
# =============================================================================
from __future__ import annotations

import argparse
import re
import sqlite3
import time

import scraper

# 条件戦（重賞でない）を示すレース名パターン。これらはフェッチせずスキップ。
NON_GRADED = re.compile(
    r"新馬|未勝利|メイクデビュー|1勝クラス|2勝クラス|3勝クラス|"
    r"1勝|2勝|3勝|500万|1000万|1600万|オープン\b")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="races.grade を結果ページから取り直して埋める")
    p.add_argument("--db", default="keiba.db")
    p.add_argument("--year", help="対象年 (例 2026)。省略で全年")
    p.add_argument("--sleep", type=float, default=0.7, help="フェッチ間隔秒")
    p.add_argument("--limit", type=int, default=0, help="最大処理数（0=無制限）")
    args = p.parse_args(argv)

    conn = sqlite3.connect(args.db)
    q = ("SELECT DISTINCT ra.race_id, ra.race_name FROM races ra "
         "JOIN results r ON r.race_id = ra.race_id "
         "WHERE ra.grade IS NULL AND r.finish_position IS NOT NULL")
    params: list = []
    if args.year:
        q += " AND ra.race_date LIKE ?"
        params.append(args.year + "%")
    q += " ORDER BY ra.race_id"
    rows = conn.execute(q, params).fetchall()

    # 条件戦を除外
    targets = [(rid, nm) for rid, nm in rows if not (nm and NON_GRADED.search(nm))]
    print(f"対象候補 {len(rows)} → 条件戦除外後 {len(targets)} レースを確認します")
    if args.limit:
        targets = targets[:args.limit]

    found = 0
    for i, (rid, nm) in enumerate(targets, 1):
        try:
            html = scraper.fetch_html(rid)
            parsed = scraper.parse_race(html, rid)
            g = parsed.get("grade")
        except Exception as e:  # noqa: BLE001
            g = None
            if i <= 5:
                print(f"  [warn] {rid} {nm}: {e}")
        if g:
            conn.execute("UPDATE races SET grade=? WHERE race_id=?", (g, rid))
            conn.commit()
            found += 1
        if i % 20 == 0 or i == len(targets):
            print(f"  {i}/{len(targets)} 確認 / グレード判明 {found} 件")
        time.sleep(args.sleep)

    print(f"\n完了: {found} レースに grade を設定しました。")
    by = conn.execute(
        "SELECT grade, COUNT(*) FROM races WHERE grade IS NOT NULL "
        + ("AND race_date LIKE ? " if args.year else "")
        + "GROUP BY grade",
        ([args.year + "%"] if args.year else [])).fetchall()
    print("現在のグレード分布:", dict(by))
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
