#!/usr/bin/env python3
# =============================================================================
# オッズ更新: netkeiba のオッズAPIから単勝オッズを取得し results を更新する
#
# 出馬表(scraper --shutuba)はオッズを取れないため、オッズ公開後にこれを実行すると
# オッズありモデルで予想できるようになる。
#
# 使い方:
#   python scripts/update_odds.py --db keiba.db --date 2026-06-21   # その日の全レース
#   python scripts/update_odds.py --db keiba.db --race-id 202609030411
#   python scripts/update_odds.py --db keiba.db                     # 結果未確定の全レース
# =============================================================================
from __future__ import annotations

import argparse
import sqlite3
import sys
import time

import scraper
import mlcommon


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="オッズAPI更新")
    p.add_argument("--db", default="keiba.db")
    p.add_argument("--date", help="開催日 (YYYY-MM-DD) の全レース")
    p.add_argument("--race-id", help="このレースだけ")
    args = p.parse_args(argv)

    conn = sqlite3.connect(args.db)
    if args.race_id:
        ids = [args.race_id]
    elif args.date:
        ids = [r[0] for r in conn.execute(
            "SELECT race_id FROM races WHERE race_date=? ORDER BY race_id", (args.date,))]
    else:
        ids = mlcommon.upcoming_race_ids(args.db)

    if not ids:
        print("対象レースがありません。", file=sys.stderr)
        return 0

    ok = ng = 0
    for i, rid in enumerate(ids):
        try:
            n = scraper.update_odds(conn, rid)
            if n:
                ok += 1
                print(f"[OK] {rid}: {n}頭のオッズを更新")
            else:
                print(f"[--] {rid}: オッズ未公開（まだ取得できません）")
        except Exception as e:  # noqa: BLE001
            ng += 1
            print(f"[NG] {rid}: {e}", file=sys.stderr)
        if i < len(ids) - 1:
            time.sleep(scraper.REQUEST_INTERVAL)

    conn.close()
    print(f"\n完了: {ok}レース更新 / {ng}レース失敗")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
