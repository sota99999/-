#!/usr/bin/env python3
# =============================================================================
# レースのラップタイムを収集して race_laps に投入する（展開・ラップ分析用）
#
# 結果が確定済みでラップ未取得のレースについて、結果ページ（db.netkeiba →
# 未反映なら race.netkeiba 速報）から 200mごとのラップを抽出し、前後半・
# 上り3F・前後傾(pace_diff)を計算して保存する。
#
# 使い方:
#   # まず1レースで取得確認（HTMLのどこからラップが取れたか表示）
#   python scripts/collect_laps.py --db keiba.db --diagnose 202605030611
#   # 未取得のレースをまとめて収集（期間指定も可）
#   python scripts/collect_laps.py --db keiba.db
#   python scripts/collect_laps.py --db keiba.db --from 2025-01-01 --to 2025-12-31
#   python scripts/collect_laps.py --db keiba.db --limit 200
#
# 注意: scraper と同じくリクエスト間隔を空ける。短縮しないこと。
# =============================================================================
from __future__ import annotations

import argparse
import sqlite3
import sys
import time

import scraper

CREATE = """CREATE TABLE IF NOT EXISTS race_laps (
    race_id TEXT PRIMARY KEY, lap_seq TEXT, n_laps INTEGER,
    first_half REAL, second_half REAL, race_first3f REAL,
    race_last3f REAL, pace_diff REAL)"""


def _fetch_laps(race_id: str):
    """race.netkeiba(速報) → db.netkeiba の順でラップを取得（取れた dict を返す）。

    db.netkeiba のラップ表は JS 描画で静的HTMLが空のため、速報側を先に試す。
    """
    try:
        html = scraper.http_get(scraper.RESULT_URL.format(race_id=race_id), encoding="utf-8")
        laps = scraper.parse_laps(html)
        if laps:
            return laps, "live"
    except Exception:  # noqa: BLE001
        pass
    try:
        laps = scraper.parse_laps(scraper.fetch_html(race_id))
        if laps:
            return laps, "db"
    except Exception:  # noqa: BLE001
        pass
    return None, None


def diagnose(race_id: str) -> int:
    print(f"=== ラップ取得診断: {race_id} ===")
    for tag, getter in (("db.netkeiba", lambda: scraper.fetch_html(race_id)),
                        ("race.netkeiba", lambda: scraper.http_get(
                            scraper.RESULT_URL.format(race_id=race_id), encoding="utf-8"))):
        try:
            html = getter()
            laps = scraper.parse_laps(html)
            if laps:
                print(f"[{tag}] OK: {laps['n_laps']}本 "
                      f"前半{laps['first_half']} 後半{laps['second_half']} "
                      f"上り3F{laps['race_last3f']} pace_diff{laps['pace_diff']}")
                print(f"   ラップ: {laps['lap_seq']}")
            else:
                print(f"[{tag}] ラップ抽出できず（HTML長 {len(html)}）")
        except Exception as e:  # noqa: BLE001
            print(f"[{tag}] 取得失敗: {e}")
        time.sleep(scraper.REQUEST_INTERVAL)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ラップタイム収集")
    p.add_argument("--db", default="keiba.db")
    p.add_argument("--from", dest="date_from", help="開始日 (YYYY-MM-DD)")
    p.add_argument("--to", dest="date_to", help="終了日 (YYYY-MM-DD)")
    p.add_argument("--limit", type=int, default=0, help="最大何レースまで（0=無制限）")
    p.add_argument("--diagnose", help="このrace_idで取得確認だけ行う")
    args = p.parse_args(argv)

    if args.diagnose:
        return diagnose(args.diagnose)

    conn = sqlite3.connect(args.db)
    conn.execute(CREATE)
    q = ["""SELECT ra.race_id FROM races ra
            WHERE ra.race_id IN (SELECT race_id FROM results
                                 GROUP BY race_id HAVING COUNT(finish_position) > 0)
              AND ra.race_id NOT IN (SELECT race_id FROM race_laps)"""]
    params: list = []
    if args.date_from:
        q.append("AND ra.race_date >= ?"); params.append(args.date_from)
    if args.date_to:
        q.append("AND ra.race_date <= ?"); params.append(args.date_to)
    q.append("ORDER BY ra.race_date")
    ids = [r[0] for r in conn.execute(" ".join(q), params)]
    if args.limit:
        ids = ids[:args.limit]
    if not ids:
        print("ラップ未取得の確定レースはありません。")
        return 0

    print(f"対象 {len(ids)} レースのラップを収集します。")
    ok = ng = 0
    for i, rid in enumerate(ids, 1):
        laps, src = _fetch_laps(rid)
        if laps:
            conn.execute(
                """INSERT OR REPLACE INTO race_laps
                       (race_id, lap_seq, n_laps, first_half, second_half,
                        race_first3f, race_last3f, pace_diff)
                   VALUES (:race_id, :lap_seq, :n_laps, :first_half, :second_half,
                           :race_first3f, :race_last3f, :pace_diff)""",
                {"race_id": rid, **laps},
            )
            conn.commit()
            ok += 1
            tag = "後傾(瞬発)" if (laps["pace_diff"] or 0) > 0 else "前傾(持続)"
            print(f"   [OK {i}/{len(ids)}] {rid}: {laps['n_laps']}本 "
                  f"pace_diff{laps['pace_diff']} {tag} ({src})")
        else:
            ng += 1
            print(f"   [NG {i}/{len(ids)}] {rid}: ラップ取得できず", file=sys.stderr)
        time.sleep(scraper.REQUEST_INTERVAL)
    conn.close()
    print(f"\n完了: {ok} レース取得 / {ng} 失敗。失敗分は再実行で再取得を試みます。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
