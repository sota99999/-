#!/usr/bin/env python3
# =============================================================================
# netkeiba 差分クローラ
#
# 指定した日付（範囲）の開催レース一覧を netkeiba から取得し、
# まだ DB に無いレースだけを scraper で取り込む（=差分クロール）。
# 既に取得済みのレースはスキップするので、何度回しても無駄打ちが無い。
#
# 使い方:
#   pip install -r scripts/requirements.txt
#   # 単一日
#   python scripts/crawler.py --db keiba.db --date 2024-05-12
#   # 期間（両端含む）
#   python scripts/crawler.py --db keiba.db --from 2024-05-01 --to 2024-05-31
#   # 取得対象の確認だけ（DBへ書き込まない）
#   python scripts/crawler.py --date 2024-05-12 --dry-run
#
# 注意: scraper.py と同様、リクエスト間にウェイトを入れている。短縮しないこと。
# =============================================================================
from __future__ import annotations

import argparse
import datetime as dt
import re
import sqlite3
import sys
import time

import scraper  # 同ディレクトリ。HTTP取得/parse/upsert を再利用

# 開催日の全レースへのリンクを含む一覧ページ（kaisai_date=YYYYMMDD）
RACE_LIST_URL = "https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={ymd}"


# -----------------------------------------------------------------------------
# レース一覧の取得
# -----------------------------------------------------------------------------
def fetch_race_ids_for_date(date: dt.date) -> list[str]:
    """指定日の開催レースID（12桁）を一覧ページから抽出して返す。"""
    ymd = date.strftime("%Y%m%d")
    html = scraper.http_get(RACE_LIST_URL.format(ymd=ymd))  # リトライ付き取得
    # href の race_id=2024... もしくは /race/2024.../ の双方に対応
    ids = re.findall(r"race_id=(\d{12})", html)
    ids += re.findall(r"/race/(\d{12})", html)
    # 重複排除（出現順を維持）
    return list(dict.fromkeys(ids))


# -----------------------------------------------------------------------------
# 差分判定
# -----------------------------------------------------------------------------
def existing_race_ids(conn: sqlite3.Connection) -> set[str]:
    """DB に既に取り込み済みの race_id 集合。"""
    return {row[0] for row in conn.execute("SELECT race_id FROM races")}


def filter_new(conn: sqlite3.Connection, race_ids: list[str]) -> list[str]:
    """未取得の race_id だけを返す（順序維持）。"""
    have = existing_race_ids(conn)
    return [rid for rid in race_ids if rid not in have]


# -----------------------------------------------------------------------------
# 日付ユーティリティ
# -----------------------------------------------------------------------------
def daterange(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def _parse_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="netkeiba 差分クローラ")
    parser.add_argument("--db", default="keiba.db", help="SQLite DBファイル")
    parser.add_argument("--date", help="単一日 (YYYY-MM-DD)")
    parser.add_argument("--from", dest="date_from", help="開始日 (YYYY-MM-DD)")
    parser.add_argument("--to", dest="date_to", help="終了日 (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true",
                        help="取得対象の一覧表示のみ。DBへ書き込まない")
    args = parser.parse_args(argv)

    # 対象日の決定
    if args.date:
        days = [_parse_date(args.date)]
    elif args.date_from and args.date_to:
        days = list(daterange(_parse_date(args.date_from), _parse_date(args.date_to)))
    else:
        parser.error("--date か、--from と --to を指定してください")
        return 2

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON")
    scraper.ensure_schema(conn)   # 初回実行時に DB を自動初期化

    total_new = total_ng = 0
    for di, day in enumerate(days, 1):
        prefix = f"[{di}/{len(days)}] {day}"
        try:
            all_ids = fetch_race_ids_for_date(day)
        except Exception as e:  # noqa: BLE001
            print(f"[NG] {prefix} 一覧取得失敗: {e}", file=sys.stderr)
            time.sleep(scraper.REQUEST_INTERVAL)
            continue

        new_ids = filter_new(conn, all_ids)
        # 中断後の再実行では取得済みは自動スキップされる（差分クロール）
        print(f"== {prefix}: 開催{len(all_ids)}R / 未取得{len(new_ids)}R "
              f"（累計 取込{total_new} 失敗{total_ng}）")

        if args.dry_run:
            for rid in new_ids:
                print(f"   would fetch {rid}")
            time.sleep(scraper.REQUEST_INTERVAL)
            continue

        for rid in new_ids:
            try:
                parsed = scraper.ingest_race(conn, rid)
                total_new += 1
                print(f"   [OK] {rid}: {parsed['race'].get('race_name')} "
                      f"({len(parsed['results'])}頭)")
            except Exception as e:  # noqa: BLE001
                total_ng += 1
                print(f"   [NG] {rid}: {e}", file=sys.stderr)
            time.sleep(scraper.REQUEST_INTERVAL)

        time.sleep(scraper.REQUEST_INTERVAL)

    conn.close()
    if not args.dry_run:
        print(f"\n完了: 新規 {total_new}R 取込 / {total_ng}R 失敗。"
              "（失敗分は再実行すれば未取得として再取得を試みます）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
