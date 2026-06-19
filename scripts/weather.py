#!/usr/bin/env python3
# =============================================================================
# 天気予報を取得して馬場状態を推定し、出走前レースに反映する
#
# 無料の天気API（Open-Meteo, APIキー不要）から各競馬場の降水量予報を取得し、
# 当日＋前日の雨量から馬場状態（良/稍重/重/不良）を推定して races を更新する。
# これにより「雨予報→道悪」を予想に自動反映できる。
#
# 使い方:
#   python scripts/weather.py --db keiba.db                 # 結果未確定の全レース
#   python scripts/weather.py --db keiba.db --date 2026-06-21
#
# 注意: 馬場推定はあくまで降水量からの目安。芝/ダートや水はけで実際は変わる。
#       当日は実際の馬場発表で再実行するのが最も正確。
# =============================================================================
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import sys

import scraper  # http_get を再利用
import mlcommon

# JRA10場のおおよその緯度・経度
JRA_COORDS = {
    "01": (43.06, 141.33),  # 札幌
    "02": (41.78, 140.74),  # 函館
    "03": (37.72, 140.47),  # 福島
    "04": (37.92, 139.05),  # 新潟
    "05": (35.66, 139.48),  # 東京（府中）
    "06": (35.77, 140.22),  # 中山（船橋）
    "07": (35.12, 136.98),  # 中京（豊明）
    "08": (34.91, 135.71),  # 京都（淀）
    "09": (34.71, 135.36),  # 阪神（宝塚）
    "10": (33.86, 130.81),  # 小倉
}

OPEN_METEO = ("https://api.open-meteo.com/v1/forecast"
              "?latitude={lat}&longitude={lon}&daily=precipitation_sum"
              "&start_date={start}&end_date={end}&timezone=Asia%2FTokyo")


def fetch_precip(venue_id: str, date: str) -> tuple[float, float]:
    """(前日の降水量, 当日の降水量) mm を返す。"""
    lat, lon = JRA_COORDS[venue_id]
    d = dt.date.fromisoformat(date)
    prev = (d - dt.timedelta(days=1)).isoformat()
    url = OPEN_METEO.format(lat=lat, lon=lon, start=prev, end=date)
    data = json.loads(scraper.http_get(url, encoding="utf-8"))
    arr = data.get("daily", {}).get("precipitation_sum", [])
    prior = float(arr[0]) if len(arr) > 0 and arr[0] is not None else 0.0
    today = float(arr[1]) if len(arr) > 1 and arr[1] is not None else 0.0
    return prior, today


def estimate_condition(prior_mm: float, today_mm: float) -> tuple[str, str]:
    """降水量から (馬場状態, 天候) を推定。前日雨は0.4倍で加味。"""
    wet = today_mm + 0.4 * prior_mm
    if wet < 1:
        cond = "良"
    elif wet < 8:
        cond = "稍重"
    elif wet < 20:
        cond = "重"
    else:
        cond = "不良"
    weather = "雨" if today_mm >= 1 else ("曇" if cond != "良" else "晴")
    return cond, weather


def update_weather(conn: sqlite3.Connection, date: str | None = None) -> int:
    """出走前レースに天気予報ベースの馬場状態をセット。更新したグループ数を返す。"""
    q = """SELECT DISTINCT ra.venue_id, ra.race_date
           FROM races ra JOIN results r ON ra.race_id = r.race_id
           WHERE r.finish_position IS NULL
             AND ra.venue_id IS NOT NULL AND ra.race_date IS NOT NULL"""
    params: tuple = ()
    if date:
        q += " AND ra.race_date = ?"
        params = (date,)
    groups = conn.execute(q, params).fetchall()
    if not groups:
        print("対象レースがありません（出馬表に開催日が入っていますか?）", file=sys.stderr)
        return 0

    n = 0
    for venue, d in groups:
        if venue not in JRA_COORDS:
            continue
        try:
            prior, today = fetch_precip(venue, d)
        except Exception as e:  # noqa: BLE001
            print(f"[NG] {venue} {d}: 天気取得失敗 {e}", file=sys.stderr)
            continue
        cond, weather = estimate_condition(prior, today)
        conn.execute(
            """UPDATE races SET track_condition=?, weather=?
               WHERE venue_id=? AND race_date=?
                 AND race_id IN (SELECT race_id FROM results WHERE finish_position IS NULL)""",
            (cond, weather, venue, d),
        )
        n += 1
        print(f"[OK] {venue} {d}: 前日{prior:.0f}mm+当日{today:.0f}mm → 馬場「{cond}」/ {weather}")
    conn.commit()
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="天気予報→馬場状態を反映")
    p.add_argument("--db", default="keiba.db")
    p.add_argument("--date", help="開催日 (YYYY-MM-DD) に限定")
    args = p.parse_args(argv)

    conn = sqlite3.connect(args.db)
    try:
        n = update_weather(conn, args.date)
    finally:
        conn.close()
    print(f"\n完了: {n} 開催地×日付の馬場を更新しました。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
