#!/usr/bin/env python3
# =============================================================================
# netkeiba レース結果スクレイパー
#
# db.netkeiba.com のレース結果ページ (例: https://db.netkeiba.com/race/202405021211/)
# を取得・パースし、keiba.db の各テーブルへ UPSERT する。
#
# 使い方:
#   pip install -r scripts/requirements.txt
#   python scripts/scraper.py 202405021211                 # 1レース取り込み
#   python scripts/scraper.py 202405021211 202405021212    # 複数指定
#   python scripts/scraper.py --db mydb.db 202405021211
#
# 注意:
#   - 対象サイトの利用規約・robots.txt を確認し、個人利用の範囲で行うこと。
#   - 連続アクセスは負荷をかけるため、リクエスト間に必ずウェイトを入れている
#     (REQUEST_INTERVAL)。短縮しないこと。
#   - netkeiba の HTML 構造は変わりうる。パースが失敗する場合は
#     parse_race() のセレクタを調整する。
# =============================================================================
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://db.netkeiba.com/race/{race_id}/"
ENCODING = "euc-jp"               # db.netkeiba.com は EUC-JP
REQUEST_INTERVAL = 1.5            # リクエスト間隔(秒)。マナーとして必須
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; keiba-db/1.0; personal research)"
}

ROOT = Path(__file__).resolve().parent.parent

# 競馬場名 → 場コード（race_id 由来の場コードと突き合わせる用の逆引き）
VENUE_NAME_TO_ID = {
    "札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05",
    "中山": "06", "中京": "07", "京都": "08", "阪神": "09", "小倉": "10",
}


# -----------------------------------------------------------------------------
# 取得
# -----------------------------------------------------------------------------
def fetch_html(race_id: str) -> str:
    """レース結果ページの HTML を取得して返す。"""
    url = BASE_URL.format(race_id=race_id)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = ENCODING
    return resp.text


# -----------------------------------------------------------------------------
# パース補助
# -----------------------------------------------------------------------------
def _id_from_href(tag, kind: str) -> str | None:
    """<a href="/horse/2020xxxxxx/"> から ID 部分を抜き出す。kind='horse'等。"""
    if tag is None:
        return None
    a = tag.find("a", href=re.compile(rf"/{kind}/"))
    if not a:
        return None
    m = re.search(rf"/{kind}/(\w+)", a["href"])
    return m.group(1) if m else None


def _to_float(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _to_int(s: str):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def _time_to_seconds(s: str):
    """'1:58.4' -> 118.4 秒。'58.4' -> 58.4。"""
    if not s:
        return None
    s = s.strip()
    if ":" in s:
        m, sec = s.split(":")
        try:
            return int(m) * 60 + float(sec)
        except ValueError:
            return None
    return _to_float(s)


# -----------------------------------------------------------------------------
# パース本体
# -----------------------------------------------------------------------------
def parse_race(html: str, race_id: str) -> dict:
    """結果ページの HTML から race / results / payouts を抽出する。"""
    soup = BeautifulSoup(html, "lxml")

    # ---- レース基本情報 ----
    race: dict = {"race_id": race_id, "venue_id": race_id[4:6]}

    h1 = soup.select_one("dl.racedata h1, diary_snap_cut h1, h1")
    race["race_name"] = h1.get_text(strip=True) if h1 else None

    # "ダ右1600m / 天候 : 晴 / ダート : 良 / 発走 : 15:40" のような行
    cond = soup.select_one("diary_snap_cut span, dl.racedata span, .racedata span")
    cond_text = cond.get_text(" ", strip=True) if cond else ""
    if "芝" in cond_text:
        race["surface"] = "芝"
    elif "ダ" in cond_text:
        race["surface"] = "ダート"
    elif "障" in cond_text:
        race["surface"] = "障害"
    m = re.search(r"(\d{3,4})m", cond_text)
    race["distance"] = _to_int(m.group(1)) if m else None
    if "右" in cond_text:
        race["direction"] = "右"
    elif "左" in cond_text:
        race["direction"] = "左"
    elif "直線" in cond_text or "直" in cond_text:
        race["direction"] = "直線"
    m = re.search(r"天候\s*:\s*(\S+)", cond_text)
    race["weather"] = m.group(1) if m else None
    m = re.search(r"(?:芝|ダート|馬場)\s*:\s*(良|稍重|重|不良)", cond_text)
    race["track_condition"] = m.group(1) if m else None

    # 日付・開催（"2024年6月2日 2回東京11日目 ..." ）
    small = soup.select_one("p.smalltxt")
    small_text = small.get_text(" ", strip=True) if small else ""
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", small_text)
    race["race_date"] = (
        f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else None
    )
    m = re.search(r"(G1|G2|G3|G[ⅠⅡⅢ])", small_text + " " + (race.get("race_name") or ""))
    race["grade"] = m.group(1) if m else None
    # レース番号は race_id 末尾2桁
    race["race_number"] = _to_int(race_id[-2:])

    # ---- 結果テーブル ----
    table = soup.select_one("table.race_table_01")
    results = []
    if table:
        header_cells = [th.get_text(strip=True) for th in table.select("tr")[0].find_all(["th", "td"])]
        col = {name: i for i, name in enumerate(header_cells)}

        def cell(cells, name):
            i = col.get(name)
            return cells[i] if i is not None and i < len(cells) else None

        for tr in table.select("tr")[1:]:
            cells = tr.find_all("td")
            if not cells:
                continue
            row = {"race_id": race_id}
            row["finish_position"] = _to_int((cell(cells, "着順") or _blank()).get_text(strip=True))
            if row["finish_position"] is None:
                row["finish_status"] = (cell(cells, "着順") or _blank()).get_text(strip=True) or None
            row["post_position"] = _to_int((cell(cells, "枠番") or _blank()).get_text(strip=True))
            row["horse_number"] = _to_int((cell(cells, "馬番") or _blank()).get_text(strip=True))

            horse_cell = cell(cells, "馬名")
            row["horse_id"] = _id_from_href(horse_cell, "horse")
            row["horse_name"] = horse_cell.get_text(strip=True) if horse_cell else None

            sexage = (cell(cells, "性齢") or _blank()).get_text(strip=True)
            row["sex"] = sexage[0] if sexage else None

            jockey_cell = cell(cells, "騎手")
            row["jockey_id"] = _id_from_href(jockey_cell, "jockey")
            row["jockey_name"] = jockey_cell.get_text(strip=True) if jockey_cell else None

            trainer_cell = cell(cells, "調教師")
            row["trainer_id"] = _id_from_href(trainer_cell, "trainer")
            row["trainer_name"] = trainer_cell.get_text(strip=True) if trainer_cell else None

            row["weight_carried"] = _to_float((cell(cells, "斤量") or _blank()).get_text(strip=True))
            row["time_seconds"] = _time_to_seconds((cell(cells, "タイム") or _blank()).get_text(strip=True))
            row["margin"] = (cell(cells, "着差") or _blank()).get_text(strip=True) or None
            row["passing"] = (cell(cells, "通過") or _blank()).get_text(strip=True) or None
            row["last_3f"] = _to_float((cell(cells, "上り") or _blank()).get_text(strip=True))
            row["odds"] = _to_float((cell(cells, "単勝") or _blank()).get_text(strip=True))
            row["popularity"] = _to_int((cell(cells, "人気") or _blank()).get_text(strip=True))

            # 馬体重 "480(+4)" を分解
            bw = (cell(cells, "馬体重") or _blank()).get_text(strip=True)
            mbw = re.match(r"(\d+)\(([-+]?\d+)\)", bw)
            if mbw:
                row["horse_weight"] = _to_int(mbw.group(1))
                row["weight_change"] = _to_int(mbw.group(2))
            else:
                row["horse_weight"] = _to_int(bw) if bw.isdigit() else None
                row["weight_change"] = None

            results.append(row)

    race["field_size"] = len(results) or None

    # ---- 払戻 ----
    payouts = []
    for pay_table in soup.select("table.pay_table_01"):
        for tr in pay_table.select("tr"):
            th = tr.find("th")
            tds = tr.find_all("td")
            if not th or len(tds) < 2:
                continue
            bet_type = th.get_text(strip=True)
            combos = list(tds[0].stripped_strings)
            payouts_v = [re.sub(r"[^\d]", "", x) for x in tds[1].stripped_strings if re.search(r"\d", x)]
            pops = list(tds[2].stripped_strings) if len(tds) > 2 else []
            for i, combo in enumerate(combos):
                payouts.append({
                    "race_id": race_id,
                    "bet_type": bet_type,
                    "combination": combo.replace(" ", ""),
                    "payout": _to_int(payouts_v[i]) if i < len(payouts_v) else None,
                    "popularity": _to_int(pops[i]) if i < len(pops) else None,
                })

    return {"race": race, "results": results, "payouts": payouts}


class _Blank:
    def get_text(self, *a, **k):
        return ""


def _blank():
    return _Blank()


# -----------------------------------------------------------------------------
# DB 投入（UPSERT）
# -----------------------------------------------------------------------------
def upsert_race(conn: sqlite3.Connection, parsed: dict) -> None:
    """parse_race の結果を各テーブルへ UPSERT する。"""
    race = parsed["race"]
    rows = parsed["results"]

    # マスタ（騎手・調教師・馬）を先に登録
    for r in rows:
        if r.get("jockey_id"):
            conn.execute(
                "INSERT OR IGNORE INTO jockeys(jockey_id, name) VALUES (?, ?)",
                (r["jockey_id"], r.get("jockey_name")),
            )
        if r.get("trainer_id"):
            conn.execute(
                "INSERT OR IGNORE INTO trainers(trainer_id, name) VALUES (?, ?)",
                (r["trainer_id"], r.get("trainer_name")),
            )
        if r.get("horse_id"):
            conn.execute(
                """INSERT INTO horses(horse_id, name, sex, trainer_id) VALUES (?, ?, ?, ?)
                   ON CONFLICT(horse_id) DO UPDATE SET
                       name=excluded.name,
                       sex=COALESCE(excluded.sex, horses.sex),
                       trainer_id=COALESCE(excluded.trainer_id, horses.trainer_id)""",
                (r["horse_id"], r.get("horse_name"), r.get("sex"), r.get("trainer_id")),
            )

    # レース
    conn.execute(
        """INSERT INTO races
               (race_id, race_date, venue_id, race_number, race_name, grade,
                surface, distance, direction, weather, track_condition, field_size)
           VALUES (:race_id, :race_date, :venue_id, :race_number, :race_name, :grade,
                   :surface, :distance, :direction, :weather, :track_condition, :field_size)
           ON CONFLICT(race_id) DO UPDATE SET
               race_date=excluded.race_date, venue_id=excluded.venue_id,
               race_number=excluded.race_number, race_name=excluded.race_name,
               grade=excluded.grade, surface=excluded.surface,
               distance=excluded.distance, direction=excluded.direction,
               weather=excluded.weather, track_condition=excluded.track_condition,
               field_size=excluded.field_size""",
        {**{k: race.get(k) for k in
            ("race_id", "race_date", "venue_id", "race_number", "race_name", "grade",
             "surface", "distance", "direction", "weather", "track_condition", "field_size")}},
    )

    # 成績
    for r in rows:
        conn.execute(
            """INSERT INTO results
                   (race_id, horse_id, jockey_id, trainer_id, post_position, horse_number,
                    finish_position, finish_status, time_seconds, margin, passing, last_3f,
                    weight_carried, horse_weight, weight_change, odds, popularity)
               VALUES (:race_id, :horse_id, :jockey_id, :trainer_id, :post_position, :horse_number,
                       :finish_position, :finish_status, :time_seconds, :margin, :passing, :last_3f,
                       :weight_carried, :horse_weight, :weight_change, :odds, :popularity)
               ON CONFLICT(race_id, horse_number) DO UPDATE SET
                   finish_position=excluded.finish_position, time_seconds=excluded.time_seconds,
                   last_3f=excluded.last_3f, odds=excluded.odds, popularity=excluded.popularity""",
            {k: r.get(k) for k in
             ("race_id", "horse_id", "jockey_id", "trainer_id", "post_position", "horse_number",
              "finish_position", "finish_status", "time_seconds", "margin", "passing", "last_3f",
              "weight_carried", "horse_weight", "weight_change", "odds", "popularity")},
        )

    # 払戻
    for p in parsed["payouts"]:
        conn.execute(
            """INSERT OR REPLACE INTO payouts(race_id, bet_type, combination, payout, popularity)
               VALUES (:race_id, :bet_type, :combination, :payout, :popularity)""",
            p,
        )

    conn.commit()


# -----------------------------------------------------------------------------
# 1レース取り込み（取得→パース→投入をまとめたヘルパ）
# -----------------------------------------------------------------------------
def ingest_race(conn: sqlite3.Connection, race_id: str) -> dict:
    """race_id を取得・パースして DB へ投入し、parse 結果を返す。"""
    html = fetch_html(race_id)
    parsed = parse_race(html, race_id)
    upsert_race(conn, parsed)
    return parsed


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="netkeiba レース結果スクレイパー")
    parser.add_argument("race_ids", nargs="+", help="取り込むレースID（12桁）")
    parser.add_argument("--db", default="keiba.db", help="SQLite DBファイル (default: keiba.db)")
    args = parser.parse_args(argv)

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON")

    for i, race_id in enumerate(args.race_ids):
        try:
            parsed = ingest_race(conn, race_id)
            n = len(parsed["results"])
            print(f"[OK] {race_id}: {parsed['race'].get('race_name')} ({n}頭) を取り込み")
        except Exception as e:  # noqa: BLE001  個別レースの失敗で全体を止めない
            print(f"[NG] {race_id}: {e}", file=sys.stderr)
        if i < len(args.race_ids) - 1:
            time.sleep(REQUEST_INTERVAL)

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
