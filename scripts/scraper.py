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
SHUTUBA_URL = "https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"  # 出馬表
RESULT_URL = "https://race.netkeiba.com/race/result.html?race_id={race_id}"    # 速報結果
# 単勝・複勝オッズのJSON API（type=1）。出馬表ページのJS描画オッズの代わりに使う
ODDS_API = ("https://race.netkeiba.com/api/api_get_jra_odds.html"
            "?race_id={race_id}&type=1&action=update")
ENCODING = "euc-jp"               # db.netkeiba.com は EUC-JP
REQUEST_INTERVAL = 1.5            # リクエスト間隔(秒)。マナーとして必須
MAX_RETRIES = 4                   # ネットワーク/一時エラー時の最大リトライ回数
# ブラウザ相当の UA。素っ気ない UA だと弾かれやすいため。
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# リトライ対象の一時的ステータス（恒久エラーの 403/404 は即時あきらめる）
RETRY_STATUS = {429, 500, 502, 503, 504}

ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# HTTP 取得（リトライ + 指数バックオフ + ジッタ）
# -----------------------------------------------------------------------------
def http_get(url: str, encoding: str | None = None, max_retries: int = MAX_RETRIES) -> str:
    """URL を取得して本文(str)を返す。一時エラーは指数バックオフで再試行する。

    - タイムアウト/接続エラーや 429,5xx はリトライ（2,4,8,16秒 + ジッタ）。
    - 403/404 等の恒久エラーは即座に例外送出（リトライしても無駄なため）。
    """
    import random
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code in RETRY_STATUS:
                raise requests.HTTPError(f"{resp.status_code} (一時エラー)", response=resp)
            resp.raise_for_status()
            resp.encoding = encoding or resp.apparent_encoding or "utf-8"
            return resp.text
        except requests.RequestException as e:
            # 恒久エラー（403/404 等）はリトライせず即送出
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and status not in RETRY_STATUS:
                raise
            last_err = e
            if attempt < max_retries:
                wait = 2 ** (attempt + 1) + random.uniform(0, 1.0)
                print(f"   ...retry {attempt + 1}/{max_retries} after {wait:.1f}s ({e})",
                      file=sys.stderr)
                time.sleep(wait)
    raise last_err if last_err else RuntimeError("http_get failed")

# 競馬場名 → 場コード（race_id 由来の場コードと突き合わせる用の逆引き）
VENUE_NAME_TO_ID = {
    "札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05",
    "中山": "06", "中京": "07", "京都": "08", "阪神": "09", "小倉": "10",
}


# -----------------------------------------------------------------------------
# 取得
# -----------------------------------------------------------------------------
def fetch_html(race_id: str) -> str:
    """レース結果ページの HTML を取得して返す（EUC-JP）。"""
    return http_get(BASE_URL.format(race_id=race_id), encoding=ENCODING)


# -----------------------------------------------------------------------------
# パース補助
# -----------------------------------------------------------------------------
def _id_from_href(tag, kind: str) -> str | None:
    """リンクから末尾のIDを抜き出す。kind='horse'/'jockey'/'trainer' 等。

    netkeiba のリンクは /horse/2021105250/ のほか
    /jockey/result/recent/05339/ のように中間語(result/recent)が入る形があるため、
    末尾のパスセグメントを ID とみなす。
    """
    if tag is None:
        return None
    a = tag.find("a", href=re.compile(rf"/{kind}/"))
    if not a:
        return None
    href = a["href"].split("?")[0].split("#")[0].rstrip("/")
    seg = href.rsplit("/", 1)[-1]
    return seg or None


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


def _lap_metrics(laps: list[float]) -> dict:
    """200mラップ列から前後半・上り3F・前後傾(pace_diff)を計算して返す。"""
    import math
    n = len(laps)
    h = math.ceil(n / 2)
    first_half, second_half = sum(laps[:h]), sum(laps[h:])
    n1, n2 = h, n - h
    pace_diff = (first_half / n1 - second_half / n2) if n2 else None
    return {
        "lap_seq": ",".join(f"{v:.1f}" for v in laps),
        "n_laps": n,
        "first_half": round(first_half, 1),
        "second_half": round(second_half, 1),
        "race_first3f": round(sum(laps[:3]), 1),
        "race_last3f": round(sum(laps[-3:]), 1),
        "pace_diff": round(pace_diff, 3) if pace_diff is not None else None,
    }


def parse_laps(html: str) -> dict | None:
    """結果ページHTMLから 200mごとのラップ列を抽出して指標化する。

    race.netkeiba は table.Race_HaronTime にラップを持つ:
      Header行(200m,400m..) / 累計行(12.7,24.0..) / ラップ行(12.7,11.3,11.5..)。
    各セルが全て 8〜16秒の行＝200mごとのラップ行として採用する（累計行は
    値が16超のため自然に除外される）。取れなければ ' - ' 区切りの旧式も試す。
    取れなければ None。pace_diff>0=後傾(瞬発力勝負), <0=前傾(持続力勝負)。
    """
    soup = BeautifulSoup(html, "lxml")
    best: list[float] | None = None
    for tbl in soup.select("table.Race_HaronTime, table[summary='ラップタイム']"):
        for tr in tbl.find_all("tr"):
            vals = []
            ok = True
            for td in tr.find_all("td"):
                t = td.get_text(strip=True)
                try:
                    v = float(t)
                except ValueError:    # 累計の "1:10.4" 等は弾く
                    ok = False
                    break
                vals.append(v)
            if ok and len(vals) >= 4 and all(8.0 <= v <= 16.0 for v in vals):
                if best is None or len(vals) > len(best):
                    best = vals
    if best is None:   # フォールバック: "12.4 - 10.9 - ..." 形式
        for m in re.finditer(r"(?:\d{1,2}\.\d\s*[-–]\s*){3,}\d{1,2}\.\d", html):
            seq = [float(x) for x in re.findall(r"\d{1,2}\.\d", m.group(0))]
            if len(seq) >= 4 and all(8.0 <= v <= 16.0 for v in seq):
                if best is None or len(seq) > len(best):
                    best = seq
    if not best:
        return None
    return _lap_metrics(best)


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

    h1 = soup.select_one("dl.racedata h1, .racedata h1, .data_intro h1, h1")
    race["race_name"] = h1.get_text(strip=True) if h1 else None
    if not race["race_name"]:
        # フォールバック: ページタイトル先頭をレース名とする
        t = soup.select_one("title")
        if t:
            race["race_name"] = (t.get_text(strip=True)
                                 .split("｜")[0].split("|")[0]
                                 .split("結果")[0].strip() or None)

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

    return {"race": race, "results": results, "payouts": payouts,
            "laps": parse_laps(html)}


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
               race_date=COALESCE(excluded.race_date, races.race_date),
               venue_id=COALESCE(excluded.venue_id, races.venue_id),
               race_number=COALESCE(excluded.race_number, races.race_number),
               race_name=COALESCE(excluded.race_name, races.race_name),
               grade=COALESCE(excluded.grade, races.grade),
               surface=COALESCE(excluded.surface, races.surface),
               distance=COALESCE(excluded.distance, races.distance),
               direction=COALESCE(excluded.direction, races.direction),
               weather=COALESCE(excluded.weather, races.weather),
               track_condition=COALESCE(excluded.track_condition, races.track_condition),
               field_size=COALESCE(excluded.field_size, races.field_size)""",
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
                   horse_id=excluded.horse_id,
                   jockey_id=COALESCE(excluded.jockey_id, results.jockey_id),
                   trainer_id=COALESCE(excluded.trainer_id, results.trainer_id),
                   post_position=COALESCE(excluded.post_position, results.post_position),
                   weight_carried=COALESCE(excluded.weight_carried, results.weight_carried),
                   finish_position=excluded.finish_position, finish_status=excluded.finish_status,
                   time_seconds=excluded.time_seconds, margin=excluded.margin,
                   passing=excluded.passing, last_3f=excluded.last_3f,
                   horse_weight=COALESCE(excluded.horse_weight, results.horse_weight),
                   weight_change=COALESCE(excluded.weight_change, results.weight_change),
                   odds=excluded.odds, popularity=excluded.popularity""",
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

    # ラップ（取得できた場合のみ）
    laps = parsed.get("laps")
    if laps:
        conn.execute(
            """INSERT OR REPLACE INTO race_laps
                   (race_id, lap_seq, n_laps, first_half, second_half,
                    race_first3f, race_last3f, pace_diff)
               VALUES (:race_id, :lap_seq, :n_laps, :first_half, :second_half,
                       :race_first3f, :race_last3f, :pace_diff)""",
            {"race_id": race["race_id"], **laps},
        )

    conn.commit()


# -----------------------------------------------------------------------------
# 出馬表（出走前）のパース
#   結果がまだ無いレースを、results に finish_position=NULL で登録するための
#   情報を抽出する。レース条件・出走馬・朝のオッズ/人気を取得。
# -----------------------------------------------------------------------------
def _header_index(header_cells: list[str]):
    """ヘッダ文字列 → 列インデックス。部分一致で柔軟に解決する。"""
    def find(*keys):
        for i, h in enumerate(header_cells):
            if any(k in h for k in keys):
                return i
        return None
    return {
        "枠": find("枠"),
        "馬番": find("馬番"),
        "馬名": find("馬名"),
        "性齢": find("性齢"),
        "斤量": find("斤量"),
        "騎手": find("騎手"),
        "厩舎": find("厩舎", "調教師"),
        "馬体重": find("馬体重"),
        "オッズ": find("オッズ", "単勝"),
        "人気": find("人気"),
    }


def parse_shutuba(html: str, race_id: str) -> dict:
    """出馬表ページから race（結果未確定）と出走馬を抽出する。"""
    soup = BeautifulSoup(html, "lxml")
    race: dict = {"race_id": race_id, "venue_id": race_id[4:6],
                  "race_number": _to_int(race_id[-2:])}

    name_el = soup.select_one(".RaceName, .RaceList_Item02 .RaceName, h1")
    race["race_name"] = name_el.get_text(strip=True) if name_el else None
    if not race["race_name"]:   # 枠順前等で取れない場合 og:title から（"... 出馬表 | ..."）
        m = re.search(r'og:title"\s+content="([^"|]+)', html)
        if m:
            race["race_name"] = re.sub(r"\s*出馬表\s*$", "", m.group(1)).strip() or None

    cond = soup.select_one(".RaceData01")
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
    elif "直" in cond_text:
        race["direction"] = "直線"
    m = re.search(r"天候\s*:\s*(\S+)", cond_text)
    race["weather"] = m.group(1) if m else None
    m = re.search(r"馬場\s*:\s*(良|稍重|重|不良)", cond_text)
    race["track_condition"] = m.group(1) if m else None

    # 開催日は kaisai_date=YYYYMMDD から取得（NOT NULL のため必須）
    m = re.search(r"kaisai_date=(\d{8})", html)
    if m:
        d = m.group(1)
        race["race_date"] = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    else:
        race["race_date"] = None

    m = re.search(r"(G[ⅠⅡⅢ123])", (race.get("race_name") or "") + " " + cond_text)
    race["grade"] = m.group(1) if m else None

    # ---- 出走馬テーブル ----
    table = soup.select_one("table.Shutuba_Table, table.ShutubaTable, table.RaceTable01")
    results = []
    if table:
        rows_tr = table.select("tr")
        header_cells = [c.get_text(strip=True) for c in rows_tr[0].find_all(["th", "td"])]
        idx = _header_index(header_cells)

        def cell(cells, key):
            i = idx.get(key)
            return cells[i] if i is not None and i < len(cells) else None

        seq = 0
        saw_umaban = False   # 馬番セルが1つも無い＝枠順確定前（特別登録リスト）
        for tr in rows_tr[1:]:
            cells = tr.find_all("td")
            if not cells:
                continue  # ヘッダー2行目(th のみ)等はスキップ
            # 馬名セル＝行内の /horse/ リンクを持つ td（枠順前で列がずれても拾える）
            hc = cell(cells, "馬名")
            if hc is None or not _id_from_href(hc, "horse"):
                a = tr.select_one('a[href*="/horse/"]')
                hc = a.find_parent("td") if a else hc
            horse_id = _id_from_href(hc, "horse")
            if not horse_id:
                continue  # 出走馬でない行（区切り行・除外馬等）はスキップ
            seq += 1
            row = {"race_id": race_id, "finish_position": None, "finish_status": None}
            # 馬番・枠は td class="Umaban*/Waku*" から読む（枠順確定版）。無ければヘッダ→連番
            uta = tr.select_one('td[class*="Umaban"]')
            wta = tr.select_one('td[class*="Waku"]')
            hn = _to_int(uta.get_text(strip=True)) if uta else None
            if hn:
                saw_umaban = True
            row["post_position"] = (_to_int(wta.get_text(strip=True)) if wta else None) \
                or _to_int((cell(cells, "枠") or _blank()).get_text(strip=True))
            # 枠順確定前は馬番が無いので連番を仮置き（確定後の再取得で本来の馬番に更新される）
            row["horse_number"] = hn \
                or _to_int((cell(cells, "馬番") or _blank()).get_text(strip=True)) or seq
            row["horse_id"] = horse_id
            row["horse_name"] = hc.get_text(strip=True) if hc else None
            sexage = (cell(cells, "性齢") or _blank()).get_text(strip=True)
            row["sex"] = sexage[0] if sexage else None
            jc = cell(cells, "騎手")
            row["jockey_id"] = _id_from_href(jc, "jockey")
            row["jockey_name"] = jc.get_text(strip=True) if jc else None
            tc = cell(cells, "厩舎")
            row["trainer_id"] = _id_from_href(tc, "trainer")
            row["trainer_name"] = tc.get_text(strip=True) if tc else None
            row["weight_carried"] = _to_float((cell(cells, "斤量") or _blank()).get_text(strip=True))
            row["odds"] = _to_float((cell(cells, "オッズ") or _blank()).get_text(strip=True))
            row["popularity"] = _to_int((cell(cells, "人気") or _blank()).get_text(strip=True))

            bw = (cell(cells, "馬体重") or _blank()).get_text(strip=True)
            mbw = re.match(r"(\d+)\(([-+]?\d+)\)", bw)
            if mbw:
                row["horse_weight"] = _to_int(mbw.group(1))
                row["weight_change"] = _to_int(mbw.group(2))
            else:
                row["horse_weight"] = _to_int(bw) if bw.isdigit() else None
                row["weight_change"] = None
            # 出走前なので走破系は無し
            row.update({"time_seconds": None, "margin": None, "passing": None, "last_3f": None})
            results.append(row)

    race["field_size"] = len(results) or None
    # 枠順確定前（馬番セル無し）や JRAフルゲート(18頭)超は「特別登録リスト」段階。
    # 登録リストは他レースとの重複登録・登録落ち馬を含むため、呼び出し側で警告する。
    provisional = bool(results) and (not saw_umaban or len(results) > 18)
    return {"race": race, "results": results, "payouts": [],
            "provisional": provisional}


# -----------------------------------------------------------------------------
# 速報結果（race.netkeiba.com）のパース
#   db.netkeiba.com（アーカイブ側）は反映が遅いため、レース直後に確定する
#   速報サイトの結果ページから着順を取得するフォールバック経路。
# -----------------------------------------------------------------------------
def _result_header_index(header_cells: list[str]):
    """結果テーブルのヘッダ文字列 → 列インデックス（部分一致で解決）。"""
    def find(*keys):
        for i, h in enumerate(header_cells):
            if any(k in h for k in keys):
                return i
        return None
    return {
        "着順": find("着順", "着 順"),
        "枠": find("枠"),
        "馬番": find("馬番"),
        "馬名": find("馬名"),
        "性齢": find("性齢"),
        "斤量": find("斤量"),
        "騎手": find("騎手"),
        "タイム": find("タイム"),
        "着差": find("着差"),
        "人気": find("人気"),
        "単勝": find("単勝", "オッズ"),
        "上り": find("後3F", "上り", "上がり"),
        "通過": find("通過"),
        "厩舎": find("厩舎", "調教師"),
        "馬体重": find("馬体重"),
    }


_BET_CLASS = {"Tansho": "単勝", "Fukusho": "複勝", "Wakuren": "枠連", "Umaren": "馬連",
              "Wide": "ワイド", "Umatan": "馬単", "Fuku3": "三連複", "Tan3": "三連単"}


def parse_payouts_live(soup, race_id: str) -> list:
    """race.netkeiba 結果ページの払戻表(Payout_Detail_Table)から払戻を抽出する。

    各 tr は券種クラス(Tansho/Fukusho/…)を持つ。Result列=組番、Payout列=払戻金。
    複勝・ワイド等は複数行ぶんが1セルに入るので stripped_strings で対応付ける。
    """
    payouts = []
    for tr in soup.select("table.Payout_Detail_Table tr"):
        classes = tr.get("class") or []
        bet_type = next((_BET_CLASS[c] for c in classes if c in _BET_CLASS), None)
        if not bet_type:
            continue
        res_td = tr.select_one("td.Result")
        pay_td = tr.select_one("td.Payout")
        if not res_td or not pay_td:
            continue
        combos = [s.replace(" ", "") for s in res_td.stripped_strings if re.search(r"\d", s)]
        pays = [re.sub(r"[^\d]", "", s) for s in pay_td.stripped_strings if re.search(r"\d", s)]
        for i, combo in enumerate(combos):
            payouts.append({
                "race_id": race_id, "bet_type": bet_type, "combination": combo,
                "payout": _to_int(pays[i]) if i < len(pays) else None, "popularity": None,
            })
    return payouts


def parse_result_live(html: str, race_id: str) -> dict:
    """race.netkeiba.com の結果ページから race / results を抽出する。

    着順が1つも無い（=まだ確定前）場合は results を空で返す。払戻は
    db.netkeiba 反映後にまとめて取得する想定で、ここでは扱わない。
    """
    soup = BeautifulSoup(html, "lxml")
    race: dict = {"race_id": race_id, "venue_id": race_id[4:6],
                  "race_number": _to_int(race_id[-2:])}

    name_el = soup.select_one(".RaceName, .RaceList_Item02 .RaceName, h1")
    race["race_name"] = name_el.get_text(strip=True) if name_el else None
    if not race["race_name"]:   # 枠順前等で取れない場合 og:title から（"... 出馬表 | ..."）
        m = re.search(r'og:title"\s+content="([^"|]+)', html)
        if m:
            race["race_name"] = re.sub(r"\s*出馬表\s*$", "", m.group(1)).strip() or None

    cond = soup.select_one(".RaceData01")
    cond_text = cond.get_text(" ", strip=True) if cond else ""
    if "芝" in cond_text:
        race["surface"] = "芝"
    elif "ダ" in cond_text:
        race["surface"] = "ダート"
    elif "障" in cond_text:
        race["surface"] = "障害"
    else:
        race["surface"] = None
    m = re.search(r"(\d{3,4})m", cond_text)
    race["distance"] = _to_int(m.group(1)) if m else None
    if "右" in cond_text:
        race["direction"] = "右"
    elif "左" in cond_text:
        race["direction"] = "左"
    elif "直" in cond_text:
        race["direction"] = "直線"
    else:
        race["direction"] = None
    m = re.search(r"天候\s*[:：]\s*(\S+)", cond_text)
    race["weather"] = m.group(1) if m else None
    m = re.search(r"馬場\s*[:：]\s*(良|稍重|重|不良)", cond_text)
    race["track_condition"] = m.group(1) if m else None

    # 開催日（ページ内の "YYYY年M月D日"。取れなければ None＝既存値を保持）
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", html)
    race["race_date"] = (f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
                         if m else None)
    m = re.search(r"(G[ⅠⅡⅢ123])", (race.get("race_name") or "") + " " + cond_text)
    race["grade"] = m.group(1) if m else None

    # ---- 結果テーブル ----
    table = soup.select_one(
        "table.RaceTable01, table.RaceCommon_Table, div.ResultTableWrap table")
    results = []
    if table:
        rows_tr = table.select("tr")
        header_cells = [c.get_text(strip=True) for c in rows_tr[0].find_all(["th", "td"])]
        idx = _result_header_index(header_cells)

        def cell(cells, key):
            i = idx.get(key)
            return cells[i] if i is not None and i < len(cells) else None

        for tr in rows_tr[1:]:
            cells = tr.find_all("td")
            if not cells:
                continue
            row = {"race_id": race_id}
            fin = (cell(cells, "着順") or _blank()).get_text(strip=True)
            row["finish_position"] = _to_int(fin)
            row["finish_status"] = (fin or None) if row["finish_position"] is None else None
            row["post_position"] = _to_int((cell(cells, "枠") or _blank()).get_text(strip=True))
            row["horse_number"] = _to_int((cell(cells, "馬番") or _blank()).get_text(strip=True))
            if row["horse_number"] is None:
                continue
            hc = cell(cells, "馬名")
            row["horse_id"] = _id_from_href(hc, "horse")
            row["horse_name"] = hc.get_text(strip=True) if hc else None
            sexage = (cell(cells, "性齢") or _blank()).get_text(strip=True)
            row["sex"] = sexage[0] if sexage else None
            jc = cell(cells, "騎手")
            row["jockey_id"] = _id_from_href(jc, "jockey")
            row["jockey_name"] = jc.get_text(strip=True) if jc else None
            tc = cell(cells, "厩舎")
            row["trainer_id"] = _id_from_href(tc, "trainer")
            row["trainer_name"] = tc.get_text(strip=True) if tc else None
            row["weight_carried"] = _to_float((cell(cells, "斤量") or _blank()).get_text(strip=True))
            row["time_seconds"] = _time_to_seconds((cell(cells, "タイム") or _blank()).get_text(strip=True))
            row["margin"] = (cell(cells, "着差") or _blank()).get_text(strip=True) or None
            row["passing"] = (cell(cells, "通過") or _blank()).get_text(strip=True) or None
            row["last_3f"] = _to_float((cell(cells, "上り") or _blank()).get_text(strip=True))
            row["odds"] = _to_float((cell(cells, "単勝") or _blank()).get_text(strip=True))
            row["popularity"] = _to_int((cell(cells, "人気") or _blank()).get_text(strip=True))
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
    return {"race": race, "results": results,
            "payouts": parse_payouts_live(soup, race_id),
            "laps": parse_laps(html)}


def _has_finish(parsed: dict) -> bool:
    """着順が1つでも確定しているか（=結果が取得できたか）。"""
    return any(r.get("finish_position") is not None for r in parsed.get("results", []))


# -----------------------------------------------------------------------------
# 1レース取り込み（取得→パース→投入をまとめたヘルパ）
# -----------------------------------------------------------------------------
def ingest_race(conn: sqlite3.Connection, race_id: str) -> dict:
    """確定済みレースを取得・パースして DB へ投入し、parse 結果を返す。

    まず db.netkeiba.com（アーカイブ）から取得し、未反映なら race.netkeiba.com
    （速報）の結果ページにフォールバックする。どちらも着順が無い場合は既存
    データを壊さないよう upsert せずに例外を送出する（時間をおいて再実行）。
    """
    parsed = parse_race(fetch_html(race_id), race_id)
    if not _has_finish(parsed):
        # db.netkeiba 未反映 → 速報サイトの結果ページから着順を取得
        try:
            live_html = http_get(RESULT_URL.format(race_id=race_id), encoding="utf-8")
            live = parse_result_live(live_html, race_id)
            if _has_finish(live):
                parsed = live
        except Exception:  # noqa: BLE001  速報側の失敗は致命的でない
            pass
    if not _has_finish(parsed):
        raise ValueError("結果が未掲載です（db.netkeiba/race.netkeiba とも未反映。時間をおいて再実行）")
    upsert_race(conn, parsed)
    return parsed


def ingest_shutuba(conn: sqlite3.Connection, race_id: str, race_date: str | None = None) -> dict:
    """出馬表（出走前）を取得・パースして DB へ投入する。finish は NULL。

    race_date を渡すと、ページから日付が取れない場合の補完に使う
    （出走前ページは開催日を取りにくいため、クローラの指定日で補う）。
    """
    html = http_get(SHUTUBA_URL.format(race_id=race_id))
    parsed = parse_shutuba(html, race_id)
    if race_date:   # クローラ指定日を開催日の正とする（出走前ページは日付が不確実なため）
        parsed["race"]["race_date"] = race_date
    upsert_race(conn, parsed)
    # --- 再取得時の掃除 ---------------------------------------------------
    #   登録段階(仮馬番・重複登録)で取り込んだ行は、枠順確定後の再取得で
    #   出走馬から消えても upsert では残る。今回の出走馬に無い未確定行を削除する。
    #   （パース0頭の時は消さない＝取得失敗でデータを壊さないため）
    if parsed["results"]:
        keep = {(r["horse_id"], r["horse_number"]) for r in parsed["results"]}
        old = conn.execute(
            "SELECT horse_id, horse_number FROM results "
            "WHERE race_id=? AND finish_position IS NULL", (race_id,)).fetchall()
        stale = [(h, n) for (h, n) in old if (h, n) not in keep]
        for h, n in stale:
            conn.execute(
                "DELETE FROM results WHERE race_id=? AND horse_id=? "
                "AND horse_number=? AND finish_position IS NULL", (race_id, h, n))
        if stale:
            print(f"    [掃除] {race_id}: 登録落ち・馬番変更の旧行 {len(stale)}件を削除")
        conn.execute("UPDATE races SET field_size=? WHERE race_id=?",
                     (len(parsed["results"]), race_id))
    if parsed.get("provisional"):
        print(f"    [注意] {race_id}: 枠順確定前の登録馬リスト"
              f"({len(parsed['results'])}頭)。確定後に再取得してください")
    return parsed


# -----------------------------------------------------------------------------
# オッズ取得（JSON API）
#   出馬表ページのオッズはJSで動的描画され静的取得できないため、
#   netkeiba の単勝オッズ JSON API から取得して results を更新する。
# -----------------------------------------------------------------------------
def fetch_odds(race_id: str) -> dict[int, float]:
    """{馬番: 単勝オッズ} を返す。取得できなければ空 dict。"""
    import json
    raw = http_get(ODDS_API.format(race_id=race_id), encoding="utf-8")
    data = json.loads(raw)
    win = (data.get("data", {}) or {}).get("odds", {}).get("1", {})  # "1"=単勝
    out: dict[int, float] = {}
    for k, v in win.items():
        try:
            num = int(k)
            # v は ["3.2", "1", ...] のような配列。先頭が単勝オッズ
            od = float(v[0] if isinstance(v, (list, tuple)) else v)
            if od > 0:
                out[num] = od
        except (ValueError, TypeError, IndexError):
            continue
    return out


def update_odds(conn: sqlite3.Connection, race_id: str) -> int:
    """オッズAPIから単勝オッズを取得し、results の odds/popularity を更新する。

    人気は単勝オッズの昇順から算出する（API依存を避ける）。更新した頭数を返す。
    """
    om = fetch_odds(race_id)
    if not om:
        return 0
    order = sorted(om, key=lambda n: om[n])
    pop = {n: i + 1 for i, n in enumerate(order)}
    for num, od in om.items():
        conn.execute(
            "UPDATE results SET odds=?, popularity=? WHERE race_id=? AND horse_number=?",
            (od, pop[num], race_id, num),
        )
    conn.commit()
    return len(om)


# -----------------------------------------------------------------------------
# スキーマ自動初期化
# -----------------------------------------------------------------------------
def ensure_schema(conn: sqlite3.Connection) -> None:
    """races テーブルが無ければ schema.sql / seed_venues.sql を適用して初期化する。

    初回実行時に「no such table: races」で落ちないよう、取り込み前に必ず呼ぶ。
    既に初期化済みなら何もしない（冪等）。
    """
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='races'"
    ).fetchone()
    if exists:
        return
    for fname in ("schema/schema.sql", "schema/seed_venues.sql"):
        conn.executescript((ROOT / fname).read_text(encoding="utf-8"))
    conn.commit()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="netkeiba レース結果/出馬表スクレイパー")
    parser.add_argument("race_ids", nargs="+", help="取り込むレースID（12桁）")
    parser.add_argument("--db", default="keiba.db", help="SQLite DBファイル (default: keiba.db)")
    parser.add_argument("--shutuba", action="store_true",
                        help="確定結果でなく出馬表（出走前）を取り込む（finish=NULL）")
    args = parser.parse_args(argv)

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON")
    ensure_schema(conn)   # 初回実行時に DB を自動初期化
    ingest = ingest_shutuba if args.shutuba else ingest_race

    for i, race_id in enumerate(args.race_ids):
        try:
            parsed = ingest(conn, race_id)
            n = len(parsed["results"])
            kind = "出馬表" if args.shutuba else "結果"
            print(f"[OK] {race_id}: {parsed['race'].get('race_name')} ({n}頭, {kind}) を取り込み")
        except Exception as e:  # noqa: BLE001  個別レースの失敗で全体を止めない
            print(f"[NG] {race_id}: {e}", file=sys.stderr)
        if i < len(args.race_ids) - 1:
            time.sleep(REQUEST_INTERVAL)

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
