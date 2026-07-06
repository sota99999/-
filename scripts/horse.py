#!/usr/bin/env python3
# =============================================================================
# 馬プロファイル検索: 名前(部分一致可)で1頭のフル評価を表示する
#
#   ・現在の相対R / 最高perf(天井) / ムラ度pstd / 条件補正の対象か
#   ・能力プロファイル(power_top と 6能力: 瞬発/持続/トップ/パワー/スタミナ/先行)
#   ・競馬場×芝ダ別成績（出走数・勝利・複勝率・平均perf・最高perf）
#   ・距離帯別成績 / 道悪成績
#   ・近走一覧（日付・レース・条件・着順・perf・オッズ）
#
# 使い方:
#   python scripts/horse.py --db keiba.db クロワデュノール
#   python scripts/horse.py --db keiba.db クロワ            # 部分一致
# =============================================================================
from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

VENUE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
         "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
BAND = [(0, 1400, "短距離(〜1400)"), (1400, 1800, "マイル(1400-1800)"),
        (1800, 2200, "中距離(1800-2200)"), (2200, 9999, "長距離(2200〜)")]


def _f(v, fmt="5.1f", na="    -"):
    try:
        if v is None or pd.isna(v):
            return na
        return format(v, fmt)
    except (TypeError, ValueError):
        return na


def profile(db: str, name: str) -> int:
    con = sqlite3.connect(db)
    hits = pd.read_sql(
        "SELECT horse_id, name FROM horses WHERE name LIKE ? ORDER BY name",
        con, params=[f"%{name}%"])
    if hits.empty:
        print(f"『{name}』に一致する馬が見つかりません。")
        con.close(); return 1
    if len(hits) > 1 and (hits["name"] == name).sum() != 1:
        print(f"候補が{len(hits)}頭: " + " / ".join(hits["name"].head(20)))
        if len(hits) > 20:
            print("  …他。名前を絞ってください。")
        con.close(); return 0
    row = hits[hits["name"] == name].iloc[0] if (hits["name"] == name).any() else hits.iloc[0]
    hid, hname = row["horse_id"], row["name"]

    runs = pd.read_sql("""
        SELECT ra.race_date, ra.race_name, ra.venue_id, ra.surface, ra.distance,
               ra.track_condition, r.finish_position fin, r.odds, r.popularity,
               hr.r_before, hr.perf, hr.pstd
        FROM results r JOIN races ra ON ra.race_id = r.race_id
        LEFT JOIN horse_relative_r hr ON hr.race_id = r.race_id AND hr.horse_id = r.horse_id
        WHERE r.horse_id = ? ORDER BY ra.race_date""", con, params=[hid])
    ab = pd.read_sql("""
        SELECT race_date, power_top, good_avg, shunpatsu, jizoku, toppspeed,
               power, stamina, start, stamina_show, off_show, runs_prior, good_prior
        FROM v_ability WHERE horse_id = ? ORDER BY race_date DESC LIMIT 1""",
        con, params=[hid])
    con.close()

    fin = runs[runs["fin"].notna()]
    latest_r = runs["r_before"].dropna().iloc[-1] if runs["r_before"].notna().any() else None
    pmax = runs["perf"].dropna().max() if runs["perf"].notna().any() else None
    pstd = runs["pstd"].dropna().iloc[-1] if runs["pstd"].notna().any() else None

    print(f"\n━━━ {hname} ━━━")
    n = len(fin)
    w = int((fin["fin"] == 1).sum()); p3 = int((fin["fin"] <= 3).sum())
    print(f"通算: {n}戦{w}勝 複勝圏{p3}回（複勝率 {p3/n*100:.0f}%）" if n else "出走データなし")
    mura = "対象(条件補正あり)" if (pstd is not None and pstd >= 14) else "対象外(堅実型)"
    print(f"相対R(現在): {_f(latest_r)} ｜ 最高perf(天井): {_f(pmax)} ｜ "
          f"ムラ度pstd: {_f(pstd)} → ムラ馬{mura}")

    if not ab.empty:
        a = ab.iloc[0]
        print(f"\n〔能力プロファイル〕（{a['race_date']}時点 / 出走{int(a['runs_prior'])}・好走{int(a['good_prior'] or 0)}）")
        print(f"  総合(power_top) {_f(a['power_top'])} ｜ 好走平均 {_f(a['good_avg'])}")
        print(f"  瞬発(上3F) {_f(a['shunpatsu'],'4.1f')}(小=良) ｜ 持続 {_f(a['jizoku'])} ｜ "
              f"トップ {_f(a['toppspeed'])} ｜ パワー {_f(a['power'])}")
        print(f"  スタミナ {_f(a['stamina'])}(複勝率{_f(a['stamina_show'],'4.2f')}) ｜ "
              f"先行(隊列) {_f(a['start'],'4.2f')}(小=前) ｜ 道悪複勝率 {_f(a['off_show'],'4.2f')}")

    if n:
        print("\n〔競馬場×芝ダ別〕      出走  勝  複勝率   平均perf  最高perf")
        g = fin.assign(v=fin["venue_id"].map(lambda x: VENUE.get(str(x).zfill(2), x)))
        for (v, s), gg in g.groupby(["v", "surface"], sort=False):
            print(f"  {v}{s:<2}            {len(gg):>4}{int((gg['fin']==1).sum()):>4}"
                  f"{(gg['fin']<=3).mean()*100:>6.0f}%  {_f(gg['perf'].mean(),'8.1f')}"
                  f"  {_f(gg['perf'].max(),'8.1f')}")
        print("\n〔距離帯別〕")
        for lo, hi, lab in BAND:
            gg = fin[(fin["distance"] > lo) & (fin["distance"] <= hi)]
            if len(gg):
                print(f"  {lab:<16} {len(gg):>3}戦{int((gg['fin']==1).sum())}勝 "
                      f"複勝率{(gg['fin']<=3).mean()*100:>4.0f}%  平均perf {_f(gg['perf'].mean())}")
        off = fin[fin["track_condition"].notna() & (fin["track_condition"] != "良")]
        if len(off):
            print(f"〔道悪〕 {len(off)}戦{int((off['fin']==1).sum())}勝 "
                  f"複勝率{(off['fin']<=3).mean()*100:.0f}%  平均perf {_f(off['perf'].mean())}")

        print("\n〔近走〕")
        for _, r in runs.tail(10).iloc[::-1].iterrows():
            v = VENUE.get(str(r["venue_id"]).zfill(2), r["venue_id"])
            finlab = f"{int(r['fin']):>2}着" if pd.notna(r["fin"]) else " 出走予定"
            print(f"  {r['race_date']} {str(r['race_name'])[:14]:<14} "
                  f"{v}{r['surface']}{int(r['distance'])} {r['track_condition'] or '-':<2} "
                  f"{finlab}  perf{_f(r['perf'],'6.1f')}  R{_f(r['r_before'],'6.1f')}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="馬プロファイル検索")
    ap.add_argument("--db", default="keiba.db")
    ap.add_argument("name", help="馬名（部分一致可）")
    args = ap.parse_args(argv)
    return profile(args.db, args.name)


if __name__ == "__main__":
    raise SystemExit(main())
