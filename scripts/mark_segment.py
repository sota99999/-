#!/usr/bin/env python3
# =============================================================================
# 指定した印（既定 ▲）の成績を「レース質別」に分解する。
#   ▲の高回収はどのレース質で稼いでいるかを特定し、「▲を買うべきレース」を絞る。
#   クラス（未勝利・新馬／1〜3勝／OP・特別／重賞）と 出走頭数 で層別に集計。
#   OOS（学習期間外, 既定 2026-03-07 以降）で評価する。
#
# 使い方:
#   python scripts/mark_segment.py --db keiba.db --mark ▲ --from 2026-03-07
# =============================================================================
from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

import mlcommon
import card


def race_class(name, grade) -> str:
    g = (grade or "").replace("Ⅰ", "1").replace("Ⅱ", "2").replace("Ⅲ", "3")
    if g in ("G1", "G2", "G3"):
        return "重賞"
    n = name or ""
    if "新馬" in n or "未勝利" in n:
        return "未勝利・新馬"
    if any(k in n for k in ["3勝", "1600万"]):
        return "3勝クラス"
    if any(k in n for k in ["2勝", "1000万"]):
        return "2勝クラス"
    if any(k in n for k in ["1勝", "500万"]):
        return "1勝クラス"
    return "OP・特別"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="印のレース質別 成績分解")
    p.add_argument("--db", default="keiba.db")
    p.add_argument("--model", default="model_win_cond.pkl")
    p.add_argument("--show-model", default="model_show_cond.pkl")
    p.add_argument("--from", dest="date_from", default="2026-03-07")
    p.add_argument("--to", dest="date_to")
    p.add_argument("--mark", default="▲", help="分析する印（▲/◎/○/△）")
    p.add_argument("--mark-max-odds", type=float, default=20.0)
    p.add_argument("--sub-max-odds", type=float, default=30.0)
    p.add_argument("--mark-min-runs", type=int, default=2)
    args = p.parse_args(argv)

    bundle = mlcommon.load_model(args.model)
    show_bundle = mlcommon.load_model(args.show_model)
    df = mlcommon.load_data(args.db, None)

    conn = sqlite3.connect(args.db)
    races = pd.read_sql_query(
        "SELECT race_id, race_date, race_name, grade, field_size FROM races", conn)
    actual = pd.read_sql_query(
        "SELECT race_id, horse_id, finish_position AS finish FROM results "
        "WHERE finish_position IS NOT NULL", conn)
    ppay = pd.read_sql_query(
        "SELECT race_id, combination, payout FROM payouts WHERE bet_type='複勝'", conn)
    conn.close()
    ppay["hn"] = pd.to_numeric(ppay["combination"], errors="coerce")
    pmap = {(r, h): p for r, h, p in zip(ppay["race_id"], ppay["hn"], ppay["payout"])}

    rr = races[races["race_id"].isin(set(actual["race_id"]))].copy()
    if args.date_from:
        rr = rr[rr["race_date"] >= args.date_from]
    if args.date_to:
        rr = rr[rr["race_date"] <= args.date_to]

    sub = df[df["race_id"].isin(rr["race_id"])].copy()
    if sub.empty:
        print("対象レースがありません。")
        return 0
    sub = card.compute_scores(sub, bundle, show_bundle, {})
    sub = sub.merge(actual, on=["race_id", "horse_id"], how="left")
    sub["odds"] = pd.to_numeric(sub.get("odds"), errors="coerce")
    if "horse_number" in sub.columns:
        hn = pd.to_numeric(sub["horse_number"], errors="coerce")
        sub["ppay"] = [pmap.get((r, h)) for r, h in zip(sub["race_id"], hn)]
    else:
        sub["ppay"] = None

    marked = []
    for _, gg in sub.groupby("race_id"):
        vm = pd.to_numeric(gg.get("odds"), errors="coerce").notna().any()
        marked.append(card.assign_marks(gg, value_mode=vm, strength_col="show_p",
                                        min_runs=args.mark_min_runs,
                                        max_odds=args.mark_max_odds,
                                        sub_max_odds=args.sub_max_odds))
    sub = pd.concat(marked, ignore_index=True)

    cmap = {r: race_class(n, g) for r, n, g in
            zip(rr["race_id"], rr["race_name"], rr["grade"])}
    fmap = {r: f for r, f in zip(rr["race_id"], rr["field_size"])}
    sub["cls"] = sub["race_id"].map(cmap)
    sub["fs"] = pd.to_numeric(sub["race_id"].map(fmap), errors="coerce")

    VEN = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
           "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}

    # --mark ALL: ◎○▲△を横並び（各セル=単回収/複回収%）で1表にまとめる
    if args.mark == "ALL":
        base = sub[sub["finish"].notna()].copy()
        dist = pd.to_numeric(base.get("distance"), errors="coerce")
        tc = base.get("track_condition")
        dims = [
            ("クラス", [(c, base["cls"] == c) for c in
                      ["未勝利・新馬", "1勝クラス", "2勝クラス", "3勝クラス", "OP・特別", "重賞"]]),
            ("馬場種別", [(s, base.get("surface") == s) for s in ["芝", "ダート"]]),
            ("距離帯", [("短 ~1300", dist < 1400), ("マイル 14-17", (dist >= 1400) & (dist < 1800)),
                     ("中 18-21", (dist >= 1800) & (dist < 2200)), ("長 2200~", dist >= 2200)]),
            ("馬場状態", [("良", tc == "良"), ("道悪", tc.isin(["稍重", "重", "不良"]))]),
            ("競馬場", [(vn, base.get("venue_id") == vid) for vid, vn in VEN.items()]),
            ("頭数", [("≤12", base["fs"] <= 12), ("13-15", (base["fs"] >= 13) & (base["fs"] <= 15)),
                    ("≥16", base["fs"] >= 16)]),
        ]

        def cell(seg, mk):
            s = seg[seg["mark"] == mk]
            n = len(s)
            if not n:
                return "    -    "
            roi = s.loc[s["finish"] == 1, "odds"].fillna(0).sum() / n * 100
            proi = pd.to_numeric(s["ppay"], errors="coerce").fillna(0).sum() / (100 * n) * 100
            return f"{roi:>3.0f}/{proi:<3.0f}({n})"

        def line(label, seg):
            return (f"{label:<13}{cell(seg, '◎'):>13}{cell(seg, '○'):>13}"
                    f"{cell(seg, '▲'):>13}{cell(seg, '△'):>13}")

        print(f"\n=== 全印 条件別マップ（{args.date_from}〜・各セル=単回収/複回収%(本数)）===")
        head = f"{'区分':<13}{'◎':>13}{'○':>13}{'▲':>13}{'△':>13}"
        for title, segs in dims:
            print(f"\n[{title}]"); print(head)
            for label, mask in segs:
                seg = base[mask]
                if len(seg):
                    print(line(label, seg))
        print("\n[全体]"); print(head); print(line("全レース", base))
        print("\n※各セル=単勝回収率/複勝回収率%(本数)。◎は複勝、▲は単勝で見るのが基本。"
              "100超で利益。OOS（学習外）評価。")
        return 0

    m = sub[(sub["mark"] == args.mark) & sub["finish"].notna()].copy()

    def summ(s):
        n = len(s)
        if not n:
            return None
        win = (s["finish"] == 1)
        plc = (s["finish"] <= 3)
        roi = s.loc[win, "odds"].fillna(0).sum() * 100 / (100 * n) * 100
        proi = pd.to_numeric(s["ppay"], errors="coerce").fillna(0).sum() / (100 * n) * 100
        return n, win.mean() * 100, plc.mean() * 100, roi, proi

    print(f"\n=== 印「{args.mark}」のレース質別 成績"
          f"（{args.date_from}〜{args.date_to or '最後'}・全{len(m)}本）===")
    hdr = f"{'区分':<14}{'本数':>5}{'単的中':>7}{'複的中':>7}{'単回収':>8}{'複回収':>8}"

    def row(label, s):
        r = summ(s)
        if r:
            print(f"{label:<14}{r[0]:>5}{r[1]:>6.1f}%{r[2]:>6.1f}%{r[3]:>7.1f}%{r[4]:>7.1f}%")

    print("\n[クラス別]"); print(hdr)
    for cls in ["未勝利・新馬", "1勝クラス", "2勝クラス", "3勝クラス", "OP・特別", "重賞"]:
        row(cls, m[m["cls"] == cls])

    print("\n[馬場種別]"); print(hdr)
    for sf in ["芝", "ダート", "障害"]:
        row(sf, m[m.get("surface") == sf])

    print("\n[距離帯]"); print(hdr)
    dist = pd.to_numeric(m.get("distance"), errors="coerce")
    row("短距離 ~1300", m[dist < 1400])
    row("マイル 1400-1700", m[(dist >= 1400) & (dist < 1800)])
    row("中距離 1800-2100", m[(dist >= 1800) & (dist < 2200)])
    row("長距離 2200~", m[dist >= 2200])

    print("\n[馬場状態]"); print(hdr)
    tc = m.get("track_condition")
    row("良", m[tc == "良"])
    row("道悪(稍重/重/不良)", m[tc.isin(["稍重", "重", "不良"])])

    print("\n[競馬場]"); print(hdr)
    VEN = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
           "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
    vc = m.get("venue_id")
    for vid, vn in VEN.items():
        row(vn, m[vc == vid])

    print("\n[出走頭数別]"); print(hdr)
    row("少頭数 ≤12", m[m["fs"] <= 12])
    row("中 13-15", m[(m["fs"] >= 13) & (m["fs"] <= 15)])
    row("多頭数 ≥16", m[m["fs"] >= 16])
    print("\n[全体]"); print(hdr); row("全レース", m)
    print("\n※単回収100%超で利益。▲は買うレースを絞るほど効く想定。OOS（学習外）評価。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
