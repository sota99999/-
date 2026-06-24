#!/usr/bin/env python3
# =============================================================================
# 指定年のG1（または任意グレード）を一括予想し、結果が出ていれば答え合わせする。
#
# 各レースについて、card.py と同じ採点で 印(◎○▲△)・〔穴〕(無印高天井)・
# 〔市場〕本命 を出し、着順が確定していれば 1-3着 と的中可否を表示する。
# 末尾に ◎/○/▲ の的中率・単回収率の集計も出す。
#
# 使い方:
#   python scripts/g1_report.py --db keiba.db \
#       --model model_win_cond.pkl --show-model model_show_cond.pkl --year 2026
# =============================================================================
from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import pandas as pd

import mlcommon
import card

VEN = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
       "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}

# JRA平地G1のレース名キーワード（grade列が空でも race_name で判定するため）
G1_KEYWORDS = [
    "フェブラリー", "高松宮記念", "大阪杯", "桜花賞", "皐月賞", "天皇賞",
    "ＮＨＫマイル", "NHKマイル", "ヴィクトリアマイル", "オークス", "優駿牝馬",
    "東京優駿", "安田記念", "宝塚記念", "スプリンターズ", "秋華賞",
    "菊花賞", "エリザベス女王杯", "マイルチャンピオンシップ", "ジャパンカップ",
    "ジャパンＣ", "チャンピオンズ", "阪神ジュベナイル", "朝日杯", "有馬記念", "ホープフル",
]


def _nm(r):
    return str(r.get("horse_name") or "")[:8]


def _fin(r):
    f = r.get("finish")
    return f"{int(f)}着" if pd.notna(f) else "－"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="年内G1の一括予想＋結果照合")
    p.add_argument("--db", default="keiba.db")
    p.add_argument("--model", default="model_win_cond.pkl")
    p.add_argument("--show-model", default="model_show_cond.pkl")
    p.add_argument("--year", default="2026")
    p.add_argument("--grade", default="G1")
    p.add_argument("--mark-max-odds", type=float, default=20.0)
    p.add_argument("--sub-max-odds", type=float, default=50.0)
    p.add_argument("--mark-min-runs", type=int, default=2)
    args = p.parse_args(argv)

    bundle = mlcommon.load_model(args.model)
    show_bundle = mlcommon.load_model(args.show_model)
    df = mlcommon.load_data(args.db, None)

    conn = sqlite3.connect(args.db)
    all_races = pd.read_sql_query(
        "SELECT race_id, race_date, race_name, venue_id, surface, distance, grade FROM races",
        conn)
    actual = pd.read_sql_query(
        "SELECT race_id, horse_id, finish_position AS finish FROM results "
        "WHERE finish_position IS NOT NULL", conn)
    place_pay = pd.read_sql_query(
        "SELECT race_id, combination, payout FROM payouts WHERE bet_type = '複勝'", conn)
    conn.close()
    place_pay["hn"] = pd.to_numeric(place_pay["combination"], errors="coerce")
    pmap = {(r, h): p for r, h, p in
            zip(place_pay["race_id"], place_pay["hn"], place_pay["payout"])}

    def _norm(gr):  # GⅠ/GⅡ/GⅢ → G1/G2/G3 に正規化
        if not gr:
            return None
        return gr.replace("Ⅰ", "1").replace("Ⅱ", "2").replace("Ⅲ", "3")

    all_races["ng"] = all_races["grade"].map(_norm)
    target = _norm(args.grade)
    # レース名→グレード（グレードが入っている全レースから多数決）
    graded = all_races[all_races["ng"].notna()]
    name2grade = (graded.groupby("race_name")["ng"].agg(lambda s: s.value_counts().idxmax()).to_dict()
                  if len(graded) else {})
    yr = all_races[all_races["race_date"].str.startswith(args.year)].copy()
    # 当該レース自身のグレード→無ければ過去同名から推定
    yr["g2"] = [ng if ng else name2grade.get(nm)
                for ng, nm in zip(yr["ng"], yr["race_name"])]
    races = yr[yr["g2"] == target].sort_values("race_date")
    # それでも0件かつG1なら、レース名キーワードでG1を救済
    if races.empty and target == "G1":
        pat = "|".join(G1_KEYWORDS)
        races = yr[yr["race_name"].fillna("").str.contains(pat, regex=True)].sort_values("race_date")
        print("（grade情報が無いため、レース名でG1を判定）")
    if races.empty:
        ng_n = int(all_races["ng"].notna().sum())
        print(f"{args.year}年の{args.grade}がDBに見つかりません。"
              f"（グレード入りレース数={ng_n}。0ならグレード未取得で、"
              f"G2/G3判定にはグレードの再取得が必要です）")
        return 0
    print(f"（{args.grade}={target} を {len(races)}レース抽出）")

    sub = df[df["race_id"].isin(races["race_id"])].copy()
    if sub.empty:
        print("対象レースの出走表データがありません。")
        return 0
    sub = card.compute_scores(sub, bundle, show_bundle, {})
    sub = sub.merge(actual, on=["race_id", "horse_id"], how="left")
    sub["odds"] = pd.to_numeric(sub.get("odds"), errors="coerce")
    # 複勝の確定払戻を (race_id, 馬番) で付与（3着以内のみ値が入る）
    if "horse_number" in sub.columns:
        hn = pd.to_numeric(sub["horse_number"], errors="coerce")
        sub["place_payout"] = [pmap.get((r, h)) for r, h in zip(sub["race_id"], hn)]
    else:
        sub["place_payout"] = None
    # 〔穴〕用に最高SPのレース内z
    g = sub.groupby("race_id")["best_speed_prior"]
    sd = g.transform("std").replace(0, np.nan)
    sub["bs_z"] = ((sub["best_speed_prior"] - g.transform("mean")) / sd).fillna(0.0)

    print(f"\n========== {args.year}年 {args.grade} 一括予想＋結果 ==========")
    bets = []  # (mark, finish, odds) for aggregate
    n_done = 0
    for _, ra in races.iterrows():
        gg = sub[sub["race_id"] == ra["race_id"]].copy()
        head = (f"\n[{ra['race_date']}] {ra['race_name']} "
                f"（{VEN.get(ra['venue_id'], ra['venue_id'])}{ra['surface']}{int(ra['distance'])}m）")
        if gg.empty:
            print(head + "  ※出走表データなし")
            continue
        vm = gg["odds"].notna().any()
        gg = card.assign_marks(gg, value_mode=vm, strength_col="show_p",
                               min_runs=args.mark_min_runs,
                               max_odds=args.mark_max_odds, sub_max_odds=args.sub_max_odds)
        print(head)
        # 印（◎○▲△）と着順
        picks = []
        for mk in ["◎", "○", "▲", "△"]:
            for _, r in gg[gg["mark"] == mk].iterrows():
                picks.append(f"{mk}{_nm(r)}({_fin(r)})")
                bets.append((mk, r.get("finish"), r.get("odds"), r.get("place_payout")))
        if picks:
            print("  予想: " + " ".join(picks))
        # 〔穴〕無印・高天井
        ana = [f"{_nm(r)}({_fin(r)})" for _, r in
               gg[(gg["mark"] == "") & (gg["bs_z"] >= 1.0)].iterrows()][:3]
        if ana:
            print("  〔穴〕 " + " ".join(ana))
        # 〔市場〕オッズ最上位
        favs = gg[gg["odds"].notna()].nsmallest(2, "odds")
        mk = [f"{_nm(r)}({card._f(r['odds'], '.1f')}倍/{_fin(r)})" for _, r in favs.iterrows()]
        if mk:
            print("  〔市場〕 " + " ".join(mk))
        # 結果 1-3着
        res = gg[gg["finish"].notna()].nsmallest(3, "finish")
        if len(res):
            n_done += 1
            order = " ".join(f"{int(r['finish'])}.{_nm(r)}" for _, r in res.iterrows())
            print("  結果: " + order)

    # 集計
    bd = pd.DataFrame(bets, columns=["mark", "finish", "odds", "ppay"])
    bd = bd[bd["finish"].notna()]
    if len(bd):
        print(f"\n---------- 集計（結果確定 {n_done} レース） ----------")
        print(f"{'印':<3}{'本数':>5}{'単勝的中':>9}{'複勝的中':>9}{'単回収率':>9}{'複回収率':>9}")
        for mk in ["◎", "○", "▲", "△"]:
            s = bd[bd["mark"] == mk]
            if not len(s):
                continue
            win = (s["finish"] == 1)
            plc = (s["finish"] <= 3)
            roi = s.loc[win, "odds"].fillna(0).sum() * 100 / (100 * len(s)) * 100
            proi = pd.to_numeric(s["ppay"], errors="coerce").fillna(0).sum() / (100 * len(s)) * 100
            print(f"{mk:<3}{len(s):>5}{win.mean()*100:>8.1f}%{plc.mean()*100:>8.1f}%"
                  f"{roi:>8.1f}%{proi:>8.1f}%")
    print("\n※印は事前情報のみで採点（リークなし）。複回収は複勝の確定払戻ベース。"
          "秋以降のG1は出馬表公開後に予想可能。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
