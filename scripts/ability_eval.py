#!/usr/bin/env python3
# =============================================================================
# 能力2指標(最高SP・Elo)の印の回顧
#
# card._ability_marks と同じ印付け（各指標で ⭐◎○▲△、オッズ妙味に ✅）を
# 結果確定レースに適用し、印ごとに
#   単勝的中率 / 複勝的中率 / 単勝回収率 / 複勝回収率
# を集計する。最高SP・Elo それぞれの印別に出し、さらに
#   「両指標で◎/⭐が一致した馬（=二冠軸）」と「✅妙味馬」も別枠で出す。
#
# 使い方:
#   python scripts/ability_eval.py --db keiba.db --from 2026-06-27 --to 2026-06-28
#
# 注意: 単勝回収=最終単勝オッズ、複勝回収=複勝の確定払戻(payouts)。
#       払戻未取得のレースは0円扱い。各印を100円ずつ買った前提。
# =============================================================================
from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

import mlcommon
import card


def _summ(b: pd.DataFrame) -> dict:
    """的中率・回収率（単勝オッズと複勝確定払戻から）。"""
    n = len(b)
    if n == 0:
        return {"n": 0}
    win = b["finish"] == 1
    place = b["finish"] <= 3
    payout = pd.to_numeric(b.loc[win, "odds"], errors="coerce").fillna(0).sum() * 100
    pp = pd.to_numeric(b.get("place_payout"), errors="coerce").fillna(0)
    return {"n": n, "win": win.mean() * 100, "place": place.mean() * 100,
            "roi": payout / (100 * n) * 100, "proi": pp.sum() / (100 * n) * 100}


def _row(label: str, s: dict) -> str:
    if not s.get("n"):
        return f"{label:<10}{0:>5}      該当なし"
    return (f"{label:<10}{s['n']:>5}{s['win']:>8.1f}%{s['place']:>8.1f}%"
            f"{s['roi']:>8.1f}%{s['proi']:>8.1f}%")


# 指定の9組み合わせ（Elo側・SP側の印）。df から真偽マスクのリストを作る
def _combo_masks(df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    sp_m = df["best_speed_prior_mark"]; el_m = df["elo_before_mark"]
    sp_v = df["best_speed_prior_val"]; el_v = df["elo_before_val"]
    return [
        ("Elo⭐ SP⭐", (el_m == "⭐") & (sp_m == "⭐")),
        ("Elo⭐ SP◎", (el_m == "⭐") & (sp_m == "◎")),
        ("Elo◎ SP⭐", (el_m == "◎") & (sp_m == "⭐")),
        ("Elo◎ SP◎", (el_m == "◎") & (sp_m == "◎")),
        ("Elo✅ SP✅", (el_v == "✅") & (sp_v == "✅")),
        ("Elo⭐ SP✅", (el_m == "⭐") & (sp_v == "✅")),
        ("Elo✅ SP⭐", (el_v == "✅") & (sp_m == "⭐")),
        ("Elo◎ SP✅", (el_m == "◎") & (sp_v == "✅")),
        ("Elo✅ SP◎", (el_v == "✅") & (sp_m == "◎")),
    ]


HEAD = f"{'印':<10}{'本数':>5}{'単的中':>9}{'複的中':>9}{'単回収':>9}{'複回収':>9}"

VENUE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
         "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
CLASS_ORDER = ["新馬", "未勝利", "1勝", "2勝", "3勝", "OP/L", "重賞", "その他"]
ODDS_ORDER = ["〜2.0", "2.0〜5.0", "5.0〜10", "10〜20", "20〜50", "50〜"]


def _class_bucket(cl) -> str:
    cl = int(cl) if pd.notna(cl) else 0
    return {1: "新馬", 2: "未勝利", 3: "1勝", 4: "2勝", 5: "3勝",
            6: "OP/L"}.get(cl, "重賞" if cl >= 7 else "その他")


def _odds_bucket(o) -> str:
    o = pd.to_numeric(o, errors="coerce")
    if pd.isna(o):
        return ""
    for hi, lab in [(2.0, "〜2.0"), (5.0, "2.0〜5.0"), (10.0, "5.0〜10"),
                    (20.0, "10〜20"), (50.0, "20〜50")]:
        if o < hi:
            return lab
    return "50〜"


# --- コース地形の層別（v_course が必要）-------------------------------------
STRAIGHT_ORDER = ["〜300m", "300-400m", "400-500m", "500m〜"]
HILL_ORDER = ["急坂", "緩坂", "平坦"]
TURN_ORDER = ["小回り", "広い", "直線"]
_HILL_JP = {"steep": "急坂", "mild": "緩坂", "flat": "平坦"}
_TURN_JP2 = {"tight": "小回り", "wide": "広い", "none": "直線"}


def _straight_bucket(m) -> str:
    m = pd.to_numeric(m, errors="coerce")
    if pd.isna(m):
        return ""
    if m < 300:
        return "〜300m"
    if m < 400:
        return "300-400m"
    if m < 500:
        return "400-500m"
    return "500m〜"


# --- 世代別・条件別（レース名＋class_levelから判定）--------------------------
AGE_ORDER = ["2歳", "3歳", "3歳上", "4歳上", "不明"]
_CLASS_SHORT = {1: "新馬", 2: "未勝利", 3: "1勝", 4: "2勝", 5: "3勝",
                6: "OP・L", 7: "G3", 8: "G2", 9: "G1"}
_AGE_PRE = {"2歳": "2歳", "3歳": "3歳", "3歳上": "3上", "4歳上": "4上"}
COND_ORDER = ["2歳新馬", "2歳未勝利", "3歳新馬", "3歳未勝利",
              "3歳1勝", "3歳2勝", "3歳OP・L",
              "3上未勝利", "3上1勝", "3上2勝", "3上3勝", "3上OP・L",
              "4上1勝", "4上2勝", "4上3勝", "4上OP・L",
              "1勝", "2勝", "3勝", "OP・L", "G3", "G2", "G1", "その他"]


def _age_bucket(name) -> str:
    n = str(name or "")
    if "２歳" in n or "2歳" in n:
        return "2歳"
    if "４歳以上" in n or "4歳以上" in n or "４歳上" in n:
        return "4歳上"
    if "３歳以上" in n or "3歳以上" in n or "３歳上" in n:
        return "3歳上"
    if "３歳" in n or "3歳" in n:
        return "3歳"
    return "不明"   # 特別・重賞は名称に年齢が無いことが多い


def _cond_bucket(name, cl) -> str:
    cls = _CLASS_SHORT.get(int(cl) if pd.notna(cl) else 0, "その他")
    apre = _AGE_PRE.get(_age_bucket(name), "")
    # 年齢が取れない特別/OP/重賞はクラスのみ（重賞・OPは年齢不明が普通）
    if apre == "":
        return cls
    return f"{apre}{cls}"


def _load_vcourse(db: str) -> pd.DataFrame:
    """v_course（race_id→地形）を DataFrame で返す。無ければ空。"""
    conn = sqlite3.connect(db)
    try:
        return pd.read_sql_query(
            "SELECT race_id, straight_m, hill_grade, turn_size FROM v_course", conn)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=["race_id", "straight_m", "hill_grade", "turn_size"])
    finally:
        conn.close()


def _print_combos(df: pd.DataFrame, title: str) -> None:
    nr = df["race_id"].nunique()
    print(f"\n【{title}】 {nr}レース / 印付き{len(df)}頭")
    print(HEAD)
    for label, mask in _combo_masks(df):
        print(_row(label, _summ(df[mask])))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="能力2指標の印の回顧")
    p.add_argument("--db", default="keiba.db")
    p.add_argument("--from", dest="date_from")
    p.add_argument("--to", dest="date_to")
    p.add_argument("--by", help="層別バックテスト。class/venue/odds/straight/hill/turn を"
                                 "カンマ区切りで指定（例 --by straight,hill）。各層で9組み合わせを出す。"
                                 "straight/hill/turn は course_master(v_course) が必要")
    args = p.parse_args(argv)

    df = mlcommon.load_data(args.db, None)
    conn = sqlite3.connect(args.db)
    q = ["""SELECT race_id FROM races WHERE race_id IN
            (SELECT race_id FROM results GROUP BY race_id
             HAVING COUNT(finish_position) > 0)"""]
    params: list = []
    if args.date_from:
        q.append("AND race_date >= ?"); params.append(args.date_from)
    if args.date_to:
        q.append("AND race_date <= ?"); params.append(args.date_to)
    ids = [r[0] for r in conn.execute(" ".join(q), params)]
    actual = pd.read_sql_query(
        "SELECT race_id, horse_id, finish_position AS finish FROM results "
        "WHERE finish_position IS NOT NULL", conn)
    place_pay = pd.read_sql_query(
        "SELECT race_id, combination, payout FROM payouts WHERE bet_type = '複勝'", conn)
    conn.close()
    place_pay["hn"] = pd.to_numeric(place_pay["combination"], errors="coerce")
    pmap = {(r, h): p for r, h, p in
            zip(place_pay["race_id"], place_pay["hn"], place_pay["payout"])}

    sub = df[df["race_id"].isin(ids)].copy()
    if sub.empty:
        print("対象レースがありません（結果確定・期間指定を確認）"); return 0
    sub = sub.merge(actual, on=["race_id", "horse_id"], how="left")
    sub["odds"] = pd.to_numeric(sub.get("odds"), errors="coerce")
    hn = pd.to_numeric(sub["horse_number"], errors="coerce")
    sub["place_payout"] = [pmap.get((r, h)) for r, h in zip(sub["race_id"], hn)]

    marked = [card._ability_marks(g) for _, g in sub.groupby("race_id")]
    sub = pd.concat(marked, ignore_index=True)
    ran = sub[sub["finish"].notna()].copy()

    n_races = sub["race_id"].nunique()
    # 複勝回収の信頼度: 3着内の馬のうち複勝払戻が取れている割合（低いと複回収は過小評価）
    placed = ran[ran["finish"] <= 3]
    cov = (pd.to_numeric(placed.get("place_payout"), errors="coerce").notna().mean()
           if len(placed) else 0.0)
    dmin = sub["race_date"].min(); dmax = sub["race_date"].max()
    print(f"\n=== 能力2指標の印別 回顧（{args.date_from or '最初'}〜{args.date_to or '最後'}） ===")
    print(f"対象レース数: {n_races}（結果確定 / 実日付 {dmin}〜{dmax}）")
    print(f"複勝払戻カバー率: {cov*100:.0f}%（低いと『複回収』は過小評価。単回収・的中率は影響なし）")

    if args.by:
        # 層別バックテスト: 指定の各次元で、バケットごとに9組み合わせを出す
        ran = ran.copy()
        ran["_class"] = ran["class_level"].map(_class_bucket)
        ran["_venue"] = ran["venue_id"].map(lambda v: VENUE.get(str(v).zfill(2), str(v)))
        ran["_odds"] = ran["odds"].map(_odds_bucket)
        # コース地形（v_course があれば結合）
        vc = _load_vcourse(args.db)
        if not vc.empty:
            ran = ran.merge(vc, on="race_id", how="left")
            ran["_straight"] = ran["straight_m"].map(_straight_bucket)
            ran["_hill"] = ran["hill_grade"].map(lambda g: _HILL_JP.get(g, ""))
            ran["_turn"] = ran["turn_size"].map(lambda t: _TURN_JP2.get(t, ""))
        else:
            ran["_straight"] = ran["_hill"] = ran["_turn"] = ""
        # 世代別・条件別（レース名＋class_level）
        conn2 = sqlite3.connect(args.db)
        rn = pd.read_sql_query("SELECT race_id, race_name FROM races", conn2)
        conn2.close()
        ran = ran.merge(rn, on="race_id", how="left")
        ran["_age"] = ran["race_name"].map(_age_bucket)
        ran["_cond"] = [_cond_bucket(n, c) for n, c in zip(ran["race_name"], ran["class_level"])]
        dims = {"class": ("クラス別", "_class", CLASS_ORDER),
                "venue": ("競馬場別", "_venue", [VENUE[k] for k in sorted(VENUE)]),
                "odds": ("オッズ帯別", "_odds", ODDS_ORDER),
                "straight": ("直線長別", "_straight", STRAIGHT_ORDER),
                "hill": ("ゴール前坂別", "_hill", HILL_ORDER),
                "turn": ("小回り/広い別", "_turn", TURN_ORDER),
                "age": ("世代別", "_age", AGE_ORDER),
                "cond": ("条件別", "_cond", COND_ORDER)}
        for key in [d.strip() for d in args.by.split(",")]:
            if key not in dims:
                print(f"\n[skip] 未知の層別キー: {key}"
                      "（class/venue/odds/straight/hill/turn）"); continue
            title, colname, order = dims[key]
            print(f"\n========== {title} ==========")
            present = [b for b in order if b in set(ran[colname])]
            present += [b for b in ran[colname].dropna().unique()
                        if b and b not in present]   # 想定外バケットも拾う
            for bucket in present:
                if not bucket:
                    continue
                _print_combos(ran[ran[colname] == bucket], f"{title}: {bucket}")
        print("\n※各層で9組み合わせ。単回収=単勝オッズ、複回収=複勝確定払戻、100%超で利益。"
              "本数が少ない層は数字が振れるので n を見て判断。")
        return 0

    for col, label, _ in card.ABILITY_COLS:
        print(f"\n― {label} の印 ―")
        print(HEAD)
        for mk in ["⭐", "◎", "○", "▲", "△"]:
            print(_row(mk, _summ(ran[ran[col + "_mark"] == mk])))

    print("\n― 2指標の組み合わせ ―")
    print(HEAD)
    for label, mask in _combo_masks(ran):
        print(_row(label, _summ(ran[mask])))

    print("\n※単回収=最終単勝オッズ、複回収=複勝確定払戻。各印を100円ずつ買った前提。"
          "払戻未取得レースは0円扱い。100%超で利益。✅は指標ごと。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
