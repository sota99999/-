#!/usr/bin/env python3
# =============================================================================
# 予想の的中率・回収率を集計する（バックテスト/振り返り）
#
# card.py と全く同じ採点（compute_scores）で各レースの印(◎○▲△)を再現し、
# 実際の着順と突き合わせて、印ごとの単勝的中率・複勝的中率・単勝回収率を出す。
# 結果が確定したレースだけを対象にする（finish_position が入っているレース）。
#
# ※ 特徴量はすべてレース前情報（elo_before 等は当該レース直前値）なので、
#    結果を取り込んだ後に集計しても予想内容は変わらない（リークしない）。
#
# 使い方:
#   python scripts/evaluate.py --db keiba.db \
#       --model model_win_noodds.pkl --show-model model_show_noodds.pkl
#   # 期間を絞る
#   python scripts/evaluate.py --db keiba.db --from 2026-06-20 --to 2026-06-21 \
#       --model model_win_noodds.pkl --show-model model_show_noodds.pkl
#
# 注意: 回収率は単勝のみ（複勝・馬連等の払戻データは未取得のため複勝は的中率のみ）。
#       オッズは結果ページの最終オッズを使用（実際の購入時とは多少ずれ得る）。
# =============================================================================
from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import pandas as pd

import mlcommon
import card

MARKS = card.MARKS   # ["◎","○","▲","△","△","×"]


def _resulted_race_ids(conn, date_from=None, date_to=None) -> list[str]:
    """結果が確定した（finish_position が1件以上ある）レースID。期間で絞れる。"""
    q = ["""SELECT ra.race_id FROM races ra
            WHERE ra.race_id IN (SELECT race_id FROM results
                                 GROUP BY race_id HAVING COUNT(finish_position) > 0)"""]
    params: list = []
    if date_from:
        q.append("AND ra.race_date >= ?"); params.append(date_from)
    if date_to:
        q.append("AND ra.race_date <= ?"); params.append(date_to)
    q.append("ORDER BY ra.race_id")
    return [r[0] for r in conn.execute(" ".join(q), params)]


def _summ(bets: pd.DataFrame) -> dict:
    """賭け対象の DataFrame から的中率・回収率を計算。

    単勝回収率は odds 列（最終単勝オッズ）から、複勝回収率は place_payout 列
    （複勝の確定払戻金, 100円あたり。3着以内の馬のみ値が入る）から算出する。
    """
    n = len(bets)
    if n == 0:
        return {"n": 0, "win": 0.0, "place": 0.0, "roi": 0.0, "proi": 0.0}
    win = bets["finish"] == 1
    place = bets["finish"] <= 3
    payout = bets.loc[win, "odds"].fillna(0).sum() * 100   # 単勝: 100円賭けの払戻合計
    # 複勝: 確定払戻(円/100円)を直接合計（3着外は払戻0）
    pp = pd.to_numeric(bets.get("place_payout"), errors="coerce").fillna(0)
    return {
        "n": n,
        "win": win.mean() * 100,
        "place": place.mean() * 100,
        "roi": payout / (100 * n) * 100,
        "proi": pp.sum() / (100 * n) * 100,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="予想の的中率・回収率を集計")
    p.add_argument("--db", default="keiba.db")
    p.add_argument("--model", default="model_win_noodds.pkl")
    p.add_argument("--show-model", help="複勝モデル（印を複勝率で付ける場合）")
    p.add_argument("--from", dest="date_from", help="開始日 (YYYY-MM-DD)")
    p.add_argument("--to", dest="date_to", help="終了日 (YYYY-MM-DD)")
    p.add_argument("--mark-by", choices=["show", "win"], default="show",
                   help="全頭診断(オッズ無)の強さ順の基準: show=複勝率(既定), win=勝率")
    p.add_argument("--mark-min-runs", type=int, default=3,
                   help="最終結論で印(妙味馬)を付ける最低出走回数")
    p.add_argument("--mark-max-odds", type=float, default=20.0,
                   help="最終結論で印(妙味馬)を付ける単勝オッズ上限")
    p.add_argument("--hon-mode", choices=["strong", "value"], default="strong",
                   help="◎の選び方: strong=強い馬(人気すぎ除外/既定), value=妙味(穴)")
    p.add_argument("--hon-min-odds", type=float, default=3.0,
                   help="◎(strong時)の単勝オッズ下限。人気を背負いすぎた本命を除外")
    p.add_argument("--weights", help="能力重視リウェイトの比率（card と同じ書式）")
    p.add_argument("--lean", type=float, default=0.0,
                   help="リウェイトの強さ倍率。既定0=較正優先（card と揃える）")
    p.add_argument("--min-ev", type=float, default=1.0,
                   help="EV戦略でこの値以上のEVの馬を単勝買いした場合の成績を出す")
    p.add_argument("--min-runs", type=int, default=3,
                   help="EV戦略で対象にする最低出走回数（新馬・能力未知馬を除外）")
    p.add_argument("--max-odds", type=float, default=30.0,
                   help="EV戦略で対象にする単勝オッズ上限（大穴の過大評価を除外）")
    args = p.parse_args(argv)

    weights = card.parse_weights(args.weights)
    weights = {k: v * args.lean for k, v in weights.items()}

    bundle = mlcommon.load_model(args.model)
    show_bundle = mlcommon.load_model(args.show_model) if args.show_model else None
    df = mlcommon.load_data(args.db, None)

    conn = sqlite3.connect(args.db)
    ids = _resulted_race_ids(conn, args.date_from, args.date_to)
    actual = pd.read_sql_query(
        "SELECT race_id, horse_id, finish_position AS finish FROM results "
        "WHERE finish_position IS NOT NULL", conn)
    # 複勝の確定払戻（combination=馬番, payout=円/100円）を (race_id, 馬番) で引けるように
    place_pay = pd.read_sql_query(
        "SELECT race_id, combination, payout FROM payouts WHERE bet_type = '複勝'", conn)
    conn.close()
    place_pay["hn"] = pd.to_numeric(place_pay["combination"], errors="coerce")
    pmap = {(r, h): p for r, h, p in
            zip(place_pay["race_id"], place_pay["hn"], place_pay["payout"])}

    sub = df[df["race_id"].isin(ids)].copy()
    if sub.empty:
        print("対象レースがありません（結果が取り込まれていますか? 期間指定は合っていますか?）")
        return 0

    # card と同一の採点
    sub = card.compute_scores(sub, bundle, show_bundle, weights)
    strength_col = "show_p" if (args.mark_by == "show" and show_bundle) else "p"

    # 実着順を結合（出走取消などで着順が無い馬は NaN）
    sub = sub.merge(actual, on=["race_id", "horse_id"], how="left")
    sub["odds"] = pd.to_numeric(sub.get("odds"), errors="coerce")
    # 複勝払戻を (race_id, 馬番) で引いて付与（3着以内のみ値が入る）
    if "horse_number" in sub.columns:
        hn = pd.to_numeric(sub["horse_number"], errors="coerce")
        sub["place_payout"] = [pmap.get((r, h)) for r, h in zip(sub["race_id"], hn)]
    else:
        sub["place_payout"] = None

    # card と同一の印付け（オッズがあれば妙味ベース、無ければ強さ順）
    marked = []
    for _, g in sub.groupby("race_id"):
        vm = pd.to_numeric(g.get("odds"), errors="coerce").notna().any()
        marked.append(card.assign_marks(g, value_mode=vm, strength_col=strength_col,
                                        min_runs=args.mark_min_runs,
                                        max_odds=args.mark_max_odds,
                                        hon_mode=args.hon_mode,
                                        hon_min_odds=args.hon_min_odds))
    sub = pd.concat(marked, ignore_index=True)

    n_races = sub["race_id"].nunique()
    ran = sub[sub["finish"].notna()].copy()   # 実際に出走した馬だけ

    print(f"\n=== 予想の振り返り（{args.date_from or '最初'}〜{args.date_to or '最後'}） ===")
    rwlabel = f"リウェイト×{args.lean:g}" if args.lean else "較正優先(リウェイトなし)"
    honlabel = (f"◎=強い馬(オッズ{args.hon_min_odds:g}倍以上)"
                if args.hon_mode == "strong" else "◎=妙味(穴)")
    print(f"対象レース数: {n_races}　／　{honlabel}　／　{rwlabel}")
    print(f"{'印':<3}{'本数':>5}{'単勝的中':>9}{'複勝的中':>9}{'単回収率':>9}{'複回収率':>9}")
    for mk in ["◎", "○", "▲", "△"]:
        s = _summ(ran[ran["mark"] == mk])
        if s["n"]:
            print(f"{mk:<3}{s['n']:>5}{s['win']:>8.1f}%{s['place']:>8.1f}%"
                  f"{s['roi']:>8.1f}%{s['proi']:>8.1f}%")

    # 参考: 1番人気（市場の本命）の成績
    fav = ran[pd.to_numeric(ran.get("popularity"), errors="coerce") == 1]
    sf = _summ(fav)
    if sf["n"]:
        print(f"\n参考 1番人気: {sf['n']}本 単勝的中{sf['win']:.1f}% "
              f"複勝的中{sf['place']:.1f}% 単回収{sf['roi']:.1f}% 複回収{sf['proi']:.1f}%")

    # EV戦略: EV>=min-ev の馬を単勝で買う。ただし「出走歴が一定以上」かつ
    #   「オッズ上限以下」に絞る（EV>1の正体がElo初期値の新馬・大穴に偏るため）。
    runs = pd.to_numeric(ran.get("runs_prior"), errors="coerce").fillna(0)
    evmask = (pd.to_numeric(ran.get("ev"), errors="coerce") >= args.min_ev) \
        & (runs >= args.min_runs) & (ran["odds"] <= args.max_odds)
    se = _summ(ran[evmask])
    print(f"EV≧{args.min_ev:g} 単勝買い（{args.min_runs}走以上・{args.max_odds:g}倍以下）: "
          f"{se['n']}本 "
          + (f"的中{se['win']:.1f}% 回収率{se['roi']:.1f}%" if se["n"] else "該当なし"))

    # ◎の複勝率 較正チェック（予測 vs 実績）
    honmei = ran[ran["mark"] == "◎"]
    if len(honmei) and "show_p" in honmei:
        print(f"\n較正: ◎の予測複勝率 平均{honmei['show_p'].mean()*100:.1f}% "
              f"→ 実際の複勝率{(honmei['finish']<=3).mean()*100:.1f}%")

    print("\n※回収率100%超で利益（単=単勝/複=複勝、各印を100円ずつ買った場合）。"
          "複回収は複勝の確定払戻ベース。払戻が未取得のレースは0円扱い。"
          "単勝オッズは最終オッズ基準。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
