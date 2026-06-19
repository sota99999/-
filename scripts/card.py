#!/usr/bin/env python3
# =============================================================================
# 全頭診断: 各レースの全出走馬をモデルで評価して一覧表示する
#
# 各馬を勝率(p)順に並べ、印(◎○▲△)・Eloレーティング・直近複勝率・
# 平均スピード指数・脚質 などの根拠もまとめて表示する。
#
# 使い方:
#   # 出走前(結果未確定)の全レースを全頭診断（オッズ無しモデル推奨）
#   python scripts/card.py --db keiba.db --model model_win_noodds.pkl
#   # 特定日 / 特定レースだけ
#   python scripts/card.py --db keiba.db --model model_win.pkl --date 2026-06-21
#   python scripts/card.py --db keiba.db --model model_win.pkl --race-id 202609030411
# =============================================================================
from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

import mlcommon

MARKS = ["◎", "○", "▲", "△", "△", "×"]   # 上位から印


def _f(v, fmt, default="  -"):
    """NaN/None を安全に整形。"""
    try:
        if v is None or pd.isna(v):
            return default
        return format(v, fmt)
    except (ValueError, TypeError):
        return default


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="全頭診断")
    p.add_argument("--db", default="keiba.db")
    p.add_argument("--model", default="model_win_noodds.pkl")
    p.add_argument("--show-model", help="複勝率も表示する場合の複勝モデル（例: model_show_noodds.pkl）")
    p.add_argument("--mark-by", choices=["show", "win"], default="show",
                   help="印(◎○▲△)の基準: show=複勝率(既定), win=勝率。showは--show-model必須")
    p.add_argument("--race-id", help="このレースだけ")
    p.add_argument("--date", help="この開催日の全レース (YYYY-MM-DD)")
    p.add_argument("--top", type=int, default=0, help="上位何頭まで表示（0=全頭）")
    args = p.parse_args(argv)

    bundle = mlcommon.load_model(args.model)
    model = bundle["model"]
    show_bundle = mlcommon.load_model(args.show_model) if args.show_model else None
    df = mlcommon.load_data(args.db, None)

    conn = sqlite3.connect(args.db)
    if args.race_id:
        ids = [args.race_id]
    elif args.date:
        ids = [r[0] for r in conn.execute(
            "SELECT race_id FROM races WHERE race_date=? ORDER BY race_id", (args.date,))]
    else:
        ids = mlcommon.upcoming_race_ids(args.db)
    names = dict(conn.execute(
        "SELECT race_id, race_name FROM races").fetchall()) if ids else {}
    conn.close()

    sub = df[df["race_id"].isin(ids)].copy()
    if sub.empty:
        print("対象レースがありません（出馬表を取り込みましたか? 日付指定は合っていますか?）")
        return 0

    x = mlcommon.build_features(sub, feature_columns=bundle["feature_columns"])
    sub["p_raw"] = model.predict_proba(x)[:, 1]
    # レース内で合計1に正規化（全頭診断の勝率として読みやすくする）
    sub = mlcommon.normalize_by_race(sub, prob_col="p_raw", out_col="p")
    # 複勝率（3着以内確率）。1レースで3頭が3着以内に入るので、レース内で合計3に正規化
    if show_bundle:
        xs = mlcommon.build_features(sub, feature_columns=show_bundle["feature_columns"])
        sub["show_raw"] = show_bundle["model"].predict_proba(xs)[:, 1]
        grp = sub.groupby("race_id")["show_raw"]
        s = grp.transform("sum")
        cnt = grp.transform("size").clip(upper=3)   # 出走頭数が3未満ならその頭数
        sub["show_p"] = (sub["show_raw"] / s.where(s > 0, 1.0) * cnt).clip(upper=0.99)
    # EVは較正済みの素の確率×オッズ（正規化前）で算出
    sub["ev"] = sub["p_raw"] * pd.to_numeric(sub.get("odds"), errors="coerce")

    # 印の基準（既定: 複勝率。--show-model が無ければ勝率）
    sort_col = "show_p" if (args.mark_by == "show" and show_bundle) else "p"
    mark_label = "複勝率" if sort_col == "show_p" else "勝率"
    has_odds = pd.to_numeric(sub.get("odds"), errors="coerce").notna().any()
    fuku_h = f"{'複勝率':>7}" if show_bundle else ""
    # オッズがある最終結論時のみ オッズ・EV(単勝期待値)を表示
    odds_h = f"{'オッズ':>6}{'EV':>6}" if has_odds else ""
    for rid, g in sub.groupby("race_id"):
        g = g.sort_values(sort_col, ascending=False).reset_index(drop=True)
        if args.top:
            g = g.head(args.top)
        print(f"\n=== {rid}  {names.get(rid, '')} ===")
        print(f"{'印':<2}{'馬番':>3} {'馬名':<13}{'勝率':>6}{fuku_h} "
              f"{'Elo':>5} {'平均SP':>6}{odds_h}")
        for i, r in g.iterrows():
            mark = MARKS[i] if i < len(MARKS) else "  "
            fuku = f"{_f(r.get('show_p', float('nan'))*100, '6.1f')}%" if show_bundle else ""
            odds = f"{_f(r.get('odds'), '6.1f')}{_f(r.get('ev'), '6.2f')}" if has_odds else ""
            print(f"{mark:<2}{int(r['horse_number']):>3} {str(r['horse_name'])[:13]:<13}"
                  f"{_f(r['p']*100, '5.1f')}%{fuku} "
                  f"{_f(r.get('elo_before'), '5.0f')} "
                  f"{_f(r.get('avg_speed_prior'), '6.1f')}{odds}")
    foot = "EV>1.0は期待値プラスの目安。" if has_odds else ""
    print(f"\n※ 勝率=1着, 複勝率=3着内 のモデル推定。印は{mark_label}順。{foot}馬券は自己責任で。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
