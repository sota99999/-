#!/usr/bin/env python3
# =============================================================================
# DB健全性チェック（データのバグ・欠損・異常値を一括診断）
#
#   ① 重複: results の (race_id,horse_id) 重複 / races の race_id 重複
#   ② 欠損: 結果確定済みなのに オッズ/タイム が欠けている行数
#   ③ カバレッジ: v_speed / race_laps / horse_relative_r / v_course が
#      結果確定レースの何%を覆っているか
#   ④ 異常値: quality_sp の極端な外れ値 / 着順>頭数 / オッズ<=0
#   ⑤ 未来レース: 出馬表だけのレースに相対Rが付与されているか
#
# 使い方: python scripts/doctor.py --db keiba.db
# =============================================================================
from __future__ import annotations

import argparse
import sqlite3

FIN = ("race_id IN (SELECT race_id FROM results "
       "WHERE finish_position IS NOT NULL)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DB健全性チェック")
    ap.add_argument("--db", default="keiba.db")
    args = ap.parse_args(argv)
    con = sqlite3.connect(args.db)
    issues = 0

    def q(sql):
        try:
            return con.execute(sql).fetchone()[0] or 0
        except sqlite3.OperationalError:
            return None                       # テーブル/ビュー未作成

    def report(label, bad, detail=""):
        nonlocal issues
        if bad is None:
            bad, detail = True, "テーブル/ビュー未作成（スキーマ適用が必要）"
        mark = "NG " if bad else "OK "
        if bad:
            issues += 1
        print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))

    print("== ① 重複 ==")
    d1 = q("""SELECT COUNT(*) FROM (SELECT 1 FROM results
              GROUP BY race_id,horse_id HAVING COUNT(*)>1)""")
    report("results の重複 (race_id,horse_id)",
           None if d1 is None else d1 > 0, f"{d1}組")
    d2 = q("SELECT COUNT(*) FROM (SELECT 1 FROM races GROUP BY race_id HAVING COUNT(*)>1)")
    report("races の重複 race_id", None if d2 is None else d2 > 0, f"{d2}件")

    print("== ② 欠損（結果確定行のみ）==")
    total = q("SELECT COUNT(*) FROM results WHERE finish_position IS NOT NULL") or 1
    for label, sql in [
        ("オッズ欠損", "SELECT COUNT(*) FROM results WHERE finish_position IS NOT NULL AND odds IS NULL"),
        ("タイム欠損", "SELECT COUNT(*) FROM results WHERE finish_position IS NOT NULL AND time_seconds IS NULL"),
    ]:
        c = q(sql)
        report(label, None if c is None else c > total * 0.05,
               f"{c}/{total} ({(c or 0)/total*100:.1f}%)")

    print("== ③ カバレッジ（結果確定レース比）==")
    nrace = q(f"SELECT COUNT(*) FROM races WHERE {FIN}") or 1
    for label, sql, thresh in [
        ("スピード指数 v_speed",
         f"SELECT COUNT(DISTINCT race_id) FROM v_speed WHERE speed_index IS NOT NULL AND {FIN}", 0.85),
        ("ラップ race_laps",
         f"SELECT COUNT(*) FROM race_laps WHERE race_first3f IS NOT NULL AND {FIN}", 0.7),
        ("コース形態 v_course",
         f"SELECT COUNT(*) FROM v_course WHERE straight_m IS NOT NULL AND {FIN}", 0.9),
    ]:
        c = q(sql)
        report(label, None if c is None else c < nrace * thresh,
               f"{c}/{nrace}レース ({(c or 0)/nrace*100:.1f}%)")
    # 相対Rが付かないのは設計上2種: ①リーク防止の立ち上げ期(最初の90日は
    # スナップショット不能) ②新馬戦(全馬デビュー=過去データ無し)。両方除いて測る。
    cut = q("SELECT DATE(MIN(race_date), '+90 days') FROM races")
    cond = f"{FIN} AND race_date >= '{cut}' AND race_name NOT LIKE '%新馬%'"
    n2 = q(f"SELECT COUNT(*) FROM races WHERE {cond}") or 1
    c2 = q(f"""SELECT COUNT(DISTINCT hr.race_id) FROM horse_relative_r hr
               JOIN races ra ON ra.race_id = hr.race_id
               WHERE ra.{FIN} AND ra.race_date >= '{cut}'
                 AND ra.race_name NOT LIKE '%新馬%'""")
    report("相対R（立ち上げ90日・新馬戦を除く）",
           None if c2 is None else c2 < n2 * 0.95,
           f"{c2}/{n2}レース ({(c2 or 0)/n2*100:.1f}%)")

    print("== ④ 異常値 ==")
    # v_run_adj の入口ガード(speed_index -50〜150)＋補正(±5程度)を考慮した外側だけを異常視
    out = q("SELECT COUNT(*) FROM v_run_adj WHERE quality_sp > 160 OR quality_sp < -60")
    report("quality_sp の外れ値(>160 or <-60)",
           None if out is None else out > 0, f"{out}走")
    bad_fin = q("""SELECT COUNT(*) FROM results r JOIN races ra ON ra.race_id=r.race_id
                   WHERE r.finish_position > ra.field_size""")
    report("着順 > 頭数", None if bad_fin is None else bad_fin > 0, f"{bad_fin}行")
    neg = q("SELECT COUNT(*) FROM results WHERE odds IS NOT NULL AND odds <= 0")
    report("オッズ <= 0", None if neg is None else neg > 0, f"{neg}行")

    print("== ⑤ 未来レース（出馬表のみ）==")
    up = q("""SELECT COUNT(*) FROM races WHERE race_id IN
              (SELECT race_id FROM results GROUP BY race_id
               HAVING COUNT(finish_position)=0)""") or 0
    upr = q("""SELECT COUNT(DISTINCT race_id) FROM horse_relative_r
               WHERE race_id IN (SELECT race_id FROM results GROUP BY race_id
               HAVING COUNT(finish_position)=0)""")
    report("出馬表レースへの相対R付与",
           None if upr is None else (up > 0 and upr == 0),
           f"{upr}/{up}レース（0なら compute_relative_r.py を実行）")

    print(f"\n診断完了: 問題 {issues} 件。" + ("要確認。" if issues else "健全です。"))
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
