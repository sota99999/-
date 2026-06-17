#!/usr/bin/env python3
# =============================================================================
# 毎週の更新～予想を1コマンドで回すオーケストレーター
#
# 次を順に実行する:
#   ① 新着レース結果の差分収集（前回収集日〜今日）
#   ② 指定開催日の全レース出馬表を取得（--raceday 指定時）
#   ③ Eloレーティング再計算
#   ④ 特徴量ビュー(v_features)を再作成
#   ⑤ モデル再学習（--retrain 指定時、単勝・複勝の両方）
#   ⑥ 指定開催日の全レースを予想（--raceday 指定時）
#
# 使い方:
#   # 毎週これだけ: 先週結果を取り込み→今週を予想（モデルは既存を使用）
#   python scripts/weekly.py --db keiba.db --raceday 2026-06-21 --bankroll 10000
#   # ときどき: 再学習も一緒に
#   python scripts/weekly.py --db keiba.db --raceday 2026-06-21 --retrain
#   # 何が走るか確認だけ（実行しない）
#   python scripts/weekly.py --db keiba.db --raceday 2026-06-21 --dry-run
#
# 注意: 収集はネットワークの通る環境（手元PC/Colab等）で実行すること。
# =============================================================================
from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import sys
from pathlib import Path

import scraper
import crawler
import compute_ratings
import train_predict
import predict

ROOT = Path(__file__).resolve().parent.parent


def last_finished_date(db: str) -> str | None:
    """確定結果が入っている最新の race_date。無ければ None。"""
    conn = sqlite3.connect(db)
    try:
        scraper.ensure_schema(conn)
        row = conn.execute(
            """SELECT MAX(ra.race_date) FROM races ra
               JOIN results r ON ra.race_id = r.race_id
               WHERE r.finish_position IS NOT NULL"""
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row and row[0] else None


def apply_features(db: str) -> None:
    """schema/features.sql を適用して v_features を作り直す（sqlite3 CLI不要）。"""
    sql = (ROOT / "schema" / "features.sql").read_text(encoding="utf-8")
    conn = sqlite3.connect(db)
    try:
        conn.executescript(sql)
        conn.commit()
    finally:
        conn.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="毎週の更新～予想を1コマンドで")
    p.add_argument("--db", default="keiba.db")
    p.add_argument("--raceday", help="予想したい開催日 (YYYY-MM-DD)。省略時は予想せず更新のみ")
    p.add_argument("--since", help="結果収集の開始日 (YYYY-MM-DD)。省略時は前回収集日から")
    p.add_argument("--retrain", action="store_true", help="モデルを再学習する")
    p.add_argument("--bankroll", type=float, default=None, help="ケリー資金配分に使う資金額")
    p.add_argument("--win-model", default="model_win.pkl")
    p.add_argument("--show-model", default="model_show.pkl")
    p.add_argument("--dry-run", action="store_true", help="実行せず手順だけ表示")
    args = p.parse_args(argv)

    today = dt.date.today().isoformat()
    start = args.since or last_finished_date(args.db) or \
        (dt.date.today() - dt.timedelta(days=365)).isoformat()

    steps: list[tuple[str, list[str] | None, callable]] = []
    steps.append((f"① 新着結果を収集 {start}〜{today}",
                  ["--db", args.db, "--from", start, "--to", today],
                  crawler.main))
    if args.raceday:
        steps.append((f"② 出馬表を収集 {args.raceday}（全レース）",
                      ["--db", args.db, "--shutuba", "--date", args.raceday],
                      crawler.main))
    steps.append(("③ Eloレーティング再計算", ["--db", args.db], compute_ratings.main))
    steps.append(("④ 特徴量ビュー再作成", None, lambda _a=None: apply_features(args.db)))
    if args.retrain:
        steps.append(("⑤ 単勝モデル再学習",
                      ["--db", args.db, "--calibrate", "isotonic", "--save-model", args.win_model],
                      train_predict.main))
        steps.append(("⑤ 複勝モデル再学習",
                      ["--db", args.db, "--target", "target_show", "--calibrate", "isotonic",
                       "--save-model", args.show_model],
                      train_predict.main))
    if args.raceday:
        pred_args = ["--db", args.db, "--model", args.win_model]
        if args.bankroll is not None:
            pred_args += ["--bankroll", str(args.bankroll)]
        steps.append((f"⑥ {args.raceday} の全レースを予想（単勝モデル）", pred_args, predict.main))

    print(f"=== weekly: DB={args.db} / 予想日={args.raceday or '(なし)'} / 再学習={args.retrain} ===")
    for i, (label, sargs, func) in enumerate(steps, 1):
        print(f"\n----- {label} -----")
        if args.dry_run:
            print(f"   (dry-run) {func.__module__ if hasattr(func,'__module__') else ''} args={sargs}")
            continue
        try:
            func(sargs) if sargs is not None else func()
        except SystemExit as e:   # 子スクリプトの sys.exit を致命扱いにしない
            if e.code:
                print(f"[NG] ステップ失敗: {label} ({e})", file=sys.stderr)
                return 1
        except Exception as e:    # noqa: BLE001
            print(f"[NG] ステップ失敗: {label}: {e}", file=sys.stderr)
            return 1

    print("\n=== 完了 ===")
    if args.raceday and not args.dry_run:
        print("複勝・連勝式も見るには:")
        print(f"  python scripts/predict.py --db {args.db} --model {args.show_model}")
        print(f"  python scripts/bet_optimizer.py --db {args.db} --model {args.win_model} "
              f"--predict --bet trio --topn 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
