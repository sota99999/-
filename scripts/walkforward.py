#!/usr/bin/env python3
# =============================================================================
# ウォークフォワード検証
#
# 期間を時系列に沿って複数の区間に分け、「過去で学習→次の区間で検証」を
# 窓をずらしながら繰り返す。1回の分割では分からない「特定期間だけ強い/弱い」
# を見抜き、回収率の"安定性"を確認するための検証。
#
#   区間: |== fold1学習 ==|=検証=|
#         |==== fold2学習 ====|=検証=|
#         |====== fold3学習 ======|=検証=|   … と学習期間を伸ばしていく
#
# 各 fold で AUC / Brier と「単勝EV>=閾値で100円買い」の回収率を表示し、
# 最後に平均と安定性（標準偏差）を出す。
#
# 使い方:
#   python scripts/walkforward.py --db keiba.db --folds 4
#   python scripts/walkforward.py --db keiba.db --target target_show --folds 5
#   python scripts/walkforward.py --db keiba.db --ev-threshold 1.2 --calibrate isotonic
# =============================================================================
from __future__ import annotations

import argparse
import sys

import numpy as np

import mlcommon
from train_predict import backtest_win   # 単勝回収率バックテストを再利用


def build_model(calibrate: str):
    base, name = mlcommon.make_model()
    if calibrate != "none":
        from sklearn.calibration import CalibratedClassifierCV
        return CalibratedClassifierCV(base, method=calibrate, cv=3), f"{name}+calib({calibrate})"
    return base, name


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ウォークフォワード検証")
    p.add_argument("--db", default="keiba.db")
    p.add_argument("--target", default="target_win", choices=["target_win", "target_show"])
    p.add_argument("--folds", type=int, default=4, help="検証区間の数（既定4）")
    p.add_argument("--calibrate", default="none", choices=["none", "sigmoid", "isotonic"])
    p.add_argument("--ev-threshold", type=float, default=1.1, help="単勝EVの購入閾値")
    p.add_argument("--min-runs", type=int, default=1, help="過去出走数の下限")
    args = p.parse_args(argv)

    df = mlcommon.load_data(args.db, None, args.min_runs)
    # 結果未確定（出走前）レースは検証から除外
    upcoming = set(mlcommon.upcoming_race_ids(args.db))
    if upcoming:
        df = df[~df["race_id"].isin(upcoming)].reset_index(drop=True)

    dates = np.array(sorted(df["race_date"].unique()))
    if len(dates) < args.folds + 2:
        sys.exit(f"[NG] データ期間が短すぎます（日数 {len(dates)} < folds+2={args.folds + 2}）。"
                 "収集期間を延ばすか --folds を減らしてください。")

    # 特徴量は全期間で一度だけ作る（fold間で列を揃えるため）
    x_all = mlcommon.build_features(df)
    y_all = df[args.target].astype(int)

    # 期間を folds+1 個に分割。fold i は「区間0..i-1で学習→区間iで検証」
    chunks = np.array_split(dates, args.folds + 1)
    model_name = build_model(args.calibrate)[1]
    print(f"モデル: {model_name} / ターゲット: {args.target} / folds: {args.folds}")
    print(f"{'fold':>4} {'検証期間':<25} {'学習':>7} {'検証':>6} {'AUC':>6} {'Brier':>7} "
          f"{'購入':>5} {'的中':>5} {'回収率%':>8}")

    from sklearn.metrics import brier_score_loss, roc_auc_score
    aucs, rois = [], []
    for i in range(1, args.folds + 1):
        train_dates = set(np.concatenate(chunks[:i]))
        valid_dates = set(chunks[i])
        tr = df["race_date"].isin(train_dates).to_numpy()
        va = df["race_date"].isin(valid_dates).to_numpy()
        if y_all[tr].nunique() < 2 or va.sum() == 0:
            continue

        model, _ = build_model(args.calibrate)
        model.fit(x_all[tr], y_all[tr])
        proba = model.predict_proba(x_all[va])[:, 1]

        try:
            auc = roc_auc_score(y_all[va], proba)
        except ValueError:
            auc = float("nan")
        brier = brier_score_loss(y_all[va], proba)
        bt = backtest_win(df[va].reset_index(drop=True), proba, args.ev_threshold)
        roi = bt["roi"]
        period = f"{min(valid_dates)}〜{max(valid_dates)}"
        roi_s = "-" if roi is None else f"{roi:.1f}"
        print(f"{i:>4} {period:<25} {int(tr.sum()):>7} {int(va.sum()):>6} "
              f"{auc:>6.3f} {brier:>7.4f} {bt['bets']:>5} {bt['hits']:>5} {roi_s:>8}")
        if not np.isnan(auc):
            aucs.append(auc)
        if roi is not None:
            rois.append(roi)

    print("-" * 86)
    if aucs:
        print(f"AUC      平均 {np.mean(aucs):.3f} / ばらつき(±) {np.std(aucs):.3f}")
    if rois:
        print(f"回収率%  平均 {np.mean(rois):.1f} / ばらつき(±) {np.std(rois):.1f} "
              f"/ 最低 {min(rois):.1f} / 最高 {max(rois):.1f}")
        print("\n判断の目安: 平均が100%超で、ばらつきが小さく最低値も極端に低くなければ"
              "“安定して使える”候補。1区間だけ突出は過信しないこと。")
    else:
        print("購入対象がほぼ無く回収率を評価できません。--ev-threshold を下げて再確認を。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
