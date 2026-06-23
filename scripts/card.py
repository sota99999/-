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
#
# 評価比率（能力・適性を重く / 枠・展開・騎手を軽く）:
#   各馬の「他馬比較での強さ(z)」をグループ別に重み付け合成し、モデル確率を
#   持ち上げ/抑える。既定で 能力0.22 適性0.22 / 枠0.04 展開0.04 騎手0.04
#   （能力・適性が枠/展開/騎手の約5.5倍。確率が過度に尖らない水準に調整済み）。
#   --weights "ability=0.3,jockey=0.02" で個別変更、--lean で強さを一括調整
#   （例 --lean 2 で効きを倍に、--lean 0 で純モデル）。
# =============================================================================
from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import pandas as pd

import mlcommon

MARKS = ["◎", "○", "▲", "△", "△", "×"]   # 上位から印

# 「他馬比較で目立つ強み」を判定する特徴量: (列, ラベル, サンプル数ゲート列, 最小数)
STRENGTH = [
    ("elo_before",            "実力",   None,                0),
    ("avg_speed_prior",       "時計",   None,                0),
    ("recent3_show_rate",     "近走",   None,                0),
    ("course_show_rate_prior", "当コース", "course_runs_prior", 2),
    ("dist_show_rate_prior",  "距離",   "dist_runs_prior",   2),
    ("dir_show_rate_prior",   "回り",   "dir_runs_prior",    2),
    ("off_show_rate_prior",   "道悪",   "off_runs_prior",    2),
    ("jockey_win_rate_prior", "騎手",   "jockey_rides_prior", 10),
    ("pace_fit",              "展開",   None,                0),
    ("draw_bias_fit",         "枠",     None,                0),
]

# -----------------------------------------------------------------------------
# 評価比率（リウェイト）の定義
#   各馬の「他馬比較での強さ(z)」をグループ別に平均し、weights で重み付けして
#   合成した値(tilt)でモデル確率を相対的に持ち上げ/抑える。
#   能力・適性を重く、バイアス・展開・騎手を軽くするのが既定。
#   GROUPS の各要素 = (特徴量列, サンプル数ゲート列, 最小数)
# -----------------------------------------------------------------------------
GROUPS = {
    "ability":  [("elo_before", None, 0), ("avg_speed_prior", None, 0),
                 ("class_adj_speed_prior", None, 0)],            # ①能力
    "aptitude": [("course_show_rate_prior", "course_runs_prior", 2),
                 ("dist_show_rate_prior", "dist_runs_prior", 2),
                 ("off_show_rate_prior", "off_runs_prior", 2)],  # ②コース・距離・道悪適性
    "bias":     [("draw_bias_fit", None, 0)],                    # ③トラック(枠)バイアス
    "pace":     [("pace_fit", None, 0)],                         # ④レース展開
    "jockey":   [("jockey_win_rate_prior", "jockey_rides_prior", 10)],  # ⑤騎手
}
# 既定の評価比率（数値が大きいほど予想で重視）。--weights / --lean で変更可
# 能力:適性:その他 ≈ 5.5:5.5:1。値の絶対水準は確率の尖り具合（確信の強さ）も決める。
DEFAULT_WEIGHTS = {"ability": 0.22, "aptitude": 0.22,
                   "bias": 0.04, "pace": 0.04, "jockey": 0.04}
GROUP_JP = {"ability": "能力", "aptitude": "適性",
            "bias": "枠", "pace": "展開", "jockey": "騎手"}
# 〔評価〕タグの重要度を比率に合わせるための 特徴量→グループ 対応
COL2GROUP = {
    "elo_before": "ability", "avg_speed_prior": "ability",
    "course_show_rate_prior": "aptitude", "dist_show_rate_prior": "aptitude",
    "dir_show_rate_prior": "aptitude", "off_show_rate_prior": "aptitude",
    "jockey_win_rate_prior": "jockey", "pace_fit": "pace", "draw_bias_fit": "bias",
}


def parse_weights(s: str | None) -> dict:
    """\"ability=0.5,jockey=0.1\" 形式の文字列を重み辞書に反映して返す。"""
    w = dict(DEFAULT_WEIGHTS)
    if s:
        for part in s.split(","):
            k, _, v = part.partition("=")
            k = k.strip()
            if k in w and v.strip():
                try:
                    w[k] = float(v)
                except ValueError:
                    pass
    return w


def _group_z(df: pd.DataFrame, members) -> pd.Series:
    """グループ内 特徴量の z スコアを（ゲート・欠損を除いて）平均した Series。"""
    parts = []
    for col, gate, minr in members:
        zc = col + "_z"
        if zc not in df.columns:
            continue
        z = df[zc].where(df[col].notna())
        if gate and gate in df.columns:
            z = z.where(df[gate].fillna(0) >= minr)
        parts.append(z)
    if not parts:
        return pd.Series(0.0, index=df.index)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True).fillna(0.0)


def _add_race_z(df: pd.DataFrame, cols) -> None:
    """レース内 z スコア列(col+'_z')を df に追加（既存はスキップ・in place）。"""
    for c in dict.fromkeys(cols):
        if c in df.columns and (c + "_z") not in df.columns:
            gg = df.groupby("race_id")[c]
            sd = gg.transform("std").replace(0, np.nan)
            df[c + "_z"] = ((df[c] - gg.transform("mean")) / sd).fillna(0.0)


def _samecond_boost(sub: pd.DataFrame, strength: float) -> pd.Series:
    """同条件スペシャリストの確率ブースト係数（レース内）を返す。

    「①同条件での実績が良ければ、他条件の悪さを上書きして高評価する」という
    競馬の定石を明示実装する。同条件での勝利数を主、好走率を従にした信号を
    レース内で z 化し、exp(strength × z) を掛け率として返す（平均≒1）。
    strength=0 で無効。samed_*（道悪不問の競馬場×馬場×距離帯）を主に使う。
    """
    if strength <= 0:
        return pd.Series(1.0, index=sub.index)
    sw = pd.to_numeric(sub.get("samed_wins_prior"), errors="coerce").fillna(0.0)
    ss = pd.to_numeric(sub.get("samed_show_rate_prior"), errors="coerce").fillna(0.0)
    sr = pd.to_numeric(sub.get("samed_runs_prior"), errors="coerce").fillna(0.0)
    # 厳密同条件(same_)の勝利も加点（道悪まで一致する完全同条件は更に強い）
    sw2 = pd.to_numeric(sub.get("same_wins_prior"), errors="coerce").fillna(0.0)
    # 出走1回以上の馬だけ信号を持つ（未経験は0＝中立）
    signal = (sw + 0.6 * sw2 + 0.5 * ss).where(sr >= 1, 0.0)
    sub = sub.assign(_sig=signal)
    gg = sub.groupby("race_id")["_sig"]
    sd = gg.transform("std").replace(0, np.nan)
    z = ((sub["_sig"] - gg.transform("mean")) / sd).fillna(0.0)
    return np.exp((strength * z).clip(-2.0, 2.0))


def compute_scores(sub: pd.DataFrame, bundle: dict, show_bundle, weights: dict,
                   same_boost: float = 0.0) -> pd.DataFrame:
    """モデル予測を行い p(勝率)/show_p(複勝率)/ev を付与して返す。

    表示する確率は較正済みモデルの素の値（リークなし）を基本にする:
      - p      : 単勝確率。レース内で合計1に正規化（厳密に1頭が勝つため）。
      - show_p : 複勝確率。レース内で合計 min(3, 頭数) に正規化（3頭が3着内）。
                 → モデルが自信過剰なら合計が3を超えるので、3に揃える過程で
                    自動的に少し引き締められ、較正が改善する。
      - ev     : 単勝期待値 = 正規化済み単勝確率 × オッズ。

    評価比率(weights)が全て0でない場合のみ、各グループの他馬比較z を合成した
    tilt で確率を増減させる「能力重視リウェイト」を適用する（既定は無効＝較正優先）。
    same_boost>0 のとき、同条件スペシャリスト・ブーストを掛けて①同条件の実績で
    他条件の悪さを上書きする（card 表示と evaluate で同一採点になるよう共通化）。
    """
    sub = sub.copy()
    x = mlcommon.build_features(sub, feature_columns=bundle["feature_columns"])
    sub["p_raw"] = bundle["model"].predict_proba(x)[:, 1]
    if show_bundle:
        xs = mlcommon.build_features(sub, feature_columns=show_bundle["feature_columns"])
        sub["show_raw"] = show_bundle["model"].predict_proba(xs)[:, 1]

    # 能力重視リウェイト（任意）。weights が全て0なら無効＝較正をそのまま使う
    use_tilt = any(abs(v) > 1e-9 for v in weights.values())
    if use_tilt:
        _add_race_z(sub, [col for g in GROUPS.values() for col, *_ in g])
        tilt = pd.Series(0.0, index=sub.index)
        for gname, members in GROUPS.items():
            tilt = tilt + weights.get(gname, 0.0) * _group_z(sub, members)
        sub["tilt"] = tilt
        # レース内で平均0に中心化（全体の確率水準をできるだけ保つ）
        sub["tilt"] = sub["tilt"] - sub.groupby("race_id")["tilt"].transform("mean")
        adj = np.exp(sub["tilt"].clip(-2.0, 2.0))
    else:
        adj = pd.Series(1.0, index=sub.index)

    # 同条件スペシャリスト・ブースト（①同条件の実績で他を上書き）
    adj = adj * _samecond_boost(sub, same_boost)

    # 単勝: 合計1に正規化（較正済みの単勝確率）
    sub["p_adj"] = sub["p_raw"] * adj
    sub = mlcommon.normalize_by_race(sub, prob_col="p_adj", out_col="p")
    # 複勝: 合計 min(3, 頭数) に正規化（自信過剰を引き締める＝較正改善）
    if show_bundle:
        sub["show_adj"] = sub["show_raw"] * adj
        grp = sub.groupby("race_id")["show_adj"]
        s = grp.transform("sum")
        cnt = grp.transform("size").clip(upper=3)
        sub["show_p"] = (sub["show_adj"] / s.where(s > 0, 1.0) * cnt).clip(upper=0.97)
    # EVは正規化済み単勝確率×オッズ（市場との乖離＝妙味の指標）
    sub["ev"] = sub["p"] * pd.to_numeric(sub.get("odds"), errors="coerce")
    return sub


SUB_MARKS = ["○", "▲", "△", "△", "△"]   # ◎の次以降（△の数はレースにより可変）


def assign_marks(g: pd.DataFrame, value_mode: bool, strength_col: str = "show_p",
                 min_runs: int = 3, max_odds: float = 20.0,
                 hon_mode: str = "strong", hon_min_odds: float = 1.0,
                 sub_max_odds: float | None = None) -> pd.DataFrame:
    """1レース分の出走馬に印(mark列)を付け、印→強さ順に並べ替えて返す。

    value_mode=True（最終結論・オッズあり）:
      ◎ の選び方は hon_mode で切替:
        "strong"(既定) = 単勝確率(p)が最も高い「強い馬」を本命に。ただし
            人気を背負いすぎた本命(回収率を殺す低オッズ)を hon_min_odds で除外し、
            出走数 min_runs 以上・オッズ max_odds 以下に限定する。
            → ◎の的中率と回収率を両立させたいときはこちら。
        "value" = EV≧1（オッズ以上に評価できる妙味馬）の中で p 最上位を本命に。
            → 妙味（穴）を本命に据えたいときはこちら。
      ○以降は妙味馬(EV≧1・出走数・オッズ条件)を強さ(strength_col)順に付ける:
        ○ = 最も強い妙味馬（妙味の本線）。
        ▲ 回収重視 = 妙味の2番手。市場は妙味でも#1を買い被るため、この層が
                     最も回収が高い（検証で▲が単回収130〜160%台のヤマ）。
        △ = 妙味の3番手以降（高回収の押さえ）。
      該当が無ければ強さ上位で埋める。印の数はレースで変動する。
      ※hon_mode="value" のときは◎も妙味(EV≧1)の中の単勝確率最上位にする。
      ※オッズ上限は ◎=max_odds（較正重視で堅め）、○▲△=sub_max_odds
        （既定は max_odds。広げると穴の妙味まで拾える）と分離できる。
    value_mode=False（全頭診断・オッズなし）:
      単純に強さ順（strength_col 降順）で ◎○▲△△× を付ける。
    """
    if sub_max_odds is None:
        sub_max_odds = max_odds
    g = g.copy()
    g["mark"] = ""
    if value_mode and "ev" in g.columns:
        p = pd.to_numeric(g["p"], errors="coerce")
        ev = pd.to_numeric(g["ev"], errors="coerce")
        odds = pd.to_numeric(g.get("odds"), errors="coerce")
        runs = pd.to_numeric(g.get("runs_prior"), errors="coerce").fillna(0)
        # 妙味＝オッズ以上の評価。○▲△は sub_max_odds まで拾う（穴の裾野を広げる）
        overlay = (ev >= 1.0) & (runs >= min_runs) & (odds <= sub_max_odds) & odds.notna()

        # ◎ 的中重視: 強い馬（人気すぎ除外）。◎だけは max_odds で堅めに限定
        if hon_mode == "value":
            hon = (p[overlay].idxmax() if overlay.any() else p.idxmax())
        else:
            elig = (runs >= min_runs) & (odds >= hon_min_odds) \
                & (odds <= max_odds) & odds.notna()
            hon = (p[elig].idxmax() if elig.any() else p.idxmax())
        g.loc[hon, "mark"] = "◎"

        # ○▲△: ◎を除く妙味馬を強さ順に。○=本線 ▲=2番手(回収のヤマ) △=3番手以降
        rest = g.loc[g.index[overlay].difference([hon])]
        rest = rest.sort_values(strength_col, ascending=False)
        for k, idx in enumerate(rest.index[:len(SUB_MARKS)]):
            g.loc[idx, "mark"] = SUB_MARKS[k]

        order = {"◎": 0, "○": 1, "▲": 2, "△": 3}
        g["_o"] = g["mark"].map(lambda m: order.get(m, 9))
        g = g.sort_values(["_o", strength_col], ascending=[True, False]).drop(columns="_o")
    else:
        g = g.sort_values(strength_col, ascending=False).reset_index(drop=True)
        g["mark"] = [MARKS[i] if i < len(MARKS) else "" for i in range(len(g))]
    return g


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
                   help="全頭診断(オッズ無)の強さ順の基準: show=複勝率(既定), win=勝率")
    p.add_argument("--race-id", help="このレースだけ")
    p.add_argument("--date", help="この開催日の全レース (YYYY-MM-DD)")
    p.add_argument("--top", type=int, default=0, help="上位何頭まで表示（0=全頭）")
    p.add_argument("--mark-min-runs", type=int, default=2,
                   help="最終結論で印(妙味馬)を付ける最低出走回数（能力未知馬を除外）")
    p.add_argument("--mark-max-odds", type=float, default=20.0,
                   help="◎の単勝オッズ上限（較正重視で堅め。大穴の過大評価を除外）")
    p.add_argument("--sub-max-odds", type=float, default=30.0,
                   help="○▲△の単勝オッズ上限（既定=◎と同じ。広げると穴の妙味を拾う）")
    p.add_argument("--hon-mode", choices=["strong", "value"], default="strong",
                   help="◎の選び方: strong=強い馬(人気すぎ除外/既定), value=妙味(穴)")
    p.add_argument("--hon-min-odds", type=float, default=1.0,
                   help="◎(strong時)の単勝オッズ下限。人気を背負いすぎた本命を除外")
    p.add_argument("--weights", help='能力重視リウェイトの比率。例 '
                   '"ability=0.22,aptitude=0.22,bias=0.04,pace=0.04,jockey=0.04"')
    p.add_argument("--lean", type=float, default=0.0,
                   help="リウェイトの強さ倍率。既定0=較正優先（リウェイトなし）, 1で適用")
    p.add_argument("--same-boost", type=float, default=0.0,
                   help="同条件スペシャリスト・ブーストの強さ（0=無効。0.5〜1.0で①同条件の実績を上乗せ）")
    args = p.parse_args(argv)

    weights = parse_weights(args.weights)
    weights = {k: v * args.lean for k, v in weights.items()}
    use_tilt = any(abs(v) > 1e-9 for v in weights.values())

    bundle = mlcommon.load_model(args.model)
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

    # モデル予測（card と evaluate で共通の採点）
    sub = compute_scores(sub, bundle, show_bundle, weights, same_boost=args.same_boost)

    # オッズがあれば「最終結論（妙味で印）」、無ければ「全頭診断（強さ順で印）」
    has_odds = pd.to_numeric(sub.get("odds"), errors="coerce").notna().any()
    strength_col = "show_p" if (args.mark_by == "show" and show_bundle) else "p"

    # ★/〔評価〕用の z（調整後の p/show_p/ev と、表示する Elo/最高SP）
    _add_race_z(sub, ["p", "show_p", "ev", "elo_before", "best_speed_prior"]
                + [c for c, *_ in STRENGTH])

    def st(r, c):   # 突出値マーク（出走馬中で z>=1.5）
        return "★" if r.get(c + "_z", 0) >= 1.5 else ""

    def hyoten(r):  # 他馬比較で目立つ強み(◎)/弱み(▼)。評価比率に応じて重要度を調整
        tags = []
        for col, label, gate, minr in STRENGTH:
            if col not in sub.columns or pd.isna(r.get(col)):
                continue
            if gate and (r.get(gate) or 0) < minr:
                continue
            z = r.get(col + "_z", 0)
            if z >= 1.0:
                # リウェイト時は比率の低いグループの強みを見出しに上げにくくする
                wf = weights.get(COL2GROUP.get(col, ""), 0.40) if use_tilt else 1.0
                tags.append((z * wf, z, f"{label}◎"))
        tags.sort(reverse=True)
        out = [t for *_, t in tags[:3]]
        if (r.get("runs_prior") or 0) >= 3 and (r.get("recent3_show_rate") or 0) == 0:
            out.append("近走▼")
        return out

    fuku_h = f"{'複勝率':>7}" if show_bundle else ""
    odds_h = f"{'オッズ':>6}{'EV':>7}" if has_odds else ""   # オッズあり最終結論時のみ
    for rid, g in sub.groupby("race_id"):
        g = assign_marks(g, value_mode=has_odds, strength_col=strength_col,
                         min_runs=args.mark_min_runs,
                         max_odds=args.mark_max_odds,
                         sub_max_odds=args.sub_max_odds,
                         hon_mode=args.hon_mode,
                         hon_min_odds=args.hon_min_odds).reset_index(drop=True)
        if args.top:
            g = g.head(args.top)
        print(f"\n=== {rid}  {names.get(rid, '')} ===")
        print(f"{'印':<2}{'馬番':>3} {'馬名':<12}{'勝率':>7}{fuku_h} "
              f"{'Elo':>6} {'最高SP':>7}{odds_h}")
        for _, r in g.iterrows():
            mark = r["mark"] or "  "
            wr = _f(r['p'] * 100, '5.1f') + "%" + st(r, 'p')
            fuku = (" " + _f(r.get('show_p') * 100, '5.1f') + "%" + st(r, 'show_p')) if show_bundle else ""
            el = _f(r.get('elo_before'), '5.0f') + st(r, 'elo_before')
            sp = _f(r.get('best_speed_prior'), '5.1f') + st(r, 'best_speed_prior')
            odds = (" " + _f(r.get('odds'), '5.1f')
                    + " " + _f(r.get('ev'), '5.2f') + st(r, 'ev')) if has_odds else ""
            print(f"{mark:<2}{int(r['horse_number']):>3} {str(r['horse_name'])[:12]:<12}"
                  f"{wr:>7}{fuku} {el:>6} {sp:>7}{odds}")
        # 〔評価〕印を付けた馬の目立つ強み/弱み
        lines = []
        for _, r in g[g["mark"] != ""].head(5).iterrows():
            tags = hyoten(r)
            if tags:
                lines.append(f"{r['mark']}{str(r['horse_name'])[:7]}: {'・'.join(tags)}")
        if lines:
            print("  〔評価〕 " + " ｜ ".join(lines))

    if has_odds:
        mark_rule = ("印=役割別。◎=的中・複勝軸（20倍以内で最も強い馬）、"
                     "○=妙味の本線、▲=単勝回収のヤマ（妙味の2番手）、△=高回収の押さえ"
                     "（○▲△はEV≧1の妙味馬・頭数は変動。妙味馬が居なければ印は少なくなる）")
    else:
        mark_rule = f"印は馬の強さ（{'複勝率' if strength_col=='show_p' else '勝率'}）順"
    rw = (f"能力重視リウェイト 適用（×{args.lean:g}）" if use_tilt
          else "リウェイトなし＝較正優先（勝率・複勝率はモデル較正値）")
    print(f"\n※{mark_rule}。★=出走馬中で突出して高い値。"
          f"〔評価〕は他馬比較で目立つ強み◎/弱み▼。")
    print(f"※{rw}。馬券は自己責任で。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
