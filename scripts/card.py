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
                 sub_max_odds: float | None = None, hon_by: str = "win") -> pd.DataFrame:
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

        # ◎の選定指標: hon_by="show"なら複勝確率(=馬券圏に来やすさ)で、"win"なら単勝確率で
        hp = pd.to_numeric(g.get("show_p"), errors="coerce") if hon_by == "show" else p
        if hp.isna().all():
            hp = p
        # ◎ 的中重視: 強い馬。◎だけは max_odds で堅めに限定（hon_min_odsで人気すぎ除外も可）
        if hon_mode == "value":
            hon = (hp[overlay].idxmax() if overlay.any() else hp.idxmax())
        else:
            elig = (runs >= min_runs) & (odds >= hon_min_odds) \
                & (odds <= max_odds) & odds.notna()
            hon = (hp[elig].idxmax() if elig.any() else hp.idxmax())
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


# 能力指標（最高SP・Elo の2本立て）。(列, 見出し, 数値書式)
ABILITY_COLS = [("best_speed_prior", "最高SP", "5.1f"),
                ("elo_before", "Elo", "6.0f")]

# コース形態の表示用（course_master / v_course があるとき）
VENUE_JP = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
            "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
_GRADE_JP = {"steep": "急坂", "mild": "緩坂", "flat": "平坦"}
_TURN_JP = {"tight": "小回り", "wide": "広い", "none": "直線"}
_PACE_JP = {"high": "ハイ傾向", "slow": "スロー傾向", "mid": ""}


def _load_course_map(db: str, ids) -> dict:
    """v_course から race_id→コース形態 dict を作る（無ければ空）。"""
    try:
        conn = sqlite3.connect(db)
        q = ("SELECT race_id, venue_id, surface, distance, course_type, turn_dir, "
             "straight_m, hill_m, hill_grade, turn_size, turf_type, pace_bias "
             "FROM v_course WHERE race_id IN (%s)" % ",".join("?" * len(ids)))
        rows = conn.execute(q, list(ids)).fetchall()
        conn.close()
    except Exception:  # noqa: BLE001  ビュー未作成など
        return {}
    cols = ["venue_id", "surface", "distance", "course_type", "turn_dir",
            "straight_m", "hill_m", "hill_grade", "turn_size", "turf_type", "pace_bias"]
    return {r[0]: dict(zip(cols, r[1:])) for r in rows}


def _course_line(c: dict) -> str:
    """コース形態を1行に整形。"""
    if not c or c.get("straight_m") is None:
        return ""
    v = VENUE_JP.get(str(c["venue_id"]).zfill(2), c["venue_id"])
    ct = c.get("course_type") or ""
    ctd = ct if ct in ("内", "外", "直") else ""
    parts = [f"{v}{c['surface']}{int(c['distance'])}{ctd}"]
    if c.get("turn_dir"):
        parts.append(f"{c['turn_dir']}回り")
    parts.append(f"直線{c['straight_m']:.0f}m")
    if c.get("hill_m") is not None:
        parts.append(f"坂{c['hill_m']:.1f}m({_GRADE_JP.get(c.get('hill_grade'), '')})")
    else:
        parts.append(f"坂{_GRADE_JP.get(c.get('hill_grade'), '')}")
    parts.append(_TURN_JP.get(c.get("turn_size"), ""))
    if c.get("turf_type") == "noshiba":
        parts.append("洋芝")
    p = _PACE_JP.get(c.get("pace_bias"), "")
    if p:
        parts.append(p)
    return "  〔コース〕 " + " ｜ ".join(x for x in parts if x)
RANK_MARKS = ["◎", "○", "▲", "△"]   # 非特出の1〜4番手
STANDOUT_Z = 1.5      # これ以上で「特出」⭐
NEARTIE_Z = 0.25      # 5番手以降が4番手とこの差以内なら△（上位と近い値）
VALUE_EV = 1.5        # 指標ベース期待値がこの倍率以上で「妙味」✅
VALUE_MINZ = 0.5      # かつ指標で平均より明確に上（過剰な大穴は弾く）


def _ability_marks(g: pd.DataFrame) -> pd.DataFrame:
    """最高SP・Elo の2指標で、レース内に印を付けて返す（表示・評価で共通）。

    各指標について:
      ⭐ = 特出（出走馬中 z≧1.5）。複数可。
      ◎○▲△ = 特出を除いた上位1〜4番手。
      △(追加) = 5番手以降でも4番手と僅差（z差≦0.25）なら△。
    付けた印は列  best_speed_prior_mark / elo_before_mark  に入れる。
    オッズがあれば、2指標の合成強さに対しオッズが割に合う馬に ✅（val_flag列）。
    """
    g = g.copy()
    odds = pd.to_numeric(g.get("odds"), errors="coerce")
    for col, _, _ in ABILITY_COLS:
        mark = pd.Series("", index=g.index)
        v = pd.to_numeric(g.get(col), errors="coerce")
        valid = v.notna()
        sd = v[valid].std(ddof=0) if valid.sum() >= 2 else 0.0
        if valid.sum() >= 2 and sd and sd > 0:
            z = (v - v[valid].mean()) / sd
        else:
            z = pd.Series(np.nan, index=g.index)   # 差が無い（新馬の全1500等）は印なし
        g[col + "_z"] = z
        if z.notna().any():
            standout = valid & (z >= STANDOUT_Z)
            mark[standout] = "⭐"
            rest = [i for i in g.index[valid & ~standout]]
            rest.sort(key=lambda i: v[i], reverse=True)
            rank4_z = None
            for k, idx in enumerate(rest):
                if k < 4:
                    mark[idx] = RANK_MARKS[k]
                    if k == 3:
                        rank4_z = z[idx]
                else:                                # 5番手以降は4番手と僅差のみ△
                    if rank4_z is not None and (rank4_z - z[idx]) <= NEARTIE_Z:
                        mark[idx] = "△"
                    else:
                        break                        # 値は降順なので以降は対象外
        g[col + "_mark"] = mark

        # ✅ オッズ妙味（指標ごと）: その指標の強さ z→疑似確率(softmax)×オッズ が割に合う馬
        val = pd.Series("", index=g.index)
        g[col + "_ev"] = np.nan
        if odds.notna().any() and z.notna().any():
            e = np.exp(z.fillna(z[z.notna()].min()).clip(-3, 3))
            p = e / e.sum() if e.sum() > 0 else e
            ev = p * odds
            g[col + "_ev"] = ev
            val[(ev >= VALUE_EV) & (z >= VALUE_MINZ) & odds.notna()] = "✅"
        g[col + "_val"] = val
    return g


def _ability_table(sub: pd.DataFrame, names: dict, top: int = 0,
                   course_map: dict | None = None) -> int:
    """Elo・最高SP の2指標＋印（⭐◎○▲△・✅）＋オッズ（＋着順）を全頭表示する。

    印もモデル確率も使わず、_ability_marks の印だけを出す。
    並びは Elo(相手込み実力)の高い順。Eloが無い場合は下に回す。
    ✅は指標ごと（Eloの✅／最高SPの✅）に、その指標値の右に付く。
    """
    has_fin = "finish" in sub.columns and pd.to_numeric(
        sub["finish"], errors="coerce").notna().any()
    fin_h = f"{'着順':>5}" if has_fin else ""
    for rid, g in sub.groupby("race_id"):
        g = _ability_marks(g)
        g = g.sort_values("elo_before", ascending=False, na_position="last")
        if top:
            g = g.head(top)
        print(f"\n=== {rid}  {names.get(rid, '')} ===")
        cline = _course_line((course_map or {}).get(rid, {}))
        if cline:
            print(cline)
        print(f"{'馬番':>3} {'馬名':<12}{'Elo':>6}{'印':<4}"
              f"{'最高SP':>7}{'印':<4}{'オッズ':>7}{fin_h}")
        for _, r in g.iterrows():
            elo = _f(r.get("elo_before"), "5.0f")
            elom = (r.get("elo_before_mark") or "") + (r.get("elo_before_val") or "")
            sp = _f(r.get("best_speed_prior"), "5.1f")
            spm = (r.get("best_speed_prior_mark") or "") + (r.get("best_speed_prior_val") or "")
            odds = _f(r.get("odds"), "6.1f")
            fin = ""
            if has_fin:
                fv = pd.to_numeric(pd.Series([r.get("finish")]), errors="coerce").iloc[0]
                fin = f"{int(fv):>5}" if pd.notna(fv) else f"{'-':>5}"
            print(f"{int(r['horse_number']):>3} {str(r['horse_name'])[:12]:<12}"
                  f"{elo:>6}{elom:<4}{sp:>7}{spm:<4}{odds:>7}{fin}")
    print("\n※2指標のみ。Elo=相手込みの総合実力(初期1500)、"
          "最高SP=スピード指数の自己最高(天井)。すべて当該レース前の値。並びはElo順。")
    print("※各指標で ⭐=特出(z≧1.5) / ◎○▲△=非特出の1〜4番手"
          "(5番手以降も4番手と僅差なら△)。")
    print("※✅=その指標の強さに対しオッズが割に合う妙味（指標ごと・期待値≧1.5）。")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="全頭診断")
    p.add_argument("--db", default="keiba.db")
    p.add_argument("--model", default="model_win_noodds.pkl")
    p.add_argument("--show-model", help="複勝率も表示する場合の複勝モデル（例: model_show_noodds.pkl）")
    p.add_argument("--mark-by", choices=["show", "win"], default="show",
                   help="全頭診断(オッズ無)の強さ順の基準: show=複勝率(既定), win=勝率")
    p.add_argument("--race-id", help="このレースだけ")
    p.add_argument("--date", help="この開催日の全レース (YYYY-MM-DD)")
    p.add_argument("--from", dest="date_from", help="開始日 (YYYY-MM-DD)")
    p.add_argument("--to", dest="date_to", help="終了日 (YYYY-MM-DD)")
    p.add_argument("--grade", help="グレードで絞る。'重賞'=G1/G2/G3、'G1'等の個別やカンマ区切りも可")
    p.add_argument("--top", type=int, default=0, help="上位何頭まで表示（0=全頭）")
    p.add_argument("--mark-min-runs", type=int, default=2,
                   help="最終結論で印(妙味馬)を付ける最低出走回数（能力未知馬を除外）")
    p.add_argument("--mark-max-odds", type=float, default=20.0,
                   help="◎の単勝オッズ上限（較正重視で堅め。大穴の過大評価を除外）")
    p.add_argument("--sub-max-odds", type=float, default=30.0,
                   help="○▲△の単勝オッズ上限（既定=◎と同じ。広げると穴の妙味を拾う）")
    p.add_argument("--hon-mode", choices=["strong", "value"], default="strong",
                   help="◎の選び方: strong=強い馬(人気すぎ除外/既定), value=妙味(穴)")
    p.add_argument("--hon-by", choices=["win", "show"], default="win",
                   help="◎の選定指標: win=単勝確率(既定), show=複勝確率(=馬券圏に来やすさで的中重視)")
    p.add_argument("--hon-min-odds", type=float, default=1.0,
                   help="◎(strong時)の単勝オッズ下限。人気を背負いすぎた本命を除外")
    p.add_argument("--weights", help='能力重視リウェイトの比率。例 '
                   '"ability=0.22,aptitude=0.22,bias=0.04,pace=0.04,jockey=0.04"')
    p.add_argument("--lean", type=float, default=0.0,
                   help="リウェイトの強さ倍率。既定0=較正優先（リウェイトなし）, 1で適用")
    p.add_argument("--same-boost", type=float, default=0.0,
                   help="同条件スペシャリスト・ブーストの強さ（0=無効。0.5〜1.0で①同条件の実績を上乗せ）")
    p.add_argument("--ability-only", action="store_true",
                   help="印やモデル確率を出さず、能力3指標(最高SP・好走時平均SP・Elo)と"
                        "オッズだけを表示。各指標で出走馬中の上位馬に★/＋を付ける")
    args = p.parse_args(argv)

    weights = parse_weights(args.weights)
    weights = {k: v * args.lean for k, v in weights.items()}
    use_tilt = any(abs(v) > 1e-9 for v in weights.values())

    # 能力指標だけを見るモードはモデル不要（v_features の素の列だけで表示）
    bundle = None if args.ability_only else mlcommon.load_model(args.model)
    show_bundle = mlcommon.load_model(args.show_model) \
        if (args.show_model and not args.ability_only) else None
    df = mlcommon.load_data(args.db, None)

    conn = sqlite3.connect(args.db)
    if args.race_id:
        ids = [args.race_id]
    elif args.date:
        ids = [r[0] for r in conn.execute(
            "SELECT race_id FROM races WHERE race_date=? ORDER BY race_id", (args.date,))]
    elif args.grade or args.date_from or args.date_to:
        # グレード/期間で絞る（過去レースの印・回顧用）。'重賞'=G1/G2/G3
        where = []; params = []
        if args.grade:
            roman = {"G1": "GⅠ", "G2": "GⅡ", "G3": "GⅢ"}
            toks = ["G1", "G2", "G3"] if args.grade == "重賞" \
                else [t.strip().upper() for t in args.grade.split(",") if t.strip()]
            gset = []
            for t in toks:
                gset.append(t)
                if t in roman:
                    gset.append(roman[t])
            where.append("grade IN (%s)" % ",".join("?" * len(gset))); params += gset
        if args.date_from:
            where.append("race_date >= ?"); params.append(args.date_from)
        if args.date_to:
            where.append("race_date <= ?"); params.append(args.date_to)
        sql = "SELECT race_id FROM races" + \
              (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY race_id"
        ids = [r[0] for r in conn.execute(sql, params)]
    else:
        ids = mlcommon.upcoming_race_ids(args.db)
    names = dict(conn.execute(
        "SELECT race_id, race_name FROM races").fetchall()) if ids else {}
    conn.close()

    sub = df[df["race_id"].isin(ids)].copy()
    if sub.empty:
        print("対象レースがありません（出馬表を取り込みましたか? 日付指定は合っていますか?）")
        return 0

    # --- 能力指標のみの一覧表示（印・モデル確率なし） -----------------------
    #   3指標それぞれ、出走馬中で最高=★ / 平均より明確に上(z≧1)=＋ をチェック。
    #   回顧用に、結果が確定していれば着順も結合して表示する（未確定は空欄）。
    if args.ability_only:
        conn = sqlite3.connect(args.db)
        fin = pd.read_sql_query(
            "SELECT race_id, horse_id, finish_position AS finish FROM results "
            "WHERE finish_position IS NOT NULL", conn)
        conn.close()
        sub = sub.merge(fin, on=["race_id", "horse_id"], how="left")
        course_map = _load_course_map(args.db, sub["race_id"].unique().tolist())
        return _ability_table(sub, names, args.top, course_map)

    # モデル予測（card と evaluate で共通の採点）
    sub = compute_scores(sub, bundle, show_bundle, weights, same_boost=args.same_boost)

    # オッズがあれば「最終結論（妙味で印）」、無ければ「全頭診断（強さ順で印）」
    has_odds = pd.to_numeric(sub.get("odds"), errors="coerce").notna().any()
    strength_col = "show_p" if (args.mark_by == "show" and show_bundle) else "p"

    # ★/〔評価〕用の z（調整後の p/show_p/ev と、表示する Elo/最高SP）
    fit_cols = ["hill_show_rate_prior", "io_show_rate_prior", "turn_show_rate_prior",
                "trans_show_rate_prior", "slowp_show_rate_prior", "highp_show_rate_prior",
                "shun_show_rate_prior", "mochi_show_rate_prior"]
    _add_race_z(sub, ["p", "show_p", "ev", "elo_before",
                      "best_speed_prior", "good_avg_speed_prior"]
                + [c for c, *_ in STRENGTH] + fit_cols)

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

    HILL_JP = {"steep": "急坂", "mild": "坂"}        # 平坦は省略（坂のある所だけ強調）
    IO_JP = {"inner": "内回り", "outer": "外回り"}
    TURN_JP = {"tight": "小回り"}

    def _apt(r, col, gate, minr, label):  # 適性タグ: 他馬比較で得意◎/苦手▼
        if col not in sub.columns or pd.isna(r.get(col)) or (r.get(gate) or 0) < minr:
            return None
        z = r.get(col + "_z", 0)
        return f"{label}◎" if z >= 1.0 else (f"{label}▼" if z <= -1.0 else None)

    def course_fit(r):  # 今日のコース/ペースへの適性（坂・内外・小回り・遠征・流れ）
        tags = []
        if r.get("hill_type") in HILL_JP:
            tags.append(_apt(r, "hill_show_rate_prior", "hill_runs_prior", 2, HILL_JP[r["hill_type"]]))
        if r.get("io_type") in IO_JP:
            tags.append(_apt(r, "io_show_rate_prior", "io_runs_prior", 2, IO_JP[r["io_type"]]))
        if r.get("turn_type") in TURN_JP:
            tags.append(_apt(r, "turn_show_rate_prior", "turn_runs_prior", 2, TURN_JP[r["turn_type"]]))
        if (r.get("is_transport") or 0) == 1:
            tags.append(_apt(r, "trans_show_rate_prior", "trans_runs_prior", 2, "遠征"))
        for col, gate, lab in [("slowp_show_rate_prior", "slowp_runs_prior", "スロー巧者"),
                               ("highp_show_rate_prior", "highp_runs_prior", "ハイ巧者"),
                               ("shun_show_rate_prior", "shun_runs_prior", "瞬発"),
                               ("mochi_show_rate_prior", "mochi_runs_prior", "持続")]:
            t = _apt(r, col, gate, 2, lab)
            if t and t.endswith("◎"):     # 流れ/ペースは"得意"だけ拾う
                tags.append(t)
        return [t for t in tags if t][:4]

    fuku_h = f"{'複勝率':>7}" if show_bundle else ""
    odds_h = f"{'オッズ':>6}{'EV':>7}" if has_odds else ""   # オッズあり最終結論時のみ
    for rid, g in sub.groupby("race_id"):
        g = assign_marks(g, value_mode=has_odds, strength_col=strength_col,
                         min_runs=args.mark_min_runs,
                         max_odds=args.mark_max_odds,
                         sub_max_odds=args.sub_max_odds,
                         hon_mode=args.hon_mode,
                         hon_min_odds=args.hon_min_odds,
                         hon_by=args.hon_by).reset_index(drop=True)
        if args.top:
            g = g.head(args.top)
        print(f"\n=== {rid}  {names.get(rid, '')} ===")
        print(f"{'印':<2}{'馬番':>3} {'馬名':<12}{'勝率':>7}{fuku_h} "
              f"{'最高SP':>7}{'好走平均':>8}{'Elo':>7}{'斤量':>6}{odds_h}")
        for _, r in g.iterrows():
            mark = r["mark"] or "  "
            wr = _f(r['p'] * 100, '5.1f') + "%" + st(r, 'p')
            fuku = (" " + _f(r.get('show_p') * 100, '5.1f') + "%" + st(r, 'show_p')) if show_bundle else ""
            sp = _f(r.get('best_speed_prior'), '5.1f') + st(r, 'best_speed_prior')
            gsp = _f(r.get('good_avg_speed_prior'), '5.1f') + st(r, 'good_avg_speed_prior')  # 好走時平均SP
            elo = _f(r.get('elo_before'), '5.0f') + st(r, 'elo_before')                      # Eloレーティング
            kg = _f(r.get('weight_carried'), '4.1f')   # 斤量
            odds = (" " + _f(r.get('odds'), '5.1f')
                    + " " + _f(r.get('ev'), '5.2f') + st(r, 'ev')) if has_odds else ""
            print(f"{mark:<2}{int(r['horse_number']):>3} {str(r['horse_name'])[:12]:<12}"
                  f"{wr:>7}{fuku} {sp:>7}{gsp:>8}{elo:>7}{kg:>6}{odds}")
        # 〔評価〕印を付けた馬の目立つ強み/弱み
        lines = []
        for _, r in g[g["mark"] != ""].head(5).iterrows():
            tags = hyoten(r)
            if tags:
                lines.append(f"{r['mark']}{str(r['horse_name'])[:7]}: {'・'.join(tags)}")
        if lines:
            print("  〔評価〕 " + " ｜ ".join(lines))
        # 〔適性〕今日のコース・ペースへの得意/苦手（坂/内外/小回り/遠征/流れ）
        flines = []
        for _, r in g[g["mark"] != ""].head(5).iterrows():
            ft = course_fit(r)
            if ft:
                flines.append(f"{r['mark']}{str(r['horse_name'])[:7]}: {'・'.join(ft)}")
        if flines:
            print("  〔適性〕 " + " ｜ ".join(flines))
        # 〔穴〕無印だが天井(最高SP)が出走馬中で突出＝モデルが軽視するスペシャリスト候補
        ana = []
        for _, r in g[g["mark"] == ""].iterrows():
            if r.get("best_speed_prior_z", 0) >= 1.0:
                ft = course_fit(r)
                sp = _f(r.get("best_speed_prior"), '.0f')
                tag = "・" + "・".join(ft) if ft else ""
                ana.append(f"{int(r['horse_number'])}{str(r['horse_name'])[:7]}(天井SP{sp}{tag})")
        if ana:
            print("  〔穴〕 " + " ｜ ".join(ana[:4]))
        # 〔市場〕オッズ最上位＝市場の本命。1番人気は複勝率66%と堅実なので、
        #   モデルが軽視(複勝率が場内6位以下)していても無視しないための安全弁。
        if has_odds:
            od = pd.to_numeric(g.get("odds"), errors="coerce")
            favs = g[od.notna()].assign(_o=od).nsmallest(2, "_o")
            mline = []
            for _, r in favs.iterrows():
                sp = r.get("show_p")
                fu = f"/複{_f(sp * 100, '.0f')}%" if show_bundle and pd.notna(sp) else ""
                flag = ""
                if show_bundle and pd.notna(sp):
                    rank = int((pd.to_numeric(g["show_p"], errors="coerce") > sp).sum()) + 1
                    if rank > 5:
                        flag = "（モデル軽視・割引注意）"
                mline.append(f"{int(r['horse_number'])}{str(r['horse_name'])[:7]}"
                             f"({_f(r.get('odds'), '.1f')}倍{fu}){flag}")
            if mline:
                print("  〔市場〕 " + " ｜ ".join(mline))

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
    print("※能力指標: 最高SP=スピード指数の自己最高(天井)、"
          "好走平均=好走時(3着内)だけの平均SP(凡走無視の安定実力)、"
          "Elo=相手関係込みの総合実力(初期1500)。いずれも当該レース前の値。")
    print(f"※{rw}。馬券は自己責任で。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
