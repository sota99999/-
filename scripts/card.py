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
        q = ("SELECT vc.race_id, vc.venue_id, vc.surface, vc.distance, vc.course_type, "
             "vc.turn_dir, vc.straight_m, vc.hill_m, vc.hill_grade, vc.turn_size, "
             "vc.turf_type, vc.pace_bias, ra.track_condition "
             "FROM v_course vc JOIN races ra ON ra.race_id = vc.race_id "
             "WHERE vc.race_id IN (%s)" % ",".join("?" * len(ids)))
        rows = conn.execute(q, list(ids)).fetchall()
        conn.close()
    except Exception:  # noqa: BLE001  ビュー未作成など
        return {}
    cols = ["venue_id", "surface", "distance", "course_type", "turn_dir",
            "straight_m", "hill_m", "hill_grade", "turn_size", "turf_type",
            "pace_bias", "track_condition"]
    return {r[0]: dict(zip(cols, r[1:])) for r in rows}


# レース条件→参考表示する能力・適性（列名, 見出し, 数値書式, 低いほど良か）
ABILITY_META = {
    "toppspeed":    ("ﾄｯﾌﾟ", "5.1f", False),
    "shunpatsu":    ("瞬発", "4.1f", True),    # 上がり3F 低いほど鋭い
    "jizoku":       ("持続", "5.1f", False),
    "power":        ("ﾊﾟﾜｰ", "5.1f", False),
    "stamina":      ("ｽﾀﾐﾅ", "5.1f", False),
    "start":        ("先行", "4.2f", True),    # 隊列 低いほど前
    "stamina_show": ("距離", "4.2f", False),   # 中長距離 複勝率
    "off_show":     ("道悪", "4.2f", False),   # 道悪 複勝率
}


def _relevant_abilities(c: dict, distance, going) -> list:
    """レースのコース形態・距離・馬場から、参考表示する能力/適性キー列を選ぶ。"""
    out: list[str] = []
    st = c.get("straight_m") or 0
    hill = c.get("hill_grade")
    turn = c.get("turn_size")
    if st >= 460:                       # 長い直線(東京・新潟外・阪神外) → キレ
        out += ["toppspeed", "shunpatsu"]
    if hill == "steep":                 # 急坂(中山・阪神内・中京) → パワー
        out += ["power"]
    if turn == "tight":                 # 小回り(函館・札幌・小倉・福島 等) → 先行力
        out += ["start"]
    if hill == "flat" and turn == "wide":  # 平坦広い(京都外) → ロングスパート持続
        out += ["jizoku"]
    if distance and distance >= 2200:   # 長距離 → スタミナ・距離適性
        out += ["stamina", "stamina_show"]
    if going and going != "良":          # 道悪 → 道悪適性
        out += ["off_show"]
    seen = set(); ded = []
    for k in out:
        if k not in seen:
            seen.add(k); ded.append(k)
    return ded


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


SP_COLS = {"ceiling": "best_speed_prior", "good": "good_avg_speed_prior"}
SP_LABEL = {"best_speed_prior": "最高SP", "good_avg_speed_prior": "好走平均"}

# 新・能力2指標（融合せず併記）:
#   主軸 = 相対R(r_before): 反復strength-of-scheduleレーティング。誰に勝ったか＝格。
#          複勝/BOXで全クラス優位。無ければ従来の Elo(elo_before) にフォールバック。
#   相補 = 能力(power_top): 展開・斤量補正＋経験的ベイズ縮約の総合実力(時計)。
#          未勝利・重賞の単勝1点で優位。無ければ従来の最高SP にフォールバック。
# 内部キーは既存の "elo"/"sp"（＝主軸/相補）を流用し、9分類・列名の互換を保つ。


def _has(g: pd.DataFrame, col: str) -> bool:
    """列が存在し数値が1つでも入っているか。"""
    return col in g.columns and pd.to_numeric(g[col], errors="coerce").notna().any()


def _prim_col(g: pd.DataFrame) -> str:
    """主軸列を解決（相対R優先、無ければElo）。"""
    return "r_before" if _has(g, "r_before") else "elo_before"


def _sec_col(g: pd.DataFrame, sp_col: str = "best_speed_prior") -> str:
    """相補列を解決（能力power_top優先、無ければ指定SP列）。"""
    return "power_top" if _has(g, "power_top") else sp_col


def _ind_label(col: str) -> str:
    """指標列→見出し。"""
    return {"r_before": "相対R", "elo_before": "Elo",
            "power_top": "能力"}.get(col, SP_LABEL.get(col, "SP"))


def _ability_marks(g: pd.DataFrame, sp_col: str = "best_speed_prior") -> pd.DataFrame:
    """主軸(相対R/Elo)・相補(能力/SP) の2指標で、レース内に印を付けて返す。

    印は正規化した列に入れる:  elo_mark/elo_val/elo_z（主軸）、sp_mark/sp_val/sp_z（相補）。
    列名キーは互換のため elo/sp のままだが、中身は相対R/能力（無ければElo/SP）。
      ⭐ = 特出（出走馬中 z≧1.5）／◎○▲△ = 特出を除く上位1〜4番手
      △(追加) = 5番手以降でも4番手と僅差（z差≦0.25）なら△
    オッズがあれば、各指標の強さに対しオッズが割に合う馬へ ✅。いずれも高いほど良。
    """
    g = g.copy()
    odds = (pd.to_numeric(g["odds"], errors="coerce") if "odds" in g.columns
            else pd.Series(np.nan, index=g.index))
    for key, col in (("elo", _prim_col(g)), ("sp", _sec_col(g, sp_col))):
        mark = pd.Series("", index=g.index)
        if col not in g.columns:            # 解決先の列が無い（全NaNでフォールバック時等）→ 印なし
            g[key + "_z"] = np.nan
            g[key + "_mark"] = mark
            g[key + "_val"] = ""
            g[key + "_ev"] = np.nan
            continue
        v = pd.to_numeric(g[col], errors="coerce")
        valid = v.notna()
        sd = v[valid].std(ddof=0) if valid.sum() >= 2 else 0.0
        if valid.sum() >= 2 and sd and sd > 0:
            z = (v - v[valid].mean()) / sd
        else:
            z = pd.Series(np.nan, index=g.index)   # 差が無い（新馬の全1500等）は印なし
        g[key + "_z"] = z
        if z.notna().any():
            standout = valid & (z >= STANDOUT_Z)
            mark[standout] = "⭐"
            n_star = int(standout.sum())   # ⭐が上位枠を埋め、◎○▲△はその後ろから
            rest = sorted(g.index[valid & ~standout], key=lambda i: v[i], reverse=True)
            rank4_z = None
            for k, idx in enumerate(rest):
                pos = k + n_star           # ⭐1頭→○から / ⭐2頭→▲から
                if pos < 4:
                    mark[idx] = RANK_MARKS[pos]
                    if pos == 3:
                        rank4_z = z[idx]
                elif rank4_z is not None and (rank4_z - z[idx]) <= NEARTIE_Z:
                    mark[idx] = "△"
                else:
                    break
        g[key + "_mark"] = mark

        # ✅ オッズ妙味（指標ごと）: その指標の強さ z→疑似確率(softmax)×オッズ が割に合う馬
        val = pd.Series("", index=g.index)
        g[key + "_ev"] = np.nan
        if odds.notna().any() and z.notna().any():
            e = np.exp(z.fillna(z[z.notna()].min()).clip(-3, 3))
            p = e / e.sum() if e.sum() > 0 else e
            ev = p * odds
            g[key + "_ev"] = ev
            val[(ev >= VALUE_EV) & (z >= VALUE_MINZ) & odds.notna()] = "✅"
        g[key + "_val"] = val
    return g


def _load_payouts(db: str, race_ids: list) -> dict:
    """{race_id: {'単勝': {馬番: 払戻円}, '複勝': {馬番: 払戻円}}} を返す（回収率用）。"""
    if not race_ids:
        return {}
    con = sqlite3.connect(db)
    out: dict = {}
    try:
        rows = con.execute(
            "SELECT race_id, bet_type, combination, payout FROM payouts "
            "WHERE bet_type IN ('単勝','複勝')").fetchall()
    except sqlite3.OperationalError:
        rows = []
    con.close()
    ids = set(race_ids)
    for rid, bt, combo, pay in rows:
        if rid not in ids or pay is None:
            continue
        try:
            num = int(str(combo).strip())
        except ValueError:
            continue
        out.setdefault(rid, {}).setdefault(bt, {})[num] = pay
    return out


def _print_review_summary(records: list, payouts: dict) -> None:
    """回顧の全レースを通した印別まとめ（単勝/複勝の的中率・回収率）を表示する。

    回収率は「オッズ」ではなく確定払戻(payoutsテーブル)から算出する（実額ベース）。
    1頭の印につき100円ずつ賭けた前提。本命=各レースの相対R最上位の印。
    """
    if not records:
        return
    payouts = payouts or {}

    def win_ret(rec):   # 単勝払戻: 1着なら払戻円、無ければオッズ×100で代替
        if rec["finish"] != 1:
            return 0
        p = payouts.get(rec["race_id"], {}).get("単勝", {})
        return p.get(rec["num"], int((rec["odds"] or 0) * 100))

    def place_ret(rec):  # 複勝払戻: 対象(2〜3着)なら払戻円、外は0
        return payouts.get(rec["race_id"], {}).get("複勝", {}).get(rec["num"], 0)

    def placed(rec):     # 複勝的中か（払戻対象に馬番があるか）
        return rec["num"] in payouts.get(rec["race_id"], {}).get("複勝", {})

    buckets = [("本命", lambda r: r["hon"]),
               ("⭐", lambda r: r["mark"] == "⭐"),
               ("◎", lambda r: r["mark"] == "◎"),
               ("○", lambda r: r["mark"] == "○"),
               ("▲", lambda r: r["mark"] == "▲"),
               ("△", lambda r: r["mark"] == "△"),
               ("全印", lambda r: True)]
    nrace = len({r["race_id"] for r in records})
    print(f"\n===== 回顧まとめ（{nrace}レース・印別 的中率／回収率）=====")
    print(f"{'印':<5}{'点数':>5}{'単勝的中':>10}{'複勝的中':>10}{'単回収':>8}{'複回収':>8}")
    for label, f in buckets:
        rs = [r for r in records if f(r)]
        n = len(rs)
        if not n:
            continue
        win = sum(1 for r in rs if r["finish"] == 1)
        plc = sum(1 for r in rs if placed(r))
        roi_w = sum(win_ret(r) for r in rs) / (n * 100) * 100
        roi_p = sum(place_ret(r) for r in rs) / (n * 100) * 100
        print(f"{label:<5}{n:>5}{f'{win}/{n} {win/n*100:.0f}%':>11}"
              f"{f'{plc}/{n} {plc/n*100:.0f}%':>11}{roi_w:>7.0f}%{roi_p:>7.0f}%")
    print("※点数=印が付いた延べ頭数(1頭100円換算)。回収率=払戻総額÷投資額(確定払戻ベース)。")
    print("※本命=各レースの相対R最上位の印(⭐が居れば⭐/居なければ◎)。新馬など印なしのレースは除外。")


def _ability_table(sub: pd.DataFrame, names: dict, top: int = 0,
                   course_map: dict | None = None,
                   sp_col: str = "best_speed_prior",
                   payouts: dict | None = None) -> int:
    """Elo・SP の2指標＋印（⭐◎○▲△・✅）＋オッズ（＋着順）を全頭表示する。

    SP指標は sp_col で切替（最高SP=天井 / 好走平均SP）。印は _ability_marks の
    正規化列（elo_*/sp_*）を使う。並びは Elo(相手込み実力)の高い順。
    """
    has_fin = "finish" in sub.columns and pd.to_numeric(
        sub["finish"], errors="coerce").notna().any()
    fin_h = f"{'着順':>5}" if has_fin else ""
    prim = _prim_col(sub)                    # 主軸(相対R or Elo)
    sec = _sec_col(sub, sp_col)              # 相補(能力 or SP)
    plab, slab = _ind_label(prim), _ind_label(sec)
    pfmt = "6.1f" if prim == "r_before" else "5.0f"
    has_pmax = _has(sub, "pmax")            # 最高perf(天井)の参考列
    pmax_h = f"{'最高':>7}" if has_pmax else ""
    hit_h = f"{'的中':>4}" if has_fin else ""   # 回顧: 単勝/複勝 的中馬の印
    records: list = []                            # 回顧まとめ用（印×着順×払戻）
    for rid, g in sub.groupby("race_id"):
        g = _ability_marks(g, sp_col)
        g = g.sort_values(prim, ascending=False, na_position="last")
        honmei_taken = False                      # 各レースの相対R最上位の印=本命
        if top:
            g = g.head(top)
        cinfo = (course_map or {}).get(rid, {})
        rel = _relevant_abilities(cinfo, cinfo.get("distance"),
                                  cinfo.get("track_condition")) if cinfo else []
        # 人気=レース内のオッズ昇順順位
        onum = pd.to_numeric(g.get("odds"), errors="coerce")
        ninki = onum.rank(method="min")
        print(f"\n=== {rid}  {names.get(rid, '')} ===")
        cline = _course_line(cinfo)
        if cline:
            print(cline)
        abil_h = "".join(f"{ABILITY_META[k][0]:>6}" for k in rel)
        print(f"{'馬番':>3} {'馬名':<12}{plab:>7}{'印':<5}{pmax_h}"
              f"{slab + '(参考)':>10}{abil_h}{'オッズ':>7}{'人気':>4}{fin_h}{hit_h}")
        for idx, r in g.iterrows():
            pv = _f(r.get(prim), pfmt)
            pm = ("条" if r.get("mura") else "") \
                + (r.get("elo_mark") or "") + (r.get("elo_val") or "")
            mx = f"{_f(r.get('pmax'), '6.1f'):>7}" if has_pmax else ""
            sv = _f(r.get(sec), "5.1f") + ((r.get("sp_mark") or "") == "⭐" and "⭐" or "")
            av = "".join(f"{_f(r.get(k), ABILITY_META[k][1]):>6}" for k in rel)
            odds = _f(r.get("odds"), "6.1f")
            nk = ninki.get(idx)
            nks = f"{int(nk):>3}人" if pd.notna(nk) else f"{'-':>4}"
            fin = ""; hit = ""
            if has_fin:
                fv = pd.to_numeric(pd.Series([r.get("finish")]), errors="coerce").iloc[0]
                fin = f"{int(fv):>5}" if pd.notna(fv) else f"{'-':>5}"
                marked = bool(r.get("elo_mark"))   # ⭐◎○▲△ が付いた馬か
                if pd.notna(fv) and marked and fv <= 3:
                    hit = f"{'◎単' if fv == 1 else '○複':>4}"
                else:
                    hit = f"{'':>4}"
                if marked:
                    od = pd.to_numeric(pd.Series([r.get("odds")]),
                                       errors="coerce").iloc[0]
                    records.append({
                        "race_id": rid, "num": int(r["horse_number"]),
                        "mark": r.get("elo_mark"),
                        "finish": int(fv) if pd.notna(fv) else None,
                        "odds": float(od) if pd.notna(od) else None,
                        "hon": not honmei_taken,
                    })
                    honmei_taken = True
            print(f"{int(r['horse_number']):>3} {str(r['horse_name'])[:12]:<12}"
                  f"{pv:>7}{pm:<5}{mx}{sv:>10}{av}{odds:>7}{nks}{fin}{hit}")
    pdesc = ("反復レーティング＝どれくらい強い相手に勝ったか(格・対戦網)"
             if prim == "r_before" else "相手込みの総合実力(初期1500)")
    sdesc = ("展開・斤量補正＋縮約の総合実力(時計)。参考"
             if sec == "power_top"
             else ("スピード指数の自己最高(天井)" if sec == "best_speed_prior"
                   else "好走時(3着内)だけの平均SP"))
    print(f"\n※主指標は{plab}のみ（着差・時計は順位より弱く一本化）。{plab}={pdesc}。")
    if has_pmax:
        print("※最高=その馬が過去に出した実力点perfの最高値(天井)。相対Rが低くても最高が高い"
              "馬は条件が向けば一発ある(参考・スコア非加算)。")
    print("※印「条」=ムラ馬(実力点のばらつき大)を、今回条件に合致した過去走だけで再評価した値。")
    print(f"※{slab}は参考列（{sdesc}）。⭐=時計が特出。相対Rが高く時計も⭐なら妙味。")
    print(f"※{plab}で ⭐=特出(z≧1.5)＝最上位枠。◎○▲△は⭐の後ろから順に"
          "(⭐1頭なら○から/2頭なら▲から。5番手以降も僅差なら△) / "
          "✅=オッズ妙味(期待値≧1.5)。並びは{}順。".format(plab))
    print("※ﾄｯﾌﾟ/瞬発/持続/ﾊﾟﾜｰ/ｽﾀﾐﾅ=好走時の実力値(quality)、距離/道悪=複勝率、"
          "先行=平均隊列。※瞬発(上がり3F)と先行(隊列)は小さいほど良、他は大きいほど良。"
          "そのレースに効く能力・適性だけを表示(参考・スコア非加算)。")
    if has_fin:
        print("※的中: 印(⭐◎○▲△)が付いた馬が馬券圏に来た時だけ表示。◎単=その印が1着 / "
              "○複=2・3着。無印馬が来ても付かない。人気=最終オッズ順。")
        _print_review_summary(records, payouts)
    else:
        print("※人気=現在のオッズ順。")
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
                   help="印やモデル確率を出さず、能力2指標(SP・Elo)とオッズだけを表示")
    p.add_argument("--sp-col", choices=["ceiling", "good"], default="ceiling",
                   help="SP指標: ceiling=最高SP(天井,既定) / good=好走平均SP(まぐれ天井を除く)")
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
            "SELECT race_id, horse_id, finish_position AS finish, popularity FROM results "
            "WHERE finish_position IS NOT NULL", conn)
        # 相対R(r_before/pmax)は当該レースに直接付与済み(出馬表含む)→直接join
        try:
            rr = pd.read_sql_query(
                "SELECT race_id, horse_id, r_before, pmax FROM horse_relative_r", conn)
            sub = sub.merge(rr, on=["race_id", "horse_id"], how="left")
        except Exception:  # noqa: BLE001
            pass
        # 6能力/power_top は v_ability に「結果のあるレース」しか行が無い。
        # 各馬の「対象レース日以前で最新」の行を merge_asof で付与:
        #   予想(未来レース)→直近の能力プロファイル / 回顧(過去)→as-of行でリーク無し。
        acols = ["power_top", "shunpatsu", "jizoku", "toppspeed", "power",
                 "stamina", "start", "stamina_show", "off_show"]
        try:
            va = pd.read_sql_query(
                "SELECT horse_id, race_date AS _vd, " + ", ".join(acols)
                + " FROM v_ability", conn)
            rd = pd.read_sql_query("SELECT race_id, race_date AS _td FROM races", conn)
            sub = sub.merge(rd, on="race_id", how="left")
            va["_vd"] = pd.to_datetime(va["_vd"], errors="coerce")
            sub["_td"] = pd.to_datetime(sub["_td"], errors="coerce")
            va = va.dropna(subset=["_vd"]).sort_values("_vd")
            sub = sub.sort_values("_td")
            sub = pd.merge_asof(sub, va, left_on="_td", right_on="_vd",
                                by="horse_id", direction="backward")
            sub = sub.drop(columns=["_td", "_vd"], errors="ignore")
        except Exception:  # noqa: BLE001  v_ability未作成なら従来指標にフォールバック
            pass
        conn.close()
        sub = sub.merge(fin, on=["race_id", "horse_id"], how="left")
        # ムラ馬だけ条件補正: 相対R(r_before) を条件評価 r_adj に差し替え、印「条」を付ける
        try:
            import cond_rating
            adj = cond_rating.adjusted(args.db, sub["race_id"].unique().tolist())
            if not adj.empty:
                sub = sub.merge(adj[["race_id", "horse_id", "r_adj", "is_mura"]],
                                on=["race_id", "horse_id"], how="left")
                use = sub["r_adj"].notna()
                sub.loc[use, "r_before"] = sub.loc[use, "r_adj"]
                sub["mura"] = sub.get("is_mura").fillna(False) & use
        except Exception:  # noqa: BLE001  モジュール未配置・perf/pstd未計算なら通常のr_before
            pass
        course_map = _load_course_map(args.db, sub["race_id"].unique().tolist())
        payouts = _load_payouts(args.db, sub["race_id"].unique().tolist())
        return _ability_table(sub, names, args.top, course_map,
                              SP_COLS.get(args.sp_col, "best_speed_prior"), payouts)

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
