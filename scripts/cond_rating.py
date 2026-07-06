#!/usr/bin/env python3
# =============================================================================
# 条件補正レーティング（ムラ馬だけ、今回条件に合致した過去走で評価）
#
# 相対R(r_before)は堅実馬には強いが、特定条件でのみ本領を発揮する「ムラ馬」は
# 条件外の大敗に評価を引きずられる（例: メイショウタバルの有馬13着）。
# そこで pstd(実力点のばらつき=ムラ度)が閾値以上の馬に限り、今回のレース条件
# （芝ダ・坂・回り・距離±tol）に合致した過去走の perf 平均で置き換える。
#   ・ムラ度が小さい堅実馬は r_before のまま（集計を壊さない）
#   ・合致走が無いムラ馬も r_before のまま
# 使い方（card.py から import）:
#   from cond_rating import adjusted
#   df = adjusted("keiba.db", race_ids, thresh=22.0)
# 前提: compute_relative_r.py 適用済み（horse_relative_r に perf/pstd 有り）。
# =============================================================================
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

THRESH = 22.0     # ムラ度(pstd)がこれ以上の馬だけ条件評価に切替
DIST_TOL = 300    # 距離が今回とこれ以内なら同条件とみなす(m)


def adjusted(db: str, race_ids, thresh: float = THRESH,
             tol: int = DIST_TOL) -> pd.DataFrame:
    """race_ids 各出走馬の条件補正レーティング r_adj を返す。

    返り値: race_id, horse_id, r_before, pstd, cm, is_mura, r_adj
    r_adj = 条件評価cm（ムラ度≥thresh かつ合致走あり）/ それ以外は r_before。
    """
    ids = list(dict.fromkeys(race_ids))
    if not ids:
        return pd.DataFrame(columns=["race_id", "horse_id", "r_before",
                                     "pstd", "cm", "is_mura", "r_adj"])
    con = sqlite3.connect(db)
    ph = ",".join("?" * len(ids))
    tgt = pd.read_sql(
        f"""SELECT ra.race_id, ra.race_date td, ra.distance tdist,
               CASE WHEN ra.surface='芝' THEN 0 ELSE 1 END tdirt,
               vc.hill_grade thill, vc.turn_size tturn
            FROM races ra LEFT JOIN v_course vc ON vc.race_id = ra.race_id
            WHERE ra.race_id IN ({ph})""", con, params=ids)
    ent = pd.read_sql(
        f"""SELECT race_id, horse_id, r_before, pstd
            FROM horse_relative_r WHERE race_id IN ({ph})""", con, params=ids)
    if ent.empty:
        con.close()
        ent["cm"] = ent.get("cm"); ent["is_mura"] = False; ent["r_adj"] = ent.get("r_before")
        return ent
    ent = ent.merge(tgt, on="race_id", how="left")
    ent["is_mura"] = pd.to_numeric(ent["pstd"], errors="coerce").fillna(0) >= thresh

    cm_map: dict = {}
    mur = ent[ent["is_mura"]]
    if len(mur):
        hids = list(mur["horse_id"].unique())
        hp = ",".join("?" * len(hids))
        hist = pd.read_sql(
            f"""SELECT hr.horse_id, ra.race_date rd, ra.distance dist,
                   CASE WHEN ra.surface='芝' THEN 0 ELSE 1 END dirt,
                   vc.hill_grade hill, vc.turn_size turn, hr.perf
                FROM horse_relative_r hr JOIN races ra ON ra.race_id = hr.race_id
                LEFT JOIN v_course vc ON vc.race_id = hr.race_id
                WHERE hr.horse_id IN ({hp}) AND hr.perf IS NOT NULL""",
            con, params=hids)
        if len(hist):
            m = mur[["race_id", "horse_id", "td", "tdist",
                     "tdirt", "thill", "tturn"]].merge(hist, on="horse_id", how="left")
            hh = m["hill"].fillna("?"); th = m["thill"].fillna("?")
            tt = m["turn"].fillna("?"); ttt = m["tturn"].fillna("?")
            keep = ((m["rd"] < m["td"]) & (m["dirt"] == m["tdirt"]) & (hh == th)
                    & (tt == ttt) & ((m["dist"] - m["tdist"]).abs() <= tol))
            cm_map = m[keep].groupby(["race_id", "horse_id"])["perf"].mean().to_dict()
    con.close()

    ent["cm"] = [cm_map.get((r, h)) for r, h in zip(ent["race_id"], ent["horse_id"])]
    ent["r_adj"] = np.where(ent["is_mura"] & ent["cm"].notna(),
                            ent["cm"], ent["r_before"])
    return ent[["race_id", "horse_id", "r_before", "pstd", "cm", "is_mura", "r_adj"]]
