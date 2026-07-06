#!/usr/bin/env python3
# =============================================================================
# 静的サイト生成: DBから閲覧用HTML一式を docs/ に生成する
#
# 生成物:
#   docs/index.html            トップ（今週のカードへのリンク・検索・ランキング）
#   docs/cards/<date>.html     開催日ごとの確定版カード（card.py --ability-only）
#   docs/horse/<horse_id>.html 馬プロファイル（horse.py と同内容）
#   docs/ranking.html          相対R / 天井 / 能力 ランキング
#   docs/horses.json           馬名検索用インデックス（クライアントJSで絞込）
#
# 使い方:
#   python scripts/site_build.py --db keiba.db --out docs
#   （生成後にコミット&プッシュし、GitHub Pages のソースを
#     ブランチ + /docs に設定すると公開される）
# =============================================================================
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sqlite3
from pathlib import Path

import card
import horse as horse_mod
import ranking as ranking_mod

STYLE = """
:root{color-scheme:light dark}
body{font-family:sans-serif;margin:0;background:#fafafa;color:#222}
@media (prefers-color-scheme:dark){body{background:#151515;color:#ddd}
 a{color:#8cf} .box{background:#1e1e1e!important;border-color:#333!important}}
header{background:#173f2f;color:#fff;padding:10px 14px;font-weight:bold}
header a{color:#cfe;text-decoration:none;margin-right:14px}
main{padding:12px;max-width:1100px;margin:auto}
pre{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;
    overflow-x:auto;line-height:1.45;background:inherit}
.box{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px 14px;
     margin:10px 0}
input{font-size:16px;padding:8px;width:100%;box-sizing:border-box;
      border:1px solid #aaa;border-radius:6px}
#hits a{display:block;padding:6px 4px;text-decoration:none;border-bottom:1px solid #eee}
h2{font-size:16px;margin:14px 0 6px}
.small{font-size:12px;opacity:.75}
"""

PAGE = """<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{style}</style></head><body>
<header>🐎 競馬DB ｜ <a href="{root}index.html">ホーム</a>
<a href="{root}ranking.html">ランキング</a></header>
<main>{body}</main>
<p class="small" style="padding:0 14px">確定版モデル: 相対R(格)＋ムラ馬条件補正。
馬券は自己責任で。</p></body></html>"""


def _capture(func, argv) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            func(argv)
        except SystemExit:
            pass
        except Exception as e:  # noqa: BLE001  1ページの失敗でビルド全体を止めない
            print(f"(生成エラー: {e})")
    return buf.getvalue()


def build(db: str, out: str, active_days: int, card_days: int) -> int:
    outp = Path(out)
    (outp / "cards").mkdir(parents=True, exist_ok=True)
    (outp / "horse").mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)

    # --- 対象開催日: 直近 card_days 日＋未来(出馬表) --------------------------
    dates = [r[0] for r in con.execute(
        """SELECT DISTINCT race_date FROM races
           WHERE race_date >= (SELECT DATE(MAX(race_date), ?) FROM races
                               WHERE race_id IN (SELECT race_id FROM results
                                                 WHERE finish_position IS NOT NULL))
           ORDER BY race_date""", (f"-{card_days} days",))]

    # --- 開催日カード ---------------------------------------------------------
    for d in dates:
        txt = _capture(card.main, ["--db", db, "--ability-only", "--date", d])
        (outp / "cards" / f"{d}.html").write_text(PAGE.format(
            title=f"{d} カード", style=STYLE, root="../",
            body=f"<h2>{d} 全レースカード</h2><div class='box'><pre>{txt}</pre></div>"),
            encoding="utf-8")

    # --- 馬プロファイル（直近 active_days 日に出走・出走予定の馬）--------------
    horses = [tuple(r) for r in con.execute(
        """SELECT DISTINCT h.horse_id, h.name FROM horses h
           JOIN results r ON r.horse_id = h.horse_id
           JOIN races ra ON ra.race_id = r.race_id
           WHERE ra.race_date >= (SELECT DATE(MAX(race_date), ?) FROM races)
           ORDER BY h.name""", (f"-{active_days} days",))]
    for hid, name in horses:
        txt = _capture(horse_mod.main, ["--db", db, name])
        (outp / "horse" / f"{hid}.html").write_text(PAGE.format(
            title=name, style=STYLE, root="../",
            body=f"<div class='box'><pre>{txt}</pre></div>"), encoding="utf-8")
    (outp / "horses.json").write_text(
        json.dumps([{"i": h, "n": n} for h, n in horses], ensure_ascii=False),
        encoding="utf-8")

    # --- ランキング -----------------------------------------------------------
    parts = []
    for by, lab in (("r", "相対R（現在の格）"), ("pmax", "最高perf（天井）"),
                    ("power", "能力（時計）")):
        txt = _capture(ranking_mod.main, ["--db", db, "--by", by, "--top", "100"])
        parts.append(f"<h2>{lab}</h2><div class='box'><pre>{txt}</pre></div>")
    (outp / "ranking.html").write_text(PAGE.format(
        title="ランキング", style=STYLE, root="", body="\n".join(parts)),
        encoding="utf-8")

    # --- トップ（検索＋カード一覧）-------------------------------------------
    links = "\n".join(
        f"<a href='cards/{d}.html'>📅 {d} のカード</a><br>" for d in reversed(dates))
    body = f"""
<h2>馬名検索</h2>
<input id="q" placeholder="馬名を入力（部分一致）" autocomplete="off">
<div id="hits" class="box" style="display:none"></div>
<h2>開催日カード</h2><div class="box">{links or 'まだありません'}</div>
<h2>ランキング</h2><div class="box"><a href="ranking.html">相対R・天井・能力 上位100 →</a></div>
<script>
let H=[];fetch('horses.json').then(r=>r.json()).then(j=>H=j);
const q=document.getElementById('q'),hits=document.getElementById('hits');
q.addEventListener('input',()=>{{const v=q.value.trim();
 if(!v){{hits.style.display='none';return}}
 const m=H.filter(h=>h.n.includes(v)).slice(0,30);
 hits.innerHTML=m.map(h=>`<a href="horse/${{h.i}}.html">${{h.n}}</a>`).join('')||'該当なし';
 hits.style.display='block'}});
</script>"""
    (outp / "index.html").write_text(
        PAGE.format(title="競馬DB", style=STYLE, root="", body=body), encoding="utf-8")
    con.close()
    print(f"生成完了: {outp}/  カード{len(dates)}日分 / 馬{len(horses)}頭 / ランキング / 検索")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="静的サイト生成")
    ap.add_argument("--db", default="keiba.db")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--active-days", type=int, default=400,
                    help="馬ページを作る対象: 直近N日に出走した馬")
    ap.add_argument("--card-days", type=int, default=30,
                    help="カードを作る対象: 最新結果日からN日前まで＋未来")
    args = ap.parse_args(argv)
    return build(args.db, args.out, args.active_days, args.card_days)


if __name__ == "__main__":
    raise SystemExit(main())
