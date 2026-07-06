#!/usr/bin/env python3
# =============================================================================
# 1ファイル完結の閲覧ページ生成（自分専用・サーバー不要）
#
# DBの内容を単一の keiba.html に埋め込む。Google Drive に置けば費用ゼロ・非公開で
# PC/スマホからいつでも閲覧できる（オフラインでも動く）。
#   ・馬名検索（部分一致→プロファイル表示）
#   ・開催日カード（確定版カードの全文）
#   ・ランキング（相対R / 天井 / 能力）
#
# 使い方:
#   python scripts/site_single.py --db keiba.db --out keiba.html
#   （Colabなら --out /content/drive/MyDrive/keiba.html で直接Driveへ）
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

TPL = """<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>競馬DB</title><style>
:root{color-scheme:light dark}
body{font-family:sans-serif;margin:0;background:#fafafa;color:#222}
@media (prefers-color-scheme:dark){body{background:#151515;color:#ddd}
 .box{background:#1e1e1e!important;border-color:#333!important}
 nav button{background:#333;color:#ddd;border-color:#555}}
header{background:#173f2f;color:#fff;padding:10px 14px;font-weight:bold}
nav{padding:8px 12px}
nav button{font-size:14px;padding:8px 14px;margin-right:6px;border:1px solid #aaa;
  border-radius:6px;background:#fff;cursor:pointer}
nav button.on{background:#173f2f;color:#fff}
main{padding:0 12px 24px;max-width:1100px;margin:auto}
pre{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;
    overflow-x:auto;line-height:1.45}
.box{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px 14px;margin:10px 0}
input,select{font-size:16px;padding:8px;border:1px solid #aaa;border-radius:6px;
  width:100%;box-sizing:border-box;margin:4px 0}
#hits a{display:block;padding:7px 4px;text-decoration:none;border-bottom:1px solid #8883}
.small{font-size:12px;opacity:.7}
</style></head><body>
<header>🐎 競馬DB <span class="small">（%%STAMP%% 更新）</span></header>
<nav>
<button id="b0" class="on" onclick="tab(0)">馬検索</button>
<button id="b1" onclick="tab(1)">カード</button>
<button id="b2" onclick="tab(2)">ランキング</button>
</nav>
<main>
<div id="t0">
 <input id="q" placeholder="馬名を入力（部分一致）" autocomplete="off">
 <div id="hits" class="box" style="display:none"></div>
 <div id="prof" class="box" style="display:none"><pre id="proftxt"></pre></div>
</div>
<div id="t1" style="display:none">
 <select id="dsel" onchange="showCard()"></select>
 <div class="box"><pre id="cardtxt"></pre></div>
</div>
<div id="t2" style="display:none">
 <select id="rsel" onchange="showRank()">
  <option value="r">相対R（現在の格）</option>
  <option value="pmax">最高perf（天井）</option>
  <option value="power">能力（時計）</option></select>
 <div class="box"><pre id="ranktxt"></pre></div>
</div>
<p class="small">確定版モデル: 相対R(格)＋ムラ馬条件補正。⭐=抜けた本命(複勝向き)・
◎=混戦の筆頭(単勝妙味)・条=条件補正・最高=天井。馬券は自己責任で。</p>
</main>
<script>
const HORSES=%%HORSES%%, PROF=%%PROF%%, CARDS=%%CARDS%%, RANKS=%%RANKS%%;
function tab(i){for(let k=0;k<3;k++){
 document.getElementById('t'+k).style.display=k===i?'':'none';
 document.getElementById('b'+k).className=k===i?'on':'';}}
const q=document.getElementById('q'),hits=document.getElementById('hits');
q.addEventListener('input',()=>{const v=q.value.trim();
 document.getElementById('prof').style.display='none';
 if(!v){hits.style.display='none';return}
 const m=HORSES.filter(h=>h.n.includes(v)).slice(0,30);
 hits.innerHTML=m.map(h=>`<a href="#" onclick="prof('${h.i}');return false">${h.n}</a>`).join('')||'該当なし';
 hits.style.display='block'});
function prof(i){hits.style.display='none';
 document.getElementById('proftxt').textContent=PROF[i]||'データなし';
 document.getElementById('prof').style.display=''}
const dsel=document.getElementById('dsel');
Object.keys(CARDS).sort().reverse().forEach(d=>{
 const o=document.createElement('option');o.value=d;o.textContent=d;dsel.appendChild(o)});
function showCard(){document.getElementById('cardtxt').textContent=CARDS[dsel.value]||''}
function showRank(){document.getElementById('ranktxt').textContent=
 RANKS[document.getElementById('rsel').value]||''}
if(dsel.options.length)showCard();showRank();
</script></body></html>"""


def _capture(func, argv) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            func(argv)
        except SystemExit:
            pass
        except Exception as e:  # noqa: BLE001
            print(f"(生成エラー: {e})")
    return buf.getvalue()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="1ファイル閲覧ページ生成")
    ap.add_argument("--db", default="keiba.db")
    ap.add_argument("--out", default="keiba.html")
    ap.add_argument("--active-days", type=int, default=400)
    ap.add_argument("--card-days", type=int, default=30)
    args = ap.parse_args(argv)
    con = sqlite3.connect(args.db)

    dates = [r[0] for r in con.execute(
        """SELECT DISTINCT race_date FROM races
           WHERE race_date >= (SELECT DATE(MAX(race_date), ?) FROM races
                               WHERE race_id IN (SELECT race_id FROM results
                                                 WHERE finish_position IS NOT NULL))
           ORDER BY race_date""", (f"-{args.card_days} days",))]
    cards = {d: _capture(card.main, ["--db", args.db, "--ability-only", "--date", d])
             for d in dates}

    horses = [tuple(r) for r in con.execute(
        """SELECT DISTINCT h.horse_id, h.name FROM horses h
           JOIN results r ON r.horse_id = h.horse_id
           JOIN races ra ON ra.race_id = r.race_id
           WHERE ra.race_date >= (SELECT DATE(MAX(race_date), ?) FROM races)
           ORDER BY h.name""", (f"-{args.active_days} days",))]
    prof = {hid: _capture(horse_mod.main, ["--db", args.db, name])
            for hid, name in horses}

    ranks = {by: _capture(ranking_mod.main, ["--db", args.db, "--by", by, "--top", "100"])
             for by in ("r", "pmax", "power")}
    stamp = con.execute("SELECT MAX(race_date) FROM races").fetchone()[0]
    con.close()

    html = (TPL
            .replace("%%STAMP%%", str(stamp))
            .replace("%%HORSES%%", json.dumps(
                [{"i": h, "n": n} for h, n in horses], ensure_ascii=False))
            .replace("%%PROF%%", json.dumps(prof, ensure_ascii=False))
            .replace("%%CARDS%%", json.dumps(cards, ensure_ascii=False))
            .replace("%%RANKS%%", json.dumps(ranks, ensure_ascii=False)))
    Path(args.out).write_text(html, encoding="utf-8")
    mb = Path(args.out).stat().st_size / 1e6
    print(f"生成完了: {args.out} ({mb:.1f}MB)  カード{len(dates)}日 / 馬{len(horses)}頭")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
