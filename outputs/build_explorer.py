"""Build a self-contained offline HTML explorer for the per-question results CSV.

Two pages (tabs), both offline (no server, no internet — just open the file):
  * Questions      — item-level explorer: filter/sort/scroll every question and see
                     its Human / Ours / Faithful answer distributions + metadata.
  * Task summaries — topline (split-avg, pooled, pop, grouped) + a by-task overview
                     and, per selected task type, score-breakdown / score-histogram /
                     dispersion (entropy) plots.

Reads outputs/simbench_question_predictions.csv -> outputs/simbench_explorer.html.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "outputs/simbench_question_predictions.csv"
OUT = ROOT / "outputs/simbench_explorer.html"


def _seg_label(seg_json: str) -> str:
    try:
        seg = json.loads(seg_json) if seg_json else {}
    except json.JSONDecodeError:
        seg = {}
    return ", ".join(f"{k}={v}" for k, v in seg.items()) if seg else ""


def _aligned(dist_json: str, options: list[str]) -> list[float]:
    try:
        d = json.loads(dist_json)
    except (json.JSONDecodeError, TypeError):
        d = {}
    return [round(float(d.get(o, 0.0)), 4) for o in options]


def main() -> None:
    df = pd.read_csv(CSV)
    rows = []
    for r in df.itertuples(index=False):
        options = json.loads(r.options)
        os_, fs_ = round(float(r.our_score), 2), round(float(r.faithful_score), 2)
        rows.append({
            "d": r.dataset, "k": r.task_kind, "b": r.eval_bucket, "f": r.simbench_file,
            "pop": bool(r.is_population),
            "seg": _seg_label(r.segment if isinstance(r.segment, str) else ""),
            "g": "" if pd.isna(r.group_size) else int(r.group_size),
            "rq": r.required_question if isinstance(r.required_question, str) else "",
            "q": r.question_text, "o": options,
            "h": _aligned(r.human_dist, options),
            "u": _aligned(r.our_pred, options),
            "fa": _aligned(r.faithful_pred, options),
            "os": os_, "fs": fs_, "us": round(float(r.uniform_score), 2),
            "e": round(float(r.truth_entropy), 3), "lift": round(os_ - fs_, 2),
        })

    meta = {
        "n": len(rows),
        "kinds": sorted(df["task_kind"].unique().tolist()),
        "datasets": sorted(df["dataset"].unique().tolist()),
        "buckets": ["dev", "val", "test"],
    }
    html = (TEMPLATE
            .replace("/*__META__*/", json.dumps(meta, ensure_ascii=False))
            .replace("/*__DATA__*/", json.dumps(rows, ensure_ascii=False, separators=(",", ":"))))
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({len(rows)} questions, {OUT.stat().st_size/1e6:.1f} MB)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SimBench explorer</title>
<style>
  :root{
    --bg:#0f1117; --panel:#171a23; --panel2:#1e222d; --line:#2b3140;
    --txt:#e6e8ee; --mut:#9aa3b2; --human:#e5e9f0; --ours:#34d399; --faith:#f59e0b;
    --uni:#6b7280; --pos:#34d399; --neg:#f87171; --acc:#60a5fa;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--txt);display:flex;flex-direction:column;
    height:100vh;overflow:hidden;
    font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  header{flex:none;padding:9px 16px;border-bottom:1px solid var(--line);background:var(--panel);
    display:flex;gap:18px;align-items:center}
  header h1{font-size:15px;margin:0;font-weight:650;letter-spacing:.2px}
  .tabs{display:flex;gap:6px}
  .tab{background:transparent;border:1px solid transparent;color:var(--mut);border-radius:8px;
    padding:6px 14px;font-size:13px;cursor:pointer}
  .tab.active{background:var(--panel2);border-color:var(--line);color:var(--txt)}
  .page{flex:1;min-height:0}
  #page-questions:not(.hidden){display:flex;flex-direction:column}
  #page-tasks:not(.hidden){display:block;overflow:auto}
  .hidden{display:none!important}
  .subbar{flex:none;display:flex;gap:14px;align-items:center;padding:8px 16px;
    border-bottom:1px solid var(--line);background:var(--panel)}
  .stat{color:var(--mut);font-size:12.5px}
  .stat b{color:var(--txt);font-variant-numeric:tabular-nums}
  .ours-c{color:var(--ours)} .faith-c{color:var(--faith)}
  .pos{color:var(--pos)} .neg{color:var(--neg)}
  .controls{flex:none;display:flex;flex-wrap:wrap;gap:8px;padding:10px 16px;
    border-bottom:1px solid var(--line);background:var(--panel);align-items:center}
  .controls label{color:var(--mut);font-size:12.5px;display:flex;gap:6px;align-items:center}
  select,input{background:var(--panel2);color:var(--txt);border:1px solid var(--line);
    border-radius:7px;padding:6px 9px;font-size:13px;outline:none}
  input[type=search]{min-width:220px}
  label.chk{display:flex;gap:6px;align-items:center;color:var(--mut);font-size:12.5px;cursor:pointer}
  .wrap{flex:1;min-height:0;display:grid;grid-template-columns:minmax(300px,420px) 1fr}
  .list{overflow:auto;border-right:1px solid var(--line);position:relative}
  .vp{position:relative;width:100%}
  .row{position:absolute;left:0;right:0;padding:8px 12px;border-bottom:1px solid var(--line);
    cursor:pointer;display:flex;flex-direction:column;gap:3px;overflow:hidden}
  .row:hover{background:var(--panel2)}
  .row.sel{background:#243049;box-shadow:inset 3px 0 0 var(--acc)}
  .row .top{display:flex;justify-content:space-between;gap:8px;align-items:center}
  .badge{font-size:10.5px;color:var(--mut);background:var(--panel2);border:1px solid var(--line);
    border-radius:5px;padding:1px 6px;white-space:nowrap}
  .row .sc{font-variant-numeric:tabular-nums;font-size:12px;white-space:nowrap}
  .row .qt{color:var(--mut);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .detail{overflow:auto;padding:18px 22px}
  h2{font-size:13px;color:var(--mut);font-weight:600;margin:0 0 6px;text-transform:uppercase;letter-spacing:.6px}
  h3{font-size:12px;color:var(--mut);font-weight:600;margin:0 0 4px}
  .qhead{display:flex;justify-content:space-between;align-items:center}
  .qhead button{font-size:11px;padding:2px 9px;text-transform:none;letter-spacing:0}
  .qbox{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px;
    white-space:pre-wrap;max-height:260px;overflow:auto;font-size:13.5px}
  .qbox.collapsed{max-height:3.4em;overflow:hidden}
  .meta{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}
  .chip{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:5px 10px;font-size:12px}
  .chip b{color:var(--mut);font-weight:500}
  .scores{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0 18px}
  .scard{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 16px;min-width:96px}
  .scard.wide{min-width:140px}
  .scard .lab{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px}
  .scard .val{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums;margin-top:2px}
  .scard .sub{font-size:11px;color:var(--mut);margin-top:3px;font-variant-numeric:tabular-nums}
  .legend{display:flex;gap:16px;margin:6px 0 4px;font-size:12.5px;color:var(--mut);align-items:center;flex-wrap:wrap}
  .legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
  .chartbox{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px;
    height:210px;display:flex;align-items:center;justify-content:center}
  .nav{display:flex;gap:8px;align-items:center;margin-left:auto}
  button{background:var(--panel2);color:var(--txt);border:1px solid var(--line);border-radius:7px;
    padding:6px 11px;font-size:13px;cursor:pointer}
  button:hover{background:#2a3142}
  .empty{color:var(--mut);padding:40px;text-align:center}
  text{fill:var(--mut);font-size:11px}
  .axis{stroke:var(--line)} .zero{stroke:#3a4254;stroke-width:1.5}
  .uni{stroke:var(--uni);stroke-dasharray:5 4;stroke-width:1.5}
  .tpad{padding:18px 22px;max-width:1180px}
  .tgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px}
</style>
</head>
<body>
<header>
  <h1>SimBench explorer</h1>
  <nav class="tabs">
    <button class="tab active" data-p="questions">Questions</button>
    <button class="tab" data-p="tasks">Task summaries</button>
  </nav>
</header>

<!-- ============ PAGE: QUESTIONS ============ -->
<section id="page-questions" class="page">
  <div class="subbar">
    <span class="stat" id="summary"></span>
    <span class="nav">
      <button id="prev">&larr; Prev</button>
      <span class="stat" id="pos"></span>
      <button id="next">Next &rarr;</button>
    </span>
  </div>
  <div class="controls">
    <select id="kind"></select>
    <select id="dataset"></select>
    <select id="bucket"></select>
    <select id="sort">
      <option value="os_desc">Sort: Ours (best first)</option>
      <option value="os_asc">Sort: Ours (worst first)</option>
      <option value="lift_desc">Sort: biggest lift over faithful</option>
      <option value="lift_asc">Sort: biggest loss vs faithful</option>
      <option value="e_desc">Sort: most diverse (entropy)</option>
      <option value="e_asc">Sort: most peaked (entropy)</option>
    </select>
    <label class="chk"><input type="checkbox" id="reqonly"> required Qs only</label>
    <input type="search" id="search" placeholder="search question text…">
  </div>
  <div class="wrap">
    <div class="list" id="list"><div class="vp" id="vp"></div></div>
    <div class="detail" id="detail"></div>
  </div>
</section>

<!-- ============ PAGE: TASK SUMMARIES ============ -->
<section id="page-tasks" class="page hidden">
  <div class="controls">
    <label>Bucket <select id="t_bucket">
      <option value="test">test (held-out)</option>
      <option value="val">val</option>
      <option value="dev">dev</option>
      <option value="">all</option>
    </select></label>
    <label>Task type <select id="t_kind"></select></label>
  </div>
  <div class="tpad">
    <h2>Topline — <span id="t_scope"></span> · <span class="ours-c">Ours</span> vs <span class="faith-c">Faithful</span></h2>
    <div class="scores" id="t_topline"></div>

    <h2 style="margin-top:8px">Mean SimBench score by task</h2>
    <div class="legend" id="t_overlegend"></div>
    <div class="chartbox" id="t_overview" style="height:240px"></div>

    <h2 style="margin-top:20px" id="t_taskhead">Task</h2>
    <div class="tgrid">
      <div><h3>Score breakdown</h3>
        <div class="legend" id="t_breaklegend"></div>
        <div class="chartbox" id="t_break"></div></div>
      <div><h3>Per-question score distribution</h3>
        <div class="legend" id="t_histlegend"></div>
        <div class="chartbox" id="t_hist"></div></div>
      <div><h3>Dispersion — mean answer entropy (0=peaked, 1=uniform)</h3>
        <div class="legend" id="t_entlegend"></div>
        <div class="chartbox" id="t_ent"></div></div>
    </div>
  </div>
</section>

<script>
const META = /*__META__*/;
const DATA = /*__DATA__*/;
const ROW_H = 56, BUF = 6;
const $ = id => document.getElementById(id);
const C = {human:'#e5e9f0', ours:'#34d399', faith:'#f59e0b', uni:'#6b7280'};
const esc = s => s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const opt = (v,t)=>{const o=document.createElement('option');o.value=v;o.textContent=t;return o;};

/* ---------- tabs ---------- */
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===t));
  const p=t.dataset.p;
  $('page-questions').classList.toggle('hidden',p!=='questions');
  $('page-tasks').classList.toggle('hidden',p!=='tasks');
  if(p==='tasks') renderTasks(); else renderList();
});

/* ================= PAGE 1: QUESTIONS ================= */
let filtered=[], sel=0;
(function initQ(){
  const k=$('kind'); k.appendChild(opt('','All tasks')); META.kinds.forEach(x=>k.appendChild(opt(x,x)));
  const d=$('dataset'); d.appendChild(opt('','All datasets')); META.datasets.forEach(x=>d.appendChild(opt(x,x)));
  const b=$('bucket'); b.appendChild(opt('','All buckets')); META.buckets.forEach(x=>b.appendChild(opt(x,x)));
  ['kind','dataset','bucket','sort','reqonly','search'].forEach(id=>
    $(id).addEventListener(id==='search'?'input':'change',applyQ));
  $('prev').onclick=()=>move(-1); $('next').onclick=()=>move(1);
  $('list').addEventListener('scroll',renderList);
  document.addEventListener('keydown',e=>{
    if(/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName))return;
    if(!$('page-questions').classList.contains('hidden')){
      if(e.key==='ArrowDown'){e.preventDefault();move(1);}
      if(e.key==='ArrowUp'){e.preventDefault();move(-1);}
    }
  });
  applyQ();
})();

function applyQ(){
  const k=$('kind').value,d=$('dataset').value,b=$('bucket').value,
        req=$('reqonly').checked,q=$('search').value.trim().toLowerCase();
  filtered=DATA.filter(r=>(!k||r.k===k)&&(!d||r.d===d)&&(!b||r.b===b)
    &&(!req||r.rq)&&(!q||r.q.toLowerCase().includes(q)));
  const cmp={os_desc:(a,b)=>b.os-a.os,os_asc:(a,b)=>a.os-b.os,
    lift_desc:(a,b)=>b.lift-a.lift,lift_asc:(a,b)=>a.lift-b.lift,
    e_desc:(a,b)=>b.e-a.e,e_asc:(a,b)=>a.e-b.e}[$('sort').value];
  filtered.sort(cmp); sel=0; summaryQ();
  $('vp').style.height=(filtered.length*ROW_H)+'px'; $('list').scrollTop=0;
  renderList(); renderDetail();
}
function summaryQ(){
  const n=filtered.length;
  if(!n){$('summary').textContent='no questions match';return;}
  const mean=f=>filtered.reduce((s,r)=>s+r[f],0)/n;
  const o=mean('os'),f=mean('fs'),l=o-f;
  $('summary').innerHTML=`<b>${n.toLocaleString()}</b> questions &nbsp;·&nbsp; `
    +`mean <span class="ours-c">Ours ${o.toFixed(1)}</span> &nbsp; `
    +`<span class="faith-c">Faithful ${f.toFixed(1)}</span> &nbsp; `
    +`lift <span class="${l>=0?'pos':'neg'}">${l>=0?'+':''}${l.toFixed(1)}</span>`;
}
function renderList(){
  const vp=$('vp'),lc=$('list');
  if(!filtered.length){vp.innerHTML='';$('pos').textContent='';return;}
  const st=lc.scrollTop,h=lc.clientHeight;
  let a=Math.max(0,Math.floor(st/ROW_H)-BUF),z=Math.min(filtered.length,Math.ceil((st+h)/ROW_H)+BUF);
  let html='';
  for(let i=a;i<z;i++){const r=filtered[i];
    html+=`<div class="row${i===sel?' sel':''}" style="top:${i*ROW_H}px;height:${ROW_H}px" data-i="${i}">`
      +`<div class="top"><span class="badge">${r.d} · ${r.b}${r.rq?' · '+r.rq:''}</span>`
      +`<span class="sc"><span class="ours-c">${r.os.toFixed(1)}</span> / `
      +`<span class="faith-c">${r.fs.toFixed(1)}</span> `
      +`<span class="${r.lift>=0?'pos':'neg'}">(${r.lift>=0?'+':''}${r.lift.toFixed(1)})</span></span></div>`
      +`<div class="qt">${esc(r.q.slice(0,140))}</div></div>`;}
  vp.innerHTML=html;
  vp.querySelectorAll('.row').forEach(el=>el.onclick=()=>{sel=+el.dataset.i;renderList();renderDetail();});
  $('pos').textContent=`${sel+1} / ${filtered.length}`;
}
function move(dir){
  if(!filtered.length)return;
  sel=Math.min(filtered.length-1,Math.max(0,sel+dir));
  const lc=$('list'),y=sel*ROW_H;
  if(y<lc.scrollTop) lc.scrollTop=y;
  else if(y+ROW_H>lc.scrollTop+lc.clientHeight) lc.scrollTop=y+ROW_H-lc.clientHeight;
  renderList(); renderDetail();
}
function renderDetail(){
  const el=$('detail');
  if(!filtered.length){el.innerHTML='<div class="empty">No questions match these filters.</div>';return;}
  const r=filtered[sel], liftCls=r.lift>=0?'pos':'neg';
  const meta=[['dataset',r.d],['task',r.k],['bucket',r.b],['file',r.f],
    ['group',r.pop?'population':(r.seg||'segment')],['respondents',r.g||'—'],
    ['options',r.o.length],['entropy',r.e.toFixed(2)],r.rq?['required',r.rq]:null].filter(Boolean);
  el.innerHTML=
    `<h2 class="qhead">Question <button id="qtoggle">Expand</button></h2>`
    +`<div class="qbox collapsed" id="qbox">${esc(r.q)}</div>`
    +`<div class="meta">${meta.map(m=>`<span class="chip"><b>${m[0]}:</b> ${esc(String(m[1]))}</span>`).join('')}</div>`
    +`<div class="scores">${scard('Ours',r.os,'ours-c')}${scard('Faithful',r.fs,'faith-c')}`
    +`${scard('Uniform',r.us,'')}<div class="scard"><div class="lab">Lift vs faithful</div>`
    +`<div class="val ${liftCls}">${r.lift>=0?'+':''}${r.lift.toFixed(1)}</div></div></div>`
    +`<h2>Answer distribution</h2>`
    +`<div class="legend"><span><i style="background:var(--human)"></i>Human (truth)</span>`
    +`<span><i style="background:var(--ours)"></i>Ours</span>`
    +`<span><i style="background:var(--faith)"></i>Faithful</span>`
    +`<span><svg width="22" height="10"><line x1="0" y1="5" x2="22" y2="5" class="uni"/></svg> Uniform</span></div>`
    +`<div class="chartbox">${distChart(r)}</div>`;
  const qb=$('qbox'),qt=$('qtoggle');
  if(qb.scrollHeight<=qb.clientHeight+2){qt.style.display='none';}
  else{qt.onclick=()=>{const c=qb.classList.toggle('collapsed');qt.textContent=c?'Expand':'Collapse';};}
}
function scard(lab,val,cls){
  return `<div class="scard"><div class="lab">${lab}</div><div class="val ${cls||''}">${val.toFixed(1)}</div></div>`;
}
function distChart(r){
  const opts=r.o,n=opts.length;
  const series=[['h',C.human],['u',C.ours],['fa',C.faith]];
  const maxP=Math.max(0.12,...r.h,...r.u,...r.fa), yMax=Math.min(1,Math.ceil(maxP*10)/10);
  const W=Math.max(360,Math.min(880,90+n*78)),H=200,mL=42,mR=12,mT=8,mB=34;
  const pw=W-mL-mR,ph=H-mT-mB,gw=pw/n,bw=Math.min(26,(gw*0.74)/3),uni=1/n,uy=mT+ph-(uni/yMax)*ph;
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%" height="100%" preserveAspectRatio="xMidYMid meet">`;
  for(let t=0;t<=4;t++){const v=yMax*t/4,y=mT+ph-(v/yMax)*ph;
    s+=`<line class="axis" x1="${mL}" y1="${y}" x2="${W-mR}" y2="${y}"/>`;
    s+=`<text x="${mL-6}" y="${y+3}" text-anchor="end">${Math.round(v*100)}%</text>`;}
  for(let i=0;i<n;i++){const gx=mL+i*gw+gw/2;
    series.forEach((ser,j)=>{const v=r[ser[0]][i],bh=(v/yMax)*ph,x=gx-(bw*1.5+2)+j*(bw+1);
      s+=`<rect x="${x.toFixed(1)}" y="${(mT+ph-bh).toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(0,bh).toFixed(1)}" fill="${ser[1]}" rx="1.5"><title>${['Human','Ours','Faithful'][j]} ${opts[i]}: ${(v*100).toFixed(1)}%</title></rect>`;});
    s+=`<text x="${gx}" y="${H-mB+16}" text-anchor="middle">${esc(String(opts[i]).slice(0,8))}</text>`;}
  s+=`<line class="uni" x1="${mL}" y1="${uy}" x2="${W-mR}" y2="${uy}"/></svg>`;
  return s;
}

/* ================= PAGE 2: TASK SUMMARIES ================= */
(function initT(){
  const k=$('t_kind'); META.kinds.forEach(x=>k.appendChild(opt(x,x)));
  k.value = META.kinds.includes('opinion_survey') ? 'opinion_survey' : META.kinds[0];
  $('t_bucket').addEventListener('change',renderTasks);
  $('t_kind').addEventListener('change',renderTaskDetail);
})();
const mean=(a,key)=>a.length?a.reduce((s,r)=>s+r[key],0)/a.length:0;
function topl(rows,key){
  const p=rows.filter(r=>r.f==='pop'),g=rows.filter(r=>r.f==='grouped');
  const pop=mean(p,key),grp=mean(g,key);
  return {splitavg:(pop+grp)/2,pooled:mean(rows,key),pop:pop,grouped:grp};
}
function nEnt(a){const p=a.filter(x=>x>0);if(p.length<=1)return 0;
  const h=-p.reduce((s,x)=>s+x*Math.log(x),0);return h/Math.log(a.length);}

function scope(){const b=$('t_bucket').value;return b?DATA.filter(r=>r.b===b):DATA;}
function renderTasks(){
  const rows=scope();
  $('t_scope').textContent = ($('t_bucket').selectedOptions[0].textContent)+` · ${rows.length.toLocaleString()} Q`;
  // topline cards
  const o=topl(rows,'os'),f=topl(rows,'fs');
  const card=(t,key)=>{const lift=o[key]-f[key];
    return `<div class="scard wide"><div class="lab">${t}</div>`
      +`<div class="val ours-c">${o[key].toFixed(1)}</div>`
      +`<div class="sub">faithful ${f[key].toFixed(1)} · <span class="${lift>=0?'pos':'neg'}">${lift>=0?'+':''}${lift.toFixed(1)}</span></div></div>`;};
  $('t_topline').innerHTML=card('Split-avg','splitavg')+card('Pooled','pooled')+card('Pop','pop')+card('Grouped','grouped');
  // by-task overview
  $('t_overlegend').innerHTML=leg([['Ours',C.ours],['Faithful',C.faith],['Uniform',C.uni]]);
  const cats=META.kinds, sk=k=>rows.filter(r=>r.k===k);
  $('t_overview').innerHTML=gbars(cats,[
    {color:C.ours,values:cats.map(k=>mean(sk(k),'os'))},
    {color:C.faith,values:cats.map(k=>mean(sk(k),'fs'))},
    {color:C.uni,values:cats.map(k=>mean(sk(k),'us'))}],
    {H:240,fmt:v=>v.toFixed(0),rot:cats.length>5});
  renderTaskDetail();
}
function renderTaskDetail(){
  const kind=$('t_kind').value, rows=scope().filter(r=>r.k===kind);
  $('t_taskhead').textContent=`Task: ${kind}  ·  ${rows.length.toLocaleString()} questions`;
  if(!rows.length){['t_break','t_hist','t_ent'].forEach(id=>$(id).innerHTML='<div class="empty">no questions</div>');return;}
  // breakdown
  const o=topl(rows,'os'),f=topl(rows,'fs'),u=topl(rows,'us');
  const order=['splitavg','pop','grouped','pooled'],labels=['split-avg','pop','grouped','pooled'];
  $('t_breaklegend').innerHTML=leg([['Ours',C.ours],['Faithful',C.faith],['Uniform',C.uni]]);
  $('t_break').innerHTML=gbars(labels,[
    {color:C.ours,values:order.map(k=>o[k])},
    {color:C.faith,values:order.map(k=>f[k])},
    {color:C.uni,values:order.map(k=>u[k])}],{fmt:v=>v.toFixed(0)});
  // histogram of per-question scores
  const vals=[...rows.map(r=>r.os),...rows.map(r=>r.fs)];
  let lo=Math.floor(Math.min(0,...vals)/20)*20; const hi=100, step=20;
  const edges=[]; for(let x=lo;x<hi;x+=step)edges.push(x);
  const binlab=edges.map(x=>`${x}`);
  const cnt=key=>{const c=edges.map(()=>0);
    rows.forEach(r=>{let idx=Math.floor((r[key]-lo)/step);idx=Math.min(edges.length-1,Math.max(0,idx));c[idx]++;});return c;};
  $('t_histlegend').innerHTML=leg([['Ours',C.ours],['Faithful',C.faith]])+'<span class="stat">score bins (lower edge)</span>';
  $('t_hist').innerHTML=gbars(binlab,[
    {color:C.ours,values:cnt('os')},{color:C.faith,values:cnt('fs')}],
    {fmt:v=>v.toFixed(0),min:0,rot:edges.length>7});
  // dispersion: mean entropy human/ours/faithful
  $('t_entlegend').innerHTML=leg([['Human',C.human],['Ours',C.ours],['Faithful',C.faith]]);
  const he=mean(rows,'e'),oe=rows.reduce((s,r)=>s+nEnt(r.u),0)/rows.length,fe=rows.reduce((s,r)=>s+nEnt(r.fa),0)/rows.length;
  $('t_ent').innerHTML=gbars(['mean entropy'],[
    {color:C.human,values:[he]},{color:C.ours,values:[oe]},{color:C.faith,values:[fe]}],
    {fmt:v=>v.toFixed(2),min:0,max:1});
}
function leg(items){return items.map(it=>`<span><i style="background:${it[1]}"></i>${it[0]}</span>`).join('');}

/* generic grouped bar chart (supports negative values) */
function gbars(cats,series,o){
  o=o||{}; const n=cats.length,m=series.length;
  const W=o.W||Math.max(320,80+n*Math.max(64,m*22+26)),H=o.H||210;
  const mL=46,mR=12,mT=12,mB=o.rot?52:34,pw=W-mL-mR,ph=H-mT-mB;
  let vals=[]; series.forEach(s=>s.values.forEach(v=>vals.push(v)));
  let dmin=o.min!=null?o.min:Math.min(0,...vals), dmax=o.max!=null?o.max:Math.max(...vals,0.0001);
  if(dmax<=dmin)dmax=dmin+1;
  const pad=(dmax-dmin)*0.08; if(o.max==null)dmax+=pad; if(o.min==null&&dmin<0)dmin-=pad;
  const Y=v=>mT+ph-((v-dmin)/(dmax-dmin))*ph;
  const gw=pw/n,gap=2,bw=Math.min(34,(gw*0.78-(m-1)*gap)/m);
  const fmt=o.fmt||(v=>v.toFixed(0));
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%" height="100%" preserveAspectRatio="xMidYMid meet">`;
  for(let t=0;t<=4;t++){const v=dmin+(dmax-dmin)*t/4,y=Y(v);
    s+=`<line class="axis" x1="${mL}" y1="${y.toFixed(1)}" x2="${W-mR}" y2="${y.toFixed(1)}"/>`;
    s+=`<text x="${mL-6}" y="${(y+3).toFixed(1)}" text-anchor="end">${fmt(v)}</text>`;}
  const y0=Y(Math.min(Math.max(0,dmin),dmax));
  if(dmin<0&&dmax>0) s+=`<line class="zero" x1="${mL}" y1="${y0.toFixed(1)}" x2="${W-mR}" y2="${y0.toFixed(1)}"/>`;
  for(let i=0;i<n;i++){const gx0=mL+i*gw+(gw-(m*bw+(m-1)*gap))/2;
    series.forEach((ser,j)=>{const v=ser.values[i],yv=Y(v),x=gx0+j*(bw+gap);
      const top=Math.min(yv,y0),hh=Math.abs(yv-y0);
      s+=`<rect x="${x.toFixed(1)}" y="${top.toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(0,hh).toFixed(1)}" fill="${ser.color}" rx="1.5"><title>${fmt(v)}</title></rect>`;});
    const cx=mL+i*gw+gw/2, lbl=esc(String(cats[i]));
    if(o.rot) s+=`<text x="${cx}" y="${H-mB+14}" text-anchor="end" transform="rotate(-35 ${cx} ${H-mB+14})">${lbl}</text>`;
    else s+=`<text x="${cx}" y="${H-mB+16}" text-anchor="middle">${lbl.slice(0,14)}</text>`;}
  s+=`</svg>`; return s;
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
