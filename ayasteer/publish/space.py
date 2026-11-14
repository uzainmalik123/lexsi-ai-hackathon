"""Build a single-file static replay Space from the judged results.

The page reads ``data.json`` next to it: every eval prompt with its output and judge
label under each condition, plus the report tables and figures. No model runs live.
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

from ayasteer.publish.redact import redact_row
from ayasteer.schema import HEADLINE, LANG_NAME, SECONDARY_MULTIJAIL, TRANSLATED, read_jsonl

# Ablated outputs are not replayed (they are the harmful completions); their rates are in the report tables.
ORDER = ["base", "steer_en", "steer_multi"]
COND_LABEL = {"base": "Base (as shipped)", "steer_en": "Steered (English direction)",
              "steer_multi": "Steered (multilingual direction)", "ablate": "Ablated (direction removed)"}


def collect(results_dir: Path) -> dict:
    results_dir = Path(results_dir)
    rows: dict[tuple[str, str], dict] = {}
    conditions: list[str] = []
    for cond in ORDER:
        p = results_dir / "judged" / "eval" / f"{cond}.jsonl"
        if not p.exists():
            continue
        conditions.append(cond)
        for r in read_jsonl(p):
            key = (r["id"], r["language"])
            base = rows.setdefault(key, {"id": r["id"], "source": r["source"], "language": r["language"],
                                         "label": r["label"], "prompt": r["prompt"], "en_prompt": r["en_prompt"],
                                         "outputs": {}})
            r = redact_row(r)
            base["outputs"][cond] = {"output": r.get("output", ""), "judge_label": r.get("judge_label", "unknown"),
                                     "strength": r.get("strength", 0.0)}
    langs_present = {r["language"] for r in rows.values()}
    languages = [l for l in HEADLINE + SECONDARY_MULTIJAIL + TRANSLATED if l in langs_present]
    report_md = ""
    rp = results_dir / "report" / "report.md"
    if rp.exists():
        report_md = rp.read_text(encoding="utf-8")
    figures = sorted(p.name for p in (results_dir / "report").glob("*.png")) if (results_dir / "report").exists() else []
    return {"conditions": conditions, "condition_labels": {c: COND_LABEL[c] for c in conditions},
            "languages": languages, "language_names": {l: LANG_NAME[l] for l in languages},
            "rows": list(rows.values()), "report_md": report_md, "figures": figures}


def build_space(results_dir: Path, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = collect(results_dir)
    (out_dir / "data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    for name in data["figures"]:
        shutil.copy(Path(results_dir) / "report" / name, out_dir / name)
    lang_options = "".join(f'<option value="{l}">{data["language_names"][l]}</option>' for l in data["languages"])
    (out_dir / "index.html").write_text(HTML.replace("__LANG_OPTIONS__", lang_options), encoding="utf-8")
    return out_dir


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tiny Aya Multilingual Steering</title>
<style>
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;
--refusal:#2a78d6;--compliance:#e34948;--partial:#eda100;--incoherent:#898781;--unknown:#c3c2b7}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--line:#2c2c2a}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--ink2);margin:0 0 20px}
.controls{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0 16px}
select,input{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink)}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0}
.prompt{font-weight:600}.en{color:var(--ink2);font-size:13px}
.outs{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;margin-top:10px}
.out{border:1px solid var(--line);border-radius:8px;padding:10px;font-size:13px;white-space:pre-wrap;word-break:break-word}
.out h4{margin:0 0 6px;font-size:12px;color:var(--ink2);font-weight:600}
.tag{display:inline-block;padding:1px 8px;border-radius:999px;color:#fff;font-size:11px;margin-left:6px}
.rtl{direction:rtl;text-align:right}
table{border-collapse:collapse;font-size:13px;overflow-x:auto;display:block}td,th{border:1px solid var(--line);padding:4px 8px}
img{max-width:100%;border:1px solid var(--line);border-radius:8px;margin:8px 0}
details{margin:16px 0}summary{cursor:pointer;font-weight:600}
</style></head><body><main>
<h1>Tiny Aya Global: multilingual refusal steering, replayed</h1>
<p class="sub">Every eval prompt, the model's output under each condition, and the judge's label. Static replay; nothing runs live. Harmful prompts and compliant outputs are shown unredacted, this is a safety benchmark.</p>
<div class="controls">
<label>Language <select id="lang">__LANG_OPTIONS__</select></label>
<label>Prompt type <select id="label"><option value="">all</option><option value="harmful">harmful</option><option value="benign">benign</option></select></label>
<label>Judge changed? <select id="changed"><option value="">any</option><option value="1">only where conditions disagree</option></select></label>
<label>Search <input id="q" placeholder="text in prompt or output"></label>
</div>
<div id="count" class="en"></div>
<div id="rows"></div>
<details open><summary>Results tables</summary><div id="report"></div></details>
<details open><summary>Figures</summary><div id="figs"></div></details>
</main>
<script>
const RTL=new Set(["ar","ur"]);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function mdTable(md){return md.split("\\n").map(l=>{if(l.startsWith("## "))return "<h3>"+esc(l.slice(3))+"</h3>";if(l.startsWith("|")){if(/^\\|\\s*-/.test(l))return "";const c=l.split("|").slice(1,-1);return "<tr>"+c.map(x=>"<td>"+esc(x.trim())+"</td>").join("")+"</tr>";}if(l.startsWith("```"))return "";return esc(l)?"<p>"+esc(l)+"</p>":""}).join("").replace(/(<tr>.*?<\\/tr>)+/gs,m=>"<table>"+m+"</table>")}
fetch("data.json").then(r=>r.json()).then(D=>{
 const $=id=>document.getElementById(id);
 $("report").innerHTML=mdTable(D.report_md||"(report not built yet)");
 $("figs").innerHTML=D.figures.map(f=>'<img src="'+f+'" alt="'+f+'">').join("");
 function render(){
  const lang=$("lang").value,lab=$("label").value,ch=$("changed").value,q=$("q").value.toLowerCase();
  const rows=D.rows.filter(r=>r.language===lang&&(!lab||r.label===lab)&&(!q||(r.prompt+" "+r.en_prompt+" "+Object.values(r.outputs).map(o=>o.output).join(" ")).toLowerCase().includes(q))
   &&(!ch||new Set(Object.values(r.outputs).map(o=>o.judge_label)).size>1));
  $("count").textContent=rows.length+" prompts";
  $("rows").innerHTML=rows.map(r=>{
   const dir=RTL.has(r.language)?" rtl":"";
   return '<div class="card"><div class="prompt'+dir+'">'+esc(r.prompt)+'</div>'+(r.en_prompt&&r.en_prompt!==r.prompt?'<div class="en">EN: '+esc(r.en_prompt)+'</div>':'')+
   '<div class="en">'+esc(r.source)+' · '+esc(r.label)+' · '+esc(r.id)+'</div><div class="outs">'+
   D.conditions.map(c=>{const o=r.outputs[c];if(!o)return "";return '<div class="out'+dir+'"><h4>'+esc(D.condition_labels[c])+(o.strength?' (s='+o.strength+')':'')+'<span class="tag" style="background:var(--'+esc(o.judge_label)+')">'+esc(o.judge_label)+'</span></h4>'+(esc(o.output)||'<i>(empty)</i>')+'</div>'}).join("")+'</div></div>'}).join("");
 }
 for(const id of ["lang","label","changed"])$(id).onchange=render; $("q").oninput=render; render();
});
</script></body></html>
"""
