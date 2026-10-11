"""NII RDC: DR が作った研究計画と maDMP から、AI-Scientist の題材(idea)を 1 件作る。

    python rdc_ideate.py <exchange_dir> [--model gpt-4o] [--retries 3]

それまで研究実行(④)は、計画と無関係な固定の題材(ext_data_demo.json: 人工データでの MLP の幅の
研究)を毎回回していた。論文(⑤)もその題材になり、研究計画(例: 認知症の家族参加型 ICF リハビリ)と
つながらなかった(2026-10-11 に確認)。ここで計画から題材を作る。

入力(DG-runner が <exchange_dir>/input/ に置く。コンテナでは /rdc/input/):
  plan.md            研究計画書(DR が生成し eLabFTW に置いた版)
  madmp.json         maDMP の現行版(DG)
  data_manifest.json 取得済みの再利用データ(DG が GRDM から渡し、sha256 で照合したもの)の一覧と中身の形
  data/              その本体

計画の主データ(研究で新しく集めるデータ)は、まだ存在しない。ここで作る題材は「いま計算だけで
進められる研究」に限る(例: 計画した試験の設計と解析の検証を、計画に書かれた数値にもとづく
シミュレーションで行う)。取得済みのデータは、実際の行数・列を題材に書き込み、作りもの(mock)で
置き換えないことを実験の指示に固定で足す(2026-10-10 の実験は、門番の rdc_data を呼ばずに
MockRDCData で 3 行の表をでっち上げていた)。

出力: <exchange_dir>/idea.json(AI-Scientist の --load_ideas 形式。1 件のリスト)と
      <exchange_dir>/ideation.json(状態。実行器が読む)。成功したときだけ終了コード 0。
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import sys
import time
import traceback

REQUIRED = ["Name", "Title", "Short Hypothesis", "Related Work", "Abstract", "Experiments",
            "Risk Factors and Limitations"]


def status(xdir: str, state: str, step: str, **extra) -> None:
    path = osp.join(xdir, "ideation.json")
    cur = {"state": state, "step": step, "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra}
    with open(path + ".tmp", "w") as f:
        json.dump(cur, f, ensure_ascii=False, indent=2)
    os.replace(path + ".tmp", path)
    print(f"[rdc] ideation: {state} — {step}", flush=True)


def condense_madmp(m: dict) -> dict:
    """maDMP から、題材づくりに要る項目だけを抜く(全体は 40 KB を超え、要らない項目が多い)。"""
    d = m.get("dmp", m) if isinstance(m, dict) else {}
    own, reuse = [], []
    for ds in d.get("dataset") or []:
        r = ds.get("x_nii_reuse")
        if r:
            reuse.append({
                "title": ds.get("title"),
                "identifier": (ds.get("dataset_id") or {}).get("identifier"),
                "access_class": r.get("access_class"),
                "intended_use": (r.get("intended_use") or {}).get("description"),
            })
        else:
            own.append({
                "title": ds.get("title"), "type": ds.get("type"), "description": ds.get("description"),
                "keyword": ds.get("keyword"), "personal_data": ds.get("personal_data"),
                "sensitive_data": ds.get("sensitive_data"),
                "format": [f for dist in ds.get("distribution") or [] for f in dist.get("format") or []],
            })
    return {
        "title": d.get("title"),
        "description": d.get("description"),
        "ethical_issues": d.get("ethical_issues_description"),
        "project": [{"title": p.get("title"), "description": p.get("description"),
                     "start": p.get("start"), "end": p.get("end")} for p in d.get("project") or []],
        "own_datasets_to_be_collected": own,
        "reuse_datasets": reuse,
    }


_PROMPT = """You design ONE computational research study that an automated AI scientist will implement and
run now, inside an isolated container, as the first step of the research plan below.

=== RESEARCH PLAN (Markdown; may be in Japanese) ===
{plan}

=== DATA MANAGEMENT PLAN (maDMP, condensed) ===
{madmp}

=== DATA FILES AVAILABLE INSIDE THE CONTAINER (already acquired through the data gatekeeper) ===
{data}

=== HARD CONSTRAINTS ===
- The plan's own primary data ("own_datasets_to_be_collected") will be collected in the future and DO NOT
  exist yet. Do not assume access to them, and do not pretend that simulated data is real data.
- No internet access. Python with numpy, pandas, scipy, scikit-learn, torch, matplotlib, seaborn.
  statsmodels is NOT installed (implement regression/ANCOVA with numpy, scipy or scikit-learn).
- Each experiment run must finish within a few minutes.
- The available data files are exactly as described above (row counts and columns are measured).
  If they are small or are demo substitutes, the study must not draw substantive conclusions from them;
  use them only for what they can support, and say so in the limitations.
- A good choice is a simulation study that tests the plan's own design and analysis before any participant
  is recruited, using the numbers stated in the plan (sample sizes, expected effects, outcomes, follow-up,
  analysis methods): e.g. power and sample size for the stated hypotheses, the planned ANCOVA versus a
  repeated-measures alternative under realistic drop-out, or sensitivity to the assumed effect sizes.
- Write everything in English. Keep the study tied to a specific research question or hypothesis of the plan.

Answer with JSON only, inside triple backticks:
```json
{{"Name": "snake_case_ascii_name_max_50_chars",
  "Title": "...",
  "Short Hypothesis": "...",
  "Related Work": "...",
  "Abstract": "...",
  "Experiments": "numbered, concrete steps the AI scientist will implement",
  "Risk Factors and Limitations": "...",
  "Plan Link": "which research question / hypothesis of the plan this study addresses, quoted from the plan"}}
```"""


def data_block(manifest: list[dict]) -> str:
    if not manifest:
        return "(none)"
    lines = []
    for d in manifest:
        lines.append(
            f"- {d.get('path')} : \"{d.get('title')}\" (dataset {d.get('datasetIdentifier')}, access class "
            f"{d.get('accessClass')}); measured {d.get('rows')} data rows, columns {d.get('columns')}; "
            f"intended use in the plan: {d.get('intendedUse')}"
            + (f"; the file itself says: {' / '.join(d.get('notes') or [])}" if d.get("notes") else ""))
    return "\n".join(lines)


def data_rules(manifest: list[dict]) -> str:
    """実験の指示に固定で足す、データの扱いの決まり(モデルの出力に頼らない)。"""
    if not manifest:
        return ("\n\nDATA RULES (fixed by the platform): No external data files are provided for this study. "
                "Any data you use must be generated by your own code and clearly labelled as simulated. "
                "Never claim that simulated data are real observations. There is no internet access.")
    files = "\n".join(f"  - {d.get('path')} ({d.get('rows')} data rows; columns {d.get('columns')}; "
                      f"\"{d.get('title')}\")" for d in manifest)
    return (
        "\n\nDATA RULES (fixed by the platform):\n"
        "The following files were acquired through the data gatekeeper and are mounted read-only:\n"
        f"{files}\n"
        "Read them directly from these paths with pandas (lines starting with '#' are comments: "
        "use comment='#'). Your code MUST load every file listed above at the start and print a data "
        "inventory (path, number of rows, column names), even if the study does not use a file further. "
        "Do NOT create mock, fake, placeholder or simulated replacements of these files, do NOT define your "
        "own rdc_data client, and do NOT copy their contents into your code. If a file cannot be read, raise "
        "the error instead of substituting data. Any additional data must be generated by your own code and "
        "clearly labelled as simulated. There is no internet access.")


def slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")
    return s[:50].strip("_") or "rdc_plan_study"


def validate(idea: dict) -> str | None:
    for k in REQUIRED:
        if not isinstance(idea.get(k), str) or len(idea[k].strip()) < 20 and k != "Name":
            return f"{k} が空か短すぎます"
    if re.search(r"[぀-ヿ一-鿿]", idea["Experiments"] + idea["Title"]):
        return "題名か実験の指示に日本語が残っています(実験のコードを書くモデルが読み違えるため英語にする)"
    return None


def ideate(xdir: str, model: str, retries: int) -> int:
    from ai_scientist.llm import create_client, extract_json_between_markers, get_response_from_llm

    inp = osp.join(xdir, "input")
    status(xdir, "running", "研究計画と DMP を読んでいます")
    try:
        plan = open(osp.join(inp, "plan.md"), encoding="utf-8").read()
        madmp = json.load(open(osp.join(inp, "madmp.json"), encoding="utf-8"))
        manifest = json.load(open(osp.join(inp, "data_manifest.json"), encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        status(xdir, "failed", f"入力を読めませんでした: {e}"[:300])
        return 1
    if len(plan.strip()) < 200:
        status(xdir, "failed", "研究計画書の本文がありません")
        return 1
    prompt = _PROMPT.format(plan=plan[:20000],
                            madmp=json.dumps(condense_madmp(madmp), ensure_ascii=False, indent=1)[:15000],
                            data=data_block(manifest))
    client, client_model = create_client(model)
    last = ""
    for attempt in range(1, retries + 1):
        status(xdir, "running", f"研究計画から題材を作っています({attempt}/{retries} 回目)", attempt=attempt)
        try:
            out, _ = get_response_from_llm(
                prompt=prompt, client=client, model=client_model,
                system_message="You are a careful research scientist. Never invent data that does not exist.",
                temperature=0.4)
            idea = extract_json_between_markers(out)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            last = f"{type(e).__name__}: {e}"
            continue
        if not isinstance(idea, dict):
            last = "題材の JSON を読み取れませんでした"
            continue
        why = validate(idea)
        if why:
            last = why
            print(f"[rdc] 題材を作り直します: {why}", flush=True)
            continue
        idea["Name"] = slug(idea["Name"])
        idea["Experiments"] = idea["Experiments"].rstrip() + data_rules(manifest)
        idea["x_rdc_source"] = {
            "plan_title": (condense_madmp(madmp).get("project") or [{}])[0].get("title"),
            "plan_link": idea.pop("Plan Link", None),
            "data_files": [d.get("path") for d in manifest],
        }
        with open(osp.join(xdir, "idea.json"), "w", encoding="utf-8") as f:
            json.dump([idea], f, ensure_ascii=False, indent=2)
        status(xdir, "done", "研究計画から題材を作りました", name=idea["Name"], title=idea["Title"],
               plan_link=idea["x_rdc_source"]["plan_link"], data_files=len(manifest), attempt=attempt)
        return 0
    status(xdir, "failed", f"{retries} 回試しましたが、題材を作れませんでした: {last}"[:300])
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exchange_dir")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--retries", type=int, default=3)
    a = ap.parse_args()
    return ideate(osp.abspath(a.exchange_dir), a.model, a.retries)


if __name__ == "__main__":
    sys.exit(main())
