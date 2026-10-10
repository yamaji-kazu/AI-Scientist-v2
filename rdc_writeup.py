"""NII RDC: 研究実行(④ 実験)の結果から、論文執筆(⑤)と自己査読(⑥)を 1 段ずつ回す。

    python rdc_writeup.py writeup <idea_dir> [--model gpt-4o]
    python rdc_writeup.py review  <idea_dir> [--model gpt-4o]

launch_scientist_bfts.py の論文執筆の段をそのまま使わない理由(2026-10-11 に .14 で確かめた):
- 実験は generate_report: False で回しており、論文執筆が読む要約(logs/0-run/*_summary.json)が
  作られていない。ここで manager.pkl から作る。ログの置き場所も logs/0-run ではなく
  logs/0-<exp_name> なので、logs/0-run をそこへのリンクにする
- 8 ページ形式の perform_writeup.py は LLM の呼び方が古い(msg=)ままで動かない。4 ページの
  ワークショップ形式(perform_icbinb_writeup.py)を使う
- 文献の検索(Semantic Scholar)は外向き通信を要する。.14 は外向き通信を閉じているので
  num_cite_rounds=0 で回し、引用の検索をしなかったことを状態に残す
- 査読の図の確認(perform_imgs_cap_ref_review)は画像を読むモデルを要する。Qwen3-Coder は
  画像を読めないので、本文の査読だけを必須とし、図の確認は失敗しても段を落とさない

- 最後の「ページ数に収める」推敲で、モデルが実験の中身を捨てて作り話に書き換えることがある
  (2026-10-11 の試し: 推敲 1 回目は two-moons の MLP の幅の研究、最後の版は実験に無い
  CIFAR-10/ResNet-18 の話で図も 0 枚)。推敲の各版のうち、題材(idea)の語をいちばん多く含む版を
  論文として採り、rdc_paper.pdf に写す。半分に満たなければ論文執筆は失敗とする
- 査読には「実験の記録との照合」を足す。題材の設計と実験の要約に照らして、食い違う主張(作り話の
  疑い)と、記録では確かめられない主張を分けて挙げる。食い違いが 0 件のときだけ「整合」とする

進み具合と結果は <idea_dir>/rdc_phase.json に書く(実行器が読み、DG に知らせる)。
成功したとき(論文の PDF / 査読の結果が実在するとき)だけ終了コード 0 を返す。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import pickle
import re
import shutil
import subprocess
import sys
import time
import traceback


def status(idea_dir: str, phase: str, state: str, step: str, **extra) -> None:
    path = osp.join(idea_dir, "rdc_phase.json")
    try:
        cur = json.load(open(path))
    except Exception:  # noqa: BLE001
        cur = {}
    cur[phase] = {"state": state, "step": step, "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra}
    with open(path + ".tmp", "w") as f:
        json.dump(cur, f, ensure_ascii=False, indent=2)
    os.replace(path + ".tmp", path)
    print(f"[rdc] {phase}: {state} — {step}", flush=True)


def logs_dir(idea_dir: str) -> str:
    cands = [d for d in glob.glob(osp.join(idea_dir, "logs", "*")) if osp.exists(osp.join(d, "manager.pkl"))]
    if not cands:
        raise RuntimeError("実験のログ(logs/*/manager.pkl)が見つかりません")
    return sorted(cands)[0]


def ensure_summaries(idea_dir: str, model: str) -> list[str]:
    """論文執筆が読む要約を用意する。既にあれば作り直さない。"""
    ld = logs_dir(idea_dir)
    run_link = osp.join(idea_dir, "logs", "0-run")
    if osp.basename(ld) != "0-run" and not osp.exists(run_link):
        os.symlink(osp.basename(ld), run_link)
    names = ["draft_summary", "baseline_summary", "research_summary", "ablation_summary"]
    if all(osp.exists(osp.join(ld, f"{n}.json")) for n in names):
        return names
    from omegaconf import OmegaConf

    from ai_scientist.treesearch.log_summarization import overall_summarize

    cfg = OmegaConf.load(osp.join(idea_dir, "bfts_config.yaml"))
    if cfg.agent.get("summary", None) is None:
        cfg.agent.summary = {"model": model, "temp": 0.3}
    with open(osp.join(ld, "manager.pkl"), "rb") as f:
        saved = pickle.load(f)
    # 実験の版によって、AgentManager そのものか journals.items() が入っている
    journals = saved.journals.items() if hasattr(saved, "journals") else saved
    results = overall_summarize(list(journals), cfg)
    for n, r in zip(names, results):
        with open(osp.join(ld, f"{n}.json"), "w") as f:
            json.dump(r, f, indent=2)
    return names


PAPER = "rdc_paper.pdf"
_STOP = {"with", "from", "that", "this", "study", "using", "based", "small", "set", "by", "an",
         "the", "and", "for", "into", "over", "under", "via", "data", "dataset", "size", "table"}


def pdf_text(pdf: str) -> str:
    r = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True, timeout=60)
    return r.stdout if r.returncode == 0 else ""


def idea_terms(idea_dir: str) -> list[str]:
    """題材(idea)を表す語。名前(snake_case)と題名の 4 文字以上の語から、ありふれた語を除く。"""
    try:
        idea = json.load(open(osp.join(idea_dir, "idea.json")))
        if isinstance(idea, list):
            idea = idea[0]
    except Exception:  # noqa: BLE001
        return []
    words = re.split(r"[^A-Za-z0-9]+", f"{idea.get('Name', '')} {idea.get('Title', '')}".replace("_", " "))
    out: list[str] = []
    for w in words:
        w = w.lower()
        if (len(w) >= 4 or w == "mlp") and w not in _STOP and w not in out:
            out.append(w)
    return out


def grounding(text: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    low = text.lower()
    return sum(1 for t in terms if t in low) / len(terms)


def choose_paper(idea_dir: str) -> tuple[str | None, list[dict]]:
    """推敲の各版から、題材の語をいちばん多く含む版を選ぶ(同点なら新しい版)。"""
    terms = idea_terms(idea_dir)
    cands = []
    for f in os.listdir(idea_dir):
        if f.endswith(".pdf") and f != PAPER:
            p = osp.join(idea_dir, f)
            cands.append({"pdf": f, "score": round(grounding(pdf_text(p), terms), 2),
                          "mtime": osp.getmtime(p)})
    cands.sort(key=lambda c: (c["score"], c["mtime"]), reverse=True)
    for c in cands:
        c.pop("mtime")
    return (cands[0]["pdf"] if cands else None), cands


def writeup(idea_dir: str, model: str, retries: int) -> int:
    status(idea_dir, "writeup", "running", "実験の要約を作っています")
    try:
        ensure_summaries(idea_dir, model)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        status(idea_dir, "writeup", "failed", f"実験の要約を作れませんでした: {e}"[:300])
        return 1
    # 8 ページ形式(perform_writeup.py)は LLM の呼び方が古いまま(msg=)で、この版では必ず
    # TypeError になる(2026-10-11 実測)。4 ページのワークショップ形式(ICBINB)を使う。
    from ai_scientist.perform_icbinb_writeup import perform_writeup

    cands: list[dict] = []
    for attempt in range(1, retries + 1):
        status(idea_dir, "writeup", "running", f"論文を書いています({attempt}/{retries} 回目)",
               attempt=attempt, citations="none (no external network)")
        try:
            ok = perform_writeup(base_folder=idea_dir, citations_text="", num_cite_rounds=0,
                                 small_model=model, big_model=model, n_writeup_reflections=2,
                                 page_limit=4)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            ok = False
        best, cands = choose_paper(idea_dir)
        score = cands[0]["score"] if cands else 0.0
        if best and score >= 0.5:
            shutil.copyfile(osp.join(idea_dir, best), osp.join(idea_dir, PAPER))
            status(idea_dir, "writeup", "done", "論文を書き終えました", pdf=PAPER, chosen=best,
                   grounding=score, candidates=cands, terms=idea_terms(idea_dir), attempt=attempt,
                   citations="none (no external network)")
            return 0
        if best:
            print(f"[rdc] どの版も題材の語が半分に満たない(最高 {score}): {cands}", flush=True)
    status(idea_dir, "writeup", "failed",
           f"{retries} 回試しましたが、実験に沿った論文の PDF を組めませんでした",
           candidates=cands)
    return 1


_FACT_PROMPT = """You are checking a research paper against what was actually planned and run.

RESEARCH DESIGN (the idea the experiments implemented):
{design}

EXPERIMENT RECORDS (JSON summaries written while the experiments ran):
{records}

PAPER (text extracted from the PDF):
{paper}

Classify the paper's concrete claims (datasets, models, settings, numbers, findings):
- "contradicted": the design or the records say something different (e.g. another dataset or model,
  a number that does not match). These may be fabricated.
- "unverifiable": not contradicted, but neither the design nor the records mention it.
Do not list claims that the design or records support. Answer only with JSON in triple backticks:
```json
{{"contradicted": [{{"claim": "...", "evidence": "what the design/records say instead"}}],
  "unverifiable": [{{"claim": "..."}}],
  "supported_points": ["..."]}}
```"""


def fact_check(idea_dir: str, paper_text: str, client, model: str) -> dict:
    """論文を実験の記録(要約)と照らし合わせ、裏付けの無い主張を挙げる(自己査読の一部)。"""
    from ai_scientist.llm import extract_json_between_markers, get_response_from_llm

    ld = logs_dir(idea_dir)
    rec = {}
    for n in ("baseline_summary", "research_summary", "ablation_summary"):
        try:
            rec[n] = json.load(open(osp.join(ld, f"{n}.json")))
        except Exception:  # noqa: BLE001
            pass
    records = json.dumps(rec, ensure_ascii=False)[:50000]
    design = ""
    for name in ("idea.md", "research_idea.md"):
        if osp.exists(osp.join(idea_dir, name)):
            design = open(osp.join(idea_dir, name)).read()[:12000]
            break
    out, _ = get_response_from_llm(
        prompt=_FACT_PROMPT.format(design=design, records=records, paper=paper_text[:40000]),
        client=client, model=model,
        system_message="You are a careful scientific fact checker. Do not invent records.",
        temperature=0.2)
    j = extract_json_between_markers(out)
    if not isinstance(j, dict) or not isinstance(j.get("contradicted"), list):
        return {"error": "照合の結果を読み取れませんでした"}
    j["consistent"] = len(j["contradicted"]) == 0
    return j


def review(idea_dir: str, model: str) -> int:
    pdf = osp.join(idea_dir, PAPER)
    if not osp.exists(pdf):
        status(idea_dir, "review", "failed", "査読する論文(PDF)がありません")
        return 1
    status(idea_dir, "review", "running", "論文を読んで査読しています", pdf=osp.basename(pdf))
    from ai_scientist.llm import create_client
    from ai_scientist.perform_llm_review import load_paper, perform_review

    try:
        client, client_model = create_client(model)
        text = load_paper(pdf)
        rv = perform_review(text, client_model, client, num_reflections=1, num_fs_examples=1)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        status(idea_dir, "review", "failed", f"査読できませんでした: {e}"[:300])
        return 1
    if not isinstance(rv, dict) or not rv:
        status(idea_dir, "review", "failed", "査読の結果を読み取れませんでした")
        return 1
    with open(osp.join(idea_dir, "review_text.txt"), "w") as f:
        f.write(json.dumps(rv, indent=4, ensure_ascii=False))
    figs = "skipped"
    try:
        from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review

        r2 = perform_imgs_cap_ref_review(client, client_model, pdf)
        with open(osp.join(idea_dir, "review_img_cap_ref.json"), "w") as f:
            json.dump(r2, f, indent=4)
        figs = "done"
    except Exception as e:  # noqa: BLE001 画像を読めないモデルでは落ちる。本文の査読は済んでいる
        figs = f"skipped: {type(e).__name__}"
    status(idea_dir, "review", "running", "論文を実験の記録と照らし合わせています")
    try:
        check = fact_check(idea_dir, text, client, client_model)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        check = {"error": f"{type(e).__name__}: {e}"[:200]}
    with open(osp.join(idea_dir, "review_fact_check.json"), "w") as f:
        json.dump(check, f, indent=2, ensure_ascii=False)
    keep = {k: rv.get(k) for k in ("Overall", "Decision", "Soundness", "Presentation",
                                   "Contribution", "Confidence") if k in rv}
    def n(key):
        v = check.get(key) if isinstance(check, dict) else None
        return len(v) if isinstance(v, list) else None
    status(idea_dir, "review", "done", "査読を終えました", pdf=PAPER, scores=keep,
           figure_review=figs, fact_check={
               "consistent": check.get("consistent") if isinstance(check, dict) else None,
               "contradicted": n("contradicted"), "unverifiable": n("unverifiable"),
               "error": check.get("error") if isinstance(check, dict) else None})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["writeup", "review"])
    ap.add_argument("idea_dir")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args()
    idea_dir = osp.abspath(a.idea_dir)
    return writeup(idea_dir, a.model, a.retries) if a.phase == "writeup" else review(idea_dir, a.model)


if __name__ == "__main__":
    sys.exit(main())
