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
  num_cite_rounds=0 で回す。文献は、研究計画のもとになった DR の調査の参考文献と、計画で再利用する
  データ(引用の義務 rdc:cite)を、実行器が <idea_dir>/rdc_references.bib に置く(2026-10-11)。
  無ければ文献なしで書き、そのことを状態に残す
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


def abstract_of_tex(tex: str | None) -> str | None:
    """LaTeX の abstract 環境を、投稿の要旨に使える平文にする(無ければ None)。

    2026-10-14、エージェントが OJS に送った投稿が「要旨は必須」で断られた(OJS の雑誌は既定で要旨を必須にする)。
    DG は論文の中身を持たないので、ここで採った版の LaTeX から取り出し、段の記録に残して実行器から DG に渡す
    (aifs ADR-35 §11)。引用・参照は外し、書体の命令は中身だけ残す。数式($…$)はそのまま残す
    """
    if not tex:
        return None
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL)
    if not m:
        return None
    t = m.group(1)
    for esc, ch in (("\\%", "\x00P"), ("\\&", "&"), ("\\_", "_"), ("\\#", "#"), ("\\$", "\x00D")):
        t = t.replace(esc, ch)
    t = re.sub(r"%.*", "", t)  # コメント
    t = re.sub(r"\\(?:cite[a-z]*|ref|eqref|[cC]ref|label|autoref)\*?(?:\[[^\]]*\])*\{[^}]*\}", "", t)
    for _ in range(3):  # \textbf{x} → x(入れ子は 3 段まで)
        t = re.sub(r"\\[A-Za-z]+\*?\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\[A-Za-z]+\*?", " ", t)
    t = t.replace("~", " ").replace("{", "").replace("}", "")
    t = t.replace("\x00P", "%").replace("\x00D", "$")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.;:、。)\]])", r"\1", t)
    return t[:4000] or None


def abstract_of_file(path: str) -> str | None:
    try:
        return abstract_of_tex(open(path, encoding="utf-8").read())
    except OSError:
        return None


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


REFS = "rdc_references.bib"
REFS_META = "rdc_references.json"


def references(idea_dir: str) -> tuple[str, list[str]]:
    """実行器が置いた文献(BibTeX)と、その鍵の一覧。無ければ空。"""
    p = osp.join(idea_dir, REFS)
    if not osp.exists(p):
        return "", []
    bib = open(p, encoding="utf-8").read()
    return bib, re.findall(r"@\w+\{([^,\s]+),", bib)


def cited_in(pdf: str, idea_dir: str) -> dict:
    """論文の PDF に、渡した文献が何件載ったか(題名の頭で照らす)。データの引用は義務なので分けて数える。"""
    try:
        meta = json.load(open(osp.join(idea_dir, REFS_META), encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    text = re.sub(r"\s+", "", pdf_text(pdf))
    def shown(e):
        t = re.sub(r"\s+", "", e.get("title") or "")[:12]
        return bool(t) and t in text
    lit = [e for e in meta if e.get("kind") == "literature"]
    ds = [e for e in meta if e.get("kind") == "dataset"]
    return {"literature": sum(1 for e in lit if shown(e)), "literature_given": len(lit),
            "datasets": sum(1 for e in ds if shown(e)), "datasets_given": len(ds),
            "datasets_not_cited": [e.get("key") for e in ds if not shown(e)],
            "unresolved": pdf_text(pdf).count("[?]") + len(re.findall(r"\(\?\)|\s\?\.", pdf_text(pdf)))}


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

    bib, keys = references(idea_dir)
    cites = f"{len(keys)} entries (DR survey and reused datasets)" if keys else "none (no references given)"
    cands: list[dict] = []
    last_reason = None
    for attempt in range(1, retries + 1):
        status(idea_dir, "writeup", "running", f"論文を書いています({attempt}/{retries} 回目)",
               attempt=attempt, citations=cites)
        try:
            ok = perform_writeup(base_folder=idea_dir, citations_text=bib, num_cite_rounds=0,
                                 small_model=model, big_model=model, n_writeup_reflections=2,
                                 page_limit=4)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            ok = False
        # 黙って失敗していた箇所の理由(出力の上限で切れた など)を、最後の失敗の文言に使う
        import ai_scientist.perform_icbinb_writeup as icbinb
        last_reason = icbinb.LAST_FAILURE or last_reason
        best, cands = choose_paper(idea_dir)
        score = cands[0]["score"] if cands else 0.0
        if best and score >= 0.5:
            shutil.copyfile(osp.join(idea_dir, best), osp.join(idea_dir, PAPER))
            status(idea_dir, "writeup", "done", "論文を書き終えました", pdf=PAPER, chosen=best,
                   grounding=score, candidates=cands, terms=idea_terms(idea_dir), attempt=attempt,
                   citations=cites, cited=cited_in(osp.join(idea_dir, PAPER), idea_dir),
                   # 採った版の LaTeX(PDF の隣に残る)の要旨。投稿に使う(aifs ADR-35 §11)
                   abstract=abstract_of_file(osp.join(idea_dir, osp.splitext(best)[0] + ".tex")))
            # 日本語版は英語の論文の訳。失敗しても英語の論文は完了のまま(理由は writeup.ja に残る)
            translate_ja(idea_dir, model)
            return 0
        if best:
            print(f"[rdc] どの版も題材の語が半分に満たない(最高 {score}): {cands}", flush=True)
    # PDF ができたが題材に沿わなかったのか、PDF まで行かなかったのかを分けて出す
    if cands:
        step = f"{retries} 回試しましたが、実験に沿った論文の PDF を組めませんでした(題材の語が半分に満たない)"
    elif last_reason:
        step = f"{retries} 回試しましたが、論文の PDF を組む前に止まりました: {last_reason}"
    else:
        step = f"{retries} 回試しましたが、論文の PDF を組めませんでした"
    status(idea_dir, "writeup", "failed", step[:300], candidates=cands, reason=last_reason)
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


# ─────────────────────── 自己査読を題材の分野に合わせる ───────────────────────
# AI-Scientist の査読は NeurIPS の書式と機械学習の論文の査読例で、ICLR の査読として行う。機械学習で
# ない題材は書き方に関わらず「範囲外」で落ち、点数と判定が意味を持たなかった(run-41bc6352: 認知症
# リハビリの RCT の検出力分析で、Weaknesses の大半が「機械学習の範囲外」)。題材から分野を選び、
# その分野の査読の観点で行う(2026-10-12)。出力の項目と尺度(Overall 1–10、Decision など)は
# 変えないので、実行器・DG・書き直しはそのまま動く。機械学習は今までどおりの書式と査読例を使う。
FIELD_FILE = "rdc_review_field.json"
REVIEW_FIELDS = {
    "machine_learning": {
        "ja": "機械学習・AI", "en": "Machine learning / AI",
        "scope": "new machine learning or AI methods, models, training or evaluation",
    },
    "health_medicine": {
        "ja": "医学・保健・看護・リハビリテーション", "en": "Health, medicine, nursing and rehabilitation",
        "scope": "clinical research, public health, epidemiology, nursing, rehabilitation, care and trial design",
        "venue": "a peer-reviewed journal in clinical medicine and health sciences",
        "criteria": [
            "Is the study design appropriate to the question, and is it reported according to the relevant guideline "
            "(CONSORT for randomized trials, SPIRIT for trial protocols, STROBE for observational studies, PRISMA for reviews)?",
            "Are the sample size and statistical power justified, with the assumptions (effect size, variance, attrition) stated and sourced?",
            "Are the outcome measures valid, reliable and clinically meaningful (e.g. minimal clinically important difference)?",
            "Are bias, confounding, randomization, blinding and missing data handled and discussed?",
            "Are ethics, consent and data protection addressed?",
            "Is clinical significance distinguished from statistical significance, and is generalizability discussed?",
        ],
    },
    "social_sciences": {
        "ja": "社会科学(教育・経済・政策・心理)", "en": "Social sciences (education, economics, policy, psychology)",
        "scope": "sociology, education, economics, political science, policy, psychology, survey and qualitative research",
        "venue": "a peer-reviewed journal in the social sciences",
        "criteria": [
            "Is the research question grounded in theory and prior literature?",
            "Are concepts operationalized and measured validly and reliably?",
            "Is the sampling appropriate, and are its limits for inference stated?",
            "Is the causal claim, if any, supported by an appropriate identification strategy?",
            "For qualitative work: are the methods transparent and the interpretation credible?",
            "Are ethics, consent and the protection of participants addressed?",
        ],
    },
    "natural_sciences": {
        "ja": "自然科学(物理・化学・生物・地球)", "en": "Natural sciences (physics, chemistry, biology, earth)",
        "scope": "physics, chemistry, biology, earth and environmental sciences, laboratory or field experiments",
        "venue": "a peer-reviewed journal in the natural sciences",
        "criteria": [
            "Is the experimental or observational design sound, with appropriate controls and replicates?",
            "Are measurement uncertainties quantified and propagated?",
            "Are the results consistent with established theory, or is any discrepancy explained?",
            "Are the methods described well enough to reproduce the work?",
            "Are the conclusions proportionate to the evidence?",
        ],
    },
    "statistics_methodology": {
        "ja": "統計学・方法論", "en": "Statistics and methodology",
        "scope": "statistical methods, simulation studies, methodology for study design or data analysis",
        "venue": "a peer-reviewed journal in statistics or research methodology",
        "criteria": [
            "Are the assumptions of the methods stated and checked?",
            "For simulation studies: are the aims, data-generating mechanisms, estimands, methods and performance measures "
            "(ADEMP) clearly specified, and is the Monte Carlo error reported?",
            "Is the method compared with appropriate existing methods?",
            "Are the results reproducible (code, seeds, software versions)?",
            "Are the practical implications and the limits of the method stated?",
        ],
    },
    "humanities": {
        "ja": "人文学", "en": "Humanities",
        "scope": "history, literature, philosophy, linguistics, cultural studies",
        "venue": "a peer-reviewed journal in the humanities",
        "criteria": [
            "Are the sources and materials appropriate and properly cited?",
            "Is the argument coherent, and are interpretations supported by the sources?",
            "Is the work positioned within the existing scholarship?",
            "Are the limits of the interpretation acknowledged?",
        ],
    },
    "general": {
        "ja": "分野を問わない一般の観点", "en": "General (field-independent)",
        "scope": "anything that does not fit the other fields",
        "venue": "a peer-reviewed scholarly journal in the paper's own field",
        "criteria": [
            "Is the research question clear and worth asking in its field?",
            "Are the methods appropriate to the question and described well enough to reproduce?",
            "Are the claims supported by the evidence presented?",
            "Are the limitations stated honestly?",
        ],
    },
}


def _field_form(key: str) -> tuple[str, str, int]:
    """分野の査読の(system prompt, 書式, 査読例の数)。出力の項目と尺度は上流の template_instructions のまま。"""
    from ai_scientist.perform_llm_review import neurips_form, reviewer_system_prompt_neg, template_instructions
    if key == "machine_learning":
        return reviewer_system_prompt_neg, neurips_form, 1
    f = REVIEW_FIELDS.get(key) or REVIEW_FIELDS["general"]
    system = (f"You are an expert reviewer for {f['venue']}. Review the paper by the standards of its own field "
              f"({f['en']}), not by the standards of machine learning. Be critical and cautious in your decision. "
              "If a paper is bad or you are unsure, give it bad scores and reject it.")
    criteria = "\n".join(f"  - {c}" for c in f["criteria"])
    form = f"""
## Review Form
You are reviewing a paper for {f['venue']}. Judge it by the standards of this field ({f['en']}).
Do not judge whether the paper belongs to machine learning; that is not a criterion here.

1. Summary: Briefly summarize the paper and its contributions.
  - Strengths and Weaknesses: assess the paper on the following field-specific points:
{criteria}
  - Originality, Quality, Clarity and Significance, as understood in this field.
2. Questions: questions for the authors whose answers could change your opinion.
3. Limitations: have the authors addressed the limitations and potential negative impact of their work?
4. Ethical concerns: flag the paper if there are ethical issues.
5. Soundness (1-4): are the methods appropriate and the central claims supported by evidence? 4 excellent, 3 good, 2 fair, 1 poor.
6. Presentation (1-4): clarity of writing and contextualization relative to prior work in the field.
7. Contribution (1-4): importance of the question and value of the results to the field.
8. Overall (1-10): 10 exceptional, 8 strong accept, 7 accept, 6 weak accept, 5 borderline accept, 4 borderline reject,
   3 reject, 2 strong reject, 1 very strong reject — judged as a submission to {f['venue']}.
9. Confidence (1-5): how certain you are of your assessment.
""" + template_instructions
    return system, form, 0  # 機械学習の論文の査読例は、ほかの分野では外す


def review_field(idea_dir: str, client, model: str) -> dict:
    """題材(idea)から査読の分野を選ぶ。一度選んだら rdc_review_field.json に残し、書き直しの後の査読でも同じ分野にする。
    環境変数 RDC_REVIEW_FIELD で決め打ちもできる。選べなかったとき・一覧に無い答えは general にする。"""
    path = osp.join(idea_dir, FIELD_FILE)
    if osp.exists(path):
        try:
            got = json.load(open(path))
            if got.get("key") in REVIEW_FIELDS:
                return got
        except Exception:  # noqa: BLE001
            pass
    forced = os.environ.get("RDC_REVIEW_FIELD", "").strip()
    if forced in REVIEW_FIELDS:
        got = {"key": forced, "source": "env", "reason": "RDC_REVIEW_FIELD"}
    else:
        try:
            idea = json.load(open(osp.join(idea_dir, "idea.json")))
            idea = idea[0] if isinstance(idea, list) else idea
        except Exception:  # noqa: BLE001
            idea = {}
        topic = "\n".join(f"{k}: {idea.get(k)}" for k in ("Title", "Short Hypothesis", "Abstract") if idea.get(k))[:4000]
        choices = "\n".join(f"- {k}: {v['scope']}" for k, v in REVIEW_FIELDS.items())
        from ai_scientist.llm import extract_json_between_markers, get_response_from_llm
        got = {"key": "general", "source": "fallback", "reason": "題材を読めなかった"}
        if topic:
            try:
                out, _ = get_response_from_llm(
                    prompt=(f"Which research field should review this paper? Choose exactly one key.\n\n{choices}\n\n"
                            f"PAPER TOPIC:\n{topic}\n\n"
                            'Answer in JSON between ```json and ```: {"field": "<key>", "reason": "<one sentence>"}'),
                    client=client, model=model, system_message="You classify research papers by field.",
                    temperature=0.0)
                j = extract_json_between_markers(out) or {}
                if j.get("field") in REVIEW_FIELDS:
                    got = {"key": j["field"], "source": "llm", "reason": str(j.get("reason") or "")[:300]}
                else:
                    got = {"key": "general", "source": "fallback", "reason": f"一覧に無い答え: {str(j.get('field'))[:60]}"}
            except Exception as e:  # noqa: BLE001
                got = {"key": "general", "source": "fallback", "reason": f"分野を選べなかった: {type(e).__name__}"}
    got["ja"], got["en"] = REVIEW_FIELDS[got["key"]]["ja"], REVIEW_FIELDS[got["key"]]["en"]
    json.dump(got, open(path, "w"), ensure_ascii=False, indent=2)
    return got


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
        field = review_field(idea_dir, client, client_model)
        status(idea_dir, "review", "running", f"論文を読んで査読しています({field['ja']}の観点)",
               pdf=osp.basename(pdf), field=field)
        system, form, n_fs = _field_form(field["key"])
        rv = perform_review(text, client_model, client, num_reflections=1, num_fs_examples=n_fs,
                            reviewer_system_prompt=system, review_instruction_form=form)
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
    status(idea_dir, "review", "running", "論文を実験の記録と照らし合わせています", field=field)
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
    status(idea_dir, "review", "done", "査読を終えました", pdf=PAPER, scores=keep, field=field,
           figure_review=figs, fact_check={
               "consistent": check.get("consistent") if isinstance(check, dict) else None,
               "contradicted": n("contradicted"), "unverifiable": n("unverifiable"),
               "error": check.get("error") if isinstance(check, dict) else None})
    return 0


# ─────────────────────── 日本語版(英語の論文の訳) ───────────────────────
# 英語の論文ができた後に、その LaTeX を節ごとに日本語に訳し、rdc_paper_ja.pdf に組む(2026-10-12)。
# 一から書かせず訳にするのは、数値・図・引用を英語版と同じにし、自己査読と実験の記録との照合が
# 済んだ英語版の「訳」という位置づけをはっきりさせるため。訳で数値や引用が落ちたり変わったり
# しないよう、節ごとに機械で照らし、合わなければ 1 度だけ訳し直す。それでも合わなければ日本語版は
# 出さず、理由を残す(英語の論文は完了のまま)。
PAPER_JA = "rdc_paper_ja.pdf"

# 日本語版は lualatex + luatexja で組む(2026-10-14)。それまでの pdflatex + CJKutf8 は、使える和文の字形が
# 和田研明朝しか無く字形にむらがあり、和欧間の空きや禁則も組まなかった(run-30a1f12c の日本語版で確かめた)。
# 和文の字形は luatexja の既定の原ノ味明朝・角ゴシック(texlive-lang-japanese に入っている。像は ais-exp:rdc-tex-ja)。
# fontspec で欧文を差し替えると、像に OpenType の Times 系(TeX Gyre)が無く欧文が組めなかったので、欧文は
# 英語版と同じ times(Nimbus)のまま、T1 で組む(lualatex の既定の TU には times の字形が無く Latin Modern になる)。
# 図と表の見出し(Figure / Table)はクラスが出すので、訳では変わらない。ここで「図」「表」にする
_JA_PREAMBLE = ("\\usepackage[T1]{fontenc}\n\\usepackage{luatexja}\n"
                "\\renewcommand{\\figurename}{図}\n\\renewcommand{\\tablename}{表}\n")


def _strip_cjk(tex: str) -> str:
    """英語版を組むときに差し込んだ CJKutf8 の包みを外す(改行の有無に関わらず)。"""
    tex = tex.replace("\\usepackage{CJKutf8}\n", "")
    tex = re.sub(r"\\begin\{CJK\}\{UTF8\}\{min\}\n?", "", tex)
    return re.sub(r"\\end\{CJK\}\n?", "", tex)


JA_CHUNK = 6000  # 1 回に訳す LaTeX の長さ(文字)。出力の上限(WRITEUP_MAX_TOKENS)に十分収まる

_JA_SYSTEM = (
    "You translate scientific papers written in LaTeX from English into Japanese, in a formal academic "
    "style (である調). Rules:\n"
    "- Translate only the natural-language text. Keep every LaTeX command, environment, math, \\cite{...}, "
    "\\citep{...}, \\citet{...}, \\ref{...}, \\label{...}, \\includegraphics{...}, \\url{...} and file name "
    "exactly as given.\n"
    "- Keep every number exactly as written (do not convert, round, or spell out numbers).\n"
    "- Also translate the titles of \\section, \\subsection and \\paragraph, and captions. Write "
    "Figure~\\ref{...} as 図~\\ref{...} and Table~\\ref{...} as 表~\\ref{...}.\n"
    "- Do not add, remove, summarize or explain anything. Keep the same structure and order.\n"
    "- Write only in Japanese (kanji, hiragana, katakana). Never use Korean Hangul or other scripts.\n"
    "- Output only the translated fragment inside one ```latex ... ``` block."
)


def _braced(tex: str, start: int) -> int:
    """tex[start] が '{' のとき、対応する '}' の位置を返す(無ければ -1)。"""
    depth = 0
    for i in range(start, len(tex)):
        c = tex[i]
        if c == "\\":
            continue
        if c == "{" and (i == 0 or tex[i - 1] != "\\"):
            depth += 1
        elif c == "}" and tex[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _tex_facts(tex: str) -> dict:
    """訳の前後で変わってはならないもの。"""
    def keys(pattern):
        out = set()
        for m in re.finditer(pattern, tex):
            out.update(k.strip() for k in m.group(1).split(",") if k.strip())
        return out
    no_math_cmd = re.sub(r"\\[A-Za-z]+", " ", tex)
    return {
        "cite": keys(r"\\cite[pt]?\*?(?:\[[^\]]*\])*\{([^}]*)\}"),
        "ref": keys(r"\\(?:ref|eqref|autoref|cref|Cref)\{([^}]*)\}"),
        "label": keys(r"\\label\{([^}]*)\}"),
        "graphics": keys(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}"),
        "numbers": set(re.findall(r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?(?![0-9])", no_math_cmd.replace(",", ""))),
        "begin": len(re.findall(r"\\begin\{", tex)),
        "end": len(re.findall(r"\\end\{", tex)),
    }


def _check_translation(en: str, ja: str) -> list[str]:
    """英語の断片と訳を照らし、食い違いを文で返す(空なら合っている)。"""
    a, b = _tex_facts(en), _tex_facts(ja)
    problems = []
    for key, label in (("cite", "引用"), ("ref", "参照"), ("label", "ラベル"), ("graphics", "図のファイル")):
        if a[key] != b[key]:
            lost, added = sorted(a[key] - b[key]), sorted(b[key] - a[key])
            problems.append(f"{label}が変わった(消えた {lost[:5]}、増えた {added[:5]})")
    lost = sorted(a["numbers"] - b["numbers"])
    if lost:
        problems.append(f"数値が消えた {lost[:8]}")
    # 英語の three などを 3 と書くのは許す(10 以下の整数)。それ以外に増えた数値は作り話の疑い
    added = sorted(n for n in b["numbers"] - a["numbers"] if not (n.isdigit() and int(n) <= 10))
    if added:
        problems.append(f"英語版に無い数値が増えた {added[:8]}")
    if (a["begin"], a["end"]) != (b["begin"], b["end"]):
        problems.append(f"環境の数が変わった(begin {a['begin']}→{b['begin']}、end {a['end']}→{b['end']})")
    # 和文の字形(wadalab)は JIS の文字集合の分しか無い。ハングルなどが混じると PDF を組めない
    # (2026-10-12 の試し: 「実践者들이」で udmjb4 の字形が無く止まった)
    foreign = sorted({ch for ch in ja if ord(ch) > 127 and not _in_jis(ch)})
    if foreign:
        problems.append(f"日本語の文字集合に無い文字が混じった {foreign[:8]}")
    if not re.search(r"[\u3040-\u30ff]", ja) and re.search(r"[A-Za-z]{4,}", re.sub(r"\\[A-Za-z]+|\{[^}]*\}", "", en)):
        problems.append("日本語になっていない")
    return problems


def _in_jis(ch: str) -> bool:
    try:
        ch.encode("euc_jp")
        return True
    except UnicodeEncodeError:
        return False


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{6,}", text.lower()))


def _split_body(body: str) -> list[str]:
    """本文を節(\\section)ごとに分け、長い節は段落の切れ目でさらに分ける。"""
    parts = [p for p in re.split(r"(?m)^(?=\\section\*?\{)", body) if p]
    out = []
    for p in parts:
        while len(p) > JA_CHUNK:
            cut = p.rfind("\n\n", 0, JA_CHUNK)
            if cut <= 0:
                cut = p.find("\n\n", JA_CHUNK)
            if cut <= 0:
                break
            out.append(p[:cut + 2])
            p = p[cut + 2:]
        out.append(p)
    return out


# Qwen3-Coder は日本語の訳に、韓国語の助詞(ハングル)を混ぜることがある。頼み直しても、温度を変えても、
# 直させても同じ所に同じ字を出した(2026-10-12: practitioners を「実践者들이」)。出力の形を縛る
# (structured_outputs)と節や本文が落ちた。日本語の直後の短い助詞だけを決まった表で置き換え、
# 置き換えたことを状態に残す。表に無いもの・長いハングルは置き換えず、日本語版を出さない
_HANGUL_PARTICLES = {"들이": "たちが", "들은": "たちは", "들을": "たちを", "들의": "たちの", "들에게": "たちに",
                     "들": "たち", "이": "が", "가": "が", "은": "は", "는": "は", "을": "を", "를": "を",
                     "의": "の", "에": "に", "에서": "で", "와": "と", "과": "と", "도": "も"}
_HANGUL_RUN = re.compile(r"(?<=[\u3040-\u30ff\u4e00-\u9fff])([\uac00-\ud7af]{1,3})")


# 2026-10-13: 中国語の簡体字(Qwen3-Coder が Rehabilitation を「康复」と訳した。run-c008e3ee、3 回とも同じ)。
# 言葉で直せるものは言葉で(康复→リハビリテーション。字だけ直すと「康復」という日本語に無い語になる)、
# 残りは字を日本の字体に直す。読み込むときに「元が日本語の文字集合に無く、置き換え先は有る」組だけ残す
# (表の書き誤りで、正しい日本語を壊さないため)
_ZH_WORDS = {"康复": "リハビリテーション", "恢复": "回復", "反复": "反復", "重复": "重複", "复杂": "複雑",
             "复制": "複製", "复合": "複合", "复数": "複数", "答复": "回答", "复习": "復習", "数据": "データ",
             "数据集": "データセット", "随机": "無作為", "样本": "標本", "显著": "有意", "检验": "検定"}
_ZH_CHARS = dict(zip(
    "复测试验实现发结论设计统证评价预应对观关认识疗护组显样变归机时间长动运练训质问题习脑记忆诊线图级层处务产类选择执构规则标准确误异优势效响贡讨议总广际东华养营资经济历纳输获达进过阶头齐单两个这们为与种队员议边见视觉读写说话语词连续维护项详细缓减轻负担荷亲属伙伴协调参与资质据检随区医药剂量饮食睡眠障碍认知功能痴呆症状严重程度变化趋势显示统计数据收集样本",
    "復測試験実現発結論設計統証評価予応対観関認識療護組顕様変帰機時間長動運練訓質問題習脳記憶診線図級層処務産類選択執構規則標準確誤異優勢効響貢討議総広際東華養営資経済歴納輸獲達進過階頭斉単両個這們為与種隊員議辺見視覚読写説話語詞連続維護項詳細緩減軽負担荷親属伙伴協調参与資質据検随区医薬剤量飲食睡眠障碍認知功能痴呆症状厳重程度変化趨勢顕示統計数据収集様本"))
_ZH_CHARS = {k: v for k, v in _ZH_CHARS.items() if k != v and not _in_jis(k) and _in_jis(v)}
_ZH_WORDS = {k: v for k, v in _ZH_WORDS.items() if any(not _in_jis(c) for c in k) and all(_in_jis(c) for c in v)}


def _repair_zh(ja: str) -> tuple[str, list[str]]:
    """簡体字を、言葉の表(長いものから)、次に字の表で日本語に直す。置き換えた記録を返す。"""
    notes: list[str] = []
    for w in sorted(_ZH_WORDS, key=len, reverse=True):
        if w in ja:
            notes.append(f"{w}→{_ZH_WORDS[w]}")
            ja = ja.replace(w, _ZH_WORDS[w])
    out = []
    for ch in ja:
        if ch in _ZH_CHARS:
            notes.append(f"{ch}→{_ZH_CHARS[ch]}")
            out.append(_ZH_CHARS[ch])
        else:
            out.append(ch)
    return "".join(out), notes


def _repair_foreign(ja: str) -> tuple[str, list[str]]:
    """日本語の文字集合に無い文字の修復: ハングルの助詞と、中国語の簡体字。"""
    ja, a = _repair_hangul(ja)
    ja, b = _repair_zh(ja)
    return ja, a + b


def _repair_hangul(ja: str) -> tuple[str, list[str]]:
    """日本語の直後の短いハングル(助詞)を表で置き換える。置き換えた記録を返す。"""
    notes: list[str] = []
    def sub(m):
        run = m.group(1)
        if run in _HANGUL_PARTICLES:
            notes.append(f"{run}→{_HANGUL_PARTICLES[run]}")
            return _HANGUL_PARTICLES[run]
        return run
    return _HANGUL_RUN.sub(sub, ja), notes


def _repair_summary(notes: list[str]) -> str:
    """置き換えの記録を「康复→リハビリテーション ×2、들이→たちが」の形にまとめる。
    2026-10-13 まで、中国語を直したときも「韓国語の助詞を置き換えた」と出ていた(run-c008e3ee の比較で見つけた)。"""
    from collections import Counter
    return "、".join(n + (f" ×{c}" if c > 1 else "") for n, c in Counter(notes).items())


def _feedback_en(en: str, ja: str) -> str:
    """照合の食い違いを、モデルに渡す英語の指示にする(日本語の状態の文とは別に作る)。"""
    a, b = _tex_facts(en), _tex_facts(ja)
    out = []
    foreign = sorted({ch for ch in ja if ord(ch) > 127 and not _in_jis(ch)})
    if foreign:
        out.append(f"These characters are not Japanese and must not appear: {foreign}. "
                   "Rewrite the sentences that contain them in Japanese only.")
    for key in ("cite", "ref", "label", "graphics"):
        if a[key] != b[key]:
            out.append(f"The {key} keys must be exactly {sorted(a[key])}.")
    lost = sorted(a["numbers"] - b["numbers"])
    if lost:
        out.append(f"These numbers from the original are missing: {lost}.")
    added = sorted(n for n in b["numbers"] - a["numbers"] if not (n.isdigit() and int(n) <= 10))
    if added:
        out.append(f"These numbers are not in the original and must be removed: {added}.")
    if (a["begin"], a["end"]) != (b["begin"], b["end"]):
        out.append("Keep every \\begin{...} and \\end{...} of the original.")
    return " ".join(out) or "Follow the rules exactly."


def _translate_piece(text: str, client, model: str) -> tuple[str | None, list[str], list[str]]:
    """1 つの断片を訳す。照合に通らなければ、(2 回目)前の訳と食い違いを渡して直させ、
    (3 回目)温度を変えて訳し直す。同じ頼み方の繰り返しでは、同じ誤りを返した
    (2026-10-12: Qwen3-Coder が practitioners を、温度を変えても「実践者들이」と訳した)。"""
    from ai_scientist.llm import get_response_from_llm
    from ai_scientist.perform_icbinb_writeup import WRITEUP_MAX_TOKENS
    first = f"Translate this LaTeX fragment into Japanese.\n\n```latex\n{text}\n```"
    problems: list[str] = []
    prev = None
    for attempt in (1, 2, 3):
        if attempt == 2 and prev is not None:
            prompt = (f"Original:\n```latex\n{text}\n```\n\nYour Japanese translation:\n```latex\n{prev}\n```\n\n"
                      f"Problems: {_feedback_en(text, prev)}\n"
                      "Fix only these problems and output the whole corrected translation.")
            temperature = 0.2
        else:
            prompt, temperature = first, (0.2 if attempt == 1 else 0.7)
        resp, _ = get_response_from_llm(prompt=prompt, client=client, model=model, system_message=_JA_SYSTEM,
                                        temperature=temperature, max_tokens=WRITEUP_MAX_TOKENS)
        m = re.search(r"```latex(.*?)```", resp, re.DOTALL)
        if not m:
            problems = ["訳に LaTeX のブロックが無い"]
            continue
        ja = m.group(1).strip("\n")
        problems = _check_translation(text, ja)
        if not problems:
            return ja, [], []
        if all(p.startswith("日本語の文字集合に無い文字") for p in problems):
            fixed, notes = _repair_foreign(ja)
            if notes and not _check_translation(text, fixed):
                return fixed, [], notes
        prev = ja
    return None, problems, []


def set_ja(idea_dir: str, **ja) -> None:
    """論文執筆(writeup)の状態に、日本語版の結果を足す(英語の論文の状態は変えない)。"""
    path = osp.join(idea_dir, "rdc_phase.json")
    cur = json.load(open(path))
    cur.setdefault("writeup", {})["ja"] = {**ja, "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(path + ".tmp", "w") as f:
        json.dump(cur, f, ensure_ascii=False, indent=2)
    os.replace(path + ".tmp", path)
    print(f"[rdc] writeup(ja): {ja.get('state')} — {ja.get('step')}", flush=True)


# 日本語化だけを別のモデルに任せる(2026-10-13)。run-c008e3ee の論文を Qwen3-Coder と llm-jp-4.1-8B で
# 訳し比べたところ、Qwen は題名で 3 回とも「康复」、本文でハングルを出し、「力分析」「可行性」と訳した。
# llm-jp は日本語でない字を出さず、「検出力分析」「ランダム化比較試験」と訳した。llm-jp の vLLM は
# run_qwen_paper.sh が段の間だけ GPU 2 に起こす。起きていなければ執筆と同じモデルで訳し、そう書き残す
JA_MODEL_WAIT = int(os.environ.get("RDC_JA_WAIT", "300"))


def _ja_model(idea_dir: str, model: str, source_note: str) -> tuple[str, str]:
    """(訳に使うモデル, 状態に足す注記) を返す。RDC_JA_MODEL が無ければ執筆と同じモデル。"""
    want = os.environ.get("RDC_JA_MODEL", "").strip()
    if not want or want == model:
        return model, ""
    if not want.startswith("llmjp/"):
        return want, ""
    import urllib.request
    served = want.split("/", 1)[1]
    url = os.environ.get("LLMJP_BASE_URL", "http://localhost:8001/v1").rstrip("/") + "/models"
    wait = int(os.environ.get("RDC_JA_WAIT", JA_MODEL_WAIT))  # 0 = 起こしていないので待たない
    if wait > 0:
        set_ja(idea_dir, state="running", step=f"日本語化に使うモデル({served})の起動を待っています",
               source=source_note)
    deadline = time.time() + wait
    while True:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if served in r.read().decode():
                    return want, ""
        except Exception:  # noqa: BLE001
            pass
        if time.time() >= deadline:
            break
        time.sleep(10)
    print(f"[rdc] {served} が使えない(待ち {wait} 秒)ので、{model} で訳す", flush=True)
    return model, f"{served} が使えなかったため、執筆と同じモデルで訳しました"


def translate_ja(idea_dir: str, model: str) -> int:
    """英語の論文(rdc_paper.pdf として採った版)の LaTeX を日本語に訳し、rdc_paper_ja.pdf に組む。"""
    try:
        wu = json.load(open(osp.join(idea_dir, "rdc_phase.json"))).get("writeup") or {}
    except Exception:  # noqa: BLE001
        wu = {}
    if wu.get("state") != "done":
        set_ja(idea_dir, state="skipped", step="英語の論文が無いので、日本語版は作りません")
        return 1
    chosen = wu.get("chosen") or ""
    src = osp.join(idea_dir, osp.splitext(chosen)[0] + ".tex") if chosen else ""
    source_note = osp.basename(src) if src else ""
    if not (src and osp.exists(src)):
        # 組んだ版ごとの LaTeX を残す前の実行。最後の版(latex/template.tex)は、最後の推敲で作り話に
        # 書き換わっていることがある(run-41bc6352 では採った版は検出力分析、最後の版は事前学習の
        # 規模の話だった)。採った版の PDF の語と比べ、違えば訳さない
        src = osp.join(idea_dir, "latex", "template.tex")
        source_note = "latex/template.tex(採った版の LaTeX が残っていない実行のため、最後の版)"
        try:
            pdf_w = _words(pdf_text(osp.join(idea_dir, PAPER)))
            tex_w = _words(re.sub(r"\\[A-Za-z]+", " ", open(src, encoding="utf-8").read()))
            same = len(pdf_w & tex_w) / max(1, len(pdf_w | tex_w))
        except Exception:  # noqa: BLE001
            same = 0.0
        if same < 0.6:
            set_ja(idea_dir, state="failed", source=source_note, similarity=round(same, 2),
                   step=("採った版の論文の LaTeX が残っておらず、残っている最後の版は採った版と中身が違うため"
                         f"(語の一致 {round(same * 100)}%)、日本語版を作りませんでした"))
            return 1
    ja_model, model_note = _ja_model(idea_dir, model, source_note)
    set_ja(idea_dir, state="running", step="英語の論文を日本語に訳しています", source=source_note)
    try:
        from ai_scientist.llm import create_client
        from ai_scientist.perform_icbinb_writeup import compile_latex

        tex = open(src, encoding="utf-8").read()
        # 文献(filecontents)は訳さない。いったん抜いて、組む前に戻す
        held: list[str] = []
        def hold(m):
            held.append(m.group(0))
            return f"%%RDC_HELD_{len(held) - 1}%%"
        tex_wo = re.sub(r"\\begin\{filecontents\*?\}.*?\\end\{filecontents\*?\}", hold, tex, flags=re.DOTALL)
        # 英語版を組むときに足した CJK の包みは外す(日本語版は luatexja で組むので要らない)
        tex_wo = _strip_cjk(tex_wo)
        if "\\begin{document}" not in tex_wo:
            raise RuntimeError("英語の論文の LaTeX に \\begin{document} がありません")
        pre, body = tex_wo.split("\\begin{document}", 1)
        client, client_model = create_client(ja_model)
        repaired: list[str] = []  # 日本語でない文字(ハングルの助詞・簡体字)を置き換えた記録

        t = pre.find("\\title{")
        if t >= 0:
            end = _braced(pre, t + len("\\title"))
            title_ja, probs, notes = _translate_piece(pre[t:end + 1], client, client_model)
            repaired += notes
            if title_ja is None:
                raise RuntimeError(f"題名の訳が英語版と合いません: {'; '.join(probs)}")
            pre = pre[:t] + title_ja + pre[end + 1:]

        pieces = _split_body(body)
        out = []
        for i, piece in enumerate(pieces, 1):
            set_ja(idea_dir, state="running", step=f"英語の論文を日本語に訳しています({i}/{len(pieces)})",
                   source=source_note)
            if not re.search(r"[A-Za-z]{4,}", re.sub(r"\\[A-Za-z]+\*?|%%RDC_HELD_\d+%%", "", piece)):
                out.append(piece)  # 訳す文の無い断片(命令だけ)はそのまま
                continue
            ja, probs, notes = _translate_piece(piece, client, client_model)
            repaired += notes
            if ja is None:
                head = re.search(r"\\section\*?\{([^}]*)\}", piece)
                where = f"「{head.group(1)}」" if head else f"{i} 番目の断片"
                raise RuntimeError(f"{where}の訳が英語版と合いません: {'; '.join(probs)}"[:300])
            out.append(ja + ("\n" if not ja.endswith("\n") else ""))
        ja_tex = pre + "\\begin{document}" + "".join(out)
        # 全体の照合は、文献を戻す前に、文献を抜いた英語版と比べる(両方とも同じ置き場所の印を持つ)
        whole = _check_translation(tex_wo, ja_tex)
        if whole:
            raise RuntimeError(f"訳した論文の全体が英語版と合いません: {'; '.join(whole)}"[:300])
        for n, block in enumerate(held):
            ja_tex = ja_tex.replace(f"%%RDC_HELD_{n}%%", block)
        # luatexja の前置きは照合の後に足す(英語版に無い命令と数値を照合に混ぜない)
        ja_tex = ja_tex.replace("\\begin{document}", _JA_PREAMBLE + "\\begin{document}", 1)

        latex_ja = osp.join(idea_dir, "latex_ja")
        if osp.exists(latex_ja):
            shutil.rmtree(latex_ja)
        shutil.copytree(osp.join(idea_dir, "latex"), latex_ja)
        for f in glob.glob(osp.join(latex_ja, "template.*")) + glob.glob(osp.join(latex_ja, "references.bib")):
            if not f.endswith("template.tex"):
                os.remove(f)
        with open(osp.join(latex_ja, "template.tex"), "w", encoding="utf-8") as f:
            f.write(ja_tex)
        out_pdf = osp.join(idea_dir, PAPER_JA)
        if osp.exists(out_pdf):
            os.remove(out_pdf)
        compile_latex(latex_ja, out_pdf, timeout=120, engine="lualatex")
        if not osp.exists(out_pdf):
            raise RuntimeError("日本語版の LaTeX を PDF に組めませんでした")
        text = pdf_text(out_pdf)
        kana = len(re.findall(r"[\u3040-\u30ff]", text))
        en_unres = pdf_text(osp.join(idea_dir, PAPER)).count("?")
        if kana < 200:
            raise RuntimeError(f"日本語版の PDF に日本語がほとんどありません(かな {kana} 字。和文の字形が組めていない疑い)")
        set_ja(idea_dir, state="done",
               step="日本語版を作りました" + (f"(訳に混じった日本語でない文字を置き換えた: {_repair_summary(repaired)})"
                                          if repaired else "") + (f"({model_note})" if model_note else ""),
               repaired=repaired, pdf=PAPER_JA, source=source_note, model=ja_model,
               pieces=len(pieces), kana=kana, question_marks={"en": en_unres, "ja": text.count("?")},
               abstract=abstract_of_tex(ja_tex))
        return 0
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        set_ja(idea_dir, state="failed", step=f"日本語版を作れませんでした: {e}"[:300], source=source_note,
               model=ja_model)
        return 1


# ─────────────────────── 論文の書き直し(自己査読の差し戻し) ───────────────────────
# 自己査読で、実験の記録と食い違う主張が見つかったら、その一覧を渡して論文を書き直す(2026-10-12)。
# 実行器が「書き直し → 査読し直し」を上限まで回し、食い違いが減らなければ restore で元に戻す。
# 差し戻しの判定に査読の判定(Accept/Reject)を使わないのは、査読が ICLR(機械学習)の査読として
# 書かれていて、機械学習でない題材は書き方に関わらず「範囲外」で落とすため(run-41bc6352 の
# Weaknesses の大半がそれだった)。書き直しで直せるのは、記録との食い違いと裏付けの無い主張である。
ROUNDS_DIR = "rdc_rounds"
# 日本語版の LaTeX(rdc_paper_ja.tex)も残す。残していなかったので、run-30a1f12c で書き直す前の論文に戻したとき、
# PDF は戻ったが隣の LaTeX は書き直した版の訳のままで、採った版の日本語版を組み直せなかった(2026-10-14)
_ROUND_FILES = (PAPER, PAPER_JA, osp.splitext(PAPER_JA)[0] + ".tex", "review_text.txt", "review_fact_check.json")

_REVISE_PROMPT = """You wrote the LaTeX paper below from an experiment. A fact check against the experiment
records found problems. Revise the paper so that every claim is supported by the records.

Rules:
- For each CONTRADICTED claim: correct it to match the records, or remove it.
- For each UNVERIFIABLE claim: remove it, or state plainly that it was not tested.
- Do not add any new number, dataset, method or result that is not in the records.
- Keep the figures (\\includegraphics), citations and the filecontents bibliography as they are,
  unless a claim using them must be removed.
- Reviewer weaknesses about the writing may also be addressed. Ignore comments about the venue's scope.
- Output the whole revised paper in one ```latex ... ``` block.

CONTRADICTED (claim / evidence from the records):
{contradicted}

UNVERIFIABLE:
{unverifiable}

REVIEWER WEAKNESSES:
{weaknesses}

EXPERIMENT RECORDS (summaries):
{records}

CURRENT PAPER:
```latex
{paper}
```"""


def _merge_phase(idea_dir: str, phase: str, **fields) -> None:
    """rdc_phase.json の段の値に、項目を足す(ほかの項目は残す)。"""
    path = osp.join(idea_dir, "rdc_phase.json")
    cur = json.load(open(path))
    cur.setdefault(phase, {}).update(fields)
    with open(path + ".tmp", "w") as f:
        json.dump(cur, f, ensure_ascii=False, indent=2)
    os.replace(path + ".tmp", path)


def _experiment_records(idea_dir: str) -> str:
    rec = {}
    try:
        ld = logs_dir(idea_dir)
    except Exception:  # noqa: BLE001
        return "{}"
    for n in ("baseline_summary", "research_summary", "ablation_summary"):
        try:
            rec[n] = json.load(open(osp.join(ld, f"{n}.json")))
        except Exception:  # noqa: BLE001
            pass
    return json.dumps(rec, ensure_ascii=False)[:30000]


def revise(idea_dir: str, model: str) -> int:
    """自己査読の食い違いを渡して論文を書き直し、rdc_paper.pdf(と日本語版)を差し替える。
    差し替える前の論文と査読は rdc_rounds/before_r<n>/ に残す(restore で戻せるように)。"""
    phases = json.load(open(osp.join(idea_dir, "rdc_phase.json")))
    n = int((phases.get("revision") or {}).get("round") or 0) + 1
    status(idea_dir, "revision", "running", f"自己査読の指摘で論文を書き直しています({n} 回目)", round=n)
    try:
        wu = phases.get("writeup") or {}
        chosen = wu.get("chosen") or ""
        src = osp.join(idea_dir, osp.splitext(chosen)[0] + ".tex") if chosen else ""
        if not (src and osp.exists(src)):
            raise RuntimeError("論文として採った版の LaTeX が残っていないので、書き直せません")
        fc = json.load(open(osp.join(idea_dir, "review_fact_check.json")))
        if not isinstance(fc.get("contradicted"), list):
            raise RuntimeError("実験の記録との照合の結果がありません")
        try:
            weak = json.load(open(osp.join(idea_dir, "review_text.txt"))).get("Weaknesses") or []
        except Exception:  # noqa: BLE001
            weak = []

        # 差し替える前の論文・査読・段の値を残す
        keep = osp.join(idea_dir, ROUNDS_DIR, f"before_r{n}")
        os.makedirs(keep, exist_ok=True)
        for f in _ROUND_FILES:
            if osp.exists(osp.join(idea_dir, f)):
                shutil.copyfile(osp.join(idea_dir, f), osp.join(keep, f))
        json.dump({k: phases.get(k) for k in ("writeup", "review")},
                  open(osp.join(keep, "phases.json"), "w"), ensure_ascii=False, indent=2)

        from ai_scientist.llm import create_client, get_response_from_llm
        from ai_scientist.perform_icbinb_writeup import WRITEUP_MAX_TOKENS, compile_latex
        def lines(items, keys):
            return "\n".join("- " + " / ".join(str(i.get(k, "")) for k in keys) if isinstance(i, dict) else f"- {i}"
                             for i in items) or "(none)"
        prompt = _REVISE_PROMPT.format(
            contradicted=lines(fc["contradicted"], ("claim", "evidence")),
            unverifiable=lines(fc.get("unverifiable") or [], ("claim",)),
            weaknesses=lines(weak, ()), records=_experiment_records(idea_dir),
            paper=open(src, encoding="utf-8").read())
        client, client_model = create_client(model)
        resp, _ = get_response_from_llm(prompt=prompt, client=client, model=client_model,
                                        system_message="You are a careful scientist revising your own paper. "
                                                       "Never invent results.",
                                        temperature=0.3, max_tokens=WRITEUP_MAX_TOKENS)
        m = re.search(r"```latex(.*?)```", resp, re.DOTALL)
        if not m or "\\begin{document}" not in m.group(1):
            import ai_scientist.llm as llm_module
            raise RuntimeError("書き直した論文に LaTeX の全文がありません"
                               f"(終わり方: {llm_module.LAST_FINISH_REASON})")
        latex = osp.join(idea_dir, "latex")
        with open(osp.join(latex, "template.tex"), "w", encoding="utf-8") as f:
            f.write(m.group(1).strip())
        name = f"rdc_revision{n}.pdf"
        out = osp.join(idea_dir, name)
        if osp.exists(out):
            os.remove(out)
        compile_latex(latex, out, timeout=120)
        if not osp.exists(out):
            raise RuntimeError("書き直した論文を PDF に組めませんでした")
        score = grounding(pdf_text(out), idea_terms(idea_dir))
        if score < 0.5:
            raise RuntimeError(f"書き直した論文が題材から外れました(題材の語の一致 {round(score * 100)}%)")
        shutil.copyfile(out, osp.join(idea_dir, PAPER))
        _merge_phase(idea_dir, "writeup", chosen=name, grounding=score, revision=n,
                     abstract=abstract_of_tex(m.group(1)),
                     cited=cited_in(osp.join(idea_dir, PAPER), idea_dir))
        # 日本語版は、書き直した英語の論文から作り直す(古い訳を残さない)
        if osp.exists(osp.join(idea_dir, PAPER_JA)):
            os.remove(osp.join(idea_dir, PAPER_JA))
        translate_ja(idea_dir, model)
        status(idea_dir, "revision", "done", f"論文を書き直しました({n} 回目)", round=n,
               addressed={"contradicted": len(fc["contradicted"]), "unverifiable": len(fc.get("unverifiable") or [])})
        return 0
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        status(idea_dir, "revision", "failed", f"論文を書き直せませんでした: {e}"[:300], round=n)
        return 1


def restore(idea_dir: str) -> int:
    """直前の書き直しを取り消し、書き直す前の論文・日本語版・査読・段の値に戻す。"""
    phases = json.load(open(osp.join(idea_dir, "rdc_phase.json")))
    n = int((phases.get("revision") or {}).get("round") or 0)
    keep = osp.join(idea_dir, ROUNDS_DIR, f"before_r{n}")
    if n < 1 or not osp.isdir(keep):
        status(idea_dir, "revision", "failed", "戻す先(書き直す前の論文)がありません", round=n)
        return 1
    for f in _ROUND_FILES:
        src, dst = osp.join(keep, f), osp.join(idea_dir, f)
        if osp.exists(src):
            shutil.copyfile(src, dst)
        elif osp.exists(dst):
            os.remove(dst)  # 書き直す前に無かったもの(日本語版など)は消す
    saved = json.load(open(osp.join(keep, "phases.json")))
    path = osp.join(idea_dir, "rdc_phase.json")
    cur = json.load(open(path))
    for k, v in saved.items():
        if v is not None:
            cur[k] = v
    cur["revision"] = {**(cur.get("revision") or {}), "state": "restored", "round": n,
                       "step": f"書き直し({n} 回目)を取り消し、書き直す前の論文に戻しました",
                       "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(path + ".tmp", "w") as f:
        json.dump(cur, f, ensure_ascii=False, indent=2)
    os.replace(path + ".tmp", path)
    print(f"[rdc] revision: restored — round {n}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["writeup", "review", "translate", "revise", "restore"])
    ap.add_argument("idea_dir")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args()
    idea_dir = osp.abspath(a.idea_dir)
    if a.phase == "translate":  # 日本語版だけを作り直す(英語の論文ができている実行に)
        return translate_ja(idea_dir, a.model)
    if a.phase == "revise":  # 自己査読の食い違いを渡して論文を書き直す(回数は rdc_phase.json の revision.round)
        return revise(idea_dir, a.model)
    if a.phase == "restore":  # 直前の書き直しを取り消す
        return restore(idea_dir)
    return writeup(idea_dir, a.model, a.retries) if a.phase == "writeup" else review(idea_dir, a.model)


if __name__ == "__main__":
    sys.exit(main())
