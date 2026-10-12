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
                   citations=cites, cited=cited_in(osp.join(idea_dir, PAPER), idea_dir))
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


# ─────────────────────── 日本語版(英語の論文の訳) ───────────────────────
# 英語の論文ができた後に、その LaTeX を節ごとに日本語に訳し、rdc_paper_ja.pdf に組む(2026-10-12)。
# 一から書かせず訳にするのは、数値・図・引用を英語版と同じにし、自己査読と実験の記録との照合が
# 済んだ英語版の「訳」という位置づけをはっきりさせるため。訳で数値や引用が落ちたり変わったり
# しないよう、節ごとに機械で照らし、合わなければ 1 度だけ訳し直す。それでも合わなければ日本語版は
# 出さず、理由を残す(英語の論文は完了のまま)。
PAPER_JA = "rdc_paper_ja.pdf"
JA_CHUNK = 6000  # 1 回に訳す LaTeX の長さ(文字)。出力の上限(WRITEUP_MAX_TOKENS)に十分収まる

_JA_SYSTEM = (
    "You translate scientific papers written in LaTeX from English into Japanese, in a formal academic "
    "style (である調). Rules:\n"
    "- Translate only the natural-language text. Keep every LaTeX command, environment, math, \\cite{...}, "
    "\\citep{...}, \\citet{...}, \\ref{...}, \\label{...}, \\includegraphics{...}, \\url{...} and file name "
    "exactly as given.\n"
    "- Keep every number exactly as written (do not convert, round, or spell out numbers).\n"
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
            fixed, notes = _repair_hangul(ja)
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
        # 英語版を組むときに足した CJK の包みは外す(日本語版を組むときに付け直す)
        tex_wo = tex_wo.replace("\\usepackage{CJKutf8}\n", "").replace("\\begin{CJK}{UTF8}{min}\n", "").replace("\\end{CJK}\n", "")
        if "\\begin{document}" not in tex_wo:
            raise RuntimeError("英語の論文の LaTeX に \\begin{document} がありません")
        pre, body = tex_wo.split("\\begin{document}", 1)
        client, client_model = create_client(model)
        repaired: list[str] = []  # ハングルの助詞を置き換えた記録

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
        compile_latex(latex_ja, out_pdf, timeout=120)
        if not osp.exists(out_pdf):
            raise RuntimeError("日本語版の LaTeX を PDF に組めませんでした")
        text = pdf_text(out_pdf)
        kana = len(re.findall(r"[\u3040-\u30ff]", text))
        en_unres = pdf_text(osp.join(idea_dir, PAPER)).count("?")
        if kana < 200:
            raise RuntimeError(f"日本語版の PDF に日本語がほとんどありません(かな {kana} 字。和文の字形が組めていない疑い)")
        set_ja(idea_dir, state="done",
               step="日本語版を作りました" + (f"(訳に混じった韓国語の助詞 {len(repaired)} か所を置き換えた)" if repaired else ""),
               repaired=repaired, pdf=PAPER_JA, source=source_note,
               pieces=len(pieces), kana=kana, question_marks={"en": en_unres, "ja": text.count("?")})
        return 0
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        set_ja(idea_dir, state="failed", step=f"日本語版を作れませんでした: {e}"[:300], source=source_note)
        return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["writeup", "review", "translate"])
    ap.add_argument("idea_dir")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args()
    idea_dir = osp.abspath(a.idea_dir)
    if a.phase == "translate":  # 日本語版だけを作り直す(英語の論文ができている実行に)
        return translate_ja(idea_dir, a.model)
    return writeup(idea_dir, a.model, a.retries) if a.phase == "writeup" else review(idea_dir, a.model)


if __name__ == "__main__":
    sys.exit(main())
