import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from rich import print

from ai_scientist.llm import create_client, get_response_from_llm
from ai_scientist.utils.token_tracker import token_tracker
from ai_scientist.perform_icbinb_writeup import (
    load_idea_text,
    load_exp_summaries,
    filter_experiment_summaries,
)

MAX_FIGURES = 12

AGGREGATOR_SYSTEM_MSG = f"""You are an ambitious AI researcher who is preparing final plots for a scientific paper submission.
You have multiple experiment summaries (baseline, research, ablation), each possibly containing references to different plots or numerical insights.
There is also a top-level 'research_idea.md' file that outlines the overarching research direction.
Your job is to produce ONE Python script that fully aggregates and visualizes the final results for a comprehensive research paper.

Key points:
1) Combine or replicate relevant existing plotting code, referencing how data was originally generated (from code references) to ensure correctness.
2) Create a complete set of final scientific plots, stored in 'figures/' only (since only those are used in the final paper).
3) Make sure to use existing .npy data for analysis; do NOT hallucinate data. If single numeric results are needed, these may be copied from the JSON summaries.
4) Only create plots where the data is best presented as a figure and not as a table. E.g. don't use bar plots if the data is hard to visually compare.
5) The final aggregator script must be in triple backticks and stand alone so it can be dropped into a codebase and run.
6) If there are plots based on synthetic data, include them in the appendix.

Implement best practices:
- Do not produce extraneous or irrelevant plots.
- Maintain clarity, minimal but sufficient code.
- Demonstrate thoroughness for a final research paper submission.
- Do NOT reference non-existent files or images.
- Use the .npy files to get data for the plots and key numbers from the JSON summaries.
- Demarcate each individual plot, and put them in separate try-catch blocks so that the failure of one plot does not affect the others.
- Make sure to only create plots that are unique and needed for the final paper and appendix. A good number could be around {MAX_FIGURES} plots in total.
- Aim to aggregate multiple figures into one plot if suitable, i.e. if they are all related to the same topic. You can place up to 3 plots in one row.
- Provide well-labeled plots (axes, legends, titles) that highlight main findings. Use informative names everywhere, including in the legend for referencing them in the final paper. Make sure the legend is always visible.
- Make the plots look professional (if applicable, no top and right spines, dpi of 300, adequate ylim, etc.).
- Do not use labels with underscores, e.g. "loss_vs_epoch" should be "loss vs epoch".
- For image examples, select a few categories/classes to showcase the diversity of results instead of showing a single category/class. Some can be included in the main paper, while the rest can go in the appendix.

Your output should be the entire Python aggregator script in triple backticks.
"""


def build_aggregator_prompt(combined_summaries_str, idea_text):
    return f"""
We have three JSON summaries of scientific experiments: baseline, research, ablation.
They may contain lists of figure descriptions, code to generate the figures, and paths to the .npy files containing the numerical results.
Our goal is to produce final, publishable figures.

--- RESEARCH IDEA ---
```
{idea_text}
```

IMPORTANT:
- The aggregator script must load existing .npy experiment data from the "exp_results_npy_files" fields (ONLY using full and exact file paths in the summary JSONs) for thorough plotting.
- It should call os.makedirs("figures", exist_ok=True) before saving any plots.
- Aim for a balance of empirical results, ablations, and diverse, informative visuals in 'figures/' that comprehensively showcase the finalized research outcomes.
- If you need .npy paths from the summary, only copy those paths directly (rather than copying and parsing the entire summary).

Your generated Python script must:
1) Load or refer to relevant data and .npy files from these summaries. Use the full and exact file paths in the summary JSONs.
2) Synthesize or directly create final, scientifically meaningful plots for a final research paper (comprehensive and complete), referencing the original code if needed to see how the data was generated.
3) Carefully combine or replicate relevant existing plotting code to produce these final aggregated plots in 'figures/' only, since only those are used in the final paper.
4) Do not hallucinate data. Data must either be loaded from .npy files or copied from the JSON summaries.
5) The aggregator script must be fully self-contained, and place the final plots in 'figures/'.
6) This aggregator script should produce a comprehensive and final set of scientific plots for the final paper, reflecting all major findings from the experiment data.
7) Make sure that every plot is unique and not duplicated from the original plots. Delete any duplicate plots if necessary.
8) Each figure can have up to 3 subplots using fig, ax = plt.subplots(1, 3).
9) Use a font size larger than the default for plot labels and titles to ensure they are readable in the final PDF paper.


Below are the summaries in JSON:

{combined_summaries_str}

Respond with a Python script in triple backticks.
"""


# NII RDC: LLM(特に llm-jp)が非 ASCII の空白やダッシュをコードに混ぜると
# `SyntaxError: invalid character` になるため、実行前にサニタイズする。
# 空白 → 半角空白、ゼロ幅 → 除去、ダッシュ/マイナス類 → ASCII '-'(U+2011 で実測の頓挫あり)。
_UNICODE_WS_FIX = {
    0x00A0: " ", 0x2002: " ", 0x2003: " ", 0x2004: " ", 0x2005: " ",
    0x2006: " ", 0x2007: " ", 0x2008: " ", 0x2009: " ", 0x200A: " ",
    0x202F: " ", 0x205F: " ", 0x3000: " ",
    0x200B: None, 0x200C: None, 0x200D: None, 0x2060: None, 0xFEFF: None,
    0x00AD: None,  # soft hyphen
    0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2014: "-",
    0x2015: "-", 0x2043: "-", 0x2212: "-",  # 各種ダッシュ・図表ダッシュ・マイナス記号
}


def sanitize_code_unicode(code: str) -> str:
    """コード中の非 ASCII 空白/ゼロ幅文字を通常空白に直す/除去する。"""
    return code.translate(_UNICODE_WS_FIX)


def rdc_cut_off(text) -> bool:
    """NII RDC: コードの囲みを開いたまま閉じていない応答(出力の上限で切れたもの)。

    2026-10-14 の 2 回の実行(16:37・21:34)で、図をまとめるスクリプトが 362 行になり、上限(4096)で
    途中の「va」「axes[0」で切れた。閉じが無いと下の取り出しは応答の全文(前置きの文ごと)を返し、
    構文エラーで図が 0 枚のまま論文が組まれた。
    """
    return isinstance(text, str) and text.count("```") % 2 == 1


def extract_code_snippet(text: str) -> str:
    """
    Look for a Python code block in triple backticks in the LLM response.
    Return only that code. If no code block is found, return the entire text.
    """
    # llm-jp が content なし(None)や空を返すことがあるので防御する。
    if not isinstance(text, str) or not text.strip():
        return ""
    # NII RDC: 切れた応答は書き出さない(前置きの文と途中までのコードを走らせない)
    if rdc_cut_off(text):
        return ""
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return sanitize_code_unicode(matches[0].strip() if matches else text.strip())


def run_aggregator_script(
    aggregator_code, aggregator_script_path, base_folder, script_name
):
    if not aggregator_code.strip():
        print("No aggregator code was provided. Skipping aggregator script run.")
        return ""
    with open(aggregator_script_path, "w") as f:
        f.write(aggregator_code)

    print(
        f"Aggregator script written to '{aggregator_script_path}'. Attempting to run it..."
    )

    aggregator_out = ""
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=base_folder,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        aggregator_out = result.stdout + "\n" + result.stderr
        print("Aggregator script ran successfully.")
    except subprocess.CalledProcessError as e:
        aggregator_out = (e.stdout or "") + "\n" + (e.stderr or "")
        print("Error: aggregator script returned a non-zero exit code.")
        print(e)
    except Exception as e:
        aggregator_out = str(e)
        print("Error while running aggregator script.")
        print(e)

    return aggregator_out


# NII RDC: 図をまとめるスクリプトだけ出力の上限を上げる(Qwen3-Coder の文脈は 65536)。
# 振り返りの会話は切れたら始め直すので、履歴が文脈を食い尽くすことはない
RDC_AGG_MAX_TOKENS = int(os.environ.get("RDC_AGG_MAX_TOKENS", "12000"))
RDC_CUT_OFF_NOTE = (
    "Your previous reply was cut off at the output limit before the closing triple backticks, "
    "so nothing could be run. Write a complete, shorter script (at most about 250 lines): "
    "combine related plots into subplots, use a small helper function for repeated styling, "
    "and do not add explanations outside the code block."
)


def rdc_fallback_figures(base_folder, summaries, limit=MAX_FIGURES):
    """NII RDC: まとめても図が 0 枚なら、実験の段が作った図(要約の plot_paths)を figures/ に入れる。

    要約の道は実験のときの「experiments/<実行>/logs/0-qwen/...」なので、作業フォルダの logs/ 以下に
    読み替える。実在するものだけを、要約の順に、名前が重ならないよう番号を付けて最大 limit 枚。
    これはこの実行の実験の図そのもので、作り直したものではない。入れた枚数を返す
    """
    paths = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "plot_paths" and isinstance(v, list):
                    paths.extend(x for x in v if isinstance(x, str))
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    # 研究の段(提案手法)の図を先に採る。基準・アブレーションは後
    if isinstance(summaries, dict):
        order = ["RESEARCH_SUMMARY", "BASELINE_SUMMARY", "ABLATION_SUMMARY"]
        for k in order + [k for k in summaries if k not in order]:
            walk(summaries.get(k))
    else:
        walk(summaries)
    figures_dir = os.path.join(base_folder, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    # 節ごとに同じ名前の図が出る(同じ実験のコードを使い回す)。名前でも中身でも重複を外す
    seen, names, n = set(), set(), 0
    for p in paths:
        i = p.find("logs/")
        cands = [p] if os.path.isabs(p) else []
        if i >= 0:
            rel = p[i:]
            cands += [os.path.join(base_folder, rel),
                      os.path.join(base_folder, rel.replace("logs/0-qwen/", "logs/0-run/", 1))]
        src = next((c for c in cands if c.lower().endswith(".png") and os.path.isfile(c)), None)
        if not src:
            continue
        with open(src, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        name = os.path.basename(src)
        if digest in seen or name in names:
            continue
        seen.add(digest)
        names.add(name)
        n += 1
        shutil.copyfile(src, os.path.join(figures_dir, f"fig{n:02d}_{os.path.basename(src)}"))
        if n >= limit:
            break
    return n


def aggregate_plots(
    base_folder: str, model: str = "o1-2024-12-17", n_reflections: int = 5
) -> None:
    filename = "auto_plot_aggregator.py"
    aggregator_script_path = os.path.join(base_folder, filename)
    figures_dir = os.path.join(base_folder, "figures")

    # Clean up previous files
    if os.path.exists(aggregator_script_path):
        os.remove(aggregator_script_path)
    if os.path.exists(figures_dir):
        shutil.rmtree(figures_dir)
        print(f"Cleaned up previous figures directory")

    idea_text = load_idea_text(base_folder)
    exp_summaries = load_exp_summaries(base_folder)
    filtered_summaries_for_plot_agg = filter_experiment_summaries(
        exp_summaries, step_name="plot_aggregation"
    )
    # Convert them to one big JSON string for context
    combined_summaries_str = json.dumps(filtered_summaries_for_plot_agg, indent=2)

    # Build aggregator prompt
    aggregator_prompt = build_aggregator_prompt(combined_summaries_str, idea_text)

    # Call LLM
    client, model_name = create_client(model)
    response, msg_history = None, []
    try:
        response, msg_history = get_response_from_llm(
            prompt=aggregator_prompt,
            client=client,
            model=model_name,
            system_message=AGGREGATOR_SYSTEM_MSG,
            print_debug=False,
            msg_history=msg_history,
            max_tokens=RDC_AGG_MAX_TOKENS,
        )
    except Exception:
        traceback.print_exc()
        print("Failed to get aggregator script from LLM.")
        return
    # NII RDC: 切れていたら、切れた応答を履歴に残さず 1 回だけ始め直す
    if rdc_cut_off(response):
        print("[rdc] 図をまとめるスクリプトが出力の上限で切れました。短く書き直させます")
        try:
            response, msg_history = get_response_from_llm(
                prompt=aggregator_prompt + "\n\n" + RDC_CUT_OFF_NOTE,
                client=client,
                model=model_name,
                system_message=AGGREGATOR_SYSTEM_MSG,
                print_debug=False,
                msg_history=[],
                max_tokens=RDC_AGG_MAX_TOKENS,
            )
        except Exception:
            traceback.print_exc()

    aggregator_code = extract_code_snippet(response)
    if not aggregator_code.strip():
        print(
            "No Python code block was found in LLM response. Full response:\n", response
        )
        rdc_finish_figures(base_folder, exp_summaries)
        return

    # First run of aggregator script
    aggregator_out = run_aggregator_script(
        aggregator_code, aggregator_script_path, base_folder, filename
    )

    # Multiple reflection loops
    for i in range(n_reflections):
        # Check number of figures
        figure_count = 0
        if os.path.exists(figures_dir):
            figure_count = len(
                [
                    f
                    for f in os.listdir(figures_dir)
                    if os.path.isfile(os.path.join(figures_dir, f))
                ]
            )
        print(f"[{i + 1} / {n_reflections}]: Number of figures: {figure_count}")
        # Reflection prompt with reminder for common checks and early exit
        reflection_prompt = f"""We have run your aggregator script and it produced {figure_count} figure(s). The script's output is:
```
{aggregator_out}
```

Please criticize the current script for any flaws including but not limited to:
- Are these enough plots for a final paper submission? Don't create more than {MAX_FIGURES} plots.
- Have you made sure to both use key numbers and generate more detailed plots from .npy files?
- Does the figure title and legend have informative and descriptive names? These plots are the final versions, ensure there are no comments or other notes.
- Can you aggregate multiple plots into one figure if suitable?
- Do the labels have underscores? If so, replace them with spaces.
- Make sure that every plot is unique and not duplicated from the original plots.

If you believe you are done, simply say: "I am done". Otherwise, please provide an updated aggregator script in triple backticks."""

        print("[green]Reflection prompt:[/green] ", reflection_prompt)
        try:
            reflection_response, msg_history = get_response_from_llm(
                prompt=reflection_prompt,
                client=client,
                model=model_name,
                system_message=AGGREGATOR_SYSTEM_MSG,
                print_debug=False,
                msg_history=msg_history,
                max_tokens=RDC_AGG_MAX_TOKENS,
            )

        except Exception:
            traceback.print_exc()
            print("Failed to get reflection from LLM.")
            break
        # NII RDC: 切れた振り返りは履歴から外し(次の呼び出しの文脈を食わない)、短く書き直させる
        if rdc_cut_off(reflection_response):
            print(f"[rdc] 振り返り {i + 1} の応答が出力の上限で切れました")
            msg_history = msg_history[:-2]
            aggregator_out = (aggregator_out or "") + "\n\n" + RDC_CUT_OFF_NOTE
            continue

        # Early-exit check
        if figure_count > 0 and "I am done" in reflection_response:
            print("LLM indicated it is done with reflections. Exiting reflection loop.")
            break

        aggregator_new_code = extract_code_snippet(reflection_response)

        # If new code is provided and differs, run again
        if (
            aggregator_new_code.strip()
            and aggregator_new_code.strip() != aggregator_code.strip()
        ):
            aggregator_code = aggregator_new_code
            aggregator_out = run_aggregator_script(
                aggregator_code, aggregator_script_path, base_folder, filename
            )
        else:
            print(
                f"No new aggregator script was provided or it was identical. Reflection step {i+1} complete."
            )

    rdc_finish_figures(base_folder, exp_summaries)


def rdc_finish_figures(base_folder, summaries):
    """NII RDC: 図をまとめる段の終わり。図が 0 枚なら、実験の段の図で埋める(rdc_fallback_figures)"""
    figures_dir = os.path.join(base_folder, "figures")
    have = [f for f in os.listdir(figures_dir) if f.lower().endswith(".png")] if os.path.isdir(figures_dir) else []
    if have:
        return
    n = rdc_fallback_figures(base_folder, summaries)
    print(f"[rdc] 図をまとめられず 0 枚でした。実験の段の図を {n} 枚入れました")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and execute a final plot aggregation script with LLM assistance."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to the experiment folder with summary JSON files.",
    )
    parser.add_argument(
        "--model",
        default="o1-2024-12-17",
        help="LLM model to use (default: o1-2024-12-17).",
    )
    parser.add_argument(
        "--reflections",
        type=int,
        default=5,
        help="Number of reflection steps to attempt (default: 5).",
    )
    args = parser.parse_args()
    aggregate_plots(
        base_folder=args.folder, model=args.model, n_reflections=args.reflections
    )


if __name__ == "__main__":
    main()
