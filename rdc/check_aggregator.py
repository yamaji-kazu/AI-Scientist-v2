"""実行: docker run --rm -v ~/AI-Scientist-v2:/repo:ro -v ~/ais-smoke-out:/runs:ro -v <このフォルダ>:/c:ro -w /repo --entrypoint python ais-exp:rdc-tex-ja /c/check_aggregator.py"""
"""図をまとめる段の切れの扱いと代わりの図(2026-10-15)。LLM は偽物。実データは 10-14 21:34 の実行の写し"""
import os, shutil, sys, tempfile, glob, io, contextlib
sys.path.insert(0, "/repo")
import ai_scientist.perform_plotting as pp

SRC = glob.glob("/runs/2026-10-14_21-34-18*")[0]
ok = fail = 0
def check(c, m, got=None):
    global ok, fail
    if c: ok += 1; print("  PASS", m)
    else: fail += 1; print("  FAIL", m, "— 実際:", got)

def figs_of(d):
    p = os.path.join(d, "figures")
    return sorted(os.listdir(p)) if os.path.isdir(p) else []

def fresh():
    d = tempfile.mkdtemp()
    shutil.copytree(os.path.join(SRC, "logs"), os.path.join(d, "logs"))
    for f in ("idea.md", "idea.json"):
        if os.path.exists(os.path.join(SRC, f)): shutil.copy(os.path.join(SRC, f), d)
    return d

F = "`" * 3
CUT = "I understand.\n\n" + F + "python\nimport os\nx = (1,\n"
GOOD = F + "python\nimport os\nos.makedirs('figures', exist_ok=True)\nopen('figures/a.png','wb').write(b'x')\n" + F
def run(replies):
    calls = []
    def fake(**kw):
        calls.append(kw)
        r = replies[min(len(calls) - 1, len(replies) - 1)]
        return r, (kw.get("msg_history") or []) + [{"role": "user", "content": kw["prompt"]}, {"role": "assistant", "content": r}]
    pp.get_response_from_llm = fake
    pp.create_client = lambda m: (None, m)
    d = fresh()
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        pp.aggregate_plots(d, model="gpt-4o", n_reflections=3)
    rdc = [l for l in log.getvalue().splitlines() if "[rdc]" in l]
    for l in rdc: print("    log:", l)
    return d, calls

print("== 判定")
check(pp.rdc_cut_off(CUT) and not pp.rdc_cut_off(GOOD) and not pp.rdc_cut_off("no code"), "閉じの無い囲みだけを切れとみなす")
check(pp.extract_code_snippet(CUT) == "", "切れた応答からは何も取り出さない", pp.extract_code_snippet(CUT))

print("== ずっと切れる(10-14 の 2 回と同じ)")
d, calls = run([CUT])
figs = figs_of(d)
check(not os.path.exists(os.path.join(d, "auto_plot_aggregator.py")), "前置きの文を含むスクリプトを書き出さない")
check(len(calls) == 2, "始め直しは 1 回だけ(切れたまま振り返りに進まない)", len(calls))
check("cut off" in calls[1]["prompt"] and calls[1]["msg_history"] == [], "始め直しは履歴を捨て、短く書くよう求める", calls[1].get("msg_history"))
check(all(c.get("max_tokens") == 12000 for c in calls), "この段の出力上限は 12000", [c.get("max_tokens") for c in calls])
check(0 < len(figs) <= pp.MAX_FIGURES and all(f.startswith("fig") and f.endswith(".png") for f in figs), f"実験の図で埋める({len(figs)} 枚)", figs)
check(len({open(os.path.join(d, "figures", f), "rb").read() for f in figs}) == len(figs), "同じ図を二度入れない")
print("    ", figs[:4])

print("== 初めは切れ、始め直しで正しい")
d, calls = run([CUT, GOOD, "I am done"])
check(figs_of(d) == ["a.png"], "まとめた図だけ(代わりの図は入れない)", figs_of(d))

print("== 振り返りの途中で切れる")
d, calls = run([GOOD, CUT, "I am done"])
check(len(calls) == 3 and calls[2]["msg_history"][-1]["content"] == GOOD, "切れた振り返りは履歴から外す", [m["content"][:20] for m in calls[2]["msg_history"]])
check("cut off" in calls[2]["prompt"], "次の振り返りで切れたことを伝える")
check(figs_of(d) == ["a.png"], "図はまとめたもののまま", figs_of(d))

print("\n結果: " + ("すべて OK" if fail == 0 else "NG") + f" (PASS {ok} / FAIL {fail})")
sys.exit(1 if fail else 0)
