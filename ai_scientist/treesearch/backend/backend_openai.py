import json
import logging
import os
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai
from rich import print

logger = logging.getLogger("ai-scientist")


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

# NII RDC: thinking モデルは生成が長く、既定タイムアウトだと backoff を誘発する。
# LLMJP_TIMEOUT(秒)で延ばす(改版提案 §12.2 の onprem 運用)。
_LLMJP_TIMEOUT = float(os.environ.get("LLMJP_TIMEOUT", "1800"))


def get_ai_client(model: str, max_retries=2) -> openai.OpenAI:
    if model.startswith("ollama/"):
        client = openai.OpenAI(
            base_url="http://localhost:11434/v1",
            max_retries=max_retries,
            timeout=_LLMJP_TIMEOUT,
        )
    elif model.startswith("llmjp/"):
        # NII RDC: onprem llm-jp (vLLM, OpenAI 互換)。tree-search コーダーを外部でなく
        # llm-jp に向ける(改版提案 §12.2)。llm.py の llmjp/ 分岐と対称。
        client = openai.OpenAI(
            base_url=os.environ.get("LLMJP_BASE_URL", "http://localhost:8001/v1"),
            api_key=os.environ.get("LLMJP_API_KEY", "EMPTY"),
            max_retries=max_retries,
            timeout=_LLMJP_TIMEOUT,
        )
    else:
        # 既定(gpt-4o 等のハードコード呼び出し)。OPENAI_BASE_URL を llm-jp に向ければ
        # ここも onprem に着地する(vLLM 側で gpt-4o 別名を served-model-name に足す)。
        client = openai.OpenAI(max_retries=max_retries, timeout=_LLMJP_TIMEOUT)
    return client


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    client = get_ai_client(model_kwargs.get("model"), max_retries=0)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message)

    # NII RDC: llm-jp は OpenAI の forced tool_choice(function-calling)に未対応
    # (独自 Harmony 形式で、vLLM の gpt-oss 用 openai parser とも不整合)。func_spec が
    # あるときは tools を送らず、スキーマをプロンプトに載せて JSON で答えさせ、本文から
    # JSON を抽出する(改版提案 §12.2 の onprem 運用。tool-calling の成熟度に依存しない)。
    llmjp_fallback = (
        filtered_kwargs.get("model", "").startswith("llmjp/") and func_spec is not None
    )

    if func_spec is not None and not llmjp_fallback:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model to use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    if llmjp_fallback:
        schema_txt = json.dumps(func_spec.json_schema, ensure_ascii=False)
        instruct = (
            f"\n\nCall the function `{func_spec.name}`: {func_spec.description}\n"
            "Respond with ONLY a single JSON object that matches this JSON schema "
            f"(no prose, no code fence):\n{schema_txt}"
        )
        if messages and messages[-1].get("role") == "user":
            messages[-1]["content"] = (messages[-1].get("content") or "") + instruct
        else:
            messages.append({"role": "user", "content": instruct})

    if filtered_kwargs.get("model", "").startswith(("ollama/", "llmjp/")):
       filtered_kwargs["model"] = filtered_kwargs["model"].split("/", 1)[1]

    t0 = time.time()
    completion = backoff_create(
        client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        # thinking モデルは content が None/空になり得る(reasoning parser の取りこぼし、
        # 思考だけで終わる等)。None を下流の extract_code(re.findall)に流すと落ちるので、
        # reasoning_content か空文字へ必ず倒す(fail-safe)。根治は parser を外して生 content を得ること。
        output = choice.message.content
        if output is None:
            output = getattr(choice.message, "reasoning_content", None) or ""
    elif llmjp_fallback:
        # 本文から JSON を抽出(thinking モデルは llmjp4 parser で reasoning が分離され、
        # content は答えになる)。抽出できなければ素の json.loads を試す。
        from ai_scientist.llm import extract_json_between_markers

        content = choice.message.content or ""
        output = extract_json_between_markers(content)
        if output is None:
            try:
                output = json.loads(content.strip())
            except json.JSONDecodeError as e:
                logger.error(f"llm-jp fallback: JSON を抽出できませんでした: {content[:500]}")
                raise e
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            print(f"[cyan]Raw func call response: {choice}[/cyan]")
            output = json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
