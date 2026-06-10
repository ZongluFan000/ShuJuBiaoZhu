from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
import time
import urllib.error
import urllib.request


@dataclass
class LLMConfig:
    provider: str
    base_url: str
    api_key: str
    model_name: str
    upstream_url: str = ""
    model_path: str = ""
    timeout_seconds: int = 180
    max_retries: int = 2
    temperature: float = 0.0
    max_tokens: int = 1024
    stream: bool = False
    enable_thinking: bool | None = None
    torch_dtype: str = "float16"
    device_map: str = "auto"
    max_memory_gpu: str = ""
    max_memory_cpu: str = ""


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._local_model = None
        self._local_tokenizer = None

    def label(self, prompt: str) -> dict[str, Any]:
        if self.config.provider == "mock":
            return self._mock_label(prompt)
        if self.config.provider == "local_transformers":
            return self._call_local_transformers(prompt)
        return self._call_api(prompt)

    def _mock_label(self, prompt: str = "") -> dict[str, Any]:
        rules = []
        current: dict[str, str] = {}
        for raw_line in prompt.splitlines():
            line = raw_line.strip()
            if line.startswith("试验注册号："):
                current["trial_register_id"] = line.split("：", 1)[1].strip()
            elif line.startswith("试验标识："):
                current["trial_id"] = line.split("：", 1)[1].strip()
            elif line.startswith("标准编号："):
                current["standard_no"] = line.split("：", 1)[1].strip()
            elif line.startswith("规则标识："):
                current["rule_type"] = line.split("：", 1)[1].strip()
                if current.get("standard_no"):
                    rules.append(current)
                    current = {}
        if not rules:
            rules = [{"trial_register_id": "", "trial_id": "", "standard_no": "1", "rule_type": "入选标准"}]
        return {
            "items": [
                {
                    "trial_register_id": rule.get("trial_register_id", ""),
                    "trial_id": rule.get("trial_id", ""),
                    "standard_no": rule.get("standard_no", ""),
                    "rule_type": rule.get("rule_type", ""),
                    "label": "未知",
                    "explanation": "mock 模式未调用真实模型，仅用于验证流程。",
                    "evidence": "mock",
                    "confidence": 0.1,
                }
                for rule in rules
            ]
        }

    def _call_api(self, prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": self.config.stream,
        }
        if self.config.upstream_url:
            payload["apiUrl"] = self.config.upstream_url
        if self.config.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.config.enable_thinking}
        last_error: Exception | None = None
        for attempt in range(max(1, self.config.max_retries)):
            try:
                content = self._request_api_content(payload)
                try:
                    return _parse_json_content(content)
                except (json.JSONDecodeError, ValueError) as exc:
                    if not self.config.stream:
                        try:
                            return self._repair_json_output(content, prompt, exc)
                        except Exception as repair_exc:
                            snippet = content.strip().replace("\r", " ").replace("\n", " ")[:800]
                            raise ValueError(
                                f"LLM returned non-JSON output: {exc}; repair_failed={repair_exc}; content_snippet={snippet!r}"
                            ) from exc
                    snippet = content.strip().replace("\r", " ").replace("\n", " ")[:800]
                    raise ValueError(f"LLM returned non-JSON output: {exc}; content_snippet={snippet!r}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                last_error = exc
                time.sleep(min(20, 2**attempt))
        raise RuntimeError(f"LLM API call failed: {last_error}")

    def _request_api_content(self, payload: dict[str, Any]) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with _direct_urlopen(req, timeout=self.config.timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
        if self.config.stream:
            return _extract_streaming_text(raw)
        data = json.loads(raw)
        return _extract_assistant_text(data)

    def _repair_json_output(self, content: str, original_prompt: str, original_error: Exception) -> dict[str, Any]:
        if "{" not in content or "}" not in content:
            raise ValueError("repair skipped because output contains no JSON-like object")
        repair_prompt = (
            "请把下面模型输出修复为一个合法 JSON 对象。不要重新判断医学含义，不要添加解释文字，"
            "只保留或补齐 JSON 结构。目标格式必须是："
            "{\"items\":[{\"trial_id\":\"\",\"standard_no\":\"\",\"label\":\"符合/不符合/未知\","
            "\"explanation\":\"\",\"evidence\":\"\",\"confidence\":0.0}]}。\n\n"
            f"原始解析错误：{original_error}\n\n"
            f"原始输出：\n{content[:8000]}\n\n"
            f"原始任务提示片段：\n{original_prompt[:2000]}"
        )
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": repair_prompt}],
            "temperature": 0.0,
            "max_tokens": min(max(int(self.config.max_tokens or 1024), 512), 2048),
            "stream": False,
        }
        if self.config.upstream_url:
            payload["apiUrl"] = self.config.upstream_url
        if self.config.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        repaired = self._request_api_content(payload)
        return _parse_json_content(repaired)


def _direct_urlopen(req: urllib.request.Request, timeout: int):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(req, timeout=timeout)


def _extract_streaming_text(raw: str) -> str:
    parts: list[str] = []
    for line in raw.splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("data:"):
            text = text[5:].strip()
        if not text or text == "[DONE]":
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            parts.append(text)
            continue
        try:
            chunk = _extract_assistant_text(data)
        except ValueError:
            chunk = ""
        if chunk:
            parts.append(chunk)
    return "".join(parts).strip()


def _extract_assistant_text(data: Any) -> str:
    if isinstance(data, list):
        for item in data:
            text = _extract_assistant_text(item)
            if text.strip():
                return text
        return ""

    if not isinstance(data, dict):
        return _stringify_content(data)

    choices = data.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            delta = choice.get("delta")
            text = _first_text(
                message.get("content") if isinstance(message, dict) else message,
                delta.get("content") if isinstance(delta, dict) else delta,
                message.get("reasoning_content") if isinstance(message, dict) else "",
                delta.get("reasoning_content") if isinstance(delta, dict) else "",
                message.get("reasoning") if isinstance(message, dict) else "",
                delta.get("reasoning") if isinstance(delta, dict) else "",
                choice.get("text"),
                choice.get("content"),
            )
            if text:
                return text

    candidates = data.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            parts = content.get("parts", []) if isinstance(content, dict) else []
            text = _first_text(*[part.get("text") for part in parts if isinstance(part, dict)])
            if text:
                return text

    content = data.get("content")
    if isinstance(content, list):
        text = _first_text(*[item.get("text") for item in content if isinstance(item, dict)])
        if text:
            return text
    elif isinstance(content, str):
        return content

    message = data.get("message")
    if isinstance(message, dict):
        text = _first_text(
            message.get("content"),
            message.get("text"),
            message.get("reasoning_content"),
            message.get("reasoning"),
        )
        if text:
            return text

    for collection_name in ("outputs", "output", "generations"):
        outputs = data.get(collection_name)
        if isinstance(outputs, list):
            for output in outputs:
                if isinstance(output, dict):
                    nested_message = output.get("message")
                    text = _first_text(
                        output.get("text"),
                        output.get("generated_text"),
                        output.get("content"),
                        nested_message.get("content") if isinstance(nested_message, dict) else "",
                    )
                    if text:
                        return text
                else:
                    text = _stringify_content(output).strip()
                    if text:
                        return text

    for key in ("response", "text", "completion", "generated_text", "answer", "result", "output_text"):
        text = _first_text(data.get(key))
        if text:
            return text

    raise ValueError("LLM response does not contain assistant text")


def _first_text(*values: Any) -> str:
    for value in values:
        text = _stringify_content(value).strip()
        if text:
            return text
    return ""


def _stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            text = _stringify_content(item).strip()
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(value, dict):
        for key in ("text", "content", "generated_text"):
            text = _stringify_content(value.get(key)).strip()
            if text:
                return text
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _parse_json_content(content: str) -> dict[str, Any]:
    text = _strip_thinking_text(content).strip()
    if not text:
        raise ValueError("empty assistant content after removing thinking text")
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    elif not text.startswith("{"):
        raise ValueError(f"assistant content does not contain a JSON object: {text[:120]!r}")
    return json.loads(text)


def _strip_thinking_text(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    value = re.sub(r"(?is)<think>.*?</think>", "", value).strip()
    if "</think>" in value:
        value = value.split("</think>", 1)[1].strip()
    return value


def _load_transformers_model(config: LLMConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if config.torch_dtype == "float16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    max_memory = None
    if config.max_memory_gpu or config.max_memory_cpu:
        max_memory = {}
        if config.max_memory_gpu:
            max_memory[0] = config.max_memory_gpu
        if config.max_memory_cpu:
            max_memory["cpu"] = config.max_memory_cpu
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=dtype,
        device_map=config.device_map,
        max_memory=max_memory,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def _messages_to_text(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def _trim_generated(prompt_ids_len: int, output_ids):
    return output_ids[0][prompt_ids_len:]


def _move_inputs_to_model(inputs, model):
    device = next(model.parameters()).device
    return {k: v.to(device) for k, v in inputs.items()}


def _generation_kwargs(config: LLMConfig) -> dict[str, Any]:
    return {
        "max_new_tokens": config.max_tokens,
        "do_sample": config.temperature > 0,
        "temperature": max(config.temperature, 1e-5),
    }


def _ensure_local_loaded(client: LLMClient):
    if client._local_model is None or client._local_tokenizer is None:
        if not client.config.model_path:
            raise RuntimeError("local_transformers provider requires model_path")
        client._local_tokenizer, client._local_model = _load_transformers_model(client.config)


def _call_local(client: LLMClient, prompt: str) -> dict[str, Any]:
    import torch

    _ensure_local_loaded(client)
    tokenizer = client._local_tokenizer
    model = client._local_model
    text = _messages_to_text(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = _move_inputs_to_model(inputs, model)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **_generation_kwargs(client.config))
    generated = _trim_generated(inputs["input_ids"].shape[-1], output_ids)
    content = tokenizer.decode(generated, skip_special_tokens=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        return _parse_json_content(content)
    except json.JSONDecodeError as exc:
        snippet = content.strip().replace("\r", " ").replace("\n", " ")[:500]
        raise RuntimeError(f"Local model returned non-JSON output: {exc}; output_snippet={snippet!r}") from exc


LLMClient._call_local_transformers = _call_local
