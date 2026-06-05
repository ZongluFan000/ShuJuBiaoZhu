from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import request, error
import json


HOST = "127.0.0.1"
PORT = 8765
ROOT = Path(__file__).resolve().parent


class ProxyHandler(SimpleHTTPRequestHandler):
    server_version = "LocalLLMProxy/1.1"

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/proxy-chat.html"
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_json({"error": "Not found"}, 404)
            return

        try:
            payload = self.read_json_body()
            upstream_url = payload.get("apiUrl", "").strip()
            if not upstream_url:
                self.send_json({"error": "Missing model API URL"}, 400)
                return

            upstream_response = self.call_upstream(upstream_url, payload)
            content = read_assistant_text(upstream_response)
            self.send_json({"content": content, "raw": upstream_response})
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self.send_json({"error": detail or f"Upstream returned HTTP {exc.code}"}, 502)
        except error.URLError as exc:
            self.send_json({"error": f"Cannot connect to upstream: {exc.reason}"}, 502)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 500)

    def read_json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def call_upstream(self, upstream_url, payload):
        messages = build_messages(payload)
        body = build_request_body(upstream_url, payload, messages)

        headers = {"Content-Type": "application/json"}
        api_key = payload.get("apiKey", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["x-api-key"] = api_key

        req = request.Request(
            upstream_url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with request.urlopen(req, timeout=120) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            return parse_response_text(text)

    def send_json(self, data, status=200):
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def build_messages(payload):
    messages = []
    system_prompt = payload.get("systemPrompt", "").strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(payload.get("messages", []))
    return messages


def build_prompt(messages):
    lines = []
    for message in messages:
        role = message.get("role", "user")
        content = stringify_content(message.get("content", ""))
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"User: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def build_request_body(upstream_url, payload, messages):
    model = payload.get("model", "")
    temperature = payload.get("temperature", 0.7)
    max_tokens = payload.get("maxTokens", 2048)
    url = upstream_url.lower()

    if "/api/chat" in url:
        return {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

    if "/api/generate" in url:
        return {
            "model": model,
            "prompt": build_prompt(messages),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

    if url.endswith("/completion") or "/completion?" in url:
        return {
            "prompt": build_prompt(messages),
            "temperature": temperature,
            "n_predict": max_tokens,
            "stream": False,
        }

    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }


def stringify_content(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if value is None:
        return ""
    return str(value)


def first_text(*values):
    for value in values:
        text = stringify_content(value).strip()
        if text:
            return text
    return ""


def read_assistant_text(data):
    if isinstance(data, list):
        for item in data:
            text = read_assistant_text(item)
            if text:
                return text
        return ""

    if not isinstance(data, dict):
        return stringify_content(data)

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or {}
            delta = choice.get("delta") or {}
            text = first_text(
                message.get("content") if isinstance(message, dict) else message,
                delta.get("content") if isinstance(delta, dict) else delta,
                choice.get("text"),
                choice.get("content"),
            )
            if text:
                return text

    candidates = data.get("candidates")
    if isinstance(candidates, list) and candidates:
        for candidate in candidates:
            content = candidate.get("content", {}) if isinstance(candidate, dict) else {}
            parts = content.get("parts", []) if isinstance(content, dict) else []
            text = first_text(*[part.get("text") for part in parts if isinstance(part, dict)])
            if text:
                return text

    content = data.get("content")
    if isinstance(content, list):
        text = first_text(*[item.get("text") for item in content if isinstance(item, dict)])
        if text:
            return text

    message = data.get("message")
    if isinstance(message, dict):
        text = first_text(message.get("content"), message.get("text"))
        if text:
            return text

    for nested_key in ("data", "reply", "message", "result"):
        nested = data.get(nested_key)
        if isinstance(nested, (dict, list)):
            text = read_assistant_text(nested)
            if text:
                return text

    outputs = data.get("outputs") or data.get("output") or data.get("generations")
    if isinstance(outputs, list) and outputs:
        for output in outputs:
            if isinstance(output, dict):
                text = first_text(
                    output.get("text"),
                    output.get("generated_text"),
                    output.get("content"),
                    output.get("message", {}).get("content") if isinstance(output.get("message"), dict) else "",
                )
                if text:
                    return text
            else:
                text = stringify_content(output).strip()
                if text:
                    return text

    for key in (
        "response",
        "text",
        "content",
        "completion",
        "generated_text",
        "answer",
        "result",
        "output_text",
    ):
        text = first_text(data.get(key))
        if text:
            return text

    return ""


def parse_response_text(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not items:
        return {"text": text}

    merged = []
    for item in items:
        piece = read_assistant_text(item)
        if piece:
            merged.append(piece)
    if merged:
        return {"content": "".join(merged), "chunks": items}
    return items[-1]


def main():
    print(f"Local proxy started: http://{HOST}:{PORT}")
    print("Close this window to stop the proxy.")
    server = ThreadingHTTPServer((HOST, PORT), ProxyHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
