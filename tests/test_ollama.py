import io
import json
import urllib.error
import urllib.request

import pytest

from ayasteer.llm.ollama import OllamaClient, OllamaError


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_chat_posts_expected_payload(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data)
        captured["timeout"] = timeout
        return _Resp(json.dumps({"message": {"role": "assistant", "content": "hola"}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = OllamaClient(model="m", base_url="http://x:1", timeout=7).chat("sys", "usr", json_mode=True, max_tokens=5)
    assert out == "hola"
    assert captured["url"] == "http://x:1/api/chat"
    body = captured["body"]
    assert body["model"] == "m" and body["stream"] is False and body["think"] is False
    assert body["format"] == "json"
    assert body["options"] == {"temperature": 0, "num_predict": 5, "num_ctx": 2048}
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
    assert captured["timeout"] == 7


def test_chat_omits_format_when_not_json(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data)
        return _Resp(json.dumps({"message": {"content": "x"}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    OllamaClient().chat("s", "u")
    assert "format" not in captured["body"]


def test_chat_raises_on_http_error(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, io.BytesIO(b"err"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(OllamaError):
        OllamaClient().chat("s", "u")


def test_chat_raises_on_malformed_body(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp(b"{}"))
    with pytest.raises(OllamaError):
        OllamaClient().chat("s", "u")
