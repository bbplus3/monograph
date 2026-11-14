"""Local LLM via Ollama (ported from spine_proposer.call_ollama). Talks only to
localhost, so no diary text leaves the machine — the privacy-preserving default
for the generative stages. Host/model come from Config.options."""
from __future__ import annotations
import json
import re
import urllib.request
import urllib.error
from .base import LLM


class OllamaLLM(LLM):
    def __init__(self, cfg=None):
        opts = (cfg.options if cfg and getattr(cfg, "options", None) else {}) or {}
        self.host = opts.get("llm_host", "http://localhost:11434")
        self.model = opts.get("llm_model", "llama3.1:8b")
        self.num_ctx = int(opts.get("num_ctx", 8192))

    def complete(self, prompt, *, system=None, temperature=0.7,
                 max_tokens=1024, json_mode=False) -> str:
        body = {
            "model": self.model, "stream": False,
            "messages": ([{"role": "system", "content": system}] if system else [])
                        + [{"role": "user", "content": prompt}],
            "options": {"num_ctx": self.num_ctx, "temperature": temperature},
        }
        if json_mode:
            body["format"] = "json"
        req = urllib.request.Request(
            self.host.rstrip("/") + "/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host} ({e}). Is it running? "
                f"Try: ollama serve  (and ollama pull {self.model})") from e
        return data.get("message", {}).get("content", "")


def parse_json(content: str):
    """Parse JSON from a model response, tolerating stray prose around it."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            return json.loads(m.group(0))
        raise ValueError("Model did not return parseable JSON:\n" + content[:2000])
