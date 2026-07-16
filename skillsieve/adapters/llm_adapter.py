"""LLM adapter supporting the model providers used by SkillSieve."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass


# Pre-configured model profiles matching the experiments reported in the paper.
# Endpoints and credentials remain configurable through LLMAdapter overrides.
MODEL_PROFILES = {
    "kimi": {
        "base_url": "https://api.kimi.com/coding/v1/messages",
        "api_type": "anthropic",
        "model": "kimi-2.5",
        "env_key": "KIMI_API_KEY",
    },
    "glm": {
        # GLM-5.1, Layer 3 juror 1.
        "base_url": "https://qianfan.baidubce.com/v2/chat/completions",
        "api_type": "openai",
        "model": "glm-5.1",
        "env_key": "BAIDU_API_KEY",
    },
    "qwen": {
        # Qwen3-235B, Layer 3 juror 2.
        "base_url": "https://qianfan.baidubce.com/v2/chat/completions",
        "api_type": "openai",
        "model": "qwen3-235b-a22b-instruct-2507",
        "env_key": "BAIDU_API_KEY",
    },
    "minimax": {
        "base_url": "https://api.minimaxi.com/v1/chat/completions",
        "api_type": "openai",
        "model": "MiniMax-M2.7",
        "env_key": "MINIMAX_API_KEY",
    },
    "baidu": {
        # DeepSeek-V3.1, Layer 3 juror 3.
        "base_url": "https://qianfan.baidubce.com/v2/chat/completions",
        "api_type": "openai",
        "model": "deepseek-v3.1-250821",
        "env_key": "BAIDU_API_KEY",
    },
}


@dataclass
class LLMAdapter:
    """Unified LLM adapter that calls Kimi, MiniMax, or Baidu Qianfan APIs.

    Parameters
    ----------
    profile:
        One of the names in ``MODEL_PROFILES`` or a custom name.
    model:
        Override the default model for this profile.
    api_key:
        API key. If not provided, reads from the environment variable
        defined in the profile (e.g. KIMI_API_KEY).
    base_url:
        Override the API endpoint URL.
    api_type:
        "openai" or "anthropic" — determines request/response format.
    temperature:
        Sampling temperature.
    max_tokens:
        Max tokens for the completion.
    timeout:
        Request timeout in seconds.
    """

    profile: str = "kimi"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    api_type: str = ""
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: float = 120.0

    def __post_init__(self):
        cfg = MODEL_PROFILES.get(self.profile, {})
        if not self.model:
            self.model = cfg.get("model", self.profile)
        if not self.base_url:
            self.base_url = cfg.get("base_url", "")
        if not self.api_type:
            self.api_type = cfg.get("api_type", "openai")
        if not self.api_key:
            env_key = cfg.get("env_key", "")
            self.api_key = os.environ.get(env_key, "") if env_key else ""

    async def complete(self, prompt: str, system: str = "", retries: int = 4) -> str:
        """Call the LLM and return the raw response text.

        Retries transient failures (timeouts, connection errors, 429 rate
        limits, 5xx server errors). Rate-limited and server-error responses
        back off exponentially (15/45/90/180s) to tolerate provider
        throttling during high-volume evaluation.
        """
        import asyncio

        for attempt in range(retries + 1):
            try:
                if self.api_type == "anthropic":
                    return await self._call_anthropic(prompt, system)
                else:
                    return await self._call_openai(prompt, system)
            except Exception as e:
                etype = type(e).__name__
                msg = str(e).lower()
                rate_limited = "429" in msg or "too many" in msg or "rate" in msg
                server_error = any(code in msg for code in ["500 ", "502 ", "503 ", "504 ", "529 "])
                transient = (
                    "timeout" in etype.lower()
                    or "timeout" in msg
                    or "connect" in etype.lower()      # httpx.ConnectError
                    or "remoteprotocol" in etype.lower()  # httpx.RemoteProtocolError
                    or "readerror" in etype.lower()
                    or rate_limited
                    or server_error
                )
                if attempt < retries and transient:
                    if rate_limited or server_error:
                        # Exponential backoff for throttling / server errors.
                        delay = [15, 45, 90, 180][min(attempt, 3)]
                    else:
                        delay = 3 * (attempt + 1)
                    await asyncio.sleep(delay)
                    continue
                raise

    async def _call_openai(self, prompt: str, system: str) -> str:
        """OpenAI-compatible API call (MiniMax, Baidu Qianfan)."""
        import httpx

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.base_url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def _call_anthropic(self, prompt: str, system: str) -> str:
        """Anthropic-compatible API call (Kimi)."""
        import httpx

        # Kimi uses Anthropic-compatible format
        messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if system:
            payload["system"] = system

        # Kimi endpoint: base_url is like https://api.kimi.com/coding/
        # The messages endpoint is at /v1/messages
        url = self.base_url
        if not url.endswith("/messages"):
            url = url.rstrip("/")
            if not url.endswith("/v1"):
                url += "/v1"
            url += "/messages"

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # Anthropic response format: {"content": [{"type": "...", ...}]}.
            # Some reasoning models emit a "thinking" block before the "text"
            # answer, so pick the first text block rather than content[0].
            content = data.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block.get("text", "")
            return ""

    @staticmethod
    def parse_json_response(raw: str) -> dict:
        """Parse a JSON dict from an LLM response string.

        Tries (in order):
        1. Extract JSON from a ```json ... ``` fenced code block.
        2. Parse the raw text directly as JSON.
        3. Find the first { ... } substring and parse it.
        4. Return {} on failure.
        """
        if not raw:
            return {}

        block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if block_match:
            try:
                return json.loads(block_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        try:
            result = json.loads(raw)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace_match:
            try:
                result = json.loads(brace_match.group(0))
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        return {}
