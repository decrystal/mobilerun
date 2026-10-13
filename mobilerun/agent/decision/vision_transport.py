"""OpenAI-compatible chat transport for the visual decision adapter.

Provider details (endpoint, credentials, the GLM thinking switch, the GLM busy
code, model token limits, JSON-mode support) live here so the decision
protocol in ``visual_fallback.py`` stays provider-neutral.
"""

from __future__ import annotations

import asyncio
import math
import os
from typing import Literal

import httpx

BIGMODEL_CHAT_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


class VisionTransport:
    def __init__(
        self,
        *,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        thinking_mode: Literal["enabled", "disabled"] | None = None,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Visual transport requires a positive finite timeout.")
        self.timeout = timeout
        self._client = client
        if thinking_mode not in {None, "enabled", "disabled"}:
            raise ValueError("thinking_mode must be enabled, disabled or None.")
        self.thinking_mode = thinking_mode
        url = httpx.URL(base_url if base_url is not None else BIGMODEL_CHAT_URL)
        if (
            url.scheme not in {"http", "https"}
            or not url.host
            or url.userinfo
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "Visual model base_url must be an HTTP(S) URL without credentials or query."
            )
        self.endpoint = str(url).rstrip("/")
        if not self.endpoint.endswith("/chat/completions"):
            self.endpoint += "/chat/completions"
        if api_key_env is not None and (
            not isinstance(api_key_env, str)
            or not api_key_env.isidentifier()
            or not api_key_env.isascii()
        ):
            raise ValueError("api_key_env must be an environment variable name.")
        # A custom provider must never inherit another provider's credentials.
        self.api_key_envs = (
            (api_key_env,) if api_key_env else ("BIGMODEL_API_KEY", "ZAI_API_KEY")
        )
        # Set once an endpoint rejects response_format; later requests omit it.
        self._json_mode_unsupported = False

    def api_key(self) -> str | None:
        return next(
            (
                value.strip()
                for name in self.api_key_envs
                if (value := os.environ.get(name)) and value.strip()
            ),
            None,
        )

    def missing_key_message(self) -> str:
        return (
            "Set " + " or ".join(self.api_key_envs) + " for the visual model provider."
        )

    async def request(
        self, payload: dict, headers: dict, *, retry_transient: bool = False
    ) -> httpx.Response:
        if payload.get("model") == "glm-4v-flash":
            payload = {**payload, "max_tokens": 1024}  # This model's output limit.
        if self.thinking_mode is not None:
            payload = {**payload, "thinking": {"type": self.thinking_mode}}
        plain = {k: v for k, v in payload.items() if k != "response_format"}
        if self._json_mode_unsupported:
            payload = plain
        response = await self._send(payload, headers, retry_transient)
        if "response_format" in payload and response.status_code in {400, 422}:
            # JSON mode is not in the documented visual schema; an endpoint that
            # rejects it falls back to prompt-only JSON with strict local parsing.
            self._json_mode_unsupported = True
            response = await self._send(plain, headers, retry_transient)
        return response

    async def _send(self, payload, headers, retry_transient) -> httpx.Response:
        if self._client is not None:
            return await asyncio.wait_for(
                self._post(self._client, payload, headers, retry_transient),
                timeout=self.timeout,
            )
        async with httpx.AsyncClient() as client:
            return await asyncio.wait_for(
                self._post(client, payload, headers, retry_transient),
                timeout=self.timeout,
            )

    async def _post(
        self,
        client: httpx.AsyncClient,
        payload: dict,
        headers: dict,
        retry_transient: bool,
    ) -> httpx.Response:
        for attempt in range(2):
            response = await client.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
                follow_redirects=False,
            )
            transient = retry_transient and response.status_code in {500, 502, 503, 504}
            if attempt or (response.status_code != 429 and not transient):
                return response
            try:
                busy = response.json()["error"]["code"] in (1305, "1305")  # GLM busy
            except (ValueError, TypeError, KeyError):
                busy = False
            if not busy and not transient:
                return response
            # Only inference is retried, before any device action. The outer
            # request deadline includes both attempts and this backoff.
            await asyncio.sleep(0.25 if transient else 2)
        raise AssertionError("unreachable")
