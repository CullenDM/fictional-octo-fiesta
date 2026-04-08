"""
Model-agnostic Ollama client for GREAT SAGE.

All model tags come from config.yaml — user picks whatever fits their RAM.
Supports the Worker and Skeptic roles with model loading/unloading for
role swap (Stage 2).
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .parser import extract_json, extract_thought

import httpx

logger = logging.getLogger("great_sage.ollama")


@dataclass
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    worker_model: str = "qwen2.5:1.5b"
    skeptic_model: str = "qwen2.5:1.5b"
    worker_ctx: int = 8192
    skeptic_ctx: int = 4096
    timeout: float = 120.0
    max_retries: int = 3


class OllamaClient:
    """Async Ollama HTTP client with model-agnostic configuration."""

    def __init__(self, config: OllamaConfig | None = None):
        self.config = config or OllamaConfig()
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=httpx.Timeout(self.config.timeout, connect=10.0),
        )
        self._total_tokens = 0

    async def close(self):
        await self._client.aclose()

    @property
    def total_tokens_consumed(self) -> int:
        return self._total_tokens

    # -------------------------------------------------------------------
    # Core generation
    # -------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        role: str = "worker",
        system_prompt: str | None = None,
        format_json: bool = True,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """
        Generate a completion from the configured model.

        Args:
            prompt: The user prompt
            role: "worker" or "skeptic" — determines which model to use
            system_prompt: Optional system message
            format_json: If True, request JSON output format
            temperature: Sampling temperature

        Returns:
            dict with keys: response (str), tokens (int), model (str)
        """
        model = (
            self.config.worker_model
            if role == "worker"
            else self.config.skeptic_model
        )
        ctx = (
            self.config.worker_ctx
            if role == "worker"
            else self.config.skeptic_ctx
        )

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": ctx,
                "temperature": temperature,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        if format_json:
            payload["format"] = "json"

        for attempt in range(self.config.max_retries):
            try:
                resp = await self._client.post("/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()

                tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
                self._total_tokens += tokens

                response_text = data.get("response", "")

                return {
                    "response": response_text,
                    "tokens": tokens,
                    "model": model,
                    "eval_duration_ms": data.get("eval_duration", 0) // 1_000_000,
                }

            except httpx.TimeoutException:
                logger.warning(
                    f"Ollama timeout (attempt {attempt + 1}/{self.config.max_retries})"
                )
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama HTTP error: {e.response.status_code}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError("Ollama generation failed after all retries")

    async def generate_json(
        self,
        prompt: str,
        role: str = "worker",
        system_prompt: str | None = None,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """
        Generate and parse JSON output.
        Handles <think> tags and conversational prose fallback.
        """
        result = await self.generate(
            prompt=prompt,
            role=role,
            system_prompt=system_prompt,
            format_json=True,
            temperature=temperature,
        )

        response_text = result["response"]
        parsed = extract_json(response_text)
        thought = extract_thought(response_text)

        if parsed is not None:
            return {**result, "parsed": parsed, "thought": thought}
        else:
            logger.warning(f"JSON extraction failed for response: {response_text[:200]}...")
            return {
                **result,
                "parsed": {"raw_response": response_text},
                "thought": thought,
            }

    # -------------------------------------------------------------------
    # Model management (for Stage 2 role swap)
    # -------------------------------------------------------------------

    async def is_model_available(self, model: str | None = None) -> bool:
        """Check if a model is available in Ollama."""
        target = model or self.config.worker_model
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return any(m["name"] == target or m["model"] == target for m in models)
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            return False

    async def load_model(self, model: str) -> bool:
        """Pre-load a model into memory."""
        try:
            resp = await self._client.post(
                "/api/generate",
                json={"model": model, "prompt": "", "keep_alive": "10m"},
                timeout=httpx.Timeout(300.0),
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Failed to load model {model}: {e}")
            return False

    async def unload_model(self, model: str) -> bool:
        """Unload a model from memory."""
        try:
            resp = await self._client.post(
                "/api/generate",
                json={"model": model, "prompt": "", "keep_alive": "0"},
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Failed to unload model {model}: {e}")
            return False

    async def check_health(self) -> bool:
        """Check if Ollama server is running."""
        try:
            resp = await self._client.get("/")
            return resp.status_code == 200
        except Exception:
            return False
