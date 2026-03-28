"""LLM client wrappers — Ollama (local) and Groq (cloud API)."""

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import json

import ollama
import structlog

from ..config import get_settings

logger = structlog.get_logger()

# Debug log file path
DEBUG_LOG_PATH = Path("data/llm_debug.log")


def _write_debug_log(label: str, content: str, thinking: str = None) -> None:
    """Write full LLM output to debug log file for debugging truncation issues."""
    settings = get_settings()
    if not getattr(settings, 'debug_log_llm_output', False):
        return

    try:
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"[{datetime.utcnow().isoformat()}] {label}\n")
            f.write(f"{'='*80}\n")
            if thinking:
                f.write(f"--- THINKING ({len(thinking)} chars) ---\n")
                f.write(thinking[:2000] + ("..." if len(thinking) > 2000 else ""))
                f.write("\n\n")
            f.write(f"--- CONTENT ({len(content)} chars) ---\n")
            f.write(content)
            f.write("\n")
    except Exception as e:
        logger.warning("debug_log_write_failed", error=str(e))


@dataclass
class LLMResponse:
    """Response from the LLM."""

    content: str
    tool_calls: list[dict] = field(default_factory=list)
    model: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class ToolCall:
    """A tool call from the LLM."""

    name: str
    arguments: dict[str, Any]
    id: str = ""


def _clean_llm_content(content: str) -> str:
    """Strip Qwen3 thinking tags and leaked CJK characters from LLM output."""
    if "</think>" in content:
        think_end = content.find("</think>")
        content = content[think_end + len("</think>"):].strip()
    elif "<think>" in content:
        think_start = content.find("<think>")
        before = content[:think_start].strip()
        after = content[think_start + len("<think>"):].strip()
        content = before if before else after

    # Strip CJK characters that Qwen occasionally leaks
    content = re.sub(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+', '', content).strip()
    return content


class OllamaClient:
    """Async wrapper around Ollama client."""

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        max_workers: int = 4,
    ):
        settings = get_settings()
        self.model = model or settings.ollama_model
        self.host = host or settings.ollama_host
        self.timeout = settings.llm_timeout
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._client = ollama.Client(host=self.host)

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        tools: Optional[list[dict]] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        json_schema: Optional[dict] = None,
        think: Optional[bool] = None,
    ) -> LLMResponse:
        """
        Send a chat request to Ollama.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            tools: Optional list of tool definitions for function calling
            max_tokens: Maximum tokens to generate
            json_mode: If True, force JSON output format (simple mode)
            json_schema: Optional JSON schema dict for structured output
            think: If False, disable Qwen3 thinking mode (<think> tags)

        Returns:
            LLMResponse with content and any tool calls
        """
        loop = asyncio.get_event_loop()

        def _sync_chat():
            options = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens

            kwargs = {
                "model": self.model,
                "messages": messages,
                "options": options,
            }

            if tools:
                kwargs["tools"] = tools

            # Structured output: prefer schema over simple json mode
            if json_schema:
                kwargs["format"] = json_schema
            elif json_mode:
                kwargs["format"] = "json"

            # Disable Qwen3 thinking mode if specified
            if think is not None:
                kwargs["think"] = think

            return self._client.chat(**kwargs)

        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(self._executor, _sync_chat),
                timeout=self.timeout,
            )

            # Log the raw response type and structure
            logger.info(
                "ollama_raw_response",
                response_type=type(response).__name__,
                has_message_attr=hasattr(response, "message"),
                is_dict=isinstance(response, dict),
            )

            # Handle both dict and object response formats
            if hasattr(response, "message"):
                # Object-style response (newer ollama library)
                message = response.message
                raw_content = getattr(message, "content", None)
                # Ollama might also have thinking in separate field
                raw_thinking = getattr(message, "thinking", None)
                logger.info(
                    "ollama_message_extracted",
                    content_type=type(raw_content).__name__ if raw_content else "None",
                    content_length=len(str(raw_content)) if raw_content else 0,
                    content_preview=str(raw_content)[:200] if raw_content else "EMPTY",
                    has_thinking=raw_thinking is not None,
                )
                # Debug log full output
                _write_debug_log(
                    "LLM_RESPONSE",
                    str(raw_content) if raw_content else "(empty)",
                    str(raw_thinking) if raw_thinking else None,
                )
                message = {"content": raw_content or "", "tool_calls": getattr(message, "tool_calls", None) or []}
            else:
                # Dict-style response
                message = response.get("message", {})
                logger.info(
                    "ollama_message_extracted",
                    content_length=len(message.get("content", "")) if message.get("content") else 0,
                    content_preview=str(message.get("content", ""))[:200] if message.get("content") else "EMPTY",
                )

            # Extract tool calls if present
            tool_calls = []

            if "tool_calls" in message:
                for tc in message["tool_calls"]:
                    func = tc.get("function", {})
                    tool_calls.append({
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", {}),
                    })

            content = _clean_llm_content(message.get("content", ""))

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                model=response.get("model", self.model),
                finish_reason=response.get("done_reason", ""),
                prompt_tokens=response.get("prompt_eval_count", 0),
                completion_tokens=response.get("eval_count", 0),
            )

        except asyncio.TimeoutError:
            logger.error("ollama_chat_timeout", timeout=self.timeout, model=self.model)
            raise TimeoutError(
                f"LLM call timed out after {self.timeout}s. "
                "Ollama may be overloaded or the model may be stuck."
            )
        except Exception as e:
            logger.error("ollama_chat_error", error=str(e))
            raise

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text from a prompt (non-chat mode).

        Args:
            prompt: The prompt to complete
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        loop = asyncio.get_event_loop()

        def _sync_generate():
            options = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens

            return self._client.generate(
                model=self.model,
                prompt=prompt,
                options=options,
            )

        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(self._executor, _sync_generate),
                timeout=self.timeout,
            )
            return response.get("response", "")
        except asyncio.TimeoutError:
            logger.error("ollama_generate_timeout", timeout=self.timeout, model=self.model)
            raise TimeoutError(
                f"LLM generate timed out after {self.timeout}s."
            )
        except Exception as e:
            logger.error("ollama_generate_error", error=str(e))
            raise

    async def is_available(self) -> bool:
        """Check if Ollama is available and the model is loaded."""
        loop = asyncio.get_event_loop()

        def _sync_check():
            try:
                models = self._client.list()
                model_names = [m.get("name", "") for m in models.get("models", [])]
                return any(self.model in name for name in model_names)
            except Exception:
                return False

        return await loop.run_in_executor(self._executor, _sync_check)

    def close(self):
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)


class GroqClient:
    """Async client for Groq cloud API (OpenAI-compatible)."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        try:
            from groq import AsyncGroq
        except ImportError:
            raise ImportError(
                "groq package required for Groq provider. Install with: pip install groq"
            )

        settings = get_settings()
        self.model = model or settings.groq_model
        self.timeout = settings.llm_timeout
        key = api_key or settings.groq_api_key
        if not key:
            raise ValueError("GROQ_API_KEY must be set when using Groq provider")
        self._client = AsyncGroq(api_key=key)

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        tools: Optional[list[dict]] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        json_schema: Optional[dict] = None,
        think: Optional[bool] = None,
    ) -> LLMResponse:
        """Send a chat request to Groq API. Same interface as OllamaClient."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        if tools:
            kwargs["tools"] = tools

        # JSON mode — Groq uses OpenAI-style response_format
        if json_schema or json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Thinking/reasoning mode for Qwen3 on Groq
        if think is False:
            # Disable thinking — hide reasoning entirely
            kwargs["reasoning_format"] = "hidden"
        elif think is True:
            # Enable thinking — parse separately so it doesn't pollute content
            kwargs["reasoning_format"] = "parsed"

        try:
            import time as _time
            _t0 = _time.monotonic()

            response = await asyncio.wait_for(
                self._client.chat.completions.create(**kwargs),
                timeout=self.timeout,
            )

            _elapsed = _time.monotonic() - _t0

            choice = response.choices[0] if response.choices else None
            message = choice.message if choice else None

            raw_content = message.content if message else ""
            raw_thinking = getattr(message, "reasoning", None)

            # Debug log with timing
            _write_debug_log(
                f"LLM_RESPONSE (api={_elapsed:.1f}s, think={'yes' if raw_thinking else 'no'})",
                raw_content or "(empty)",
                str(raw_thinking) if raw_thinking else None,
            )

            logger.info(
                "groq_response",
                content_length=len(raw_content) if raw_content else 0,
                content_preview=(raw_content or "")[:200],
                has_thinking=raw_thinking is not None,
                model=response.model,
            )

            content = _clean_llm_content(raw_content or "")

            # Extract tool calls
            tool_calls = []
            if message and message.tool_calls:
                for tc in message.tool_calls:
                    args = tc.function.arguments
                    # Groq returns arguments as JSON string
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    tool_calls.append({
                        "name": tc.function.name,
                        "arguments": args,
                    })

            usage = response.usage
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                model=response.model or self.model,
                finish_reason=choice.finish_reason if choice else "",
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            )

        except asyncio.TimeoutError:
            logger.error("groq_chat_timeout", timeout=self.timeout, model=self.model)
            raise TimeoutError(
                f"Groq API call timed out after {self.timeout}s."
            )
        except Exception as e:
            logger.error("groq_chat_error", error=str(e))
            raise

    async def is_available(self) -> bool:
        """Check if Groq API is reachable."""
        try:
            await asyncio.wait_for(
                self._client.models.list(),
                timeout=10,
            )
            return True
        except Exception:
            return False

    def close(self):
        """No-op — async client cleans up automatically."""
        pass


# Global client instance
_client = None


def get_llm_client():
    """Get the global LLM client based on configured provider."""
    global _client
    if _client is None:
        settings = get_settings()
        if settings.llm_provider == "groq":
            _client = GroqClient()
            logger.info("llm_client_init", provider="groq", model=settings.groq_model)
        else:
            _client = OllamaClient()
            logger.info("llm_client_init", provider="ollama", model=settings.ollama_model)
    return _client


# Alias for backwards compatibility
get_ollama_client = get_llm_client
