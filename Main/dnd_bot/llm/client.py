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

from ..config import get_settings, get_profile

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
        num_ctx: Optional[int] = None,
    ):
        settings = get_settings()
        self.model = model or "qwen3.5:35b-a3b"
        self.host = host or settings.ollama_host
        self.timeout = settings.llm_timeout
        self.num_ctx = num_ctx  # Cap context window to control VRAM
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
        tool_choice: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
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
            tool_choice: Tool choice mode - "auto", "required", or a specific
                tool name. "required" forces the model to call at least one tool.
                Ollama doesn't support this natively, so we hint via system message.
            frequency_penalty: Penalize tokens by how often they've appeared (0.0-2.0).
                Maps to Ollama's repeat_penalty (offset by 1.0).
            presence_penalty: Penalize any token that has appeared at all (0.0-2.0).
                Maps to Ollama's presence_penalty option.

        Returns:
            LLMResponse with content and any tool calls
        """
        loop = asyncio.get_event_loop()

        # If tool_choice is set and Ollama doesn't support it natively,
        # add a system hint to strongly encourage tool use
        if tool_choice and tool_choice != "auto" and tools:
            hint = {
                "role": "system",
                "content": (
                    "IMPORTANT: You MUST call at least one tool before responding. "
                    "Do NOT generate a text response without calling a tool first."
                ),
            }
            if tool_choice not in ("required", "any"):
                hint["content"] = (
                    f"IMPORTANT: You MUST call the '{tool_choice}' tool. "
                    "Do NOT generate a text response without calling this tool."
                )
            messages = [*messages, hint]

        def _sync_chat():
            options = {"temperature": temperature}
            if self.num_ctx:
                options["num_ctx"] = self.num_ctx
            if max_tokens:
                options["num_predict"] = max_tokens
            if frequency_penalty is not None:
                # Ollama uses repeat_penalty (default 1.0, higher = more penalty)
                # OpenAI frequency_penalty is 0.0-2.0, map to Ollama's 1.0-1.5 range
                options["repeat_penalty"] = 1.0 + frequency_penalty
            if presence_penalty is not None:
                options["presence_penalty"] = presence_penalty

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

            # Pass think mode to Ollama. Qwen3/3.5 can hang or produce
            # empty output with thinking enabled, so callers pass
            # think=False for reliability.
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

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        think: Optional[bool] = None,
        on_token: Optional[Any] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> LLMResponse:
        """
        Stream a chat response from Ollama, calling on_token(text) for each chunk.

        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            think: If False, disable Qwen3 thinking mode
            on_token: Async callback(str) called with each text chunk
            frequency_penalty: Penalize repeated tokens (0.0-2.0)
            presence_penalty: Penalize any already-used token (0.0-2.0)

        Returns:
            Complete LLMResponse after streaming finishes
        """
        loop = asyncio.get_event_loop()

        def _sync_stream():
            options = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens
            if frequency_penalty is not None:
                options["repeat_penalty"] = 1.0 + frequency_penalty
            if presence_penalty is not None:
                options["presence_penalty"] = presence_penalty

            kwargs = {
                "model": self.model,
                "messages": messages,
                "options": options,
                "stream": True,
            }
            if think is not None:
                kwargs["think"] = think

            return self._client.chat(**kwargs)

        try:
            stream = await asyncio.wait_for(
                loop.run_in_executor(self._executor, _sync_stream),
                timeout=self.timeout,
            )

            full_content = []
            prompt_tokens = 0
            completion_tokens = 0

            # Iterate over stream chunks in executor to avoid blocking
            def _collect_chunks():
                chunks = []
                for chunk in stream:
                    chunks.append(chunk)
                return chunks

            chunks = await loop.run_in_executor(self._executor, _collect_chunks)

            for chunk in chunks:
                if hasattr(chunk, "message"):
                    delta = getattr(chunk.message, "content", "") or ""
                else:
                    delta = chunk.get("message", {}).get("content", "") or ""

                if delta:
                    full_content.append(delta)
                    if on_token:
                        await on_token(delta)

                # Capture usage from final chunk
                if hasattr(chunk, "prompt_eval_count"):
                    prompt_tokens = getattr(chunk, "prompt_eval_count", 0) or 0
                    completion_tokens = getattr(chunk, "eval_count", 0) or 0
                elif isinstance(chunk, dict):
                    prompt_tokens = chunk.get("prompt_eval_count", 0) or 0
                    completion_tokens = chunk.get("eval_count", 0) or 0

            content = _clean_llm_content("".join(full_content))

            return LLMResponse(
                content=content,
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        except asyncio.TimeoutError:
            logger.error("ollama_stream_timeout", timeout=self.timeout, model=self.model)
            raise TimeoutError(f"LLM stream timed out after {self.timeout}s.")
        except Exception as e:
            logger.error("ollama_stream_error", error=str(e))
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
        self.model = model or "qwen/qwen3-32b"
        self.timeout = settings.llm_timeout
        self._max_retries = settings.groq_max_retries
        self._fallback_enabled = False  # Set by _create_client from profile
        key = api_key or settings.groq_api_key
        if not key:
            raise ValueError("GROQ_API_KEY must be set when using Groq provider")
        # Let the SDK retry up to max_retries before we fall back
        self._client = AsyncGroq(api_key=key, max_retries=self._max_retries)

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        tools: Optional[list[dict]] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        json_schema: Optional[dict] = None,
        think: Optional[bool] = None,
        tool_choice: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> LLMResponse:
        """Send a chat request to Groq API. Same interface as OllamaClient."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if frequency_penalty is not None:
            kwargs["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        if tools:
            kwargs["tools"] = tools
            # Groq/OpenAI-compatible tool_choice support
            if tool_choice:
                if tool_choice in ("auto", "required", "none"):
                    kwargs["tool_choice"] = tool_choice
                else:
                    # Force a specific tool by name
                    kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }

        # JSON mode — Groq uses OpenAI-style response_format
        if json_schema or json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Thinking/reasoning mode for Qwen3 on Groq
        # IMPORTANT: Don't use reasoning_format with JSON or tool modes.
        # "hidden" still consumes output tokens for internal reasoning,
        # leaving too few for actual content — causing empty output,
        # json_validate_failed errors, and dropped tool calls.
        if json_schema or json_mode or tools:
            pass  # Skip reasoning_format — let model output directly
        elif think is False:
            kwargs["reasoning_format"] = "hidden"
        elif think is True:
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
            error_str = str(e)
            is_rate_limit = "429" in error_str or "Too Many Requests" in error_str or "rate" in error_str.lower()

            if is_rate_limit and self._fallback_enabled:
                logger.warning(
                    "groq_rate_limited_falling_back_to_ollama",
                    error=error_str[:100],
                    retries_exhausted=self._max_retries,
                )
                return await self._fallback_ollama(
                    messages, temperature, tools, max_tokens, json_mode, json_schema, think, tool_choice
                )

            logger.error("groq_chat_error", error=error_str)
            raise

    async def _fallback_ollama(
        self,
        messages: list[dict],
        temperature: float,
        tools: Optional[list[dict]],
        max_tokens: Optional[int],
        json_mode: bool,
        json_schema: Optional[dict],
        think: Optional[bool],
        tool_choice: Optional[str] = None,
    ) -> LLMResponse:
        """Fall back to local Ollama when Groq is rate-limited."""
        if not hasattr(self, '_ollama_fallback'):
            try:
                self._ollama_fallback = OllamaClient()
                logger.info("ollama_fallback_initialized", model=self._ollama_fallback.model)
            except Exception as e:
                logger.error("ollama_fallback_init_failed", error=str(e))
                raise RuntimeError("Groq rate limited and Ollama fallback unavailable") from e

        logger.info("using_ollama_fallback")
        return await self._ollama_fallback.chat(
            messages=messages,
            temperature=temperature,
            tools=tools,
            max_tokens=max_tokens,
            json_mode=json_mode,
            json_schema=json_schema,
            think=think,
            tool_choice=tool_choice,
        )

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


class AnthropicClient:
    """Async client for Anthropic Claude API (narrator-only)."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

        settings = get_settings()
        self.model = model or "claude-sonnet-4-6"
        self.timeout = settings.llm_timeout
        key = api_key or settings.anthropic_api_key
        if not key:
            # Fallback: read directly from .env (system env may shadow with empty string)
            from dotenv import dotenv_values
            env = dotenv_values()
            key = env.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY must be set when using Anthropic provider")
        self._client = anthropic.AsyncAnthropic(api_key=key)

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        tools: Optional[list[dict]] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        json_schema: Optional[dict] = None,
        think: Optional[bool] = None,
        tool_choice: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> LLMResponse:
        """Send a chat request to Claude API. Same interface as OllamaClient.

        Note: Claude does not support frequency/presence penalties. These
        params are accepted for interface compatibility but ignored.
        """
        # Claude uses a separate system parameter, not a system message
        system_parts = []
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                chat_messages.append(msg)

        # Claude requires alternating user/assistant messages
        # Merge consecutive same-role messages
        merged = []
        for msg in chat_messages:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append({"role": msg["role"], "content": msg["content"]})

        # Ensure first message is from user
        if not merged or merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "content": "Begin."})

        kwargs = {
            "model": self.model,
            "messages": merged,
            "max_tokens": max_tokens or 2000,
            "temperature": temperature,
        }

        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        # Claude uses input_schema instead of parameters for tools
        if tools:
            claude_tools = []
            for t in tools:
                func = t.get("function", {})
                claude_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })
            kwargs["tools"] = claude_tools
            if tool_choice == "auto":
                kwargs["tool_choice"] = {"type": "auto"}
            elif tool_choice == "required":
                kwargs["tool_choice"] = {"type": "any"}

        try:
            import time as _time
            _t0 = _time.monotonic()

            response = await asyncio.wait_for(
                self._client.messages.create(**kwargs),
                timeout=self.timeout,
            )

            _elapsed = _time.monotonic() - _t0

            content = ""
            tool_calls = []
            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text
                    elif block.type == "tool_use":
                        tool_calls.append({
                            "name": block.name,
                            "arguments": block.input,
                        })

            _write_debug_log(
                f"ANTHROPIC_RESPONSE (api={_elapsed:.1f}s)",
                content or "(empty)",
            )

            logger.info(
                "anthropic_response",
                content_length=len(content),
                content_preview=content[:200],
                tool_calls=len(tool_calls),
                model=response.model,
            )

            usage = response.usage
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                model=response.model or self.model,
                finish_reason=response.stop_reason or "",
                prompt_tokens=usage.input_tokens if usage else 0,
                completion_tokens=usage.output_tokens if usage else 0,
            )

        except asyncio.TimeoutError:
            logger.error("anthropic_chat_timeout", timeout=self.timeout, model=self.model)
            raise TimeoutError(f"Anthropic API call timed out after {self.timeout}s.")
        except Exception as e:
            logger.error("anthropic_chat_error", error=str(e))
            raise

    async def is_available(self) -> bool:
        """Check if Anthropic API is reachable."""
        try:
            # Simple models list check
            return True  # If constructor didn't raise, key is set
        except Exception:
            return False

    def close(self):
        """No-op — async client cleans up automatically."""
        pass


class OpenRouterClient:
    """Async client for OpenRouter API (OpenAI-compatible)."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package required for OpenRouter. Install with: pip install openai"
            )

        settings = get_settings()
        self.model = model or "qwen/qwen3.6-plus:free"
        self.timeout = settings.llm_timeout
        key = api_key or settings.openrouter_api_key
        if not key:
            raise ValueError("OPENROUTER_API_KEY must be set when using OpenRouter provider")
        self._client = AsyncOpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
        )

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        tools: Optional[list[dict]] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        json_schema: Optional[dict] = None,
        think: Optional[bool] = None,
        tool_choice: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> LLMResponse:
        """Send a chat request to OpenRouter. Same interface as other clients."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if frequency_penalty is not None:
            kwargs["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                if tool_choice in ("auto", "required", "none"):
                    kwargs["tool_choice"] = tool_choice
                else:
                    kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }

        if json_schema or json_mode:
            kwargs["response_format"] = {"type": "json_object"}

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

            _write_debug_log(
                f"OPENROUTER_RESPONSE (api={_elapsed:.1f}s)",
                raw_content or "(empty)",
            )

            logger.info(
                "openrouter_response",
                content_length=len(raw_content) if raw_content else 0,
                content_preview=(raw_content or "")[:200],
                model=response.model if hasattr(response, 'model') else self.model,
            )

            # Extract tool calls
            tool_calls = []
            if message and message.tool_calls:
                for tc in message.tool_calls:
                    args = tc.function.arguments
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
                content=raw_content or "",
                tool_calls=tool_calls,
                model=getattr(response, 'model', self.model) or self.model,
                finish_reason=choice.finish_reason if choice else "",
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            )

        except asyncio.TimeoutError:
            logger.error("openrouter_chat_timeout", timeout=self.timeout, model=self.model)
            raise TimeoutError(f"OpenRouter API call timed out after {self.timeout}s.")
        except Exception as e:
            logger.error("openrouter_chat_error", error=str(e))
            raise

    async def is_available(self) -> bool:
        return True

    def close(self):
        pass


# Global client instances
_client = None
_narrator_client = None


def _reset_clients():
    """Clear cached client instances so they recreate from the active profile."""
    global _client, _narrator_client
    _client = None
    _narrator_client = None


def _create_client(provider: str, model: str, fallback_to_ollama: bool = False, context_size: int = 0):
    """Create an LLM client for a given provider and model."""
    settings = get_settings()

    if provider == "groq":
        client = GroqClient(model=model)
        client._fallback_enabled = fallback_to_ollama
        return client
    elif provider == "anthropic":
        return AnthropicClient(model=model)
    elif provider == "openrouter":
        return OpenRouterClient(model=model)
    else:  # ollama
        return OllamaClient(model=model, num_ctx=context_size or None)


def get_llm_client():
    """Get the brain/triage LLM client based on active profile."""
    global _client
    if _client is None:
        profile = get_profile()
        brain = profile.brain
        _client = _create_client(brain.provider, brain.model, brain.fallback_to_ollama, brain.context_size)
        logger.info(
            "brain_client_init",
            provider=brain.provider,
            model=brain.model,
            profile=profile.name,
        )
    return _client


def get_narrator_client():
    """Get the narrator LLM client based on active profile.

    Supports independent routing from the brain client — different
    provider, different model, different everything.
    """
    global _narrator_client
    if _narrator_client is None:
        profile = get_profile()
        narrator = profile.narrator

        # If narrator and brain use same provider+model, share the client
        if (narrator.provider == profile.brain.provider
                and narrator.model == profile.brain.model):
            _narrator_client = get_llm_client()
            logger.info(
                "narrator_client_init",
                provider=narrator.provider,
                model=narrator.model,
                shared_with="brain",
            )
        else:
            _narrator_client = _create_client(narrator.provider, narrator.model, context_size=narrator.context_size)
            logger.info(
                "narrator_client_init",
                provider=narrator.provider,
                model=narrator.model,
                profile=profile.name,
            )
    return _narrator_client


# Alias for backwards compatibility
get_ollama_client = get_llm_client
