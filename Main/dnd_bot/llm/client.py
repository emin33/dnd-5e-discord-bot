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

    # Prompt caching telemetry (audit #21). Providers that support caching
    # populate one or both. Anthropic distinguishes read (cheap) from
    # creation (more expensive than non-cached the first time). DeepSeek
    # reports cache_hit_tokens as cache_read_tokens; misses are just regular
    # prompt_tokens. OpenAI's auto-cache also fills cache_read_tokens.
    cache_read_tokens: int = 0   # input tokens served from cache (savings)
    cache_write_tokens: int = 0  # input tokens written to cache (this turn)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def cache_hit_ratio(self) -> float:
        """Fraction of prompt tokens served from cache. 0.0 if no caching."""
        total = self.prompt_tokens + self.cache_read_tokens
        return self.cache_read_tokens / total if total > 0 else 0.0


@dataclass
class ToolCall:
    """A tool call from the LLM."""

    name: str
    arguments: dict[str, Any]
    id: str = ""


def _clean_llm_content(content: str) -> str:
    """Strip leaked thinking tags and CJK characters from LLM output.

    Actively needed: Groq only offers Qwen 3 32B, which leaks <think>
    tags even with think=False. Also catches Qwen fine-tunes on Ollama
    that exhibit the same bug. Qwen 3.5+ on Ollama should be clean but
    the check is near-zero cost.
    """
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
    """Async wrapper around Ollama client.

    Tool-calling requests are routed through Ollama's OpenAI-compatible
    endpoint (``/v1/chat/completions``) instead of the native ``/api/chat``
    path. The native path has a known bug (ollama#14601) where tool
    definitions are rendered as Go-struct strings rather than valid JSON,
    causing models to silently ignore the tools. The compat endpoint uses
    OpenAI's serializer and works correctly. Non-tool requests continue
    to use the native client (json_schema / json_mode / think).
    """

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
        # Lazy-init OpenAI-compat client for tool-bearing requests.
        # Uses Ollama's /v1/chat/completions endpoint to bypass the
        # broken native tool serializer.
        self._openai_compat_client = None

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
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
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
            top_p: Nucleus sampling threshold (Qwen3 recommends 0.8).
            top_k: Top-K sampling cutoff (Qwen3 recommends 20).
            min_p: Min-probability cutoff (Qwen3 recommends 0).

        Returns:
            LLMResponse with content and any tool calls
        """
        # Route tool-bearing requests through Ollama's OpenAI-compat
        # endpoint to avoid the broken native tool serializer
        # (ollama#14601). Non-tool calls keep the native path because it
        # supports json_schema / json_mode / think correctly.
        if tools:
            return await self._chat_via_openai_compat(
                messages=messages,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
            )

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

    @staticmethod
    def _build_hermes_tools_block(tools: list[dict]) -> str:
        """Build a Hermes-format <tools> system-prompt block.

        Empirically required for Qwen3.6 on ollama: while the research
        suggested ollama's Qwen3 chat template auto-injects from the
        ``tools=`` arg, in practice on this build (ollama 0.x +
        qwen3.6:latest) the auto-injection is missing/broken and the
        model emits zero tool calls without an explicit Hermes block.
        Keep this injection — pair with ``num_ctx`` override so context
        bloat doesn't truncate prose.
        """
        tools_json = "\n".join(json.dumps(t) for t in tools)
        return (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within "
            "<tools></tools> XML tags:\n"
            "<tools>\n"
            f"{tools_json}\n"
            "</tools>\n\n"
            "For each function call, return a json object with function "
            "name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": "<function-name>", "arguments": <args-json-object>}\n'
            "</tool_call>"
        )

    # Models with Ollama's native tool RENDERER/PARSER directive
    # (verify with `ollama show <model> --modelfile`). For these, Ollama
    # wraps `tools=[...]` with the model-family's hard token boundaries
    # (Gemma 4: `<|tool|>`, `<|tool_call|>`, `<|tool_result|>`). Injecting
    # a Hermes XML `<tools>` block alongside that is at best dead context
    # bloat, at worst confuses the model into emitting Hermes XML instead
    # of the native special tokens — defeating determinism (audit #95).
    _NATIVE_TOOL_MODEL_PREFIXES: tuple[str, ...] = ("gemma4",)

    @classmethod
    def _build_compat_messages(
        cls,
        model: str,
        messages: list[dict],
        tools: list[dict],
    ) -> tuple[list[dict], bool]:
        """Build the message list for the OpenAI-compat chat path.

        Returns ``(messages, uses_native_tool_renderer)``. For models in
        the native-tool allowlist, returns the messages unchanged so
        Ollama's renderer can do its job. For everything else (e.g.
        Qwen3), prepends a Hermes-format `<tools>` system block before
        the first user message so the model knows what to call.

        Extracted from ``_chat_via_openai_compat`` for testability.
        """
        uses_native = any(model.startswith(p) for p in cls._NATIVE_TOOL_MODEL_PREFIXES)
        if uses_native:
            return list(messages), True
        hermes_block = cls._build_hermes_tools_block(tools)
        injected = list(messages)
        first_user_idx = next(
            (i for i, m in enumerate(injected) if m.get("role") == "user"),
            len(injected),
        )
        injected.insert(first_user_idx, {"role": "system", "content": hermes_block})
        return injected, False

    @staticmethod
    def _parse_hermes_tool_calls(content: str) -> tuple[str, list[dict]]:
        """Extract <tool_call>...</tool_call> blocks from model content.

        Returns (clean_content, tool_calls). When ollama's compat layer
        already parses tool calls into the structured ``tool_calls``
        field, this returns an empty list and the original content. We
        keep this as a fallback in case the model emits inline blocks
        without the SDK auto-parsing them.
        """
        if "<tool_call>" not in content:
            return content, []

        tool_calls: list[dict] = []
        pattern = re.compile(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            re.DOTALL,
        )
        for m in pattern.finditer(content):
            block = m.group(1).strip()
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError:
                logger.warning(
                    "hermes_tool_call_parse_failed",
                    block_preview=block[:120],
                )
                continue
            name = parsed.get("name", "")
            args = parsed.get("arguments", {}) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name:
                tool_calls.append({"name": name, "arguments": args})

        clean = pattern.sub("", content).strip()
        return clean, tool_calls

    def _get_openai_compat_client(self):
        """Lazily build an AsyncOpenAI client pointed at Ollama's
        OpenAI-compatible endpoint. Used for tool-bearing requests because
        the native /api/chat path mangles tool definitions (ollama#14601).
        """
        if self._openai_compat_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package required for Ollama tool calling. "
                    "Install with: pip install openai"
                )
            # Ollama's OpenAI-compat endpoint lives at <host>/v1.
            # The host setting is e.g. "http://localhost:11434".
            base_url = self.host.rstrip("/") + "/v1"
            self._openai_compat_client = AsyncOpenAI(
                api_key="ollama",  # Required by the SDK; ignored by ollama
                base_url=base_url,
            )
        return self._openai_compat_client

    async def _chat_via_openai_compat(
        self,
        messages: list[dict],
        temperature: float,
        tools: list[dict],
        tool_choice: Optional[str],
        max_tokens: Optional[int],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
    ) -> LLMResponse:
        """Tool-bearing chat path via Ollama's OpenAI-compat endpoint.

        Per agent research (ollama#10698, #14958, #11759):

        - **Pass ``tools`` natively**: ollama's Qwen3 chat template injects
          a Hermes-format <tools> block from the ``tools=`` arg correctly.
          Manually injecting our own duplicate bloats context AND confuses
          the model into picking the in-prompt format over the structured
          path, costing prose generation.
        - **Set ``num_ctx`` via ``extra_body``**: the compat endpoint
          silently defaults ctx to 2048 unless overridden. With our 9-tool
          schema + roster + system prompt, that truncates the prompt and
          drops the model's prose, leaving only the tool call. Threading
          our profile's ``context_size`` through here is critical.
        - **Omit ``tool_choice`` unless explicitly set**: ollama treats
          absent and "auto" identically; ``"required"`` makes the prose
          drop worse.
        - **Sampling params** (frequency/presence_penalty) are honored on
          this path; the native /api/chat path silently drops them for
          Qwen3.x.
        """
        client = self._get_openai_compat_client()

        injected_messages, uses_native_tool_renderer = self._build_compat_messages(
            self.model, messages, tools,
        )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": injected_messages,
            "temperature": temperature,
            "tools": tools,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if frequency_penalty is not None:
            kwargs["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty
        if top_p is not None:
            kwargs["top_p"] = top_p
        # Only forward tool_choice when explicitly requested; default
        # behavior matches OpenAI/ollama "auto" without the param.
        if tool_choice and tool_choice != "auto":
            if tool_choice in ("required", "none"):
                kwargs["tool_choice"] = tool_choice
            else:
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice},
                }

        # Thread Ollama's num_ctx through extra_body — the OpenAI SDK
        # doesn't have a native field for this. Without it, ollama
        # silently truncates at 2048 tokens, which drops prose alongside
        # tool calls. top_k and min_p are also Ollama-specific and live
        # under options.
        ctx_size = self.num_ctx or 32768
        extra_body = {
            "options": {
                "num_ctx": ctx_size,
            }
        }
        if max_tokens:
            extra_body["options"]["num_predict"] = max_tokens
        if top_k is not None:
            extra_body["options"]["top_k"] = top_k
        if min_p is not None:
            extra_body["options"]["min_p"] = min_p
        kwargs["extra_body"] = extra_body

        try:
            import time as _time
            _t0 = _time.monotonic()

            response = await asyncio.wait_for(
                client.chat.completions.create(**kwargs),
                timeout=self.timeout,
            )

            _elapsed = _time.monotonic() - _t0

            choice = response.choices[0] if response.choices else None
            message = choice.message if choice else None
            raw_content = (message.content if message else "") or ""

            # Source 1: SDK-parsed tool_calls (when ollama's Qwen3
            # template auto-parses inline <tool_call> XML into structured
            # output, this populates).
            sdk_tool_calls: list[dict] = []
            if message and getattr(message, "tool_calls", None):
                for tc in message.tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    sdk_tool_calls.append({
                        "name": tc.function.name,
                        "arguments": args,
                    })

            # Source 2: parse <tool_call> blocks from content. Triggers
            # when the SDK's auto-parser missed inline emissions.
            stripped_content, hermes_tool_calls = self._parse_hermes_tool_calls(raw_content)

            tool_calls = sdk_tool_calls or hermes_tool_calls
            content_for_cleanup = stripped_content if hermes_tool_calls else raw_content
            content = _clean_llm_content(content_for_cleanup)

            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0

            logger.info(
                "ollama_compat_response",
                model=self.model,
                elapsed_s=round(_elapsed, 2),
                content_length=len(content),
                tool_call_count=len(tool_calls),
                sdk_tool_calls=len(sdk_tool_calls),
                hermes_tool_calls=len(hermes_tool_calls),
                # Audit #95 verification: for gemma4 models we expect
                # sdk_tool_calls > 0 and hermes_tool_calls == 0 (i.e. the
                # native renderer is doing its job). If hermes_tool_calls
                # ever fires for a native-tool model, something is off.
                tool_path=("native" if uses_native_tool_renderer else "hermes"),
                prompt_tokens=prompt_tokens,
                num_ctx=ctx_size,
                tools_offered=[t["function"]["name"] for t in tools],
                content_preview=content[:200],
            )
            _write_debug_log(
                f"OLLAMA_COMPAT_RESPONSE (api={_elapsed:.1f}s, tools={len(tool_calls)})",
                content or "(empty)",
            )

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                model=getattr(response, "model", self.model) or self.model,
                finish_reason=choice.finish_reason if choice else "",
                prompt_tokens=prompt_tokens,
                completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            )

        except asyncio.TimeoutError:
            logger.error(
                "ollama_compat_timeout",
                timeout=self.timeout,
                model=self.model,
            )
            raise TimeoutError(
                f"Ollama OpenAI-compat call timed out after {self.timeout}s."
            )
        except Exception as e:
            logger.error(
                "ollama_compat_chat_error",
                error=str(e),
                model=self.model,
            )
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

        # Enforce JSON output when json_schema or json_mode is requested.
        # Claude doesn't have native json_mode, so we inject a system hint.
        if json_schema or json_mode:
            system_parts.append(
                "IMPORTANT: Respond with ONLY a valid JSON object. "
                "No prose, no markdown fences, no explanation."
            )

        if system_parts:
            # Audit #21: use Anthropic's ephemeral prompt caching. The first
            # system_part is typically the long static narrator template
            # (~2K tokens) — putting cache_control on it caches everything
            # up to and including that block. Subsequent system_parts
            # (per-turn instructions) stay outside the cache breakpoint so
            # the cache key stays stable across turns.
            #
            # Anthropic's minimum cacheable size is 1024 tokens (~4000 chars).
            # If the first part is shorter than that, fall back to joining as
            # one string (Anthropic ignores cache_control on too-short blocks).
            MIN_CACHEABLE_CHARS = 4000
            if len(system_parts[0]) >= MIN_CACHEABLE_CHARS:
                blocks: list[dict] = []
                for i, part in enumerate(system_parts):
                    block: dict = {"type": "text", "text": part}
                    if i == 0:
                        block["cache_control"] = {"type": "ephemeral"}
                    blocks.append(block)
                kwargs["system"] = blocks
            else:
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
            cache_read = getattr(usage, "cache_read_input_tokens", 0) if usage else 0
            cache_write = getattr(usage, "cache_creation_input_tokens", 0) if usage else 0
            if cache_read or cache_write:
                logger.info(
                    "anthropic_cache_telemetry",
                    cache_read_tokens=cache_read,
                    cache_write_tokens=cache_write,
                    uncached_input_tokens=usage.input_tokens if usage else 0,
                )
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                model=response.model or self.model,
                finish_reason=response.stop_reason or "",
                prompt_tokens=usage.input_tokens if usage else 0,
                completion_tokens=usage.output_tokens if usage else 0,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
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


class GeminiClient:
    """Async client for Google Gemini API (narrator-only)."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package required. Install with: pip install google-generativeai"
            )

        settings = get_settings()
        self.model = model or "gemini-2.5-flash"
        self.timeout = settings.llm_timeout
        key = api_key or settings.gemini_api_key
        if not key:
            # Fallback: read directly from .env (system env may shadow with empty string)
            from dotenv import dotenv_values
            env = dotenv_values()
            key = env.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError("GEMINI_API_KEY must be set when using Gemini provider")
        self._genai = genai
        genai.configure(api_key=key)

    def _convert_tools(self, tools: list[dict]) -> Optional[list]:
        """Convert OpenAI-format tools to Gemini function_declarations."""
        if not tools:
            return None
        protos = self._genai.protos
        declarations = []
        for t in tools:
            func = t.get("function", {})
            params = func.get("parameters")
            declarations.append(
                protos.FunctionDeclaration(
                    name=func.get("name", ""),
                    description=func.get("description", ""),
                    parameters=self._convert_schema(params) if params else None,
                )
            )
        return [protos.Tool(function_declarations=declarations)]

    def _convert_schema(self, schema: dict):
        """Convert JSON Schema dict to Gemini Schema proto (recursive)."""
        protos = self._genai.protos
        type_map = {
            "object": protos.Type.OBJECT,
            "string": protos.Type.STRING,
            "number": protos.Type.NUMBER,
            "integer": protos.Type.INTEGER,
            "boolean": protos.Type.BOOLEAN,
            "array": protos.Type.ARRAY,
        }
        schema_type = type_map.get(schema.get("type", ""), protos.Type.OBJECT)
        kwargs: dict[str, Any] = {"type": schema_type}
        if "description" in schema:
            kwargs["description"] = schema["description"]
        if "enum" in schema:
            kwargs["enum"] = schema["enum"]
        if "properties" in schema:
            kwargs["properties"] = {
                k: self._convert_schema(v)
                for k, v in schema["properties"].items()
            }
        if "required" in schema:
            kwargs["required"] = schema["required"]
        if "items" in schema:
            kwargs["items"] = self._convert_schema(schema["items"])
        return protos.Schema(**kwargs)

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
        """Send a chat request to Gemini API. Same interface as other clients.

        Note: Gemini does not support frequency/presence penalties. These
        params are accepted for interface compatibility but ignored.
        """
        # Separate system instructions from chat messages
        system_parts = []
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                chat_messages.append(msg)

        # Convert to Gemini format: "assistant" → "model"
        contents = []
        for msg in chat_messages:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        # Ensure first message is from user (Gemini requires it)
        if not contents or contents[0]["role"] != "user":
            contents.insert(0, {"role": "user", "parts": [{"text": "Begin."}]})

        # Merge consecutive same-role messages
        merged = []
        for msg in contents:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["parts"].extend(msg["parts"])
            else:
                merged.append({"role": msg["role"], "parts": list(msg["parts"])})

        # Build model kwargs — system_instruction and tools are set per-model
        model_kwargs: dict[str, Any] = {"model_name": self.model}
        if system_parts:
            model_kwargs["system_instruction"] = "\n\n".join(system_parts)

        gemini_tools = self._convert_tools(tools)
        if gemini_tools:
            model_kwargs["tools"] = gemini_tools

        model = self._genai.GenerativeModel(**model_kwargs)

        # Generation config
        gen_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens or 2000,
        }

        # Enforce JSON output when json_schema or json_mode is requested
        if json_schema or json_mode:
            gen_kwargs["response_mime_type"] = "application/json"

        gen_config = self._genai.GenerationConfig(**gen_kwargs)

        call_kwargs: dict[str, Any] = {
            "contents": merged,
            "generation_config": gen_config,
        }

        # Tool choice → Gemini function_calling_config
        if tools and tool_choice:
            mode_map = {"auto": "AUTO", "required": "ANY", "none": "NONE"}
            mode = mode_map.get(tool_choice, "AUTO")
            call_kwargs["tool_config"] = {
                "function_calling_config": {"mode": mode}
            }

        try:
            import time as _time
            _t0 = _time.monotonic()

            response = await asyncio.wait_for(
                model.generate_content_async(**call_kwargs),
                timeout=self.timeout,
            )

            _elapsed = _time.monotonic() - _t0

            content = ""
            tool_calls = []

            if not response.candidates:
                logger.warning(
                    "gemini_no_candidates",
                    model=self.model,
                    prompt_feedback=str(getattr(response, "prompt_feedback", "")),
                )
                return LLMResponse(content="", model=self.model, finish_reason="SAFETY")

            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    content += part.text
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append({
                        "name": fc.name,
                        "arguments": dict(fc.args) if fc.args else {},
                    })

            _write_debug_log(
                f"GEMINI_RESPONSE (api={_elapsed:.1f}s)",
                content or "(empty)",
            )

            logger.info(
                "gemini_response",
                content_length=len(content),
                content_preview=content[:200],
                tool_calls=len(tool_calls),
                model=self.model,
            )

            usage = response.usage_metadata
            finish = ""
            if response.candidates:
                fr = response.candidates[0].finish_reason
                finish = fr.name if hasattr(fr, "name") else str(fr)

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                model=self.model,
                finish_reason=finish,
                prompt_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
                completion_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            )

        except asyncio.TimeoutError:
            logger.error("gemini_chat_timeout", timeout=self.timeout, model=self.model)
            raise TimeoutError(f"Gemini API call timed out after {self.timeout}s.")
        except Exception as e:
            logger.error("gemini_chat_error", error=str(e))
            raise

    async def is_available(self) -> bool:
        """Check if Gemini API is reachable."""
        return True  # If constructor didn't raise, key is set

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


class DeepSeekClient:
    """Async client for DeepSeek API (OpenAI-compatible).

    Supports DeepSeek V4 Pro / V4 Flash. Direct integration enables their
    automatic prefix-based prompt caching (cached input tokens reported in
    usage as prompt_cache_hit_tokens vs prompt_cache_miss_tokens).

    Thinking mode (default ENABLED on DeepSeek) is disabled by default here
    because we want narrator prose without CoT leakage AND because thinking
    mode silently disables temperature, top_p, frequency_penalty, and
    presence_penalty — losing the variety controls the narrator depends on.
    Pass think=True explicitly to re-enable for reasoning-heavy tasks.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package required for DeepSeek. Install with: pip install openai"
            )

        settings = get_settings()
        self.model = model or "deepseek-v4-flash"
        self.timeout = settings.llm_timeout
        key = api_key or settings.deepseek_api_key
        if not key:
            raise ValueError("DEEPSEEK_API_KEY must be set when using DeepSeek provider")
        self._client = AsyncOpenAI(
            api_key=key,
            base_url="https://api.deepseek.com/v1",
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
        """Send a chat request to DeepSeek. Same interface as other clients.

        Note: DeepSeek's thinking mode silently ignores temperature, top_p,
        frequency_penalty, and presence_penalty when enabled. We default to
        thinking=disabled to preserve those controls; pass think=True to opt
        back in (and accept those params being dropped).
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        # Thinking mode: default OFF to preserve sampling params and avoid
        # CoT leakage in narrator prose. think=True opts into thinking.
        thinking_enabled = think is True
        extra_body = {
            "thinking": {"type": "enabled" if thinking_enabled else "disabled"},
        }

        # Sampling params are silently ignored when thinking is on — only
        # forward them when thinking is off so behavior is predictable.
        if not thinking_enabled:
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

        kwargs["extra_body"] = extra_body

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

            # Capture reasoning_content if thinking was enabled (separate field)
            reasoning_content = ""
            if message and hasattr(message, "reasoning_content"):
                reasoning_content = getattr(message, "reasoning_content", "") or ""

            _write_debug_log(
                f"DEEPSEEK_RESPONSE (api={_elapsed:.1f}s)",
                raw_content or "(empty)",
            )

            # Cache hit/miss telemetry — DeepSeek-specific in usage object
            usage = response.usage
            cache_hit = getattr(usage, "prompt_cache_hit_tokens", 0) if usage else 0
            cache_miss = getattr(usage, "prompt_cache_miss_tokens", 0) if usage else 0

            logger.info(
                "deepseek_response",
                content_length=len(raw_content) if raw_content else 0,
                content_preview=(raw_content or "")[:200],
                model=getattr(response, "model", self.model),
                thinking=thinking_enabled,
                reasoning_length=len(reasoning_content),
                cache_hit_tokens=cache_hit,
                cache_miss_tokens=cache_miss,
                cache_hit_ratio=(cache_hit / (cache_hit + cache_miss)) if (cache_hit + cache_miss) else 0,
            )

            # Extract tool calls (standard OpenAI shape)
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

            return LLMResponse(
                content=raw_content or "",
                tool_calls=tool_calls,
                model=getattr(response, "model", self.model) or self.model,
                finish_reason=choice.finish_reason if choice else "",
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                # DeepSeek auto-caches prefixes; surface hits as cache_read_tokens
                # so observability matches Anthropic. There's no explicit "write"
                # cost — misses are just regular input billing.
                cache_read_tokens=cache_hit,
            )

        except asyncio.TimeoutError:
            logger.error("deepseek_chat_timeout", timeout=self.timeout, model=self.model)
            raise TimeoutError(f"DeepSeek API call timed out after {self.timeout}s.")
        except Exception as e:
            logger.error("deepseek_chat_error", error=str(e))
            raise

    async def is_available(self) -> bool:
        return True

    def close(self):
        pass


# Global client instances
_client = None
_narrator_client = None  # Backwards-compatible standard-tier narrator
_narrator_clients_by_tier: dict = {}  # tier -> client (with fallback applied)
_clients_by_provider_model: dict = {}  # (provider, model) -> shared client instance


def _reset_clients():
    """Clear cached client instances so they recreate from the active profile."""
    global _client, _narrator_client, _narrator_clients_by_tier, _clients_by_provider_model
    _client = None
    _narrator_client = None
    _narrator_clients_by_tier = {}
    _clients_by_provider_model = {}


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
    elif provider == "deepseek":
        return DeepSeekClient(model=model)
    elif provider == "gemini":
        return GeminiClient(model=model)
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


def _resolve_narrator_tier_config(profile, tier: str):
    """Resolve a tier name to its ProviderConfig, applying fallback rules.

    Fallback chain:
        opening  -> narrator_opening or premium fallback
        premium  -> narrator_premium or standard fallback
        standard -> narrator (always present)

    Returns ``(resolved_tier_name, ProviderConfig)``. The resolved tier name
    reflects what was actually picked after fallback (useful for logging).
    """
    if tier == "opening":
        if profile.narrator_opening is not None:
            return "opening", profile.narrator_opening
        if profile.narrator_premium is not None:
            return "premium", profile.narrator_premium
        return "standard", profile.narrator

    if tier == "premium":
        if profile.narrator_premium is not None:
            return "premium", profile.narrator_premium
        return "standard", profile.narrator

    return "standard", profile.narrator


def get_narrator_client_for(tier: str = "standard"):
    """Get the narrator LLM client for a given tier.

    Tiers (with silent fallback to ``standard`` when unconfigured):
    - ``"opening"``: session opener (falls back to ``"premium"`` then ``"standard"``)
    - ``"premium"``: high-significance turns (falls back to ``"standard"``)
    - ``"standard"``: default daily driver

    Clients are cached by ``(provider, model)`` so two tiers pointing at the
    same model share an instance. The brain client is also reused if its
    ``(provider, model)`` matches a narrator tier.
    """
    global _narrator_clients_by_tier, _clients_by_provider_model

    if tier in _narrator_clients_by_tier:
        return _narrator_clients_by_tier[tier]

    profile = get_profile()
    resolved_tier, cfg = _resolve_narrator_tier_config(profile, tier)
    key = (cfg.provider, cfg.model)

    # Try (provider, model) shared cache first — covers both cross-tier sharing
    # and brain↔narrator sharing.
    client = _clients_by_provider_model.get(key)

    if client is None:
        # Also check the brain client; reuse if (provider, model) matches.
        brain = profile.brain
        if (cfg.provider == brain.provider and cfg.model == brain.model
                and _client is not None):
            client = _client
            # Bump context window if narrator needs more headroom
            if hasattr(client, "num_ctx") and cfg.context_size:
                client.num_ctx = max(client.num_ctx or 0, cfg.context_size)
        else:
            client = _create_client(
                cfg.provider, cfg.model, context_size=cfg.context_size,
            )

        _clients_by_provider_model[key] = client
        logger.info(
            "narrator_client_init",
            provider=cfg.provider,
            model=cfg.model,
            requested_tier=tier,
            resolved_tier=resolved_tier,
            profile=profile.name,
            cache_hit=False,
        )
    else:
        logger.debug(
            "narrator_client_cache_hit",
            provider=cfg.provider,
            model=cfg.model,
            requested_tier=tier,
            resolved_tier=resolved_tier,
        )

    _narrator_clients_by_tier[tier] = client
    return client


def get_narrator_client():
    """Backwards-compatible accessor: returns the standard-tier narrator client.

    Prefer ``get_narrator_client_for(tier=...)`` for new call sites that need
    to route by narrative significance.
    """
    global _narrator_client
    if _narrator_client is None:
        _narrator_client = get_narrator_client_for("standard")
    return _narrator_client


# Alias for backwards compatibility
get_ollama_client = get_llm_client
