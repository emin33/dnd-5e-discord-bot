# Testing Non-Deterministic LLM Pipelines Deterministically: Battle-Tested Patterns for a D&D Game-Master Orchestrator

## TL;DR
- **Your highest-value first move is a seam, not a tool**: inject a single `Brain`/`LLMClient` protocol into the orchestrator and swap in a scripted fake (the pattern behind LangChain's `FakeListChatModel`, Pydantic AI's `TestModel`/`FunctionModel`, AutoGen's `ReplayChatCompletionClient`, and litellm's `mock_response`). Then assert on the **sequence of tool calls and state transitions**, not on the generated prose. This alone gets you real integration tests on the main entrypoint in a day or two.
- **Record-replay (VCR cassettes) is a strong second layer** for catching prompt/serialization regressions against real provider wire formats, but it is brittle with streaming SSE and randomized request bodies, so use it sparingly and behind a stable request matcher.
- **Snapshot the structured trajectory (tool-call sequence + final state), never the raw narration**: use `inline-snapshot` or `syrupy` with matchers that scrub timestamps/IDs. Snapshotting the full natural-language transcript is too brittle and should be avoided.

## Key Findings

1. **Every mature framework ships a fake model that plugs into one narrow seam.** The universal pattern is: the framework defines a model *interface* (ABC/protocol), and the fake implements that interface returning scripted responses. Examples with real APIs: LangChain `FakeListChatModel` (cycles a `responses: list[str]`), `GenericFakeChatModel` (iterates `AIMessage`s, supports streaming/tool calls), Pydantic AI `TestModel` (auto-generates schema-valid output, calls all tools) and `FunctionModel` (you write `f(messages, info) -> ModelResponse`), litellm `completion(..., mock_response=...)`, AutoGen `ReplayChatCompletionClient(chat_completions=[...])`, DSPy `DummyLM`, and the OpenAI Agents SDK's test-suite `FakeModel`. The lesson for your orchestrator: **define one `Brain` protocol and write one fake** rather than mocking provider SDKs.

2. **The decisive technique for orchestrator correctness is asserting on effects/tool-dispatch/state, not text.** LangGraph's own docs test by invoking the compiled graph and asserting on resulting state dict and which node ran; tools like FastEval expose `result.nodes_ran` for trajectory assertions; the "functional core / imperative shell" and event/command patterns make the agent *return a list of effects as data* that you assert against. This is exactly the refactor your effect system already invites.

3. **Record-replay works but has sharp edges with LLMs.** `vcrpy` + `pytest-recording` record HTTP into YAML cassettes and replay deterministically; you must filter the `authorization` header, and you will hit a known `vcrpy` failure on streamed/`chunked` httpx responses. Request bodies containing randomized ordering or timestamps break the default matcher, so you need a custom matcher.

4. **pytest-asyncio gotchas are real and specific.** Use `asyncio_mode = "auto"`, `AsyncMock` for awaitable clients, `freeze_time(..., real_asyncio=True)` to avoid breaking `asyncio.sleep`, and patch the import path *where the name is looked up*, not where it's defined. Mocking async context managers (`__aenter__`/`__aexit__`) is a frequent source of `'coroutine' object does not support the asynchronous context manager protocol` errors.

5. **Provider abstraction layers are designed for swap-in.** The OpenAI and Anthropic Python SDKs both accept an injected `http_client` (so you can pass an httpx client backed by a mock transport); litellm normalizes everything to the OpenAI shape and has a built-in `mock_response`; Pydantic AI separates `Model`/`Provider`/`Profile` so a fake `Model` swaps cleanly. Your multi-provider layer should hide behind one interface so the fake replaces the *whole* layer.

## Details

### 1. Fake / stub LLM client patterns (with real APIs)

**LangChain / LangGraph** (`langchain_core.language_models.fake_chat_models`):
- `FakeListChatModel(responses=[...])` — a `SimpleChatModel` subclass that **cycles through `responses` in order** (internal counter `i`, incremented every call; wraps back to start at the end), with an optional `sleep` to simulate latency and `error_on_chunk_number` to inject a streaming error. Verbatim from source: `"""List of responses to **cycle** through in order."""`.
- `GenericFakeChatModel(messages=iter([...]))` — takes an `Iterator[AIMessage | str]`; "usable in both sync and async tests", "invokes `on_llm_new_token`", and "includes logic to break messages into message chunk to facilitate testing of streaming." Because it yields `AIMessage` objects you can attach `tool_calls` to script function calls. (Note the docs caveat: in some versions streaming "is not implemented yet" for the generic variant — verify against your pinned version.)
- `FakeMessagesListChatModel`, `ParrotFakeChatModel` (echoes input back), and `FakeListLLM`/`FakeStreamingListLLM` for the older completion interface.
- These plug into the `BaseChatModel` Runnable seam, so anywhere your code accepts a chat model, the fake drops in.

**Pydantic AI** (`pydantic_ai.models.test` / `pydantic_ai.models.function`) — the gold standard for your use case:
- `TestModel()` — "There's no ML or AI in TestModel, it's just plain old procedural Python code that tries to generate data that satisfies the JSON schema of a tool." By default it **calls all tools in the agent**, then returns either plain text or a structured response. Inspect what the agent exposed via `m.last_model_request_parameters.function_tools`.
- `FunctionModel(my_func)` where `my_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse` — you get full control: branch on message count/content, return `ToolCallPart(...)` on the first call and `TextPart(...)` on the next. This is the cleanest way to script multi-step tool sequences deterministically.
- Injection seam: `with my_agent.override(model=FunctionModel(...)):` and the global safety valve `pydantic_ai.models.ALLOW_MODEL_REQUESTS = False` which **blocks any real network request** so an un-mocked agent fails loudly in CI.
- `capture_run_messages()` context manager records the exact `ModelMessage` list exchanged, so you can assert on what the orchestrator sent.

**litellm** (`completion(..., mock_response="...")`): returns a normal `ModelResponse`/`ChatCompletion` object without calling the API, and **works for streaming as well** (yields chunk deltas). Per BerriAI's litellm GitHub README it is a "Python SDK, Proxy Server (AI Gateway) to call 100+ LLM APIs in OpenAI (or native) format," and the docs add "Every response follows the OpenAI Chat Completions format, regardless of provider" — so a single mock covers Ollama/Anthropic/Gemini/DeepSeek if you route them through litellm.

**AutoGen / AG2** (`from autogen_ext.models.replay import ReplayChatCompletionClient`): "A mock chat completion client that replays predefined responses." Constructor takes `chat_completions=[...]` — a list of `str` (wrapped into a `CreateResult` with text content) or `CreateResult` objects (returned directly). Replays **in order**, one per `create()`/`create_stream()` call; raises when exhausted; supports `.reset()` to rewind and exposes call inspection (e.g. `.create_calls`) and native token-usage tracking. To script a tool call, pass a `CreateResult` whose `content` is a `list[FunctionCall]` (`FunctionCall(id=..., name=..., arguments=<json-string>)`) with `finish_reason="function_calls"`. *(Exact attribute spellings and exhaustion exception type should be verified against the source-rendered module page.)*

**OpenAI Agents SDK**: ships a `FakeModel` in `tests/fake_model.py` implementing the `Model` ABC (`get_response(...) -> ModelResponse` and `stream_response(...) -> AsyncIterator[...]`, verbatim from `src/agents/models/interface.py`). Tests script a turn by queuing scripted Responses-format output items (text-message items and function tool-call items, e.g. `ResponseFunctionToolCall(name=..., arguments=<json>, ...)`). Inject via `Agent(model=fake_model)` (the agent validates "Agent model must be a string, Model, or None") or per-run via `Runner.run(agent, ..., run_config=RunConfig(model=fake_model))` which "Allows setting a global LLM model to use, irrespective of what model each Agent has." *(The exact scripting method name should be confirmed against the GitHub file.)*

**DSPy** `DummyLM` — "Dummy language model for unit testing purposes... If a list of dictionaries is provided, the dummy model will return the next dictionary in the list for each request." **Haystack** uses plain `unittest.mock.Mock` on the generator's `pipeline`/`run` plus `set_all_seeds(0)` and a `SpyingTracer` in its own conftest. **Semantic Kernel / Microsoft Agent Framework** marks the OpenAI/Anthropic client as `INJECTABLE` and reconstructs with a provided client via dependency injection — a textbook DI seam.

*Code sketch — one fake brain for your orchestrator:*
```python
# brains.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class Brain(Protocol):
    async def decide(self, ctx: "TurnContext") -> "BrainResult": ...

# A BrainResult is pure data: either a tool call request or a final narration.
@dataclass(frozen=True)
class BrainResult:
    tool_calls: list["ToolCall"]      # structured, assertable
    narration: str | None             # the non-deterministic prose

# tests/fakes.py
class ScriptedBrain:
    """Like FakeListChatModel: cycles scripted results in order."""
    def __init__(self, results: list[BrainResult]):
        self._results = iter(results)
        self.calls: list[TurnContext] = []   # spy
    async def decide(self, ctx):
        self.calls.append(ctx)               # record for assertions
        return next(self._results)

class FunctionBrain:
    """Like Pydantic AI FunctionModel: branch on context."""
    def __init__(self, fn): self._fn = fn
    async def decide(self, ctx): return self._fn(ctx)
```

### 2. Record-replay (VCR-style)

`vcrpy` records HTTP interactions to YAML "cassettes" and replays them; `pytest-recording` is the maintained pytest plugin (`@pytest.mark.vcr`, `--record-mode=once|rewrite|none`). **Do not use `pytest-vcr` and `pytest-recording` together — they are incompatible.** Key practices for LLMs:
- **Redact keys**: `vcr_config` returning `{"filter_headers": ["authorization"]}`. Some providers pass the key as a query param (`filter_query_parameters`) or post-data param. Use `record_mode="none"` (or `block_network`) in CI so a missing cassette fails rather than calling the API.
- **Custom request matching**: the default matcher compares URI/method/body. LLM request bodies often vary (message ordering, injected timestamps, randomized seeds), so register a custom matcher (`pytest_recording_configure` → `vcr.register_matcher`) that matches on a stable subset.
- **Streaming pitfall (concrete and current)**: vcrpy GitHub Issue #895 — "AssertionError in Async Env Stemming from Response Having `_decoder` Attr," opened by JSv4 on Jan 5, 2025 and still open — fails in `vcr/stubs/httpx_stubs.py` line 139 when the OpenAI response carries `'transfer-encoding': 'chunked'`. This is corroborated by issue #597 ("async httpx streaming doesn't work," raising `IndexError` in `httpx_stubs.py:72`) and #927 (google-genai + aiohttp "infinite loop when streaming from VCR cassette"). The OpenAI/Anthropic SDKs stream over SSE, so cassettes for streamed turns are fragile — prefer recording non-streamed calls, or fake the stream at the seam instead.
- LangGraph itself uses VCR cassettes for its notebook/CI tests (record once, replay in CI). There is also a niche `vcr-langchain` wrapper, but it lags LangChain's API and is not broadly maintained.

*Verdict for a solo maintainer:* VCR is worth it as a **separate, opt-in test tier** (`@pytest.mark.integration`) that you re-record deliberately when prompts change. It is not the right tool for your main-entrypoint integration tests — the scripted-brain seam is.

### 3. Golden-transcript / snapshot testing

- **`inline-snapshot`** (sponsored by Pydantic): write `assert value == snapshot()`, run `pytest --inline-snapshot=fix`, and it writes the expected value *into your test source*. The OpenAI Agents SDK adopted it to snapshot normalized trace spans (`fetch_normalized_spans() == snapshot([...])`); a single key change fails the test and is fixed by re-running. Handles dynamic fields, and keeps expected data next to the test.
- **`syrupy`** stores snapshots in a `__snapshots__/*.ambr` directory; supports `JSONSnapshotExtension`, and crucially `matcher=path_type({...})` and `exclude=props("id", ...)` to **scrub non-deterministic fields** (timestamps, UUIDs) so snapshots stay stable. LangGraph uses syrupy to snapshot **graph structure/schemas** (not LLM text) — e.g. `test_pregel.ambr`.
- `pytest-snapshot` is a simpler file-based alternative; `jest`-style is the JS origin of the idea.

**When appropriate vs. brittle:** Snapshot the **structured trajectory** — the ordered list of `(tool_name, normalized_args)`, the final game-state diff, the sequence of effects — with timestamps/dice-rolls/IDs scrubbed via matchers. **Do not** snapshot the raw narrator prose: it changes every run and the snapshot becomes noise. Snapshotting a whole transcript is only appropriate when the text is itself produced by a fake (deterministic) model.

*Code sketch:*
```python
def test_combat_turn_trajectory(snapshot):
    orch = build_orchestrator(triage=ScriptedBrain([...]), narrator=ScriptedBrain([...]))
    result = await orch.run_turn(load_fixture_state())
    # snapshot the EFFECTS, scrubbing rolls/timestamps
    assert [e.normalized() for e in result.effects] == snapshot
    # OR with syrupy matchers:
    assert result.state_diff == snapshot(matcher=path_type({"updated_at": (datetime,)}))
```

### 4. Asserting on state transitions and tool-dispatch (the key one)

This is where your effect system is a gift. The mature pattern across event-sourced and "functional core / imperative shell" systems is: **the decision logic returns effects as a pure data structure; a separate interpreter executes them.** Then tests assert on the returned effect list with a trivial `assert`, with zero mocking.

- **Functional core / imperative shell** — attributable to Gary Bernhardt's talk "Boundaries" (SCNA 2012, in Ruby), which proposes shifting design focus "from the objects and how they interact, to the values they exchange," keeping the core pure and pushing side effects to the shell. Pure functions live in the center; side-effects are pushed to the outer rim. The Elm-style loop `state, effect = App.update(state, event)` is the same idea — the update function is pure and trivially testable.
- **The decision/evolution split**: Jérémie Chassaing's functional-core formulation writes the core around two pure functions — "A decision function that takes a Command and current state, and returns Events" and "An evolution function that takes current State and an Event and returns the new State." This maps directly onto your situation: a pure `decide_turn(state, brain_result) -> [Effect]` plus a pure `apply(state, effect) -> state`, with all I/O confined to a thin interpreter.
- **Command/effect pattern**: the core "never calls I/O — it returns data describing what's needed," and "the shell interprets them." Maximum testability; every decision point is an inspectable event.
- **Spy-on-the-dispatcher**: if you can't fully refactor yet, wrap the dispatcher with `mocker.spy(dispatcher, "dispatch")` (pytest-mock's `mocker.spy` "acts exactly like the original... except the spy also tracks calls, return values and exceptions") and assert on `spy.call_args_list == [call(CastSpell(...)), call(ApplyDamage(...))]`. Use `unittest.mock.call` objects and `unittest.mock.ANY` for fields you don't care about. Note: `assert_has_calls` does **not** fail on *extra* calls, so for exact sequences assert on `call_args_list` directly.
- **LangGraph state-graph testing** (from the official Test docs): compile with `MemorySaver()`, `invoke`/`ainvoke` a tiny state, and assert on the resulting state and which node ran. You can also invoke a single node (`compiled_graph.nodes["node1"].invoke(...)`) or seed mid-graph state with `update_state(...)` to test partial paths. FastEval's harness exposes `result.nodes_ran` so you can do `assert "classifier" in result.nodes_ran` — pure trajectory assertions with no token spend.

*Code sketch — the pattern to refactor toward:*
```python
# functional core: pure, no I/O, returns effects as data (the "decision function")
def decide_turn(state: GameState, brain_result: BrainResult) -> list[Effect]:
    effects = []
    for tc in brain_result.tool_calls:
        effects.append(dispatch_table[tc.name](state, tc.args))
    return effects   # e.g. [RollDice(d=20), ApplyDamage(target="goblin", amount=...)]

# test: trivial, deterministic, no mocks of I/O
def test_attack_emits_roll_then_damage():
    effects = decide_turn(fixture_state, BrainResult(
        tool_calls=[ToolCall("attack", {"target": "goblin"})], narration=None))
    assert [type(e).__name__ for e in effects] == ["RollDice", "ApplyDamage"]
    assert effects[1].target == "goblin"

# imperative shell (interpreter) is thin and tested once with an in-memory store
async def run_turn(state, brain):
    result = await brain.decide(make_ctx(state))
    effects = decide_turn(state, result)          # pure
    for eff in effects:
        state = await interpret(eff, state)        # I/O lives here only ("evolution")
    return state
```
Separate "did the agent take the right actions" (assert on `effects`) from "did the model say nice words" (a separate, opt-in eval against a real model, or skipped entirely in unit CI).

### 5. Seam design that makes the orchestrator testable

- **Dependency injection of the brains/providers**: pass `triage_brain` and `narrator_brain` into the orchestrator constructor (or a `Deps` dataclass à la Pydantic AI's `deps_type`). Never construct provider clients inside the orchestrator. This is the single most important refactor.
- **Protocol/ABC-based provider interface**: define `class Brain(Protocol)` (or `abc.ABC`) with one async method. Real implementations (`OllamaBrain`, `AnthropicBrain`, …) and the `ScriptedBrain`/`FunctionBrain` fakes all satisfy it. This mirrors litellm's "Every response follows the OpenAI Chat Completions format, regardless of provider," LangChain's `BaseChatModel`, and Pydantic AI's `Model`/`Provider` split.
- **Humble object / hexagonal ports-and-adapters**: keep the asyncio orchestration loop "humble" — it only wires ports (brain port, effect-interpreter port, state-store port) to adapters. All branching logic lives in the pure core.
- **Effect system as pure data interpreted separately** (see §4): make tool dispatch produce `Effect` values, then interpret them. This makes both tool dispatch and state mutation *observable* (assert on the effect list) and *injectable* (swap the interpreter for an in-memory recording one in tests).
- **Provider HTTP seam as a fallback**: if you can't inject at the brain level for some provider, both the OpenAI SDK (`OpenAI(http_client=DefaultHttpxClient(transport=...))`) and Anthropic SDK accept a custom httpx client/transport, so you can inject a mock transport (`respx` mocks httpx directly and is ideal here). Pydantic AI's `AnthropicProvider(http_client=custom_http_client)` does the same. But injecting at the `Brain` seam is cleaner than at the HTTP layer.

### 6. pytest + asyncio specifics

- **Plugin**: `pytest-asyncio` with `asyncio_mode = "auto"` in `pyproject.toml` auto-detects `async def test_*` (no per-test `@pytest.mark.asyncio` needed). The `anyio` pytest plugin (`pytestmark = pytest.mark.anyio`) is the alternative Pydantic AI uses, and is better if you want to run tests under both asyncio and trio.
- **Async mocks**: `AsyncMock` (stdlib since 3.8) for awaitable methods; `mocker.AsyncMock` via pytest-mock. Set `return_value`/`side_effect=[...]` for sequenced responses. A real gotcha: mocking an **async context manager** requires setting `__aenter__`/`__aexit__` return values, or you get `'coroutine' object does not support the asynchronous context manager protocol`.
- **Async generators / streaming**: to fake a stream, make the fake return an async generator:
  ```python
  async def fake_stream():
      for chunk in ["You ", "see ", "a ", "goblin."]:
          yield Chunk(delta=chunk)
  brain.stream = lambda *a, **k: fake_stream()
  # consume:  chunks = [c async for c in orch.stream_turn(...)]
  ```
- **Freezing time**: `freezegun` (`@freeze_time("2026-06-03")`) or `time-machine` (`@time_machine.travel(...)`, C-extension, faster, plays nicer with pytest assertion rewriting). Critically, in async code use `freeze_time(..., real_asyncio=True)` so `asyncio.sleep()` isn't broken. Even better long-term: inject a `Clock` dependency rather than freezing globally.
- **Seeding randomness**: D&D dice rolls are non-deterministic by design — inject the RNG (`random.Random(seed)`) or a `Dice` port so tests control rolls. Haystack's own conftest calls `set_all_seeds(0)`.
- **The #1 gotcha**: patch where the name is **looked up**, not where it's defined — `patch("myorchestrator.AnthropicClient")`, not `patch("anthropic.AnthropicClient")`, because "with `patch()` it matters that you patch objects in the namespace where they are looked up." Also watch event-loop-scoped fixtures and forgetting `await` (a bare `async def test` without the marker silently doesn't run its assertions).

## Recommendations

**Ranked by effort-to-value for a solo maintainer.**

**Do first (highest value, ~1–3 days, do this before anything else):**
1. **Introduce one `Brain` protocol and inject the brains** (triage, narrator) into the orchestrator constructor. Stop constructing provider clients internally. Add an `ALLOW_MODEL_REQUESTS`-style global guard: a module flag that makes any real provider call raise in test mode (copy Pydantic AI's `ALLOW_MODEL_REQUESTS = False` idea).
2. **Write a `ScriptedBrain` and a `FunctionBrain`** (≈20 lines, modeled on `FakeListChatModel` and `FunctionModel`). `ScriptedBrain` cycles a list of `BrainResult`s and records `.calls` for spying.
3. **Write your first integration tests on the main entrypoint** that assert on the **tool-call sequence and the resulting game-state diff**, not on narration. This is the test coverage you currently have zero of, and it's now reachable.

**Medium-term investment (~1–2 weeks, do once the above is paying off):**
4. **Refactor toward functional-core/imperative-shell** (the decision/evolution split): make tool dispatch return `Effect` data, interpret effects in a thin shell. Then most logic is tested with two-line `assert`s and no mocks. This is the structural change that makes the 4,800-line file maintainable.
5. **Add structured-trajectory snapshot tests** with `inline-snapshot` (or `syrupy` with `path_type`/`exclude` matchers to scrub rolls/timestamps/IDs). Snapshot the effect list and state diff, never the prose.
6. **Add an opt-in VCR tier** (`@pytest.mark.integration`, `record_mode="none"` in CI) for a handful of *real* end-to-end turns per provider, to catch prompt/serialization regressions. Filter `authorization`; use a custom request matcher; avoid streamed turns (the `_decoder` bug, vcrpy #895).

**Probably overkill for a solo project (skip unless a specific need appears):**
7. Full LLM-as-judge eval harnesses (FastEval-style `correctness` scoring, LangWatch Scenario) — valuable for prose *quality* regressions but heavy to maintain; defer until you have a quality problem worth measuring.
8. Adopting a whole framework (LangGraph/Pydantic AI) *just* for its test model — you can replicate `TestModel`/`FunctionModel` in ~30 lines without the migration cost. Borrow the pattern, not the dependency.
9. Recording/replaying streaming SSE cassettes — high fragility, low payoff; fake the stream at the `Brain` seam instead.

**Benchmarks that would change the ranking:** If you add a second human contributor or your prompts change weekly, promote VCR (#6) and snapshot tests (#5) to "do first." If narration *quality* becomes a product requirement (not just correctness), promote the eval harness (#7). If you stay solo and correctness-focused, items 1–5 are the whole game.

## Caveats
- **Framework APIs drift fast.** Pydantic AI in particular "changes frequently" (e.g. `result_type` → `output_type`); pin versions and verify helper names against your installed version. `GenericFakeChatModel` streaming support and `FakeListChatModel`'s cycling behavior have varied across LangChain releases.
- **Two specifics from the framework deep-dive are reported from docs/PRs but were not fully verified verbatim from source**: the OpenAI Agents SDK `FakeModel`'s exact scripting method name (likely `set_next_output`) and AutoGen `ReplayChatCompletionClient`'s exact inspection-attribute spellings (`create_calls`, `reset()`, exhaustion exception type). Confirm against the GitHub source before relying on them; the *patterns* are solid regardless.
- **VCR with async streaming is genuinely buggy** today (vcrpy issues #895, #597, #927 all open) — treat streamed-turn cassettes as unsupported.
- **Snapshot tests can rot into rubber-stamps**: if reviewers blindly `--inline-snapshot=fix` without reading the diff, they catch nothing. Discipline (reviewing snapshot diffs like code) is required for the technique to have value.
- **These patterns test "did the agent do the right thing," not "is the model good."** Deterministic tests deliberately exclude model quality; keep a small, separate, occasionally-run eval against real providers so you don't mistake green unit tests for a working game.