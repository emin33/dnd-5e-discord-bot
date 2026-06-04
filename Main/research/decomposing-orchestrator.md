# Decomposing a 4,800-Line LLM Turn Orchestrator: Architecture Survey, Target Structure, and Incremental Migration Plan

## TL;DR
- **Adopt a hybrid: a thin async "turn coordinator" (like LlamaIndex's AgentRunner) that owns the turn loop and is the *only* writer to world-state, plus four families of pluggable collaborators — an intent Router, a Tool Registry of uniform Command-style executors, one parameterized narration Strategy, and an explicit exploration↔combat State machine.** Use a small middleware/pipeline as the spine and an in-process async event bus only for genuinely fire-and-forget side effects (dedup is *not* one of those — see below). This maps cleanly onto how every major agent framework separates routing, tool execution, state, and output generation.
- **Cut the tool-executor registry first** — it has the cleanest boundary (18 sibling handlers with a near-uniform shape), the highest leverage (adding a 19th tool becomes trivial and isolated), and it is the easiest seam to wrap in characterization tests by recording/replaying LLM tool-call fixtures. Then collapse the 3 narration paths into one Strategy, then extract the combat State transition, then move world-state behind a single-writer repository with dedup as a write-pipeline step.
- **Migrate strangler-fig / branch-by-abstraction style behind a golden-master test net built FIRST** using VCR-style record/replay of Ollama calls plus seams that inject fakes. Never big-bang. Each step leaves the system green. Defend against re-monolithization with a strict dependency rule (collaborators never import the coordinator), single-writer state discipline, and a "no new business `elif` in the coordinator" rule enforced by the registry.

## Key Findings

### The monolith holds six responsibilities that belong in four different structural categories
The orchestrator currently fuses (1) intent triage = *routing*, (2) three narration paths = *output generation*, (3) ~18 tool executors = *action/command execution*, (4) combat initiation = *mode/state transition*, (5) world-state sync = *state ownership*, and (6) entity dedup = *a state-write concern*. These are not six copies of one pattern — they are four distinct concerns (route, execute, generate, own-state) plus two cross-cutting state operations. Every mature agent framework keeps these four separate, which is strong convergent evidence for the target structure.

### How established agent frameworks separate the same concerns
- **LlamaIndex (most directly applicable).** LlamaIndex splits the agent loop into an **AgentRunner** and an **AgentWorker**. Per the official v0.10.34 docs ("Step-wise, Controllable Agents"): *"AgentRunners are orchestrators that store state (including conversational memory), create and maintain tasks, run steps through each task, and offer the user-facing, high-level interface for users to interact with. AgentWorkers control the step-wise execution of a Task."* The docs add that workers *"act upon state passed down from the Task/TaskStep objects, but do not inherently store state themselves."* This is exactly the target: a **thin, state-owning coordinator** plus a **stateless, pluggable step executor**. State is threaded through `Task` (holds query + memory + `extra_state`) and `TaskStep` objects, never held in the worker. LlamaIndex's stated benefit — *"Easier Customization: it's easy to subclass/implement new agent algorithms... by implementing an AgentWorker"* — is the same property you want for adding a 19th tool. (Note: the verbatim AgentRunner/AgentWorker API is the v0.10.x lower-level API; newer LlamaIndex docs reframe this as event-driven Workflows with a `Context` key-value store, but the structural principle is unchanged.)
- **LangGraph (state-machine decomposition).** LangGraph models the agent turn as an explicit **state machine**: nodes are pure functions that receive a typed shared **state** (a `TypedDict`/Pydantic model) and return updates; **edges** (fixed or conditional) route control; the canonical agent loop is `call_model → conditional router → execute_tools → back to call_model`. Crucially, *"Nodes are pure functions: They receive state and return updates. This immutability enables debugging, replay, and checkpointing"* and *"The routing logic is separated from the processing logic."* This is the single best mental model for your exploration↔combat mode switching and your intent→handler routing. You do not need to adopt LangGraph itself; you can borrow the discipline (explicit state object, conditional edges, separation of routing from node bodies).
- **OpenAI Agents SDK (handoffs/routing).** A `triage_agent` with `handoffs=[...]` routes a turn to a specialist that "owns" the next response; the SDK's three added layers are the execution loop, handoffs (routing), and guardrails. Borrowable: keep the **routing surface legible** — "Give each specialist a narrow job... Split only when the next branch truly needs different instructions, tools, or policy." Maps onto your rules-triage brain → handler routing.
- **LangChain LCEL / AgentExecutor.** LCEL composes `Runnable`s with the pipe operator into linear pipelines; `RunnableBranch` does `if/elif/else` routing. As of LangChain 1.0 (GA October 22, 2025), **AgentExecutor remains in maintenance mode through December 2026**, and agents now run on the LangGraph runtime: the new `create_agent` entry point *"sits on top of the LangGraph runtime, supersedes both the legacy AgentExecutor and the langgraph.prebuilt.create_react_agent shortcut, and exposes customization via a middleware array rather than subclassing."* The ecosystem itself migrated from a linear-pipeline abstraction to an explicit state graph for agent loops. Lesson: a pipeline is great for the fixed pre/post stages of a turn, but the agentic *loop* wants a state machine — and even LangChain now exposes customization as a middleware array, validating the "middleware spine + state-machine loop" split recommended below.
- **Semantic Kernel (plugins + planner + filters).** SK separates **plugins/skills** (uniform callable functions, native or semantic, marked with `@kernel_function`), the **Kernel** (orchestrator/DI container), **planners** (decide which functions to invoke), and **filters** (middleware). Per the Semantic Kernel Filters docs: *"Filters in Semantic Kernel provide a powerful middleware pattern that allows you to intercept and modify the behavior of function invocations and prompt rendering. They enable you to add cross-cutting concerns like logging, authentication, caching, and custom processing logic"* and *"Filters provide a clean way to implement cross-cutting concerns without modifying your core business logic."* The plugin model is a direct analog of your tool registry; the filter pipeline is a direct analog of the middleware spine where your PGI/NLI gates belong.

### Game-development patterns map almost 1:1 onto a D&D turn
- **Command pattern (Nystrom, *Game Programming Patterns*, 2014).** Nystrom quotes the Gang of Four definition — *"Encapsulate a request as an object, thereby letting users parameterize clients with different requests, queue or log requests, and support undoable operations"* — and offers GoF's pithier slugline, *"Commands are an object-oriented replacement for callbacks,"* plus his own: *"A command is a reified method call."* He notes the decoupling "between the AI that selects commands and the actor code that performs them gives us a lot of flexibility." This is the canonical model for your ~18 tool executors: each becomes a Command with a uniform `execute(ctx) -> result` interface, dispatched by a registry. Commands also give you a free audit/undo/replay log — valuable for testing a non-deterministic LLM loop.
- **State pattern + pushdown automaton.** Nystrom: *"Where a finite state machine has a single pointer to a state, a pushdown automaton has a stack of them."* Exploration mode vs. combat mode is textbook State pattern; combat-as-a-pushed-state (you can `push` combat onto exploration and `pop` back) models "enter encounter, resolve, return to where you were" without the inline branching you have now.
- **Turn-loop phase sequencing + update/render separation.** Games sequence a turn as input → resolve → apply → render. The "update vs. render" split — compute state changes, *then* present them — maps exactly onto your needed separation of "compute world-state changes" (deterministic, testable) from "narrate them" (the LLM narration strategy). This is the most important borrowed idea: **narration must be downstream of and separate from state mutation.**
- **Entity-Component-System (ECS).** ECS *"separates the data (components) and behavior (systems)."* You almost certainly should **not** adopt full ECS — it is over-engineering for a solo-maintained Discord bot and optimizes for cache locality you don't need. But borrow the *principle*: world-state entities are plain data; dedup and sync are **systems/steps** that operate over that data, not methods tangled into the entities or the orchestrator.

### The four candidate architectures, scored for a solo asyncio maintainer

**A. Middleware/pipeline (async chain transforming a context object — ASGI/Starlette style).**
- *Structure:* each stage is `async def __call__(self, ctx, next)`; Starlette builds "a chain of ASGI applications that call into the next one." Shared data rides on a context object (Starlette uses the `scope` dict by convention).
- *Pros:* trivially understandable for one person; perfect for the **fixed** per-turn stages (load context → triage → … → persist → narrate); natural place for cross-cutting concerns (PGI validation, NLI post-check) as middleware; async-native; easy to test a stage in isolation.
- *Cons:* a linear chain cannot express loops or branching well — and an agent turn with tool-calling is a loop. Middleware order is load-bearing and can become subtle. Putting *business logic* in middleware is an anti-pattern ("Middleware should handle cross-cutting concerns, not domain workflows").
- *Verdict:* **Use it for the spine, not the loop.** The outer turn = a short middleware pipeline; the inner tool-calling loop = a state machine/coordinator step.

**B. Command pattern + command bus/registry (each tool = a Command; a dispatcher invokes).**
- *Structure:* uniform `Command`/`Handler` interface; a registry maps intent/tool-name → handler; "CommandBus is a communication mechanism used to decouple the sender of a command from its handler." Variants: simple dict registry, command bus (one handler per command, can carry middleware), CQRS-style.
- *Pros:* **the cleanest fix for the 18 tool executors.** Uniform interface + registry means adding a 19th tool is one new class + one registration line, fully isolated. Commands are individually unit-testable. Gives audit/replay for free. Async-native (handlers are coroutines).
- *Cons:* a full command bus with DI can be ceremony overkill for one dev — prefer a plain registry/dict over a heavyweight bus. Risk of "anemic" commands if logic leaks back to the coordinator.
- *Verdict:* **Adopt for tool executors and (optionally) for top-level intents.** Keep it lightweight: a `dict[str, ToolHandler]` registry, not an enterprise bus.

**C. Service objects + thin coordinator (extract each responsibility into a focused collaborator; orchestrator becomes a wiring facade).**
- *Structure:* the classic Extract Class / Facade decomposition; the coordinator owns the loop and delegates to `NarrationService`, `WorldState`, `EntityDeduper`, `CombatMode`, `Router`, `ToolRegistry`.
- *Pros:* lowest cognitive load and least ceremony — ideal for a solo maintainer; directly attacks the god-object anti-pattern (the documented fix is "Group related methods... Divide large classes... each handling a specific responsibility"); each service independently testable; no framework lock-in.
- *Cons:* the coordinator can quietly re-bloat into a new god object if you let it accumulate `if/elif` logic or state; "thin" requires discipline.
- *Verdict:* **This is the backbone of the recommendation.** Combine with B (registry) and the State pattern so the coordinator stays thin by construction.

**D. Event-driven / in-process async event bus (emit events; handlers subscribe).**
- *Structure:* `bus.subscribe("tool.called", handler)` / `await bus.publish(...)`; in-process pub/sub with no broker. Decouples producers from consumers.
- *Pros:* great for genuinely decoupled **side effects** (telemetry, logging, cache updates); actor-style isolation.
- *Cons:* **dangerous as the primary control flow for a solo maintainer.** In-process buses don't persist events ("If the process dies, undelivered events are gone"), don't guarantee ordering across subscribers, and make the turn's control flow implicit and hard to trace — the opposite of what you want when debugging a non-deterministic LLM turn. Re-monolithization risk is replaced by "spaghetti events" risk.
- *Verdict:* **Use sparingly and only for fire-and-forget side effects**, never for the core turn sequence (route→execute→sync→narrate). World-state writes should be explicit calls, not events, so the single-writer rule stays enforceable. (Note: entity dedup is *not* a side effect — it must run synchronously in the write path; see below.)

## Details

### Recommended target structure

A **thin async coordinator** (the strangler facade) owns the turn loop and the single write-path to world-state. Everything else is a pluggable collaborator behind a `Protocol`. Proposed package layout:

```
turn/
  coordinator.py        # TurnCoordinator: owns the loop, the ONLY world-state writer
  context.py            # TurnContext dataclass: the state object passed through stages
  pipeline.py           # tiny async middleware spine (fixed pre/post stages)
  protocols.py          # Protocols/ABCs for every collaborator (the dependency boundary)
routing/
  router.py             # IntentRouter: triage brain -> intent; maps intent -> handler
narration/
  strategy.py           # NarrationStrategy (ONE parameterized path) + NarrationSpec
  prompts/              # per-variant prompt/template data (data-driven, not code-driven)
tools/
  registry.py           # ToolRegistry: dict[str, ToolHandler] + @register decorator
  base.py               # ToolHandler Protocol: async def execute(ctx, args) -> ToolResult
  executors/            # one module per tool (the ~18, now isolated siblings)
combat/
  mode.py               # ModeMachine (pushdown automaton): Exploration <-> Combat
  states.py             # ExplorationState, CombatState (State pattern)
world/
  store.py              # WorldStateStore: single-writer repository; apply(change)
  dedup.py              # EntityDeduper: a write-pipeline STEP, called by the store
  changes.py            # WorldChange dataclasses (command-sourcing-lite)
llm/
  client.py            # Ollama/Qwen3 client behind a Protocol (the key test seam)
  fakes.py             # Recorded/replay fakes for tests
events/
  bus.py               # in-process async bus for SIDE EFFECTS ONLY
```

Key Python interfaces (asyncio idioms, using `Protocol` for structural typing so collaborators never need to import the coordinator):

```python
# protocols.py
from typing import Protocol, runtime_checkable
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

@dataclass
class TurnContext:
    player_input: str
    intent: str | None = None
    world: "WorldStateView" = ...        # READ-only view for most collaborators
    pending_changes: list["WorldChange"] = field(default_factory=list)
    transcript: list["Message"] = field(default_factory=list)

@runtime_checkable
class ToolHandler(Protocol):
    name: str
    async def execute(self, ctx: TurnContext, args: dict) -> "ToolResult": ...

class NarrationStrategy(Protocol):
    async def narrate(self, ctx: TurnContext, spec: "NarrationSpec") -> AsyncIterator[str]: ...

class LLMClient(Protocol):
    async def complete(self, prompt: "Prompt") -> "Completion": ...
    def stream(self, prompt: "Prompt") -> AsyncIterator[str]: ...

class WorldStateStore(Protocol):
    def view(self) -> "WorldStateView": ...                       # read
    async def apply(self, changes: list["WorldChange"]) -> None:  # the ONLY writer
        ...
```

**Who owns state (the load-bearing rule).** The `WorldStateStore` is the *single writer*. Collaborators receive a read-only `WorldStateView` and *return* `WorldChange` objects; only the coordinator calls `store.apply(changes)`. This is "single-writer discipline" borrowed from event-sourcing systems (Akka: *"there can only be one active instance... with a given persistenceId. Otherwise, multiple instances would store interleaving events"*) and is what prevents the sync/dedup logic from re-tangling. **Entity dedup runs as a step inside `store.apply()`** (a write-pipeline stage), not as an event handler and not in the coordinator — so dedup always runs exactly when state changes and can never be bypassed or duplicated.

**The turn loop (coordinator pseudo-structure):**
```python
async def run_turn(self, player_input: str) -> AsyncIterator[str]:
    ctx = TurnContext(player_input, world=self.store.view())
    async with self.pipeline.stage("triage"):
        ctx.intent = await self.router.route(ctx)          # routing
    handler = self.registry.get(ctx.intent)                # registry dispatch
    async with asyncio.TaskGroup() as tg:                  # structured concurrency
        result = await handler.execute(ctx, args)          # command execution
    self.mode = self.mode.on_result(result)                # state transition (push/pop combat)
    await self.store.apply(ctx.pending_changes)            # single-writer + dedup step
    spec = self.mode.narration_spec(ctx, result)
    async for token in self.narration.narrate(ctx, spec):  # ONE narration strategy, streamed
        yield token
```

### Async-specific concerns
- **Structured concurrency.** Use `asyncio.TaskGroup` (Python 3.11+) or AnyIO task groups to bound the lifetime of any concurrent work in a turn (e.g., parallel tool calls, NLI post-check running alongside generation). AnyIO is worth adopting even on asyncio: it gives proper cancel scopes and *"All child tasks complete before the task group exits... Cleanup is guaranteed."* Per the official AnyIO docs ("Why you should be using AnyIO APIs instead of asyncio APIs," AnyIO 4.13.0): *"the asyncio.TaskGroup class does not offer any way to cancel, or even list all of the contained tasks, so in order to do that, you would still have to keep track of any tasks you create."* AnyIO's cancel scopes (`fail_after`, `move_on_after`) are therefore the cleaner tool for per-turn timeouts.
- **Cancellation.** A Discord turn can be abandoned; wrap the turn in a cancel scope so a dropped turn cancels in-flight Ollama calls and tool tasks. Always re-raise the cancellation exception after cleanup.
- **Streaming narration via async generators.** Narration is an `AsyncIterator[str]`. Use an `asyncio.Queue` between the Ollama token producer and the Discord consumer to handle **backpressure** — a bounded `asyncio.Queue(maxsize=...)` so a slow consumer applies backpressure instead of ballooning memory. Setting `maxsize=0` means unlimited and loses backpressure protection.
- **Async context managers** for the pipeline stages (timing, tracing, PGI/NLI gates) and for the LLM client lifecycle.

### Collapsing the 3 near-duplicate narration paths into ONE
Use **Strategy parameterized by data**, not Template Method. The three paths drifted because the variation is in *content/prompt/parameters*, not in *algorithm structure*. Refactoring guidance: the Strategy pattern *"lets you extract the varying behavior into a separate class hierarchy and combine the original classes into one, thereby reducing duplicate code."* Concretely:
1. Diff the three paths; extract the invariant skeleton (build prompt → call narrator brain → PGI validate → NLI cross-check → stream) into one `NarrationStrategy.narrate()`.
2. Capture the *differences* as a `NarrationSpec` dataclass (which prompt template, temperature, which guardrails, tone) — **data, not branches**.
3. Drive the spec from the current mode/intent (`mode.narration_spec(...)`). Adding a 4th narration variant becomes adding a data row, not copy-pasting a method.

Template Method (a base class with overridable steps) is the fallback *only if* the three paths genuinely differ in step *structure* — but here they're copy-paste drift, so data-driven Strategy is the lower-cognitive-load choice for one maintainer. (Refactoring.Guru: Strategy is composition-based and switchable at runtime; Template Method is inheritance-based and static.)

### Giving the 18 tool executors a uniform interface + registry
1. Define `ToolHandler` Protocol with `async def execute(ctx, args) -> ToolResult` and a `name`.
2. Build a `ToolRegistry` wrapping `dict[str, ToolHandler]` with a `@register` decorator (the Python registry pattern — "separates the data (the mappings) from the logic"). The registry replaces the giant `if/elif` dispatch.
3. Move each executor into `tools/executors/<name>.py` as a sibling class implementing the Protocol. Each is now independently testable with a fake `LLMClient` and a fake `WorldStateView`.
4. Adding a 19th tool = new file + `@register`. The coordinator never changes. This is the structural property that most directly kills the god object.

### Modeling combat initiation as a State transition
Replace inline combat-entry branching with a `ModeMachine` implementing State + a pushdown stack: `ExplorationState` and `CombatState`, with `push(CombatState)` on encounter start and `pop()` on combat end (returning to exactly the prior exploration context). Each state answers: which intents/tools are legal here, and which `NarrationSpec` to use. This is Nystrom's State chapter applied directly, and it removes mode-checking `if`s scattered across the orchestrator.

### How intent triage separates from execution, mapped to the dual-brain design
Router → Handler. The **rules-triage brain** is the implementation of `IntentRouter.route(ctx) -> intent`; it does *classification only* and returns an intent label. The coordinator then does registry dispatch to a handler. The **narrator brain** lives entirely in the `NarrationStrategy`, downstream of state mutation. This cleanly separates "decide what to do" (triage brain) from "do it" (tool handlers) from "describe it" (narrator brain) — the exact separation LangGraph enforces with "routing logic separated from processing logic" and OpenAI's triage-agent handoffs.

### How dedup and world-state sync relate
Dedup is a **step in the write pipeline**, invoked by `WorldStateStore.apply()` — not an event handler, not a coordinator method. Rationale: dedup is logically *part of* a consistent write (you never want state to exist un-deduped), so it must run synchronously inside the single write-path. Modeling state changes as explicit `WorldChange` objects passed to `apply()` is "command-sourcing-lite": it gives you one auditable choke point where dedup + sync + invariant checks run, in order, every time. World-state sync (keeping derived/graph state consistent — your NetworkX/ChromaDB knowledge graph) is a *projection* updated in the same `apply()` call, or via the event bus only if it can tolerate eventual consistency.

### Building the characterization/golden-master test net BEFORE refactoring
This is non-negotiable and comes first (Feathers' algorithm: identify change points → find test points → break dependencies → write characterization tests → *then* change). For a non-deterministic LLM orchestrator:
1. **Find the seam at the LLM boundary.** Put the Ollama/Qwen3 client behind the `LLMClient` Protocol. This is the "object seam" Feathers prefers — it lets you inject a fake without editing call sites.
2. **Record/replay (VCR-style "cassettes").** Run real turns once against live Ollama, record every request/response to fixture files, and replay on subsequent runs. Tools: `pytest-recording`/VCR.py, or for structured LLM calls a BAML-VCR-style approach. Per the VCR-for-LLMs writeup: *"The tests run once with live LLM API calls and subsequent test runs replay the recorded responses instead of making live calls. If the API request payload... changes in any way, the tests fail"* — turning a non-deterministic system into a deterministic regression detector. Commit cassettes to version control; refresh periodically (e.g., `record_mode="none"` in CI, `record_mode="all"` occasionally to refresh).
3. **Golden-master the whole turn.** Characterization tests capture *actual* current behavior (Feathers: a characterization test "characterizes the actual behavior of a piece of code" — you assert what it *does*, not what it *should* do). Snapshot the full turn output (narration + resulting world-state diff) for a corpus of representative inputs. Use snapshot libraries (`inline-snapshot`, or `pytest-insta` for pickleable state). Mask volatile fields (timestamps, IDs).
4. **Layered assertions for the parts you can't pin exactly.** Combine deterministic checks (tool selected, JSON tool-call schema valid, state diff correct, mode transition correct — these *are* deterministic given a replayed LLM response) with fuzzy checks (semantic-similarity threshold on narration via a sentence-transformer cosine score) only where needed.
5. **Fakes per seam.** Keep a small fixture set: one normal response, one tool-call response, one refusal, one malformed response, one empty — "Five to ten fixtures cover most agent logic paths."

### Step-by-step incremental migration order (each step ends green)
**Step 0 — Net first.** Establish the `LLMClient` seam + record cassettes + golden-master the current monolith's turn outputs. Do *no* refactoring yet. Run the suite; it's green by definition.

**Step 1 — Extract the Tool Registry (the first seam to cut).** *Why first:* cleanest boundary, highest leverage, easiest to test. Branch-by-abstraction: introduce the `ToolHandler` Protocol + `ToolRegistry`; for each of the 18 executors, extract the body into a handler class and have the old `if/elif` branch *delegate* to `registry.get(name).execute(...)`. The abstraction layer (registry) lets old and new coexist. When all 18 delegate, delete the `if/elif`. Tests: per-handler unit tests with fakes + the golden master proves the turn is unchanged.

**Step 2 — Collapse narration into one Strategy.** Introduce `NarrationStrategy` + `NarrationSpec`. Route one of the three paths through it behind a feature flag; compare output to the golden master (and to the other two paths) using semantic-similarity assertions. Migrate the second, then third path. Delete the two duplicates. This is branch-by-abstraction with the spec as the abstraction.

**Step 3 — Extract the combat State machine.** Introduce `ModeMachine`; replace combat-entry branching with `push/pop`. Characterization tests assert mode transitions for recorded turns are identical.

**Step 4 — Introduce the single-writer WorldStateStore.** Wrap existing state behind `WorldStateStore.view()`/`apply()`. Convert mutations to `WorldChange` objects returned by collaborators and applied only by the coordinator. Keep the old direct-mutation path behind the abstraction until all writers are converted, then remove it.

**Step 5 — Move dedup into the write pipeline.** Relocate dedup logic to run inside `apply()`. Characterization tests on state diffs guarantee dedup output is unchanged.

**Step 6 — Thin the coordinator / add the middleware spine.** Now that collaborators exist, reduce the coordinator to the wiring + loop shown above; introduce the small async middleware pipeline for fixed stages (PGI/NLI gates as middleware filters). Add an event bus *only* for side effects (telemetry, graph projection if eventually-consistent).

**Step 7 — Delete the strangled host.** Remove dead branches and, optionally, the temporary abstraction layers that were only there for migration (Fowler: "We may also choose to delete the abstraction layer once we no longer need it").

### Anti-patterns / re-monolith pressures and how to defend
- **Pressure: the coordinator accumulates `if/elif`.** *Defense:* a hard rule — "no business branching in the coordinator." New behavior = new handler + registration, or new state, or new `NarrationSpec` row. Lint for it in review.
- **Pressure: collaborators reach back into the coordinator or each other.** *Defense:* the **dependency rule** — collaborators depend only on `protocols.py` and data types, never import `coordinator.py`. Enforce with import-linter / a CI check. This is the structural firewall against re-coupling.
- **Pressure: multiple writers to world-state.** *Defense:* single-writer discipline; `WorldStateView` is read-only; only `store.apply()` mutates. If a collaborator "needs" to write, it returns a `WorldChange` instead.
- **Pressure: events become implicit control flow ("spaghetti events").** *Defense:* the event bus is for fire-and-forget side effects only; the core route→execute→sync→narrate sequence stays explicit in the coordinator.
- **Pressure: narration creeps back upstream of state mutation.** *Defense:* enforce the update/render split — narration only ever reads the post-`apply()` state; it never mutates and never runs before `apply()`.
- **General signs of collapse:** the coordinator file growing again; a single collaborator gaining unrelated methods (low cohesion); a "utils"/"helpers" dumping ground forming; tests needing to construct the whole world to test one piece. Treat any of these as a trigger for the next extraction. Continuous refactoring in small commits (Feathers/Fowler) beats periodic big cleanups.

## Recommendations
1. **This week — build the net.** Put Ollama behind the `LLMClient` Protocol; record VCR cassettes for ~10 representative turns (exploration, combat-entry, each tool family, a refusal, a malformed tool call); golden-master the turn outputs (narration + state diff). Ship nothing else. *Benchmark to proceed:* suite green and deterministic on replay across 3 consecutive CI runs.
2. **Next — cut the tool registry (Step 1).** Highest leverage, cleanest seam. *Benchmark to proceed:* all 18 executors delegate through the registry; the `if/elif` block is deleted; adding a trivial 19th tool touches exactly one new file + one registration line.
3. **Then narration (Step 2), combat state (Step 3), single-writer store + dedup (Steps 4–5)** in that order. *Benchmark per step:* golden master unchanged (semantic-similarity ≥ your chosen threshold for narration; exact match for state diffs and mode transitions).
4. **Finally thin the coordinator + add the middleware spine (Step 6)** and delete the strangled host (Step 7). *Benchmark:* coordinator file under a self-imposed budget (e.g., ~200 lines) with zero business `if/elif`.
5. **Adopt AnyIO** for task groups/cancel scopes if you need per-turn timeouts and clean cancellation of abandoned Discord turns; otherwise `asyncio.TaskGroup` is sufficient. Use a **bounded `asyncio.Queue`** for streaming narration backpressure.
6. **Do not adopt** full ECS, a heavyweight command bus with DI, LangGraph/LangChain as a runtime dependency, or an event-bus-driven control flow. Borrow their *patterns* (state object, command/registry, routing/handoff, plugin uniformity, filter middleware), not their weight. **Threshold that would change this:** if the bot grows to multiple concurrent narrator/triage brains or multi-agent handoffs, revisit LangGraph (explicit checkpointed state machine) or the OpenAI Agents SDK (handoffs) as a runtime — but only then.

## Caveats
- **The 4,800-line estimate and the six-responsibility breakdown are taken from the task description, not independently verified;** the exact extraction boundaries depend on the real coupling in your code (map change-points and hidden dependencies first, per Feathers).
- **LLM golden-master testing has inherent limits:** even at temperature 0, outputs aren't perfectly deterministic across hardware/model-version changes; replay (cassettes) sidesteps this for regression but means your tests validate behavior *given a fixed recorded LLM response*, not live model quality. Pair with a separate, slower eval suite for model quality.
- **Framework facts move fast:** LangChain's AgentExecutor entered maintenance mode (through December 2026) and LangGraph became the agent runtime in the 1.0 releases (Oct 22, 2025); LlamaIndex's headline API shifted from AgentRunner/AgentWorker to event-driven Workflows. The *structural lessons* are stable; specific APIs will keep changing, which is itself an argument for borrowing patterns rather than adopting frameworks wholesale.
- **Single-writer state** assumes one logical turn-writer; if you later process multiple players' turns truly concurrently against shared world-state, you'll need per-entity locking or an actor-per-entity model (out of scope here, but the `apply()` choke point is the right place to add it).
- This report assumes the dual-brain, PGI, and NLI components already work; the recommendation reorganizes *where* they're called (triage brain in the Router, NLI/PGI as middleware gates, narrator brain in the Strategy), not their internals.