# Orchestrator Decomposition + Single-Authority Refactor — Plan

Synthesized 2026-06-03 from two deep-research passes (in `research/`) that converged
independently. This is the durable playbook; it survives session boundaries.

## The one mechanism that stops the re-flag cycle

**Build a test net FIRST, then strangler-fig refactor where every step is proven
behavior-unchanged against that net.** Every prior structural fix that got re-flagged
was a *partial* change with no net to verify completeness (Stage A.2 half-applied =
quality finding #2). The net is non-negotiable and comes before any decomposition.

## What both research reports agree on (high confidence)

1. **Inject a `Brain`/`LLMClient` Protocol** at the orchestrator entry; write a
   `ScriptedBrain` (cycles scripted `BrainResult`s) + `FunctionBrain` (branch on ctx);
   add an `ALLOW_MODEL_REQUESTS = False`-style global guard so un-mocked calls fail
   loudly in tests. This single seam is the highest-value first move.
2. **Assert on the tool-call sequence + resulting state diff, NOT on narration prose.**
   Scrub rolls/timestamps/IDs. Snapshot the *structured trajectory* only.
3. **Functional core / imperative shell:** decision logic returns `Effect` objects as
   data; a thin interpreter executes them. Our `ProposedEffect`/`EffectExecutor` system
   already invites this — lean into it.
4. **Cut the tool registry FIRST** — cleanest boundary, highest leverage (a 19th tool
   becomes one file + one registration line), and it kills the 4-site dispatch drift
   (quality #5).
5. Then **narration: 3 paths → 1 `NarrationStrategy` + `NarrationSpec`** (data, not branches).
6. Then **combat → `ModeMachine`** (State pattern + pushdown: push combat onto exploration, pop back).
7. Then **single-writer `WorldStateStore`**: collaborators get a read-only view and
   *return* `WorldChange`s; only the coordinator calls `store.apply()`. **Dedup runs as a
   step inside `apply()`**, never as an event or a coordinator method.
8. **Strangler-fig / branch-by-abstraction; each step ends green; never big-bang.**
9. **Borrow patterns, not frameworks** (no LangGraph/Pydantic-AI runtime dependency;
   replicate `TestModel`/`FunctionModel` in ~30 lines).

## Sequenced steps (mapped to THIS codebase)

- **Step 0 — Net.** We already have provider clients (`OllamaClient`/`AnthropicClient`/…);
  what's missing is *injection at the orchestrator entry* + fakes + the guard + the first
  `process_action` integration tests (assert effect sequence + state diff). ~1–3 days.
- **Step 1 — Tool registry.** The ~18 `_execute_*` + the 4 parallel dispatch sites
  (`intents`, `narrator_tools`, `effects`, `orchestrator`) → one `dict[str, ToolHandler]`
  registry + Protocol. Fixes quality #5 and #11 (no-op executors) along the way.
- **Step 2 — Narration Strategy.** Collapse the 3 near-duplicate narration paths.
- **Step 3 — Combat ModeMachine.** Replace inline combat-entry branching.
- **Step 4 — Single-writer `WorldStateStore`.** This is also where the deferred
  **Stage B (#6/#7)** and the **DF-1/5/6/7 state-ownership cluster** finally resolve:
  one session-owned authority, `apply()` choke point. Per-entity locking added here later
  if true concurrency is needed.
- **Step 5 — Dedup into the write pipeline.**
- **Step 6 — Thin coordinator + middleware spine** (PGI/NLI as filter middleware).
- **Step 7 — Delete the strangled host** + temporary abstraction layers.

Per-step benchmark: golden master unchanged — *exact* match for state diffs / mode
transitions / tool-call sequences; semantic-similarity threshold for narration prose.

## Progress log

### Step 0 — Net: **DONE** (2026-06-04, commits `4b49a4b` + `42e918f`, 506 tests green)
- **Production seam (behavior-neutral; defaults preserve current behavior):**
  - `llm/client.py`: `LLMClient` Protocol + `ALLOW_MODEL_REQUESTS` guard
    (`set_model_requests_allowed`) — every real provider `chat()/chat_stream()`
    raises when armed, so an un-mocked call in a test fails loudly. Guard call
    added to all 6 providers + Ollama `chat_stream`.
  - `llm/orchestrator.py`: inject `narrator_client_factory` (defaults to
    `get_narrator_client_for`) so a fake narrator client survives the per-turn
    tier swap at `_select_narrator_client_for_turn`. Triage was already
    injectable via `client=`.
- **Net:** `tests/fakes.py` (`ScriptedBrain`, `FunctionBrain`, `brain_router`,
  response builders) + `tests/integration/test_process_action.py` (5 tests:
  attack→combat-without-narrating, social→ref_entity, cast_spell→slot-spent-&-
  persisted, skill_check→roll-scrubbed via `_narrate_outcome`, guard-fires).
- **Verified seam facts (grep-confirmed, may save the next session time):**
  - **Two** LLM seams only: `get_llm_client()` feeds triage **and** the
    state/entity extractors **and** the dedup judge (all `self.client or
    get_llm_client()`); `get_narrator_client_for(tier)` feeds narration. A test
    wires the first via one `FunctionBrain` on `orch.client` +
    `_state_extractor.client` + `_entity_extractor.client` + `get_dedup_judge()`.
  - `self.rules` is **dead** — assigned in `__init__`, never called. Triage is
    `self.client.chat(json_schema=get_triage_schema())` (~orch:1475). NLI is
    disabled (Step 3.7). Entity-extract is skipped when `world_state` is present.
  - **Test env gotcha:** the suite runs under `C:\Python\python.exe` (system),
    **not** `Main/venv` (which has no pytest). Use the controlled-file pattern.
### Step 1a — Delete the dead tool-execution path: **DONE** (2026-06-04, commit `85cafe5`, 506 green)
- **Verify-don't-trust payoff:** the audit framed Step 1 as "extract the 16
  `_execute_*`," but `_execute_tool` (the orchestrator's rules-brain tool
  dispatcher) has **zero callers** — it + the 14 `_execute_*` it dispatched are a
  dead **parallel implementation** of tool execution (dead because `self.rules`
  is dead). The research says remove parallel implementations *first*. Deleted
  them + the orphaned `ToolExecutionResult` DTO (528 lines; orchestrator
  4785→4257).
- **KEPT** the two executors live code also calls directly
  (`_execute_purchase_item` ← `_handle_purchase`, `_execute_add_item` ←
  `_handle_inventory`) and the interleaved live helper (`_resolve_character_by_name`).
- **Lesson (recorded for the redo of similar cuts):** these dead methods are
  **interleaved** with live helpers — a line-range deletion swallowed a live one
  and the net failed loudly (caught it). Use **AST-span-by-name** deletion, not
  line ranges, for interleaved clusters. The purchase/inventory live path is
  **untested** — safety came from AST-by-name + post-delete `hasattr` checks + suite.
- **Correction (2026-06-09, quality re-audit #13):** `_resolve_combatant_by_name` was
  wrongly recorded above as a live helper — zero callers (they died with `_execute_tool`);
  it and `_check_player_attack_initiation` (also zero callers; superseded by triage + the
  `start_combat` tool) are now deleted, AST-span-by-name.

- ~~**Next (the real Step 1 — the tool *registry*)**~~ — **DONE below** (2026-07-05).

### Step 1 — Tool registry: **DONE** (2026-07-05, commits `51118fb` net + `79e391c` + `2a83637` + `a116319`, 609 tests green)
- **Net first (per the rule):** `51118fb` widened the Step-0 net with 7 per-tool
  `process_action` turns — add_npc / spawn_object / update_player grant+remove
  (working, pinned green) and purchase / inventory-pickup / remove_entity
  (pinned **broken** with "flips when fixed" arrows). Only then was dispatch touched.
- **What landed:**
  - `llm/tool_registry.py` — ONE declarative `NarratorToolSpec` per narrator tool:
    name, JSON schema, tier membership, converter (args→`ProposedEffect`),
    `effect_types` it can emit, `world_sync` flag. Registered at import, canonical
    order. **Adding a tool = one registration block** (remove_entity proved it).
  - `narrator_tools.py` 881→92 lines: `get_narrator_tools_for_tier` /
    `tool_calls_to_effects` / the `NARRATOR_TOOLS*` constants are thin reads over
    the registry; the hardcoded schema list + 8-branch `_convert_tool_call` deleted.
  - `EffectExecutor`'s `dict[EffectType, method]` hoisted to `__init__` +
    `handled_effect_types()` so tests can introspect it. Keys unchanged.
  - **Exhaustiveness tests** (`tests/unit/test_tool_registry.py`): every emittable
    EffectType has an executor row; every `world_sync=True` type has a
    `_sync_effect_to_world_state` branch (source inspection — the elif chain stays
    hand-written until Step 4 owns it as data); signal set pinned to exactly
    {start_combat, request_roll}; per-tier composition pinned exactly.
  - **remove_entity wired end-to-end** (audit Duplication P0): full-tier tool +
    converter + `_execute_remove_entity` (scene-registry removal, honest failure
    when absent); the pre-existing sync + KG-bridge branches became reachable.
    The pinned-broken net test flipped to asserting the removal.
  - **No-op stub dispositions** (audit May #11): `apply_damage` kept (INTENTS
    producer live) but returns `success=False` honestly, unreachable sync branch
    deleted; `APPLY_HEALING`/`ADD_CONDITION`/`REMOVE_CONDITION`/`LOG_MEMORY`/
    `REVEAL_OBJECT` **deleted** (grep-dead from BOTH producers — tool converter
    and INTENTS parsers); `set_flag` reclassified, not a stub — its real write is
    the sync branch, gated on success, same as change_location.
- **Lessons:**
  - Copy → pin-identical → delete beats edit-in-place for a 600-line dispatch
    strangle: commit 1 duplicated schemas/converters into the registry with
    byte-equality tests (per-tier schema lists, converter `model_dump` over
    branch-covering args); commit 2's deletion then couldn't drift. One commit of
    deliberate duplication, guarded, is cheap.
  - "Every EffectType has a handler" is necessary but not sufficient — the same
    drift hides in the *sync* chain. Declaring `world_sync` on the spec and
    source-inspecting the elif chain was ~15 lines and catches the other half;
    datafying the chain itself belongs to Step 4, don't do it early.
  - Verify-don't-trust again: the audit's "6 no-op stubs" were NOT uniform — one
    (`set_flag`) was load-bearing via its sync branch; blanket-failing all six
    would have broken working INTENTS flag-setting. Per-stub disposition mattered.
  - The two ProposedEffect **producers** are the tool path and the INTENTS text
    fallback. The StateDelta extractor pipeline does NOT produce ProposedEffects —
    it is the *parallel write path* (apply_delta/DeltaBridge.convert) that Step 4's
    single-writer store must merge; don't conflate them when counting dispatch sites.
- **Deliberately left (with pins where possible):**
  - Purchase/pickup id-vs-name defect (net agent's find, not audit-named): both
    net tests still pin the broken triage-route behavior with fix-shape notes —
    `_handle_purchase`/`_handle_inventory` pass a UUID that `_execute_purchase_item`
    /`_execute_add_item` re-resolve as a NAME. Commerce route ≠ narrator-tool
    surface; separate slice. **LANDED** (2026-07-05, final-review pass): the two
    executors now take the already-resolved Character from their only callers;
    both net tests flipped to pin the working behavior.
  - `ProposedEffect` 58-field god-DTO → discriminated union (audit Type P1):
    follow-up, ideally per-tool models co-located in the registry entries.
  - `CONSUME_RESOURCE`: real executor, zero producers — dead-code-pass candidate.
  - `_handle_inventory` drop/equip/use narrative-success no-ops (TODO, no DB touch).
  - `requires_confirmation` settable only from INTENTS (audit P2): dies with
    INTENTS or the tool path gains confirmation semantics — registry makes either
    a one-entry change.

## Pre-net cleanup (safe to do NOW, before the net — reduces surface)

Pure dead-code deletion is the safest change (grep-confirm zero usages + suite green) and
it removes *parallel implementations* the decomposition would otherwise have to reason
about. From the quality audit's quick-wins:
- Delete grep-confirmed orphans — mostly **DONE** (`ddea308`): `game/combat/turn_loop.py`,
  `game/mechanics/effects.py` (+ `create_*_effect` factories), `views/CombatTurnManager`,
  empty `llm/prompts/` & `llm/tools/` packages all gone. Still pending: streaming-TTS
  subsystem, duplicate `_build_mechanics_embed`/`split_text` in `game.py`,
  `Character.proficiencies`/`CharacterProficiency`, etc.
- Either wire `image_comfyui.py` into the factory or delete it (config advertises
  `"comfyui"` → ValueError today).
- Unused imports (**DONE** `ddea308`, 84 removed); hoist in-function `import structlog`/
  `uuid` to module level (still pending).
- **DEFER** "rarely-hit fallback" deletions (INTENTS mini-language + adjudicator, ~650
  lines) until the net exists — confirm-then-delete, not blind.

Also finish here: **Stage A.2 remainder (#2)** — `_get_save_modifier`/`_is_concentrating`/
`_check_concentration` in `coordinator.py` — and **double end-of-turn processing (#3)**:
both **DONE** (`c0b3d67`; the three helpers now route via `_resolve_player_character`).

## Session strategy

- **This/current-context session:** pre-net cleanup + finish Stage A.2 (#2) + double-tick (#3).
  Low-risk, doesn't need the net, clears the deck.
- **Fresh session(s)** with full context budget: Step 0 net → Steps 1–7. Onboard via the
  order below. The big decomposition wants fresh focus + the test-net discipline.

## START HERE (fresh-session onboarding order)

1. This file (`REFACTOR_PLAN.md`).
2. `research/testing-llm-pipelines.md` and `research/decomposing-orchestrator.md` (the detail).
3. `AUDIT_QUALITY_2026_05_29.md` (quality findings + grades).
4. `AUDIT_DATAFLOW_2026_05_29.md` — esp. §0 (unified state model), §4 (state-ownership map
   = the decomposition blueprint), §5b (refactor progress).
5. `AUDIT_FULL_2026_05.md` (the #6/#7 deferred-cluster rationale).
6. Memory: `audit_quality_2026_05_29.md`, `audit_dataflow_2026_05_29.md`, `roadmap_audit_2026_05.md`.
7. `git log --oneline -10`; run the suite with the controlled-file pattern
   (`timeout 180 python -m pytest tests/ --timeout=45 -q > _testout.txt 2>&1; echo EXIT=$?`)
   — NOTE: a leaky async fixture once hung pytest on *exit* (tests passed, process didn't
   terminate); fixtures opening a `Database` must `yield` + `await db.disconnect()`.

## Operational notes (learned the hard way — don't relearn them)

- **Run tests with the controlled-file pattern** (background pytest output buffers and
  won't appear until exit):
  `timeout 180 python -m pytest tests/ -p no:cacheprovider --timeout=45 -q > _testout.txt 2>&1; echo EXIT=$?; tail -6 _testout.txt`
- **A leaky async fixture hangs pytest on exit** (tests PASS, process never terminates,
  orphaning it). Any fixture opening a `Database` must `yield` + `await db.disconnect()`,
  not `return`. This caused days of "the tests never finish" confusion.
- **Don't kill "hung" pytest reflexively** — it's usually just slow-to-flush; wait for the
  result. Genuinely stuck = a process alive >2 min for a ~20s suite.
- **Bare `python -c "import dnd_bot.bot.cogs.*"` fails** with `discord has no attribute
  'ApplicationContext'` — a py-cord vs discord.py env quirk, NOT a real error. The suite
  never imports the cogs; don't chase it.
- **Type gate exists: `run_typecheck.bat`** (= `venv\Scripts\python -m mypy dnd_bot`,
  must exit 0). Tier map in `pyproject.toml [tool.mypy]`: STRICT on `models`/`data`/
  `memory`/`config.py` — keep those at zero errors; `ignore_errors` (commented why) on
  `bot`/`voice`/`immersion`/`llm`/`game`/`main` until each gets its typing pass. Run the
  gate alongside the suite before committing typed-core changes.
- **Commit at every green checkpoint.** Branch: `audit-and-single-authority-refactor`. The
  git repo root is the PARENT of `Main/`, so use `git -C "<repo root>"`. The LF→CRLF
  warnings on commit are benign. `.gitignore` already excludes wav/mp3/model-weights/db —
  but always dry-run-check `git add -An` for stray binaries before a big commit.
- **Verify-don't-trust the audits.** They've been right on the *bug* but wrong on
  *specifics* — file:line citations drift, and a few "dead code" / "depends on X" claims
  were stale (e.g. turn_loop's "voice frontend depends on it" was false). Grep-confirm
  before deleting; read the real code before changing it.
- Baseline: **609 tests pass** (as of Step 1, 2026-07-05; was 506 at Step 0).
  Keep them passing after every step — and keep THIS number current when a
  step lands, or the plan becomes the stale doc it warns about (quality
  re-audit 2026-06-09 caught exactly that).

## Anti-re-flag rules (enforce in review)

- No business `if/elif` in the coordinator — new behavior = new handler/state/spec row.
- Collaborators import only `protocols.py` + data types; never the coordinator (import-linter).
- Single-writer: `WorldStateView` is read-only; only `store.apply()` mutates.
- Narration only ever reads post-`apply()` state; never mutates, never runs before `apply()`.
- Inject the RNG/dice and a clock; never freeze globals.
