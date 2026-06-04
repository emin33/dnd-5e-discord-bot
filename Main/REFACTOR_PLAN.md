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

## Pre-net cleanup (safe to do NOW, before the net — reduces surface)

Pure dead-code deletion is the safest change (grep-confirm zero usages + suite green) and
it removes *parallel implementations* the decomposition would otherwise have to reason
about. From the quality audit's quick-wins:
- Delete grep-confirmed orphans: `game/combat/turn_loop.py`, `game/mechanics/effects.py`
  (+ `create_*_effect` factories), `views/CombatTurnManager`, streaming-TTS subsystem,
  duplicate `_build_mechanics_embed`/`split_text` in `game.py`, empty `llm/prompts/` &
  `llm/tools/` packages, `Character.proficiencies`/`CharacterProficiency`, etc.
- Either wire `image_comfyui.py` into the factory or delete it (config advertises
  `"comfyui"` → ValueError today).
- Unused imports; hoist in-function `import structlog`/`uuid` to module level.
- **DEFER** "rarely-hit fallback" deletions (INTENTS mini-language + adjudicator, ~650
  lines) until the net exists — confirm-then-delete, not blind.

Also finish here: **Stage A.2 remainder (#2)** — `_get_save_modifier`/`_is_concentrating`/
`_check_concentration` in `coordinator.py` still read `_character_cache` directly,
bypassing the session-first `_get_character`; and **double end-of-turn processing (#3)**.

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

## Anti-re-flag rules (enforce in review)

- No business `if/elif` in the coordinator — new behavior = new handler/state/spec row.
- Collaborators import only `protocols.py` + data types; never the coordinator (import-linter).
- Single-writer: `WorldStateView` is read-only; only `store.apply()` mutates.
- Narration only ever reads post-`apply()` state; never mutates, never runs before `apply()`.
- Inject the RNG/dice and a clock; never freeze globals.
