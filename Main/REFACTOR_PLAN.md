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

### Step 2 — Narration Strategy: **DONE** (2026-07-06, commits `901553c` pins + `7c056e0` + `06f35e8`, 631 tests green)
- **Pin first (per the rule):** `901553c` golden-pinned the three paths' prompt-assembly
  inputs through the narrator-client seam BEFORE any production change — a
  sentinel-per-field "does it reach the prompt" table per path (the audit's drift table
  in executable form), message role shapes, player_action decorations, exact
  chat/stream/followup kwargs, and the only-B-streams fact. 618 green at the pin.
- **What landed:**
  - `llm/narration.py` — `NarrationSpec` (frozen dataclass: one narration turn as
    DATA — raw action, decorated player_action, per-path prompt + role, streaming
    permission, empty-prose policy) + `NarrationStrategy` (the shared skeleton ONCE:
    injected tier selection riding the Step-0 `_narrator_client_factory` seam →
    **context union via `dataclasses.replace`** → bookend/basic builder → spec prompt →
    tool reminder → chat/chat_stream → extraction → empty-prose policy → the audit-#20
    tool-followup leg → ellipsis fix). Collaborators are injected callables, so 13 unit
    tests drive it with fakes (`tests/unit/test_narration_strategy.py`).
  - The three `_narrate_*` methods are now thin spec builders ending in one
    `strategy.run` call; their hand-copied rebuilds + `_narrator_tool_followup` +
    the orchestrator's penalty constants deleted (orchestrator 3504→3323
    non-blank lines; 4081→3868 raw).
- **Drift verdicts** (full evidence in `06f35e8`'s body; every pin flip is one of these):
  - **BUG → unioned:** A's missing memory/history/summary/quests (all-paths docstrings
    claimed one architecture; session computes them every turn; the audit's named live
    cost); kg_context_yaml + narrative_memory (Step 2.75 computed them FOR the narrator,
    telemetry claimed `context_injected=True`, prompt never saw them — the KG→narrator
    bridge was dead at prompt assembly); character_stats (`_build_character_context`
    exists solely to feed the never-firing `<acting_character>` renderer; annotation
    tidied to `Optional[Union[str, dict]]`); last_turn_trace (built for the bookend
    "Last turn:" reminder that never rendered); A's missing truncation "..." fix
    (same 1500-token cap on all three).
  - **No prompt impact, unified free:** campaign/session ids, current_combatant +
    initiative_order (latent behind combat_state), recent_messages + party_status
    (shadowed aliases), combat_round (never populated — the "Current round: 0"
    cosmetic bug stays out of scope).
  - **INTENT → spec data:** streaming only on B (`allow_streaming`; stream carries no
    tools by design — followup is the recovery); prompt role (A=user mech outcome,
    B/C=system `###INSTRUCTION###`); empty-prose policy (A substitutes the mech hint
    and continues, B/C bail with placeholder + no effects); A's try/except hint
    fallback (kept at the call site — mechanics already executed, narration failure
    must not hide a committed purchase).
- **Lessons:**
  - `dataclasses.replace` IS the anti-drift mechanism: the strategy overrides only
    `player_action`/`player_name`, so "which fields does the narrator get" stopped
    being a per-path decision anyone can fumble. The audit's exact recommendation;
    it worked. Field OMISSION is no longer expressible without a spec knob — good.
  - Sentinel-per-field pins made the bug-vs-intent argument cheap: each verdict is a
    visible `False→True` diff in one table, not an assertion rewrite. Pin tables that
    converge (3 tables → 1) are themselves evidence the duplication died.
  - Telemetry lied: turn logs reported `kg_context_injected` for fields the prompt
    never contained. When auditing "is feature X live", check the PROMPT, not the logs.
- **What Step 3 (combat ModeMachine) should know:**
  - `NarratorBrain.narrate_outcome` (brains/narrator.py:112, driven by the combat
    coordinator at coordinator.py:1533 and 1710) is STILL a separate narration stack with its
    own 11-field rebuild — deliberately out of Step-2 scope. When Step 3 touches the
    combat loop, migrate that call onto a `NarrationSpec` too (pin first through the
    coordinator), then `Brain._build_messages/_build_bookend_messages` can go public
    or move out (the audit's "two access protocols for one component" note).
  - The strategy is constructor-wired in `DMOrchestrator.__init__` via bound-callable
    seams; a combat-mode narration turn should be a new spec construction, not a new
    method — "no business if/elif in the coordinator" applies to prompts too.
  - Narration now sends kg/narrative-memory/character-stats/last-turn-trace: local
    context windows are measurably heavier per turn — ~4.9KB on the B/C paths and
    ~16.5KB on the commerce path (KG-active end-game measurement; the "~1-2KB"
    first recorded here was an estimate, not a measurement). If a small local model
    regresses, the knob is upstream (what Step 2.75 computes), not a per-path field
    drop. (Follow-up: context-budget caps + local_dense 16k→24k, 2026-07-06.)
- **Correction (2026-07-06, Step-2 review):** `06f35e8`'s try-widening on the
  commerce path (A) wraps the whole `strategy.run` — the narrative-hint fallback
  now also covers failures in the context union, message builder, and tool
  reminder, not just tier selection as the commit message implied. Recorded here
  since commit messages are immutable history.

### Step 3 — Combat ModeMachine: **DONE** (2026-07-06, branch `step-3-combat-modemachine`, commits `975365b`..review-fixes, 668 tests green)
- **Pin first, three times (per the rule):**
  - `975365b` — the combat-round net the audit demanded: coordinator.py (1586
    lines) had ZERO direct round coverage. One golden-master trajectory
    (start_turn → longsword attack with the modifier math pinned via recorded
    roller REQUESTS → end_turn → scripted NPC scimitar turn → round wrap →
    killing blow → first-class combat_over) + miss/blocked-condition/surprised-
    NPC edges. Deterministic via scripted roller + stub SRD + fake inventory —
    faces scripted, structure exact (the plan's scrub-rolls rule).
  - `a70c76c` — combat-narration pins through the coordinator at the
    narrator-client seam (message shape, `[]:`-prefixed decoration, exact
    kwargs, no-followup, batch semantics).
  - `e8218a8` — the three combat-entry signals pinned end-to-end (two of them
    pinned BROKEN with flip arrows). Also fixed a Step-0 net leak: the attack
    pin had been leaking a CombatManager at `discord:99` into every later test.
- **What landed:**
  - **Combat narration (the 4th path) → NarrationSpec/Strategy** (`0ed3181` +
    `241a767`): `NarratorBrain.narrate_outcome` + `_format_outcome` deleted
    (AST-span-by-name); the coordinator's `narrate_result`/`narrate_turn_results`
    are thin spec builders over a strategy instance with no-op collaborators.
    Two new spec knobs (data, not branches): `enable_tools=False` (no tools
    kwargs, no tool reminder, no followup leg — combat effects are owned by the
    combat engine) and `think` pass-through (None = kwarg not sent, so the
    orchestrator paths' pinned kwargs stayed byte-identical). Verdicts:
    anti-repetition penalties DRIFT→unified (the stack predated them; pin
    flipped); think=False INTENT→spec data (Qwen3 truncation); the 11-field
    rebuild → `replace()` carry-all has NO prompt impact (the coordinator-built
    context populates only fields the rebuild carried, and `world_state_yaml`
    stays empty so the basic builder still fires — review-verified); ellipsis
    fix unified free. `Brain._build_messages/_build_bookend_messages` →
    public `build_messages/build_bookend_messages` (they ARE the cross-module
    prompt-assembly API; the audit's "two access protocols" note).
  - **`game/modes.py` ModeMachine** (`7721521`): pushdown stack, EXPLORATION as
    the un-poppable base; mode VALUES deliberately, not per-mode classes —
    nothing consumes per-mode policy yet, and the machine's surface won't change
    when a consumer appears. `GameSession.enter_combat_mode/exit_combat_mode`
    are the ONLY writers of the flip and its derived surfaces (state
    COMBAT/ACTIVE, combat_manager, world_state.phase). `end_combat` = the pop
    (audit #77's "long-term this is the Step 3 pop"); the dead
    `GameSessionManager.enter_combat` twin deleted; `process_message`'s inline
    `combat_triggered` branch deleted. Phase now follows the mode IMMEDIATELY
    on push/pop; the per-turn phase sync STAYS as reconciler for the OTHER
    phase writer (StateDelta.phase_change) — phase single-ownership is Step 4's.
  - **`game/combat/encounter.py` EncounterBuilder** (`3f43b93`):
    `_trigger_combat` + guess/group/CR helpers moved out of the LLM layer
    (audit #91) — byte-identical per hostile review except the declared seams.
    `start_encounter` ends in the mode push (resolving the moved code's own
    "State transition should be handled by session manager" note) = the ONE
    combat-entry decision point (audit #89). All three signals funnel through
    it; the narrator `start_combat` tool effect — which previously set the flag
    with NO encounter, flipping sessions into COMBAT with no CombatManager —
    now drafts the scene hostiles like the extractor path and REFUSES to
    trigger on an empty scene (BUG→fixed; both pinned-broken tests flipped).
    The already-exists early return now ADOPTS the registered combat
    (`enter_combat_mode(existing)`) instead of only reporting True — one small
    dent in the three-stores drift (audit #79).
- **Adversarial review (found real things again; all fixed in-range):**
  - `Main/test_combat.py` (root harness, never collected by pytest) still
    called the deleted `DMOrchestrator._detect_group_count/_singularize_name`
    statics — the "zero underscore references" sweep missed root scripts.
    Import swapped to the encounter module; harness re-run green.
  - The move had silently reordered `session.combat_manager` assignment to
    AFTER initiative/start. Resolved by going further, deliberately:
    registration AND the mode push now both happen only after a fully started
    encounter, so an exception mid-build leaves no half-state anywhere (the
    b7f8262 code stranded a SETUP-state combat in the registry that the adopt
    branch would then resurrect).
  - The strategy's context-budget warning was reserving the 5k tool-schema
    overhead on the no-tools combat path — overhead now rides only when
    `spec.enable_tools`.
  - **Correction to `3f43b93`'s message** (immutable history): "voice paths
    keep today's behavior" is wrong about the OLD behavior — the old
    START_COMBAT effect set `combat_triggered=True` even with no session; new
    code returns False there. Review traced every consumer: the only
    sessionless `process_action` caller reads `.narrative` only, so the
    True→False flip has zero observers. The fix stands; the sentence didn't.
- **Net find (not audit-named):** the unarmed-strike fallback declares
  `damage_dice="1"`, which DiceRoller rejects as notation — every unarmed HIT
  errors AFTER consuming the action. Pinned broken with flip arrows
  (`test_combat_round.py`); the fallback exists in TWO coordinator sites
  (`_get_weapon_for_attack`, `_get_equipped_weapons`). Fix spun off as its own
  slice — commerce-rule: separate defect, separate change.
- **Deliberately out of scope:** the 4-way bot-layer turn-loop driver
  consolidation (audit #87 asked Step 3 to absorb it). It is bot-layer work
  (cogs/views import only under the venv; the suite cannot hold the net for
  it) and deserves its own step with a GameFrontend-shaped seam. Also left:
  channel_id-vs-session_key keying in the encounter registry (pre-existing
  migration debt, moved as-is).
- **Lessons:**
  - "Zero references" greps must include the ROOT harness scripts, not just
    the package + tests — pytest's `testpaths` blinds the suite to them, and
    that is exactly where this wave's real breakage hid.
  - The audit's "three deciders" framing made the fix shape obvious: don't
    build three fixes, build ONE entry point and make the deviant signal
    (START_COMBAT) conform to the participant logic the other two shared.
  - Publish-last ordering (register/push only after a fully started encounter)
    fell out of the review for free — when a move reorders side effects, either
    restore the order or make the new order strictly better AND say so.
  - A mode machine without per-mode policy consumers is write-only scaffolding;
    that is fine and honest — its value TODAY is single-ownership of the flip,
    not polymorphism. Resist growing state classes until a consumer exists.

### Step 4 — Single-writer WorldStateStore: **CORE DONE** (2026-07-06, branch `step-4-worldstate-store`, commits `3b943f1`..review-fixes, 693 tests green)
- **Scoped by inventory, not by audit citation:** two fan-out maps (every
  WorldState WRITER, every READER) before any design. The decisive finding:
  all three writer families live in suite-testable code — session.py (8
  sites), the one `apply_delta` call site, and the 11-branch effect-sync
  chain. Nothing in bot/ or voice/ writes world state; the KG bridge only
  reads; no reader holds a long-lived reference and deltas apply
  post-narration. Unlike Step 3, this strangle had no venv-only blind spots.
- **Pin first:** `3b943f1` — 19 pins, one exact WorldState diff per sync-chain
  branch. CORRECTION recorded in `ada5644`: the net commit's "NO direct
  coverage" was overstated — CHANGE_LOCATION/UPDATE_ENTITY/REF_ENTITY already
  had 13 tests; the other 8 branches were bare.
- **What landed:**
  - `game/world_store.py` (`ada5644`): `WorldStateStore` wrapping one
    WorldState. `apply_effect` = the sync chain moved VERBATIM out of the
    orchestrator (deleted by AST span, 227 lines; review verified the move
    line-by-line — four deltas, all declared). `apply_delta` = the extractor
    pipeline's seam, where Step 5's dedup pass slots in (anti-re-flag rule:
    dedup runs INSIDE apply). `GameSession.world_store` derives per access,
    so `world_state` reassignment can't orphan a stale wrapper.
  - Session bookkeeping (`5d7d271`): `begin_turn` (turn counter + party
    snapshots), `reconcile_phase(in_combat)` — ONE method now serving the
    ModeMachine push/pop phase writes AND process_message's per-turn
    reconcile (their guard semantics were already identical; narrative
    phases like dialogue survive outside combat), `add_established_fact`.
    session.py's only remaining world-state write is construction.
  - Single-writer GUARD (`8ed27ae` + review fix): an AST-scanning test bans
    WorldState mutating calls / field assigns / container mutations outside
    the store across dnd_bot/ AND the Main-root harness scripts — the
    import-linter contract the plan wants, in test form. AST not regex: the
    regex draft's first catch was a docstring (which was itself stale,
    pointing at the deleted chain — fixed).
- **Adversarial review (real findings again; fixed in-range):**
  - `test_eval.py`/`test_harness.py` hand-rolled combat auto-resolve with
    direct `state`/`phase` writes — outside the guard's original fence, in
    exactly the files where Step 3's breakage hid. Both now route through
    `exit_combat_mode()` (which also clears the `combat_manager` reference
    they never dropped), and the guard's fence now covers `Main/*.py`.
  - Declared delta the review confirmed test-covered: `add_established_fact`
    gained an `if fact` empty-guard the old inline loop lacked — a semantic
    improvement, not a verbatim move; recorded here per the exact-diff
    standard.
  - Guard heuristic gaps (aliased receivers, setattr, del, tuple targets):
    probed empirically, none exist in the code today; the receiver-name
    heuristic is declared in the guard's docstring. WATCH ITEM: sub-object
    NPCState mutation through retained `_find_npc()` references is the
    likeliest future bypass (bridge.py holds such references, all reads
    today).
  - Coverage note: the orchestrator's call-site no-session guard
    (store-resolution before the effect loop) has no direct test — the old
    in-method guard pin was replaced by store-derivation pins; safe today
    because `set_session` only ever receives a real GameSession or None.
- **Deliberately deferred (with reader-inventory evidence):**
  - **WorldStateView / reader migration:** every reader is snapshot-safe
    today (to_yaml serialization, pre-filtered lists, no cross-await holds),
    so a read-only facade would be ceremony without a threat model; the
    write-side guard enforces the invariant that matters. Revisit when
    Step 5 puts policy inside apply or if async boundaries shift
    (matcher.scene_seeds is the first candidate consumer).
  - **Stage B (#6/#7 per-turn TurnContext + per-session lock):** the audit
    itself marks it biggest/riskiest and not urgent behind the global lock.
    Belongs with Step 6's thin-coordinator work, not here.
  - **ROOT-3 (WorldState/combat serialization + recover parity):** the store
    is its natural owner (save/load as store methods) but it is an
    independent feature slice with schema decisions — its own branch.
    **DONE** — see the ROOT-3 entry below (2026-07-06).
- **Lessons:**
  - Map writers AND readers before designing a store: the reader inventory
    is what licensed deferring the view — scope cut by evidence, not by
    fatigue. The two fan-out maps cost minutes and shaped every slice.
  - An AST source-scan is the cheap stand-in for import-linter, and building
    it immediately pays: its first run caught stale documentation, its
    review probe proved it non-vacuous (15 hits against the pre-branch
    orchestrator).
  - Same lesson, third wave: the Main-root harness scripts are part of the
    codebase even though pytest can't see them. Fences and sweeps must
    include them BY DEFAULT now.

### Step 5 — Dedup into the write pipeline: **DONE** (2026-07-06, branch `step-5-dedup-in-apply`, commit `f25ee13` + review cleanup, 695 tests green)
- Both dedup passes left the orchestrator god object and became store-owned
  write-pipeline steps — the anti-re-flag rule ("dedup runs as a step inside
  apply(), never as an event or a coordinator method") now holds:
  - **Extractor side:** `WorldStateStore.apply_delta` is async and runs
    `_dedup_delta` (the moved `_dedup_extractor_new_npcs`, review-verified
    verbatim) BEFORE the write — dedup → validate → write behind one seam.
    The new_npcs/roster gating moved inside; with nothing to judge, the
    judge is never consulted.
  - **Narrator side:** `WorldStateStore.dedup_effect` (the moved
    `_dedup_rewrite`, verbatim) is the pre-execution step of the effect
    pipeline — ADD_NPC→REF_ENTITY rewrite before validation/execution, now
    self-gating (two new tests prove pass-through never consults the judge
    and preserves object identity, so idempotency keying is untouched).
    Its alias write onto the existing NPCState was a world-state mutation
    on the orchestrator — the exact sub-object bypass class the Step-4
    review flagged as its watch item; store-owned now.
- **No new pins needed:** both passes already had strong nets (7 + 8 test
  sites); the move was repoint-style with every assertion preserved
  verbatim (review-diffed), one test upgraded to drive the real choke
  point (`await apply_delta(..., narrator_prose=...)`), and the dead
  orchestrator-builder helpers/mock imports deleted from both test files
  (the review caught the second file's half-finished cleanup — CRLF line
  endings had silently defeated a bulk edit; check replace counts, not
  script exit codes).
- **Adversarial review: APPROVE.** Verified empirically: verbatim bodies
  incl. all eight log event names; async flip fully propagated (exactly
  one production caller, all callers await, keyword-only narrator_prose —
  no orphaned-coroutine risk); delta object identity through the rebind
  (registry sync + KG bridge see the deduped delta); the effect-side gate
  now resolves via `session.world_store` — equivalent for real sessions
  and CONSISTENT with the Step-4 sync seam (a duck-typed session exposing
  world_state but not the property would lose both seams together, not
  one).
- Orchestrator: −156 more lines; its dedup surface is now two one-line
  store calls.

### ROOT-3 — World serialization + session recovery: **DONE** (2026-07-06, branch `root-3-world-serialization`, commits `3875c72`..`803a8fa`, 723 tests green)
- **Audit-vs-reality (verify-don't-trust, again):** the audit's fix shape —
  "recover_sessions is a thin drifted subset; make both call the same init
  helpers" — was STALE. Quality P0-10 (`ffc1b1b`, 2026-06-09) had already
  deleted recover_sessions/load_active_sessions/the snapshot repo methods and
  installed a blanket `end_stale_sessions` startup sweep. This slice BUILT
  resume properly rather than patching a deleted twin. Lesson: audits go
  stale on the FIX SHAPE, not just file:line — re-derive the shape from HEAD
  before planning a slice.
- **Pin first:** start_session had ZERO direct coverage; 6 pins (`3875c72`:
  registration/identity, fresh WorldState, exact save_session row, memory
  warm, KG load + Chroma sync incl. empty-skip, NPC preload field mapping,
  both failure isolations) written before any extraction and passed through
  it untouched — the behavior-neutrality evidence for the start path.
- **What landed:**
  - Repo (`4481a22` + `a082f5c`): the migration-001 `session_snapshot` table
    gains its first writers. `save_world_snapshot` = single-statement
    `INSERT OR REPLACE` under stable id `world:<session_id>` — one row per
    session (per-turn saves can't grow the table) and no multi-statement
    window: the shared aiosqlite connection means any interleaved `commit()`
    between awaits commits half a DELETE+INSERT; a single statement with the
    invariant in the KEY closes that torn-write class. `get_latest_snapshot`,
    `load_active_sessions` (newest-first, `rowid` tie-break — `started_at`
    is 1-second resolution).
  - Store (`defed54`): `to_snapshot`/`state_from_snapshot` — the store owns
    the wire format (it already owned every write); the session layer owns
    when/where. Round-trip tests drive dict→JSON→dict→WorldState over every
    field family and assert model_dump equality.
  - Session layer (`a0a5d63`): start_session's KG-load+Chroma-sync and
    scene-NPC-preload blocks → `_load_knowledge_graph`/`_preload_scene_npcs`
    (verbatim moves, review-verified byte-identical modulo the campaign_id
    binding). `_persist_world_snapshot` once per processed turn, after the
    extractor delta + effect sync + fact sync have all landed (review
    confirmed: no WorldState write escapes the persist point; the only
    out-of-turn writer is the combat-mode phase flip, neutralized at
    recovery). Envelope v1: world state + membership ids/names + is_dm +
    dm_user_id + session_key + guild_id; characters by id ONLY — the DB row
    stays authoritative (DF-1 lesson). persist_failed policy: log, never
    break the turn. `recover_sessions` rebuilds per row (world via
    state_from_snapshot, membership re-fetched fresh, guild_id from the
    CAMPAIGN row — DF-7) then runs the SAME init helpers (DF-16/N10 parity);
    unrecoverable rows (no snapshot / unknown version / corrupt / missing
    campaign / duplicate key / foreign frontend) are ended PER-ROW.
  - Bot wiring (`2316c66`): setup_hook recovers + rebuilds the
    active-campaign map (newest session wins a contested guild).
    `end_stale_sessions` DELETED, not kept as a backstop — run after
    recovery it would end exactly the rows just recovered; its job is done
    per-row inside recover. Its 2 pins retired by name.
- **Declared non-goals** (each logged where it bites, all in the
  recover_sessions docstring):
  - **Mid-combat resume** — NOT because the combat tables drifted
    (CombatState's values still match the schema CHECK exactly) but because
    a live encounter is bot-layer driver machinery (manager registry, turn
    coordinator, per-channel locks, Discord views) the suite cannot net;
    a recovered COMBAT session with no driver would be wedged. Combat rows
    downgrade to ACTIVE/exploration via the store's reconcile seam.
  - **Voice/web recovery** — no lifecycle owner in this process (see F1).
  - **Memory-tier crash amnesia** — tiers persist only at graceful
    end_session; a crash loses buffer/summary since the last graceful end
    while the WORLD is current. Follow-up: per-consolidation persistence.
  - **Extractor-minted NPCs** re-enter the scene registry only on next
    reference; durable registry membership is Stage-C canonical-id work.
- **Adversarial review (7 findings, no P0 — every wave finds something
  real, again):** F1 fixed — recovery ended up IMMORTALIZING voice/web rows
  (end_session reaches only `discord:` keys; the voice API is a separate
  process that never recovers; /api/game/end 404s after restart), while the
  deleted sweep had been the only bound on that population: foreign-frontend
  rows are now ended per-row, test flipped visibly. F2 fixed (guild map:
  oldest session's campaign was winning — reversed iteration). F5 fixed
  (same-second ORDER BY tie could keep the older husk in the duplicate-key
  sweep — rowid tie-break + pinned). F7 fixed (envelope construction moved
  inside its try; the docstring promised no-turn-breakage the code didn't
  fully deliver). F3/F4 declared (above). F6 no-fix with evidence: legacy
  uuid-id snapshot rows require a DB that ran this branch's unmerged
  intermediate commits; none exists — a perpetual cleanup DELETE for an
  empty population is overcorrection.
- **Stage-C facts this wave established (don't re-derive):** the
  orchestrator's per-turn `sync_to_npc_repo` (orchestrator.py:~2864) is
  DEAD — gated on `skip_entity_extraction`, which is always true now that
  world_state is always present; the scene registry reaches the npc DB only
  at graceful end_session. So recovery's registry preload can only ever see
  end_session-persisted NPCs (the F4 gap). Both belong to Stage C alongside
  the canonical-id decision.
- **Lessons:**
  - When a slice DELETES a blanket safety mechanism, inventory what
    unrelated populations it was bounding. F1 existed precisely because the
    startup sweep had been the only terminator for voice/web rows, and this
    branch removed the sweep while making those rows recoverable forever.
    Ask "what else did the deleted thing do?" before celebrating the
    per-row replacement.
  - A shared async DB connection makes every multi-statement write a
    torn-write risk (interleaved commits between awaits). Single-statement
    upserts with the invariant in the key beat transactions you don't
    actually control.
  - Shared fixtures ARE the parity argument: the start pins and the
    recovery tests drive the same fakes from one conftest, so "recovery
    runs the same init" is checked by construction, not by prose.

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
- Baseline: **723 tests pass** (as of ROOT-3, 2026-07-06; was 695 at Step 5, 693 at Step 4 core, 668 at Step 3, 631 at Step 2, 609 at Step 1, 506 at Step 0).
  Keep them passing after every step — and keep THIS number current when a
  step lands, or the plan becomes the stale doc it warns about (quality
  re-audit 2026-06-09 caught exactly that).

## Anti-re-flag rules (enforce in review)

- No business `if/elif` in the coordinator — new behavior = new handler/state/spec row.
- Collaborators import only `protocols.py` + data types; never the coordinator (import-linter).
- Single-writer: `WorldStateView` is read-only; only `store.apply()` mutates.
- Narration only ever reads post-`apply()` state; never mutates, never runs before `apply()`.
- Inject the RNG/dice and a clock; never freeze globals.
