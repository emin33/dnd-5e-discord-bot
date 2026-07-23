# Longform quality-test readiness

Status: preliminary architecture findings, 2026-07-21

This note separates changes that are safe to land now from state-ownership
work that needs pinned tests before production behavior changes.

## Scope of the narrative/memory suite

Combat is not part of the longform narrative gate. An incidental combat uses
the explicit `simulate_victory` harness policy: the production combat teardown
owner ends the encounter, the run continues, and the intervention is written
to both the turn record and manifest. `--combat-policy fail` is available when
a scenario should prohibit combat entirely. Combat mechanics remain a separate,
deterministic test suite.

A green longform process must mean all of the following:

- every requested turn produced a response;
- no loop, orchestrator, combat-policy, or player-driver failure was hidden;
- the seed was real rather than a fallback;
- every deterministic recall assertion passed;
- the process exited zero. Partial, untrusted, and invalid runs exit nonzero.

The existing `emergent_callback` scenario remains the 22-turn smoke test.
`deep_emergent_callback` is the 80-turn soak entry point: eight establishment
turns, sixty-two unrelated story/tool turns, then a ten-turn callback phase.
It is intentionally available but has not yet been run as part of this change.

`deep_seeded_callback` is the creative promotion candidate. It gives the
narrator a high-potential static campaign premise, a flawed player persona, and
six distinct dramatic acts. The simulated player carries a private rolling
continuity note so motives, loyalties, promises, possessions, and pressure can
survive beyond its recent-turn window. Actions use structured output and must
pass deterministic completeness, variety, and anti-stalling checks.

The seeded premise does **not** pre-populate the graph or domain database. The
callback target must be a separate NPC, object, or localized place invented by
the narrator after the run starts; fixed-premise elements are rejected as
untrusted. Turns 15-70 are a required washout interval in which player prose,
narrator prose, and injected KG context must all omit that seed. This preserves
the blank-slate accumulation test while giving the creative models better raw
material.

## State ownership: preliminary finding

The current turn order lets the state extractor run before narrator-declared
tools are applied. Both can describe the same mutation, while player state has
no extractor fallback at all. Adding player fields directly to `StateDelta`
would close the missing path but create a second writer for HP, spell slots,
currency, conditions, and inventory.

The safest design is a committed change receipt, not a larger uncoordinated
extractor:

1. The narrator proposes tool calls; validation does not mutate state.
2. Deterministic action handlers and valid narrator tools commit first.
3. Each successful mutation returns a receipt containing canonical field paths,
   entity IDs, before/after values, and a stable turn/tool idempotency key.
4. The extractor receives the post-commit state plus the receipt's covered
   fields. It may fill only uncovered narrative facts.
5. Failed/rejected tools do not mark a field covered, so extraction may still
   recover a clearly established qualitative change.
6. Numeric player resources are never inferred from vague prose. Damage,
   healing, currency, slots, and quantities require a deterministic mechanic or
   a validated tool with explicit values.

This ordering resolves the tool/extractor race without silently preferring one
LLM output. It should be implemented behind focused conflict tests before the
pipeline order changes.

## Memory graph convergence: preliminary finding

WorldState, the scene registry, the knowledge graph, Chroma, pinned facts, and
the turn log currently receive related changes through separate paths. The
graph should be a projection of committed domain changes rather than another
writer.

Recommended seam:

- emit one canonical `ChangeSet` after a turn commits;
- project that same event into WorldState snapshots, KG edges/nodes, vector
  embeddings, pinned-fact policy, and telemetry;
- record projection status and rejection reasons per target;
- retry projections by the stable change ID without re-running game mechanics;
- rebuild derived stores from the event stream/snapshots in a restart test.

Before landing this, pin adversarial cases for duplicate NPC aliases, failed
tool validation, extractor/tool conflicts, location transitions, entity death,
and a crash between the authoritative commit and graph projection.

## Rest feature resources

The base `Character` model and database schema do not contain Second Wind,
Action Surge, ki, Bardic Inspiration, Channel Divinity, or Wild Shape counters.
The previous rest implementation tried to assign these as unknown Pydantic
fields and then claimed that all class features recovered. The safe interim
behavior is now truthful: tracked spell slots recover, compatible extended
models can recover existing counters, and untracked features are not claimed.

Full support requires a durable feature-resource model rather than one column
per class. A suitable shape is `(character_id, resource_key, current, maximum,
recharge_rule, source)` with repository round-trip and migration tests.

## Proposed longform campaign matrix

One successful 22-turn callback proves a path, not reliability. Promotion to a
real product-quality gate should require repeated runs across these axes:

| Track | Shape | Primary gates |
|---|---|---|
| Tool reliability | 40-60 turns with explicit item, currency, NPC, location, condition, and spell-slot mutations | expected tool fires; receipt matches DB/WorldState; zero rejected/duplicate reapplication |
| Memory graph | 80-120 turns, multiple callbacks, aliases, location changes, and one process restart | callback retrieval; canonical identity; stale facts absent; projection convergence after restart |
| Creative continuity | 60-100 turns with open goals, reversals, NPC agendas, and callbacks | rubric judge plus deterministic continuity checks; no contradiction with committed state |
| Soak/reliability | matrix above repeated across seeds and narrator profiles | zero timeouts/exceptions; bounded latency/context; stable pass rate and cost |

Creative grading should never be the sole gate. Keep deterministic assertions
for tools, state, identity, recall, and lifecycle; use a rubric judge only for
narrative dimensions such as consequence, novelty, character voice, escalation,
and callback quality. Preserve the full prompt/tool/state trace for every
failed judged turn so failures are debuggable rather than just scored.

## Future sourcebook-style world packs

The long-term product should support installing a world pack that behaves like
buying a sourcebook: established lore, characters, factions, locations, maps,
items, quests, relationships, aliases, and discoverability metadata are loaded
before play. The LLM then performs selection, adaptation, and consequence more
often than invention from nothing.

Keep authored canon and campaign history as two layers:

1. Import a versioned, licensed content pack into stable canonical IDs with
   source/provenance metadata.
2. Project the same pack into the domain database, KG edges/nodes, and vector
   index through one idempotent importer; never maintain three hand-written
   seed paths.
3. Treat pack records as immutable base facts. Campaign changes live in an
   overlay/event stream: NPC state, destroyed locations, discovered lore,
   ownership, relationships, and player-created entities.
4. Resolve effective state as base canon plus campaign overlay, preserving
   uncertainty and secret/GM-only visibility rather than dumping the whole
   book into every prompt.
5. Make derived graph/vector projections rebuildable from pack version plus
   campaign events, and test pack upgrade/migration without overwriting play.

The promotion matrix should retain both tracks: blank-slate scenarios prove
that accumulation works, while pack-seeded scenarios measure the likely best
product experience and how well authored canon evolves under play.

## Remaining blockers before the expensive soak

- implement the committed mutation receipt and pin ordering/conflict tests;
- add durable feature resources before claiming class-feature rest recovery;
- add restart checkpoints and graph-projection convergence assertions;
- expand tool coverage beyond the current NPC-reference-heavy trajectory;
- cap/retract established facts and move synchronous vector work off the event
  loop so a long soak measures product quality rather than accumulated latency;
- establish a multi-run pass-rate threshold, cost budget, and p95 latency gate.
- migrate the Flash-Lite player driver from the end-of-support
  `google.generativeai` SDK to `google.genai` before it becomes a long-term CI
  dependency.

## 2026-07-22 session: first 80-turn soak + tool-omission repair

First `deep_seeded_callback` soak (manifest `20260722_230128`, $0.25, 80/80
turns): all six deep-recall assertions passed — the seed NPC survived a
55-turn washout and returned through player, narrator, and KG retrieval.
Effect execution reliability was 99.6% (269/270). Verdict was still FAIL on
four assertions, which decomposed into three defects:

1. **Narrator tool omission** (`covered=7/13`). Root cause confirmed in the
   turn logs: the generic mutation followup re-asks for tools without saying
   what was missed, and weaker narrators answer with more `ref_entity` calls.
   Fixed by the extractor-coordinated targeted followup
   (`dnd_bot/llm/state_followup.py` + orchestrator Step 3.6a + 
   `NarrationStrategy.targeted_state_followup`): after the applied StateDelta,
   high-confidence uncovered mutations produce one tool-only recovery request
   naming the exact missing calls (change_location / ref binding for
   materialized NPCs / update_entity). Conservative mirrors of the audit
   observer; delta already keeps state correct, so the leg only restores
   narrator-authored grounding. Validation run `20260722_232549`:
   `tool_omission_signal_coverage` passed for the first time (5/5).
2. **Identity split** (`Elara` twice in the graph; misbound
   `ref_entity(Lyra <- Elara)` for three turns). Chain: entity "the figure"
   got competing name claims (Lyra, then Elara) while a canonical Elara node
   existed; at T56 the returning Elara was re-created because the reanchor
   abstained and the extractor dedup judge failed open on an EMPTY qwen
   response. Two guards added: `EffectValidator._alias_canonical_conflict`
   rejects a ref whose alias is a different NPC's canonical name (production
   form of the `tool_reference_identity_grounding` gate), and
   `fuzzy_match_monster` now requires a shared token below 0.8 score (the
   soak turned NPC Elara into an SRD **Lamia** at 0.60 similarity). Remaining
   open: deterministic exact-name dedup should not depend on the LLM judge
   when the judge returns empty (fail-open path), and DeltaBridge could
   enforce `canonical_npc_identity_unique` at the write seam.
3. **Washout seed leak** (soak `leaks=[55]`, validation `leaks=[21]`): KG
   retrieval surfaced the seed entity once per run inside the washout window.
   Not yet investigated; recurring, low-magnitude, next in line.

Also fixed: `evaluate_tool_omission_signals` now uses `locations_equivalent`
so a base place vs its qualified sub-scene ("Tallow Rows" vs "Tallow Rows
alley") no longer demands change_location.

Validation run remainder: `tool_effect_execution_reliability` 95.9% vs 98%
gate — four fail-closed rejections of invented update_entity ids
(`courier`, `living-brass-compass` x3). Correct behavior, small-N gate
sensitivity; consider an id-resolution assist ("living-brass-compass" →
roster "brass-compass") only with exact-suffix conservatism.

## 2026-07-23 iteration: retrieval scoping + guard refinements + the naming-promotion root cause

Fixes landed and validated across runs 20260723_002014 / 20260723_003823:

- Vector-fallback KG seeds no longer amplify: they appear in context without
  BFS expansion (`get_context_subgraph(no_expand_ids=...)`) and no longer
  trigger narrative-episode recall (exact text matches only).
  `kg_kept_seed_out_of_irrelevant_context` passed in both validation runs
  after failing in the soak and the first validation run.
- `tool_omission_signal_coverage` passed in both runs (targeted followup
  active: change_location, ref bindings, update_entity all recovered live).
- Dedup judge runs with `think=False` (empty-thinking fail-open closed).
- Misbinding guard refined after a live false positive: only proper-named
  NPCs claim canonical ownership ("the apothecary" cannot veto an alias on
  the shop location); "apothecary" added to generic role terms.
- Omission gate skips store-rejected npc_updates ("NPC not found" for
  untracked background figures owes no narrator tool).
- New `unnamed_identity` followup signal: a strongly-cued proper name no
  store knows ("Elara Venn's eyes narrow") asks the narrator to resolve it
  (add_npc new person vs update_entity rename of a generic-labeled entity).

**Root cause now isolated — cross-store naming promotion.** Both remaining
identity failures (Elara Venn / "the apothecary"; Orris / "the older
woman") are the same event: a generic-labeled person acquires a proper
name. The scene registry resolves it correctly (merges, records alias).
The state extractor instead emits a NEW npc under the proper name; the
dedup judge accepts it (names look distinct); the graph gains a parallel
node; when the narrator's ref later legitimately promotes the generic node
("the older woman" -> "Orris"), `canonical_npc_identity_unique` collides.
Needed (design-level, next session):
1. Extractor-apply should consult the scene registry's identity resolution
   (it already merged the proper name onto the generic entity) before
   accepting a new_npc whose name the registry maps to an existing entity.
2. `kg.promote_entity_name` and DeltaBridge AddNode should enforce the
   proper-name uniqueness invariant at the write seam: a promotion or add
   that would create a second durable npc node with the same proper name
   must merge or abstain, never silently collide.
3. The dedup judge prompt should receive the scene registry's alias map as
   evidence, not just raw roster names.

Remaining known-noisy gate: `tool_structural_failure_budget` (5.7% and
8.3% vs 5% on ~110-effect runs) — the OLD generic followup leg's refs
without entity_id / non-verbatim aliases. Pre-existing model sloppiness,
worth revisiting once the targeted leg fully replaces the generic one.

## Soak #2 (20260723_005611): 24/26 — fixes hold at scale

Second 80-turn run, fresh seed (npc "Tomas Vex"), all fix batches active:
recall 6/6 again, and every previously-fixed gate passed at 80-turn scale
(omission coverage, identity grounding, canonical name uniqueness, KG seed
scoping). Two residual fails:

1. `narrator_kept_seed_out_of_memory_gap` (turn 61): the narrator
   spontaneously invented "Mira Vex, Tomas Vex's grandmother" as the plot's
   hidden figure mid-washout. KG context was clean; this is organic creative
   reincorporation via the narrator's own memory tiers — a desirable
   narrative behavior that nonetheless weakens the cold-recall evidence for
   that run. Left as a hard gate for now: across a multi-run matrix, clean-
   washout runs carry the recall claim; annotated-washout runs still verify
   consistency.
2. `tool_effect_execution_reliability` 97.7%: five fail-closed rejections,
   all narrator-embellished ids ('corvins-hallway',
   'drilled-silver-coin-ragpicker-token',
   'low-wooden-door-with-token-indentation'). Addressed post-run by
   `_resolve_invented_scene_ids`: when an unresolvable id strictly contains
   exactly one known entity's full token set (>=2 tokens), the effect is
   rewritten onto that entity; no unique containment keeps the rejection.
   Unit-tested against the live cases; next run validates in vivo.

Chroma per-turn embed/query work now runs off the event loop
(asyncio.to_thread across 9 hot-path sites; session._build_context is
async). Remaining queued: google.genai migration (venv install was deferred
while soaks ran), soak #3 for the pass-rate matrix, naming-promotion design
(chip).
