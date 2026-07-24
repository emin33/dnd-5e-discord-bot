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

## 2026-07-23 (later): cross-store naming promotion landed

The design from the previous section is implemented (commits 778898e +
d9225c4), all three seams, conservative/abstain-on-ambiguity throughout:

1. **Extractor-apply consults the scene registry.**
   `WorldStateStore.apply_delta`/`_dedup_delta` and `dedup_effect` take a
   `scene_registry`; a proposed NPC name that the registry's identity keys
   resolve (exact `resolve_unique_identity` bar, NPC-typed entities only)
   is crossed to the world NPC via the canonical `npc_id` link and
   rewritten deterministically — `new_name` promotion when a proper name
   lands on a generic label, `add_aliases` otherwise. Judge never
   consulted for these. Effect-path rewrites emit REF_ENTITY carrying the
   proper name as `ref_alias_used`, so the existing NamePromotion
   machinery does the KG rename.
2. **Proper-name uniqueness enforced at the KG write seam.**
   `_apply_add_node` merges a new NPC node into the unique existing
   holder of the same proper name (identity-key match on name+aliases) or
   abstains when ambiguous; `promote_entity_name` abstains on collision.
   Generic role labels exempt. Additionally (found live, run
   20260723_120152 T15): `promote_entity_name` abstains when the new name
   is a token-fragment of the current label — the narrator ref'd 'a Choir
   acolyte' with alias "Choir" and the node was renamed to the faction,
   misbinding the next turn's legitimate 'the acolyte' ref.
3. **Judge evidence.** The dedup judge prompt now receives the scene
   registry's alias map (`name`/`aliases`/`world_npc_id` rows) with
   guidance that those merges are authoritative.

Supporting: "older"/"younger" joined `_GENERIC_NPC_TERMS` (the live
label "the older woman" wasn't classifying as generic, which would have
blocked promotion). 1111 unit tests green (15 new pin the Orris,
Elara Venn, and Choir live cases), mypy clean.

Validation, two 30-turn targeted_relevance_callback runs:

- 20260723_120152 (seed "Orina Vex"): 23/26. `canonical_npc_identity_unique`
  PASS. `tool_reference_identity_grounding` FAIL via the Choir promotion
  hijack above — fixed by the fragment guard. Both washout gates failed
  as organic-reincorporation noise (soak-#2 "Mira Vex" class): the
  narrator minted a second "Orina" (mother of a missing son) colliding
  with the seed's first name while the player carried a sworn oath into
  the washout window.
- 20260723_121644 (seed "Vex Harlow"): 24/26. **Both target gates PASS**
  (`canonical_npc_identity_unique` collisions={},
  `tool_reference_identity_grounding` misbound=[]); washout gates PASS.
  Residual fails are documented noise: `tool_structural_failure_budget`
  5.5% vs 5% (pre-existing), and one `tool_omission_signal_coverage` miss
  (T29: extractor put names in the `id` field — 'Vex Harlow',
  'small-pouch-of-grey-ash' — updates store-rejected, narrator owed the
  tool; pre-existing extractor sloppiness class).

Neither run organically re-triggered the naming-promotion event (it is
intermittent); the registry-consult path is pinned by unit tests against
the live cases. Next: soak #3 for the pass-rate matrix should confirm
`canonical_npc_identity_unique` + `tool_reference_identity_grounding`
hold at 80 turns with the seams active.

## Soak #3 (20260723_122931, seed "Pell"): 24/26 — seams confirmed at 80 turns

Third consecutive soak with recall 6/6. With the naming-promotion seams
active: `canonical_npc_identity_unique` collisions={} — and the seams fired
live ('acolyte 1' promoted to Pell at T4; a duplicate Pell merged into the
canonical node at T61 via the narrator-reference merge). Reliability 99.7%
(336/337). KG and player washout gates clean.

Residual fails, both understood:
- `tool_reference_identity_grounding`: two gate artifacts, fixed — bare
  numerals now count as generic tokens ("acolyte 1" is a spawn-numbering
  label, and the catalog snapshot predates Step 4 promotion), and a
  generic/title alias ("Brother") is descriptive address, not an identity
  claim; 'brother'/'sister' joined the generic vocabulary.
- `narrator_kept_seed_out_of_memory_gap` (7 turns): the narrator kept Pell
  on-stage into the washout — organic continuity (Mira-Vex class, stronger).
  The player and KG gates stayed clean, so the recall core holds; run-level
  cold-recall evidence is weaker. Matrix interpretation: count clean-washout
  runs toward the recall claim rather than forcing the narrator to drop
  threads.

Soak matrix so far (deep_seeded_callback, deepseek_v4_flash_qwen9b):
| run | seed | score | recall | identity unique | omission | notes |
|-----|------|-------|--------|-----------------|----------|-------|
| 20260722_230128 | Sera Vellik | 22/26 | 6/6 | FAIL (pre-fix) | FAIL (pre-fix) | baseline |
| 20260723_005611 | Tomas Vex | 24/26 | 6/6 | PASS | PASS | pre-seams |
| 20260723_122931 | Pell | 24/26 | 6/6 | PASS | PASS | all seams active |

## 2026-07-23: third pillar closed — narrative-quality grader

`test_narrative_grader.py` grades a completed run's turn log offline with
an independent judge (Gemini 2.5 Flash, think=False — never the narrator
grading itself): rolling 6-turn windows with a carried story summary,
five dimensions (continuity, contradiction_free, npc_voice,
prose_freshness, player_agency), per-flag turn numbers, plus
deterministic prose metrics (8-gram cross-turn repetition, opening-bigram
variety). Hard gates per the plan: overall >= 4.0, no dimension mean
< 3.0, ZERO severe contradictions, repetition ratio <= 0.35, and judge
coverage fails closed. Artifacts land in data/narrative_quality/.
~$0.01/80-turn run.

First grades over existing artifacts:
| run | turns | overall | verdict | notes |
|-----|-------|---------|---------|-------|
| soak #1 (Sera Vellik) | 80 | 4.67 | PASS | freshness 4.21 lowest |
| soak #3 (Pell) | 80 | 4.70 | PASS | narrator mid-sentence cutoffs T8/T9/T26 flagged (NARRATOR_MAX_TOKENS symptom) |
| validation 20260723_003823 | 30 | 4.48 | FAIL | severe contradiction T22: wax soft/warm at T19 -> brittle "hours old" at T22 (review: continuity slip vs deliberate time-anomaly cue) |

The FAIL proves the judge is not a rubber stamp. Follow-ups it surfaced:
narrator truncation (prose hitting the 1500-token ceiling mid-sentence)
and the T22-class ambiguity between world-weirdness and contradiction —
both feed the fact-supersession design.

GeminiClient migration note: think=False now maps to
ThinkingConfig(thinking_budget=0) — 2.5 models think by default and a
small-budget JSON call otherwise returns truncated output (found live by
the grader's first run).

### Correction (same day): "narrator truncation" was a grader artifact

Root-caused via turn logs: the flagged turns' completion tokens were
356/282/505 — nowhere near the 1500 ceiling — and the real narration tails
end cleanly. The judge was grading the grader's own 1400-char excerpt
boundary. Fixed: excerpts now cut at sentence boundaries and carry an
explicit "[EXCERPT TRUNCATED BY GRADER]" marker; window excerpt budget
raised to 2600 chars. Soak #3 re-grades 4.77 with the phantom flags gone;
remaining flags are legitimate editorial nits. NARRATOR_MAX_TOKENS is NOT
implicated — retract that follow-up. The wax severe-contradiction FAIL
stands (full narrations were within excerpt limits).

## 2026-07-23: fact supersession landed

The wax contradiction (grader severe, run 20260723_003823 T22) traced to
the append-only fact ledger: state changes appended new facts while the
contradicted ones stayed live. Fix at the single-writer seam
(`WorldStateStore.apply_delta` -> `game/fact_supersession.py`): anchor-word
overlap gates candidates (recent-first, cap 8), the brain judge decides
supersession (think=False, temperature 0), default keep-both on any
uncertainty. Retired facts move to `WorldState.superseded_facts` with
{fact, superseded_by, turn} provenance — history preserved, prompts stop
seeing both sides of a contradiction. Live qwen3.5:9b calibration: wax
state change retires, movement retires the stale location, birth-history
correctly survives a new residence. 14 unit tests pin the seam.

### Supersession validated live (run 20260723_151427, seed "Lys Vane")

Best 30-turn result yet: 25/26 harness gates (sole fail: two sloppy
narrator aliases — "Warrens", a district name, attached to a person).
`fact_superseded` fired 15 times doing real work ("frost stopped
spreading" retired "frost is spreading"; corrected Wardens patrol
location retired the wrong one). Narrative grader: PASS 4.64 with ZERO
severe contradictions — the previous run of this same scenario FAILED on
the wax contradiction; contradiction_free mean rose 4.4 -> 4.8.
Watch-item: one borderline retirement merged a street-sigil fact into a
cellar-seal fact (shared spiral/seal anchors) — plausibly the story
connecting them, but the class is worth an eye in future soaks.
Remaining finding queue: prose freshness (consistently the floor,
4.0-4.2), retire the generic tool-followup leg, extractor id-field
sloppiness.

### Freshness hint + id-field fix measured (run 20260723_160347)

Run formally INVALID(player-error) — the Gemini test actor emitted a
5-word action at T26 (harness flake, not product) — but the narration is
genuine, so the grader measurement stands: **prose_freshness 4.4** (off
the 4.0-4.2 floor of every prior run), overall **4.84** (highest yet),
worst opener repetition 2x (was 6x). The cross-turn opening hint moved
the one dial that had never moved. Product gates otherwise 24/26 with
only the structural budget miss (7.1% — the generic followup leg, the
last queue item). No update_entity rejections of the id-field class.

## 2026-07-23: first fully-green run (27/27) with cross-store audit live

Run 20260723_162617 (seed "Tamsyn"): **PASS 27/27** — the first longform
run to clear every gate, including the new `cross_store_consistency`
assertion. The audit (dnd_bot/game/consistency_audit.py, run against LIVE
stores before teardown) reported zero violations, 15 live fact
supersessions, clean pinned facts, and the first-ever Chroma coverage
measurement: 8/8 described KG entities indexed. Narrative grade 4.8
(continuity 5.0, contradiction_free 5.0, zero severe), repeat ratio 0.0.

The audit exists because the "provably consistent?" question found a real
hole in minutes: the per-turn memory->world fact sync resurrected
supersession-retired facts every turn (fixed bidirectionally, 2b26065).
Remaining unproven path: restart/resume convergence — crash mid-run,
reload, assert the stores re-agree. That is the next audit to build.

## 2026-07-23 (later): generic tool-followup leg narrowed — ref grounding made deterministic

The last persistent gate miss, `tool_structural_failure_budget` (5.5-8.3%
vs 5%), is closed by making the generic followup's reference grounding
deterministic instead of trusting the model's `ref_entity` guesses.

**Corrected root cause.** The earlier framing ("the followup leg is the
tool recovery for streamed turns") is incomplete: streaming is OFF in the
longform harness (`Harness.send_action` -> `process_message` passes no
`on_narrative_token`), yet the deepseek narrator emits ZERO tools in its
primary prose call on essentially every turn (`primary_effects=0`), so the
generic leg fires on ~every turn (29/30, 72/80 in recent runs) and does
~all the tool work. `ref_entity` is ~3/4 of executed effects (66/94,
260/336) and ~3/4 of the structural errors (10x "requires entity_id", 5x
"alias not verbatim", vs 3x update_entity + 2x add_npc). The leg is
load-bearing, not a rare recovery — full retirement would collapse
`tool_effect_turn_coverage` (turns-with-nonref sit at only 65-67% vs the
60% gate even WITH the leg).

**Fix** (`NarrationStrategy._supersede_followup_refs`, wired into
`_tool_followup` before any drop-counting). Grounding a reference never
needed the model to get the id/alias right: every ref that survives
validation is reconstructable from prose + authoritative roster. So model
ref guesses became advisory. Fully-valid ref turns are returned
byte-for-byte (no augmentation — e.g. no invented current-location ref);
when any ref fails to ground, its intent is honoured the way an empty
`ref_entity({})` already was — deterministic recover-all-named via
`_recover_roster_references` — instead of charging the malformed call to
the structural budget. Mutation calls (update_*, change_location,
remove_entity, spawn_object, add_npc) pass through untouched with their
existing validation/repair, so the residual mutation-sloppiness class is
still measured. New diagnostic: `tool_followup_refs_superseded`. Streamed
turns still produce `ref_entity` — now deterministic and clean.

An adversarial multi-agent review of the change confirmed one real bug
before landing (fixed + pinned): a kept model ref carrying a display-name
id ("Bram" vs slug "bram-id") did not suppress deterministic recovery of
the same entity (exact-string dedup), so one mention could execute two
ref_entity effects and double-count the mention/importance signal feeding
name promotion. A metric-gaming claim (ungroundable invented-id refs no
longer charge the budget) was adversarially REFUTED — state output is
identical and the class stays visible in `tool_followup_refs_superseded` —
with one legitimate watch-item: that diagnostic feeds no gate, so a
regression flooding hallucinated refs would be visible but ungated;
consider a superseded-ratio budget if it ever trends up.

**Validation.** 1163 unit tests green (56 in test_narration_strategy: 2 new
pins, 2 updated to supersede semantics), mypy clean. Live:
`test_tool_reliability.py` (deepseek_v4_flash_qwen9b) — PASS 11/11,
44/44 effects executed (100%), followup fired 8/12 turns, structural
errors ZERO, repair turns ZERO, budget inputs dropped=1/45 = **2.2%**
(vs 5.5-8.3% pre-fix; the one drop is a primary-path ref to a same-turn
add_npc, out of the followup's scope), `tool_followup_refs_superseded`=1
doing real work, coverage 12/12. Narrative grade of the same run: overall
4.7, continuity 5.0, repeat ratio 0.007; its one severe-contradiction flag
is a T2 prose meta-leak from the resolved-outcome REPAIR leg ("I apologize
for the contradiction") — pre-existing repair-prose class, unrelated to
tool effects; worth its own small fix (strip meta-openers from repair
prose).

**Separate blocker found (pre-existing, now the top item):** four
consecutive longform callback runs aborted at the seed pick
("No trustworthy graph-backed emergent callback seed") — 2x
targeted_relevance_callback, 1x deep_seeded_callback, 1x
emergent_callback. In every aborted run the narrator told an
object-focused, NPC-less story (extractor `new_npcs=0` every turn,
`kg_npc_nodes=0`; seed candidates like "metallic residue" fail canonical
eligibility) while `targeted_relevance_callback`/`deep_seeded_callback`
hard-require `required_seed_type="npc"`. The supersession change is
exonerated: it was a verified NO-OP in those runs (superseded=0). This
narrator NPC-drought (model/premise drift vs the soak-era behavior) is
what currently blocks the npc-seeded scenarios, independent of tool
reliability.

## 2026-07-23 (evening): NPC-less-opening seed aborts root-caused + retry fix

Four consecutive runs (elegant-albattani worktree, 20260723_173643/174525/
175031/175539) aborted at the turn-9 seed pick. Root cause is a **story
lottery, not a regression**: with the identical Glasswake premise, prompts,
and code paths that seeded Pell/Lys Vane/Lena Harlow/Tamsyn earlier the same
day, deepseek-v4-flash sometimes opens an object-focused story in which every
human stays anonymous ("the courier", "a woman", "grey apron figure") for all
8 explore turns. The extractor faithfully reports new_npcs=0 (or
generic-named NPCs the eligibility filter correctly rejects), so
`_canonical_seed_candidates` is empty and `required_seed_type="npc"`
scenarios could not seed. Once an opening goes object-focused it
self-conditions and stays that way. The branch's supersession diff
(9c85e21) only touches followup ref grounding and was confirmed a no-op
here. The existing single turn-8 nudge fired in all three NPC runs but one
narration turn yields scenery, not a name exchange.

The fourth abort ("'metallic residue' (item) is not an exact eligible
canonical graph candidate", emergent_callback) was a different mode: eligible
candidates existed but Gemini picked off-list three times and `pick_seed`
raised.

Harness fixes (test_long_horizon.py):
- Seed pick no longer hard-aborts at turn 9. Up to `SEED_PICK_MAX_RETRY_TURNS`
  (3) extended explore turns force an escalated name-demanding player action
  and re-attempt the pick each turn; only then does the run abort.
- The soft pre-pick nudge now also fires at turn `seed_pick_after_turn - 1`,
  giving two chances to put a nameable person on stage before the boundary.
- `seed.chosen_after_turn` records the ACTUAL pick turn; washout-transition
  forcing, washout redaction, and the explore-window/gap assertions all
  follow it, so a late pick cannot fail `seed_appears_in_explore`.
- `pick_seed` falls back to the priority-sorted top canonical candidate when
  Gemini exhausts its 3 attempts against a non-empty eligible list.

## 2026-07-23 (night): merged-master validation + gate-classifier refinements

All three same-day fixes (ref supersession 9c85e21, repair meta-talk strip
4d70b39, seed retry-with-nudge 09ea374) merged to master; 1174 tests green,
mypy clean. First combined run, targeted_relevance_callback 20260723_223012
(seed "Sera Venn" — the retry fix seeded a scenario that had aborted four
times): 25/27, `tool_structural_failure_budget` **0.0%** (0/83, zero
structural errors), `narrator_no_meta_reasoning_leak` PASS, grader PASS
4.76 with zero severe contradictions.

Both residual fails were measurement gaps, fixed in 69a41b3:

- `kg_kept_seed_out_of_irrelevant_context` now separates retrieval that
  TARGETED the seed (graph yaml, seed/text/vector ids, chunk entity_ids)
  from the seed name riding inside ANOTHER entity's recalled episode (the
  player asked about the Chain Stair; the recalled turn-8 episode read
  "Sera points past the chapel..."). Episode co-mentions are annotated,
  not failed. T22/23 reclassify to co-mention; zero targeted leaks.
- `is_generic_npc_label` now classifies generic-noun + prepositional
  descriptor ("man in apron") as generic, abstaining on capitalized
  descriptor tokens ("man in Orin's shop"). The T12 "misbind"
  (man in apron <- Hesk) was a correct vocative naming promotion ("You
  talk too much, Hesk." ... "Hesk shrugs"); the identity gate's
  generic-label exemption now covers it.

Offline re-evaluation of 20260723_223012 with both refinements: 27/27.

Also added `tool_followup_supersede_budget` (<=15%): superseded followup
refs no longer charge the structural budget, so this separate bound keeps
raw ref-emission quality measured — a hallucinated-ref flood must fail
something even though every instance self-heals. Healthy runs measure
0-2%; the pre-fix noisy runs would have measured ~6-8%. This closes the
adversarial review's watch-item.

## Soak #4 (20260723_230351, seed "Gideon Hask"): 25/27 on merged master

First 80-turn soak with ALL 2026-07-23 fixes active (supersession, repair
meta-strip, seed retry, gate classifiers landed mid-run so evaluated with
the pre-refinement gates). Complete and trusted; recall 6/6; KG washout
clean; identity gates clean.

- `tool_structural_failure_budget` **3.2%** (7/220) at 80-turn scale —
  was 5.7-8.3% on comparable pre-fix runs. `tool_followup_refs_superseded`
  = 8 with 17 deterministic ref recoveries; the new supersede budget
  measures 3.5% (<=15%).
- Narrative grade **PASS 4.86 — highest soak grade yet** (4.67/4.70/4.77
  before), zero severe contradictions, continuity 5.0, freshness 4.43.
- Residual fails, both documented classes: `narrator_kept_seed_out_of_
  memory_gap` (single leak at T15, the washout boundary — organic
  thread-finish, Mira-Vex class); `tool_effect_execution_reliability`
  97.7% vs 98% — five rejections, ALL the invented/partial-id-at-
  execution class ('masked_courier', 'living-brass-compass',
  'orris-vanes-hidden-note', 'carved-wooden-door', ref 'gideon'). That
  id-resolution cluster (extractor id-field sloppiness + narrator
  embellishment at the store seam) is now the top open item.

### 2026-07-24: the soak-#4 rejection cluster diagnosed and closed

Traced all five `tool_effect_execution_reliability` rejections through the
turn log; two distinct defects, one correct rejection:

1. **Executor/validator asymmetry (T16/T23/T69, stage=execution).**
   'living-brass-compass', 'orris-vanes-hidden-note', and
   'carved-wooden-door' were NOT invented ids — each is a real KG item
   node with exactly that id (T16's is the twin of 'brass-compass' from a
   turn-1 double-spawn). `_is_known_entity` accepts scene items and
   any-type graph entities, so validation passed and
   `_resolve_invented_scene_ids` correctly left them alone — but
   `_execute_update_entity` resolved only scene-registry entities and
   NPC-typed world/graph entities. Fix: execution now falls back to
   `_resolve_known_world_reference` (items, locations, any graph type),
   identity-only, same no-materialization contract as the world-NPC path.
2. **Partial ids (T77, stage=validation).** ref 'gideon' for the tracked
   NPC 'Gideon Hask' (canonical UUID id). The rescue's strict-containment
   rule deliberately excluded subsets; `_canonicalize_npc_effect_ids`
   abstains because identity_keys('gideon') doesn't intersect
   {'gideon hask'}. Fix: `_resolve_invented_scene_ids` now also matches
   the inverse direction — proposed tokens a strict subset of exactly one
   known entity's name — under one shared uniqueness bar (a lone match
   across BOTH directions resolves; 'sera' with two Seras tracked, or a
   token shared by an item and an NPC, abstains).
3. **Correctly rejected (T1).** 'masked_courier' at turn 1: the courier
   existed only in seed prose — no NPC, no item, no graph node, and the
   turn's delta recorded only a fact. Fail-closed rejection is right;
   nothing to resolve onto.

All five live shapes pinned in `test_invented_id_rescue.py` and
`test_scene_entity_update.py`. Rerunning soak #4's tape, 4/5 rejections
now execute → hypothetical 212/213 = 99.5% (gate needs 98%).

Live validation (20260724_010137, targeted_relevance_callback, seed
"Tess Greymark", 30 turns): `tool_effect_execution_reliability` **98.8%
PASS** (80/81); the run's sole rejection is the alias-canonical-conflict
guard correctly refusing a T28 misbinding — the invented/partial-id class
had zero occurrences and the rescue fired zero times, so the 25/28 run's
three fails (seed-gap organic reincorporation, structural budget 6.9% on
a small 87-proposal denominator, and three narrator alias misbinds at
T25/T28/T30) are all pre-existing stochastic classes untouched by this
change. Off-scene `update_entity` executed 4x through the widened seam.

Soak matrix:
| run | seed | score | recall | identity unique | omission | grade | notes |
|-----|------|-------|--------|-----------------|----------|-------|-------|
| 20260722_230128 | Sera Vellik | 22/26 | 6/6 | FAIL (pre-fix) | FAIL (pre-fix) | 4.67 | baseline |
| 20260723_005611 | Tomas Vex | 24/26 | 6/6 | PASS | PASS | - | pre-seams |
| 20260723_122931 | Pell | 24/26 | 6/6 | PASS | PASS | 4.77 | naming seams |
| 20260723_230351 | Gideon Hask | 25/27 | 6/6 | PASS | PASS | 4.86 | all fixes merged |

## Soak #5 (20260724_013045, seed "Sera Venn"): 27/28 — reliability gate closed

First soak with the id-resolution fix. **Every tool gate passes at 80-turn
scale**, including the one that had never cleared:

- `tool_effect_execution_reliability` **98.7%** (224/227, 3 rejected) — was
  97.7% in soak #4 and the sole remaining product miss.
- `tool_structural_failure_budget` 2.6% (6/233); `tool_followup_supersede_
  budget` 10.0% (26/259); `tool_omission_signal_coverage` 7/7;
  `tool_reference_identity_grounding` misbound=[];
  `canonical_npc_identity_unique` collisions={}; `cross_store_consistency`
  violations=[] with 19/19 Chroma coverage and 2/2 world NPCs graph-backed.
- `kg_kept_seed_out_of_irrelevant_context` PASS with targeted_leaks=[] AND
  episode_comentions=[] — the refined classifier was not even load-bearing
  here.
- Narrative grade PASS 4.76, zero severe contradictions, repeat ratio 0.0.

Sole fail: `narrator_kept_seed_out_of_memory_gap` (turns 17-18) — the
organic-reincorporation class (Mira-Vex/Pell). Per the matrix
interpretation, clean-washout runs carry the cold-recall claim; this run's
player and KG washout gates are both clean.

### Adversarial review of the id-resolution change (post-merge)

The change had been merged on its reported result without code review. A
4-lens / 18-agent adversarial review found 12 confirmed defects; the three
HIGH ones were a single root cause, fixed here:

- **Inverse arm had no strength bar.** `(len(ct) >= 2 and ct < value) or
  value < ct` — precedence put the guard only on the forward arm, so a bare
  token (`door`, `man`, `brass`) rewrote onto whichever entity contained it,
  binding effects to unrelated entities and permanently poisoning alias
  lists. One-token ids now resolve only when the token is identity-bearing
  (>=4 chars, not a generic role noun, not a common object noun) AND the
  candidate is a proper-named NPC. `gideon` -> `Gideon Hask` still resolves.
- **Union regressed the soak-#1 fix.** Pooling both directions into one
  `matched` set let an unrelated superset count as a second match and veto a
  good containment resolution. Containment is now authoritative and tried
  first; subset is a fallback only.
- **Cross-type binding.** Candidates now carry entity type; an item-shaped
  id cannot rewrite onto an NPC.

Open (confirmed, separate seams — NOT fixed here): a LOCATION target accepts
NPC-only semantics so DeltaBridge stamps status/disposition onto locations;
`_resolve_invented_scene_ids` ignores `ref_alias_used`; non-NPC
`update_entity` populates a description/inventory `applied` receipt with no
writer behind it (and its dedup never fires, so it is non-idempotent);
`campaign_dead_npcs` is accepted by the validator but resolved by neither
executor helper.

Soak matrix:
| run | seed | score | recall | reliability | structural | grade |
|-----|------|-------|--------|-------------|-----------|-------|
| 20260722_230128 | Sera Vellik | 22/26 | 6/6 | - | - | 4.67 |
| 20260723_005611 | Tomas Vex | 24/26 | 6/6 | 97.7% | - | - |
| 20260723_122931 | Pell | 24/26 | 6/6 | 99.7% | - | 4.77 |
| 20260723_230351 | Gideon Hask | 25/27 | 6/6 | 97.7% | 3.2% | 4.86 |
| 20260724_013045 | Sera Venn | 27/28 | 6/6 | **98.7%** | 2.6% | 4.76 |
