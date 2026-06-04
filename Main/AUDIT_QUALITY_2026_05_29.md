# Code-Quality Audit Synthesis — D&D 5e Bot (~48K LOC, solo dev, late-alpha)

## 1. Overall Verdict

The system works and is unusually well-documented (most non-obvious decisions cite an audit number), but maintainability is dragged down by a recurring pattern: features were patched in repeatedly without consolidation, leaving god-objects, half-finished refactors, and multiple parallel/abandoned implementations of the same flow. The single largest liability is `DMOrchestrator` (4,783 lines, ~68 methods) concentrating the entire turn pipeline into one untestable class; around it sit several dead subsystems (streaming TTS, `turn_loop.py`, `mechanics/effects.py`, ComfyUI provider) that actively mislead. Tests are strong on pure logic and persistence round-trips but leave the orchestration core (`process_action`) and the production LLM-attribution path with zero coverage, while `strict=true` mypy is declared but unenforced (524 errors, not installed) — the green suite and strict flag both overstate real confidence.

| Area | Grade |
|---|---|
| LLM pipeline | C |
| Game/world (`game` + `memory`) | C+ |
| Voice/immersion | C+ |
| Bot/web | C |
| Data/models/config | C+ |
| Cross-cut: duplication & dead code | C |
| Cross-cut: tests & typing | C+ |
| **Overall** | **C / C+** |

## 2. Top 12 Highest-Leverage Fixes (globally ranked)

1. **`dnd_bot/llm/orchestrator.py:732` — test-quality.** Add an integration test driving `process_action()` with a deterministic fake LLM client + real WorldState/EffectExecutor (one attack, one social, one cast_spell). *Why: the single public turn entrypoint on a 4,783-line god-object has zero coverage of any kind; the suite is green but the heart of the system is unverified.*

2. **`dnd_bot/game/combat/coordinator.py:810` — consistency (correctness-adjacent).** Make all four sync sites (`_execute_spell:810`, `_get_save_modifier:1072`, `_is_concentrating:1084`, `_check_concentration:1100`) go through `_get_character()` or a session-first cached accessor. *Why: half-applied single-authority refactor — these read the private cache directly and get a stale copy or silently `None` (disabling concentration saves), re-opening the exact divergence the refactor closed.*

3. **`dnd_bot/game/combat/coordinator.py:203` — coupling (correctness-adjacent).** Pick one owner for end-of-turn effects: either `end_turn()` stops processing them or `next_turn()` does. Add a test asserting ongoing-damage ticks exactly once per cycle. *Why: `end_turn()` and the `next_turn()` it calls both run `process_end_of_turn_effects` on the same combatant — burning damage and repeating saves fire twice.*

4. **`dnd_bot/llm/orchestrator.py:521` — complexity.** Extract `NarrationService`, `CombatInitiator`, `ToolExecutor` (the 18 `_execute_*`), and `WorldStateSync` (`_sync_effect_to_world_state`, 3056-3287) as collaborators; orchestrator coordinates. *Why: 4,783 lines / 68 methods funnel every turn through one class — every change risks the whole pipeline and nothing can be unit-tested in isolation.*

5. **`dnd_bot/llm/effects.py:60` — coupling.** Convert `ProposedEffect` (50-field flat "god-DTO") into a Pydantic discriminated union per `effect_type`, and centralize type→handler wiring in one registry read by validation, execution, and world-state-sync. *Why: today every effect type is hand-wired in four parallel dispatch sites (`intents:343`, `narrator_tools:728`, `effects:686`, `orchestrator:3067`) that silently drift when one is forgotten — the direct cause of the no-op-stub confusion below.*

6. **`dnd_bot/voice/frontend.py:230` — coupling (latent crash).** Fix or delete `resolve_voice_combat()` — it builds `BrainContext(player_input=…, scene_context=…)` with kwargs that don't exist on the dataclass, so it `TypeError`s on the first voice combat command. *Why: the only consumer of the LiveKit voice-combat branch has never run; it's drifted dead code masquerading as working.*

7. **`dnd_bot/voice/tts.py:116` — dead-code.** Delete the entire streaming-TTS subsystem (`TTSSentenceQueue`, `RivaTTS.synthesize_stream*`, `split_sentences`) and the `add_token`/`_sentence_queue` wiring in `frontend.py`. *Why: ~120 lines fed tokens but never synthesized or consumed — implies low-latency streaming is wired when the real path is `_speak(narrative)`.*

8. **`tests/unit/test_immersion.py:16` — test-quality.** Test `parse_narrative_async` (the production path) and `dialogue_attributor.attribute_dialogue` (JSON/markdown-fence/null handling), not just the synchronous regex fallback. *Why: the LLM attribution engine plus `voice_resolver`/`tts_assembler` — the most complex immersion logic and the code that breaks in production — has zero coverage; tests validate only the path real requests rarely hit.*

9. **`dnd_bot/data/repositories/character_repo.py:532` — consistency (fragility).** Replace `SELECT *` + hardcoded positional indices (`row[27]`, `row[34]`) with explicit named columns or a Row factory across all `_row_to_*` methods. *Why: every read silently couples Python to exact migration column order — one mid-table column addition shifts every downstream field; this is the precise bug class the integration tests exist to catch.*

10. **`dnd_bot/bot/cogs/spells.py:369` — duplication.** Extract one `require_character(ctx) -> (character, campaign_id)` helper/decorator; replace the ~15 copy-pasted "resolve campaign + fetch character + two guards" blocks (the `# audit #52` comment appears 13×) and fold in the three divergent `_get_campaign`/`_get_campaign_id`/`_get_character` helpers. *Why: any change to guard wording or campaign-resolution rules is currently a 15-place edit.*

11. **`dnd_bot/llm/effects.py:891` — dead-code (trap).** Make `_execute_apply_damage`/`_apply_healing`/`_add_condition`/`_remove_condition`/`start_combat`/`set_flag`/`log_memory` either implement, or return explicit no-op/`was_handled` instead of `success=True`. *Why: registered in the live dispatch dict, they report success while mutating zero state — an APPLY_DAMAGE effect "succeeds" and changes nothing.*

12. **`pyproject.toml:71` — typing.** Resolve the strict-mypy fiction: either demote to `strict=false` with per-module opt-in, or keep strict, install mypy in the dev extra + CI, and burn down the 524 errors. *Why: `strict=true` is declared but mypy isn't installed and a single file yields 524 errors (53 attr-defined, 6 union-attr point at real bugs) — reviewers get false assurance the type system is enforced.*

## 3. Cross-Cutting Themes

**The orchestrator god-object (the dominant theme).** `DMOrchestrator` (4,783 lines) recurs in four separate area reports as *the* central liability — owning triage, three near-duplicate narration paths, mechanics, 18 tool executors, combat init, world-state sync, dedup, and scratchpad. It is untestable in pieces, a merge-conflict magnet, and its public `process_action()` has no test. Findings #1, #4, #5 all radiate from it. `coordinator.py` (1,580) and `combat.py` cog (1,700) are the same pattern one layer down.

**Parallel / abandoned implementations.** A consistent failure mode: a flow gets reimplemented, the old one is never deleted, and a docstring claims the dead one is canonical. Confirmed orphans: `game/mechanics/effects.py` (468 lines, zero importers — a whole second effect engine), `game/combat/turn_loop.py` (193 lines, never imported, docstring falsely claims both frontends use it), `immersion/image_comfyui.py` (207 lines, no factory branch, yet config advertises `"comfyui"` → ValueError), streaming TTS, `views/CombatTurnManager` (140 lines, dead but exported), the INTENTS mini-language + adjudicator (~650 lines, rarely-hit fallback), `_handle_sell` permanent stub. Combat-turn orchestration alone exists in **three** forms.

**Duplication clusters (the N-place-edit tax).** Distinct recurring clones: (a) the cog "resolve campaign + character" block (~15×); (b) `_build_mechanics_embed` + text splitter verbatim in two files, with ~140 dead lines in `game.py`; (c) the character-creation wizard duplicated ~380 lines between `character.py` and `campaign.py`; (d) per-provider boilerplate — TTS `synthesize_async` wrapper (6×), WAV/RIFF header (4×), no TTS/ASR base class, no `OpenAICompatClient` for the 3 OpenAI-shaped LLM providers (~100-line `chat()` bodies); (e) `slugify`/`_slugify_id` byte-identical in two modules; (f) str→`AbilityScore` map reimplemented 6+×; (g) monster stat-block parsing twice with divergent heuristics; (h) HP/progress-bar rendering 4 ways; (i) `DeathSaves` defined twice.

**Half-finished refactors as a category.** The single-authority `Character` (finding #2), the effect-consolidation (`update_player` vs legacy currency/damage fields coexisting), and `BrainContext`'s `party_members`/`party_status` + `recent_messages`/`message_history` alias pairs — all are partially applied, leaving two representations of one fact "kept in sync by hand" (and only staying in sync by luck, e.g. orchestrator setting both).

**Test-quality posture: strong core, unguarded edges.** Pure logic (combat models, dice, KG, dedup, memory buffers) and persistence round-trips are genuinely well-tested and mock-free. But the riskiest code is unguarded: `process_action` (zero), `EffectValidator`/`EffectExecutor`/`intents` parsing (no dedicated tests despite being pure and deterministic), the async LLM attribution path, voice resolution. Secondary smells: brain-stub trio copy-pasted across 3 test files (already diverging: `_MockResp` vs `_MockResponse`), tautological tests (`test_dice` nat-20 loop with `assert True`; `test_tiered_narrator` validates its own closure, never calls `load_profile`), white-box coupling to private internals (`_get_cache`, `MessageBuffer._*`), and ~11 root-level `test_*.py` scripts that pytest never collects — including a `test_combat.py` that shadows the real one and holds *unique* coordinator/multiplayer/concentration coverage the pytest suite lacks.

**Typing & docstrings drift.** Strict mypy unenforced (#12); key boundaries untyped (`EffectExecutor.__init__`, `BrainResult.proposed_effects`, `session.knowledge_graph: Optional[Any]`, bare-`dict` SRD monster shapes, untyped combat-helper params in `game.py`). Several class/module docstrings still describe the dead path as primary (orchestrator's PROSE+INTENTS flow, `narrator.py` "doesn't emit effects" when it now emits tool calls, Riva-specific voice docstrings on a provider-agnostic class, `coordinator.py`'s 4-step flow ignoring `run_npc_turn`).

## 4. Quick Wins vs Structural

### Quick wins (do-as-you-touch — low risk, mostly deletion/local)
- Delete dead modules/symbols once grep-confirmed: `turn_loop.py`, `mechanics/effects.py` (+5 `create_*_effect` factories), `image_comfyui.py` (or wire the branch + fix config comment), `CombatTurnManager`, `format_dm_response`/`split_text` + duplicate `_build_mechanics_embed` in `game.py`, `_is_auto_crit`, `_extract_referenced_entities`, `transcribe_bytes` (4×), `_FutureResolvingView`, `Currency.add_currency`/`ItemInfo`, `Character.proficiencies`/`CharacterProficiency`, empty `llm/prompts/` & `llm/tools/` packages.
- Remove unused imports (`character.py` leveling imports + `build_character_summary_embed`; `image_fal.py` `base64`/`Optional` + double `requests`; `image_openai.py` `Optional`; `discord_text.py` `build_combat_end_embed`).
- Rename/relocate root `test_*.py` scripts to `evals/` or `scripts/` (or `norecursedirs`/`collect_ignore`) so `test_*` means pytest; kill the `test_combat.py` shadow.
- Fix the two tautological tests (`test_dice` seed the RNG, drop `assert True`; `test_tiered_narrator` call real `load_profile` or delete).
- Hoist in-function `import structlog`/`uuid` to module level (`effects.py`, `intents.py`, `immersion_repo.py`, `session_repo.py`, `database.py`).
- Single-constant fixes: shared `world_setting` default; `slugify` import instead of `_slugify_id`; reconcile `npc.py` hostility thresholds to the `HOSTILITY_*` constants; drop stray `f""` prefixes on constant error strings.
- Apply existing type annotations to the `game.py` combat helpers (`CombatTurnCoordinator`/`CombatManager`/`Combatant`/`ActionResult` already imported elsewhere).
- Update the misleading docstrings (orchestrator/narrator PROSE+INTENTS, Riva-specific voice, coordinator 4-step flow) to the actual primary path.

### Structural (schedule it — touches contracts, needs a test net first)
- **Decompose `DMOrchestrator`** into NarrationService / CombatInitiator / ToolExecutor / WorldStateSync (#4) — gate behind the `process_action` integration test (#1) so the extraction is verifiable.
- **`ProposedEffect` → discriminated union + single handler registry** (#5); only then finish the effect-consolidation and delete legacy currency/damage fields + no-op executors.
- **Finish the single-authority `Character` refactor** (#2) and **resolve double end-of-turn processing** (#3) — both correctness-adjacent, both want a focused combat integration test.
- **Provider base classes**: `TTSProvider`/`ASRProvider` Protocol with a `synthesize_async` mixin + `pcm16_to_wav` util; `_OpenAICompatClient` base for Groq/OpenRouter/DeepSeek; provider capability flags (`NEEDS_LOCK`, `IS_CLIENT_SIDE`) to replace isinstance switches.
- **`BaseRepository`** (init/`_get_db`/singleton) + named-column reads (#9) across all 7 repos; add round-trip tests for `npc_repo`/`immersion_repo`.
- **Extract the character-creation wizard** into one `CharacterWizard` driver shared by `/character create` and the lobby button.
- **Cog consolidation**: `require_character` (#10), `_require_manager`, `_resolve_user_combatant`, `end_combat_if_over` teardown helper; route `game.py`'s inline combat trio through `turn_loop`; split `CombatCog` (setup vs in-turn) and move readied-actions onto the model.
- **Shared presentation module** (`mechanics.py` embed + splitter, `bars.py`, `format_dice_roll`/`format_modifier`).
- **Collapse `BrainContext` alias pairs** and the three narration near-duplicates into one `_run_narrator` helper.
- **Resolve mypy strategy** (#12) and add the missing pure-function suites (`EffectValidator`/`EffectExecutor` idempotency, `intents._parse_intent_line`) + the `tests/conftest.py` brain-stub fixture to kill the 3-file copy.
- **Confirm-then-delete the INTENTS/adjudicator fallback** (verify no configured profile relies on text-only output) or route it through the same effect registry.
