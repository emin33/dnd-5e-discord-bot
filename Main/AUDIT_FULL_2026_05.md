# Full Project Audit — May 2026

Cross-layer audit by 5 parallel Opus subagents (LLM, game/world, voice/immersion, bot/web, data/config).
93 findings total. Chip away across sessions. Mark items `[x]` when done.

Companion audit: `AUDIT_IMMERSION_2026_04.md` (29 items, 20 done) — focus there for immersion-specific items.

---

## P0 — Critical (data loss, broken features, multi-session corruption)

### 1. Character row indexing reads wrong columns
- **File:** `dnd_bot/data/repositories/character_repo.py:558-560`
- **Issue:** Reads `description=row[30]`, `portrait_url=row[31]`, `voice_id=row[32]`; real schema has those at 32/33/34. Every character load returns timestamps stuffed into `description`/`portrait_url`, description text in `voice_id`.
- **Fix:** Use `row[32]`, `row[33]`, `row[34]`. Better: switch to `sqlite3.Row` row_factory and address by name.
- [x] Fixed 2026-05-27 — indices corrected, `len(row) > N` guards dropped (columns now guaranteed by migration 005). Integration test in `tests/integration/test_character_repo_roundtrip.py` asserts the round-trip.

### 2. Migration 004 doesn't add columns it claims to add
- **File:** `migrations/004_immersion_features.sql:1-25`
- **Issue:** Header comment says "Adds voice_id to NPCs and characters" but contains only `CREATE TABLE`s — no `ALTER TABLE character ADD COLUMN description/portrait_url/voice_id`. Fresh DB crashes on character create. Local dev only works because columns were added out-of-band.
- **Fix:** Add the missing `ALTER TABLE` statements (or new migration 005). Verify with `PRAGMA table_info(character)` on fresh DB.
- [x] Fixed 2026-05-27 — new `migrations/005_character_immersion_columns.sql` adds character.description/portrait_url/voice_id and npc.voice_id. Made idempotent by adding a "duplicate column name" fallback in `Database._run_migrations` so the migration is safe to apply on dev DBs that already have the columns.

### 3. `transfer_item` non-atomic, can lose inventory
- **File:** `dnd_bot/data/repositories/inventory_repo.py:210-256`
- **Issue:** Split-stack path commits twice (update source, insert recipient). Failure between commits leaves source decremented and recipient empty.
- **Fix:** Wrap body in `async with await db.transaction():` and use raw `db.execute` (helper methods' inline commits defeat tx).
- [x] Fixed 2026-05-27 — wrapped in `db.transaction()` savepoint, replaced `update_item`/`add_item` calls with raw `db.execute`. Inlined the recipient stack-merge so a split transfer into a character who already has the same item still merges into one stack.

### 4. Tuple unpack swapped — purchase/pickup broken
- **File:** `dnd_bot/llm/orchestrator.py:1776, 1855`
- **Issue:** `character, char_id = resolved` but `_resolve_character_by_name` returns `(char_id, character)`. The Character object gets passed as `buyer_id`/`character_id` into DB lookups.
- **Fix:** `char_id, character = resolved` to match the function contract (14 other call sites already correct).
- [x] Fixed 2026-05-27 — both call sites corrected to `char_id, character = resolved`.

### 5. Idempotency keys regenerated every turn — dedup never fires
- **File:** `dnd_bot/llm/orchestrator.py:1255, 3389`
- **Issue:** `message_id = str(uuid.uuid4())` per turn makes the `f"{campaign}:{msg_id}:{i}"` key unique every call. `_applied_effects` grows unbounded; retries execute effects twice.
- **Fix:** Derive `message_id` from a stable identifier (turn number + session_id, or hash of action+turn).
- [x] Fixed 2026-05-27 — `message_id` now derived from `f"{session.session_key}:turn-{world_state.turn}"` so retries within the same turn collapse onto the same key. `_applied_effects` switched from raw `set` to a new `_BoundedKeySet` (FIFO, maxlen=1000) so memory growth is capped — old turns age out automatically.

### 6. Singleton orchestrator mutates per-turn state — races across sessions
- **File:** `dnd_bot/llm/orchestrator.py:2144, 735, 1085, 2305-2306, 3365, 4698`
- **Issue:** Global singleton writes `_on_narrative_token`, `_last_narrator_prose`, `_last_narrator_routing`, and mutates `self.narrator.client` per turn. Two concurrent player turns corrupt each other.
- **Fix:** Bind per-turn state to `BrainContext` or new `TurnContext`; stop mutating `narrator.client`.
- [ ] **Deferred** — coupled with #7. **Currently latent, not active**: the global `_processing_lock` in #7 serializes orchestrator calls, so concurrent corruption can't actually happen today. 97 references to `self._last_*` / `self._current_session` / `self._scene_registry` / `self.narrator.client` in the orchestrator. A clean fix requires threading a `TurnContext` dataclass through every method that currently mutates `self`. Estimated 3-5 hours focused refactor; non-trivial regression risk. **Tackle this only when also doing #7** — solving one without the other either gains nothing (per-session lock + shared state = active race) or stays serialized (global lock + shared state = current behavior).

### 7. Global processing lock serializes ALL campaigns
- **File:** `dnd_bot/game/session.py:167, 580`
- **Issue:** `_processing_lock = asyncio.Lock()` on the singleton manager. Slow narrator call in campaign A blocks every other campaign's turn. (Companion to #6.)
- **Fix:** Move lock onto `GameSession` per-session; pass `session` through `process_action` instead of stashing on orchestrator `self`.
- [ ] **Deferred** — coupled with #6. Moving the lock to per-session WITHOUT also doing #6 makes the orchestrator state race active (worse than today). Both must be tackled together. The current global lock is the cheap correctness guarantee; without #6, removing it would break things. Only worth doing if you actively need concurrent multi-campaign throughput — for a single-server bot the global lock costs nothing (one turn at a time is the natural pace anyway).

### 8. Scene registry collides across voice/web sessions
- **File:** `dnd_bot/game/scene/registry.py:546-549`, `dnd_bot/game/session.py:577,731`
- **Issue:** Keyed by `int channel_id`. All voice/web sessions use `channel_id=0`, so every concurrent voice/web campaign shares one registry — NPCs leak between sessions.
- **Fix:** Re-key by `session_key: str` (matches `_active_combats`/`_coordinators` pattern); pass `session.session_key` instead of raw `channel_id`.
- [x] Fixed 2026-05-27 — `_registries` re-keyed to `dict[str, SceneEntityRegistry]`. `get_scene_registry` / `clear_scene_registry` now take `session_key: str`. All 9 call sites updated: session.py uses `session.session_key`; bot/frontends/discord_text.py and immersion cog use the session's key; campaign cog (which has only an interaction, not a session yet) constructs `f"discord:{interaction.channel_id}"`. Voice/web sessions no longer collide at channel_id=0.

### 9. Conditions lost crossing combat boundary
- **File:** `dnd_bot/game/combat/manager.py:89-121` (entry), `:693-710` (exit)
- **Issue:** `add_player` never copies `character.conditions` into combatant effects. `sync_to_character` docstring claims to sync conditions but doesn't. Buffs/debuffs dropped both directions.
- **Fix:** In `add_player`, translate each `CharacterCondition` into a `CombatEffect`. In `sync_to_character`, rebuild `character.conditions` from `combatant.effects`.
- [x] Fixed 2026-05-27 — both directions implemented. Exhaustion stacks are exploded into per-stack effects on entry and aggregated back on exit. Expired-round conditions are skipped on entry. 5 new unit tests in `TestConditionCrossingCombatBoundary` lock in the behavior.

### 10. Bridge `if giver_id in known or giver_id` truthy-string bug
- **File:** `dnd_bot/game/knowledge/bridge.py:457-466`
- **Issue:** Always true for non-empty names — appends `AddEdge(QUEST_GIVER)` even with no giver node. `_apply_add_edge` then raises "Source node not found".
- **Fix:** `if giver_id and giver_id in known:` (or also add placeholder NPC node like `_handle_new_npc` does for locations).
- [x] Fixed 2026-05-27 — condition flipped to `and`. Placeholder-node fallback deferred to a follow-up (lower priority since QUEST_GIVER is just one of several edges from the quest node).

### 11. TTS cache lock declared but never used
- **File:** `dnd_bot/immersion/tts_assembler.py:52-90`
- **Issue:** `_tts_cache_lock` exists, but `_get_or_create_tts` is sync and reads/writes `_tts_cache` without it. AUDIT 2.1 ("TTS cache lock") fix is incomplete. Concurrent calls from `asyncio.gather` still create duplicate TTS instances.
- **Fix:** Make `_get_or_create_tts` `async` and wrap dict miss in `async with _get_tts_cache_lock():`, await from `_synthesize_segment`.
- [x] Fixed 2026-05-27 — function now async with double-checked locking (fast-path hit outside lock, re-check inside). Call site in `_synthesize_segment` awaits it.

### 12. Kokoro/Local TTS not thread-safe under concurrent `/api/tts`
- **File:** `dnd_bot/voice/api.py:843-848`, `dnd_bot/voice/tts_factory.py:124-127`
- **Issue:** `needs_lock()` returns True only for RivaTTS. Kokoro shares one `KPipeline` (PyTorch nn.Module) across calls; concurrent FastAPI requests invoke it from multiple ThreadPool workers → undefined behavior. Local image provider has same issue.
- **Fix:** Extend `needs_lock` to return True for any local in-process model.
- [x] Fixed 2026-05-27 — `needs_lock()` now returns True for `KokoroTTS` (in addition to `RivaTTS`), with a graceful `ImportError` fallback so the optional Kokoro dep doesn't break import. Local image provider lock is a separate file/concern; track as follow-up if it becomes a problem.

### 13. `image_guidance=0.0` profile setting silently ignored
- **File:** `dnd_bot/immersion/image_factory.py:91`
- **Issue:** `if immersion.image_guidance else 0` treats `0.0` as falsy. Flux Schnell uses `guidance=0.0` legitimately (AUDIT 6.2). Schnell profile silently broken.
- **Fix:** Explicit `None` check: `immersion.image_guidance if (immersion and immersion.image_guidance is not None) else settings.local_image_guidance`.
- [x] Fixed 2026-05-27 — switched to explicit `is not None` check. Flux Schnell profile now passes guidance=0.0 through correctly.

### 14. OpenAI TTS/ASR factory passes OpenRouter key — guaranteed 404
- **File:** `dnd_bot/voice/tts_factory.py:42`, `dnd_bot/voice/asr_factory.py:38`
- **Issue:** Both factories pass `settings.openrouter_api_key` as the OpenAI key. OpenRouter doesn't proxy `audio.speech`/`audio.transcriptions`. Non-empty key suppresses `OPENAI_API_KEY` env fallback.
- **Fix:** Pass `api_key=None` (rely on env, like `image_openai.py:23`), or add a dedicated `openai_api_key` setting.
- [x] Fixed 2026-05-27 — both factories now pass `api_key=None` so the OpenAI client picks up `OPENAI_API_KEY` from env.

### 15. OpenAI image generation has no timeout
- **File:** `dnd_bot/immersion/image_openai.py:33-44`
- **Issue:** No `timeout` kwarg. DALL-E hangs can block the executor thread for ~10min (OpenAI client default). Exception is caught downstream but the worker thread is held that whole time.
- **Fix:** `OpenAI(api_key=…, timeout=60.0)` in `_get_client()`.
- [x] Fixed 2026-05-27 — `OpenAI(timeout=60.0)` in both branches of `_get_client`.

### 16. Discord cogs loaded twice on every startup
- **File:** `dnd_bot/bot/client.py:131-156, 164-191`
- **Issue:** `create_bot()` synchronously loads all 10 cogs; `setup_hook()` then calls `_load_cogs()` which tries to load them again → `ExtensionAlreadyLoaded` for every cog.
- **Fix:** Remove the second loop. Keep one canonical path.
- [x] Fixed 2026-05-27 — synchronous loop in `create_bot()` deleted; `setup_hook()`'s `_load_cogs()` is the single canonical path (standard discord.py pattern).

### 17. Attack action consumed only on hit
- **File:** `dnd_bot/bot/cogs/combat.py:612-625`
- **Issue:** `manager.use_action(attacker.id)` inside the `if is_hit:` block. Missed attacks don't burn the action — players can `/combat attack` repeatedly until they hit.
- **Fix:** Move `use_action` outside the `if is_hit:` block.
- [x] Fixed 2026-05-27 — `use_action` moved out, PHB p.192 citation added in a comment so future-me doesn't move it back.

### 18. Lobby buttons never defer, race on double-click
- **File:** `dnd_bot/bot/views/campaign_lobby.py:68-86`, `cogs/campaign.py:73-99, 760-786`
- **Issue:** No `interaction.response.defer()`; `_handle_start` does DB queries + narrator LLM call before responding, blows past Discord's 3s window. `has_active_session` check also races — two parallel start clicks can both pass it.
- **Fix:** `await interaction.response.defer()` at top of both callbacks; use `asyncio.Lock` around `_handle_start`'s active-session check or let `SessionManager.start_session` raise on duplicate.
- [x] Fixed 2026-05-27 — both buttons now defer (DM check stays before defer for fast ephemeral reject). Added module-level `_session_start_lock`; the `has_active_session` check + `start_session` call now run under it so concurrent clicks serialize. Downstream `interaction.response.send_message` calls swapped to `interaction.followup.send` in `_handle_join`, `_show_character_select`, `_handle_start`.

### 19. 8 orphan methods in CharacterCog reference undefined `interaction`
- **File:** `dnd_bot/bot/cogs/character.py:532-825`
- **Issue:** `_show_ability_assignment`/`_show_point_buy`/`_show_race_select`/etc. use `interaction` which is never bound → `NameError`. Superseded by `*_from_interaction` versions. Also pass 3-arg `ConfirmCharacterView` that no longer matches signature.
- **Fix:** Delete lines 532-825 entirely.
- [x] Fixed 2026-05-27 — 294 lines deleted. File now 1117 lines, parses clean, full test suite still passes.

### 95. Qwen3 Hermes-block injection griefs native-tool models (Gemma 4)
- **File:** `dnd_bot/llm/client.py:455-467` (`_chat_via_openai_compat`)
- **Issue:** Discovered while investigating the question "are we using Gemma 4's hard token boundaries?" `ollama show gemma4:e2b` reveals `RENDERER gemma4` + `PARSER gemma4` — Ollama natively handles Gemma 4's `<|tool|>`/`<|tool_call|>`/`<|tool_result|>` special tokens when you pass `tools=[...]`. But our code unconditionally also injects a Hermes-format `<tools>` XML system block (a Qwen3 workaround from when its template's auto-injection was broken). For Gemma 4 this either (a) wastes context with a redundant tool-format declaration, or (b) confuses the model into emitting Hermes-style XML instead of using the special tokens — defeating the determinism the boundary tokens were designed for.
- **Fix:** Gate the Hermes block injection by model family. Skip for known native-tool models, keep for Qwen3 / unknown models.
- [x] Fixed 2026-05-27 — extracted `_build_compat_messages(model, messages, tools) -> (messages, uses_native)` as a class method. `_NATIVE_TOOL_MODEL_PREFIXES = ("gemma4",)` allowlist; gemma4 variants pass through unchanged, everything else gets the Hermes block (safe default). Log line `tool_path=native|hermes` added to `ollama_compat_response` so cache hit/miss for native-tool models is observable. 6 unit tests in `tests/unit/test_llm_client.py::TestOllamaCompatMessageBuilding` lock in gating + edge cases (no user message, pre-existing system, multiple gemma4 sizes). Verify the real win post-deploy by watching `ollama_compat_response` for `tool_path=native` + `hermes_tool_calls=0` when running a Gemma-brain profile.

### 94. `character_repo.create()` silently drops `known_spells` and `prepared_spells`
- **File:** `dnd_bot/data/repositories/character_repo.py:33-142` (INSERT statement); `:367-388` (update path)
- **Issue:** Discovered while writing the #62 integration test. `create()` does not insert into `character_spell` at all. `update()` only inserts entries for `prepared_spells`, so `known_spells` (spells known but not currently prepared) can NEVER be written. New wizards/clerics/bards lose every spell on the character-create wizard's "confirm" step.
- **Fix:** Insert all `character.known_spells` into `character_spell` inside `create()` (with `is_prepared` set per `character.prepared_spells` membership). Extend `update()` to handle adds/removes of `known_spells` too. Then unrestore the spell assertions in `test_create_then_get_by_id_round_trips_all_fields`.
- [x] Fixed 2026-05-27 — `create()` now inserts every entry in `character.known_spells` with `is_prepared` flagged per `prepared_spells` membership. `update()` reconciles via set diff (delete unlearned, insert newly-known, flip is_prepared per pass). New `test_update_syncs_known_and_prepared_spells` integration test covers learn/unlearn/prepare cycles.

---

## P1 — Important (fix soon, but not blocking)

### LLM pipeline

### 20. Two-pass tool followup discards conversation context
- **File:** `dnd_bot/llm/orchestrator.py:2244-2291`
- **Issue:** `_narrator_tool_followup` ignores the `messages` it receives; rebuilds a 2-message prompt with just prose. Roster, world-state YAML, `[id: ...]` tags all lost — model hallucinates entity IDs.
- **Fix:** Reuse `messages` and append a "now declare your tools" system message, or at minimum pass through `self._scene_registry.get_triage_context()`.
- [x] Fixed 2026-05-27 — followup reuses the original message stack, appends the assistant's prose as an assistant turn, and adds a user turn instructing tool-call-only. Roster + YAML + entity context preserved so `ref_entity` IDs resolve correctly.

### 21. Anthropic narrator has no prompt caching — 5K tokens re-billed every turn
- **File:** `dnd_bot/llm/client.py:991-1116`
- **Issue:** Full system prompt + bookend re-sent every request. At Sonnet 4.6 $3/M and ~2K static tokens, ~$0.006/turn wasted; caching cuts 90% off repeated input.
- **Fix:** Wrap `system` as content block list with `cache_control: {type: "ephemeral"}` on static portion. Report `cache_read_input_tokens` in `LLMResponse`.
- [x] Fixed 2026-05-27 — broader sweep than original audit scope: (a) `LLMResponse` extended with `cache_read_tokens` / `cache_write_tokens` + `cache_hit_ratio` property. (b) AnthropicClient: first system_part now sent as a content block with `cache_control: {"type": "ephemeral"}` when it's >= 4000 chars (~1024 tokens, Anthropic's min cacheable size); subsequent system_parts stay outside the breakpoint so the cache key is stable across turns. Reads `cache_read_input_tokens` and `cache_creation_input_tokens` from usage. (c) DeepSeekClient: existing `prompt_cache_hit_tokens` telemetry now also populates the `cache_read_tokens` field on `LLMResponse`. (d) Orchestrator: cache stats logged in the per-turn record (`narrator_cache_read`, `narrator_cache_write`, `narrator_cache_hit_ratio`), only shown when the provider reports them so Ollama/Groq logs stay clean. Note: Gemini explicit caching API (`cachedContent`) and any other provider's caching are not yet wired — defer until cost justifies the work.

### 22. `referenced_entity_ids` mixes UUIDs and human names
- **File:** `dnd_bot/llm/orchestrator.py:1105-1114`, `extractors/state_extractor.py:132-137`
- **Issue:** `REF_ENTITY` effects contribute UUIDs, `ADD_NPC` effects contribute raw name strings — both go into the same dedup-hint list. LLM can't resolve UUIDs.
- **Fix:** Look up entities by ID first, pass canonical names everywhere (or split into two labeled lists).
- [ ] Fixed

### 23. ChromaDB entity re-embed every turn unconditionally
- **File:** `dnd_bot/llm/orchestrator.py:1161-1184, 1302-1330`
- **Issue:** Every `AddNode`/`UpdateNode` with a description triggers ChromaDB upsert + re-embed (~10-50ms per entity, hot path), even when only `location` or `disposition` changed.
- **Fix:** Track description hash on Entity; skip upsert when description didn't change.
- [ ] Fixed

### 24. `EntityNameMatcher` re-instantiated twice per turn
- **File:** `dnd_bot/llm/orchestrator.py:1006, 1192`, `game/knowledge/matcher.py:30-34`
- **Issue:** Matcher cache is lazy but rebuilt twice per turn from `get_all_names()`. Cache never survives.
- **Fix:** Construct one matcher per session (attach to GameSession or memoize on KnowledgeGraph) with a graph-version counter to invalidate.
- [ ] Fixed

### 25. `_handle_purchase`/`_handle_inventory` not the only places to grep — verify all `_resolve_character_by_name` callers
- See P0 #4. Quick sweep for other call sites unpacking the wrong direction.
- [ ] Verified

### 26. `RulesBrain` instantiated, never called; 5 brain methods dead
- **File:** `dnd_bot/llm/orchestrator.py:528`, `llm/brains/rules.py:515-657`
- **Issue:** `self.rules` referenced nowhere else. `RulesBrain.process`/`.resolve_*` have zero call sites. 466-line `RULES_TOOLS` schema duplicated in orchestrator's `_execute_tool`.
- **Fix:** Remove `RulesBrain` instantiation + unused methods. Keep `RULES_TOOLS` only if some code uses the schema.
- [ ] Fixed

### 27. NarratorBrain has 4 unreachable methods
- **File:** `dnd_bot/llm/brains/narrator.py:112-188, 276-352`
- **Issue:** `.process`, `.process_streaming`, `.describe_scene`, `.roleplay_npc` have no callers. Orchestrator builds messages inline.
- **Fix:** Delete the dead methods. Trim PROSE/INTENTS stripping in `narrate_outcome:243-268` (legacy from text-format era).
- [ ] Fixed

### 28. Text-fallback PROSE/INTENTS path is dead in practice
- **File:** `dnd_bot/llm/orchestrator.py:2293-2333`, `llm/intents.py`, `llm/brains/adjudicator.py`
- **Issue:** `_extract_prose_and_effects` calls intents parser only when there are no `tool_calls` — never triggers with tool-capable providers. 600-line intent mini-language + adjudicator kept alive for unreachable path.
- **Fix:** Delete adjudicator + intents module, rely on two-pass followup at line 2419. Also drop unused `strip_planning_text_fallback` import at orchestrator:22.
- [ ] Fixed

### 29. NLIValidator initialized, imported, never invoked
- **File:** `dnd_bot/llm/orchestrator.py:32, 535, 1237-1243`
- **Issue:** Step 3.7 explicitly disabled with TODO. Still logs `pairs_checked=0` every turn.
- **Fix:** Remove `_nli_validator`, import, and `record_nli(0, ...)`. Re-add when correction loop is wired.
- [ ] Fixed

### Game/World state

### 30. `world_state.established_facts` grows unbounded with O(n²) dedup
- **File:** `dnd_bot/game/session.py:639-642`
- **Issue:** Per-turn list copy with `not in` membership check on append-only list. Both lists feed into narrator YAML.
- **Fix:** Use `set` for membership, cap to last 30, sync only newly-added pinned facts (track index pointer).
- [ ] Fixed

### 31. `scene_items` never clears on location change
- **File:** `dnd_bot/game/world_state.py:189-217, 263-268`
- **Issue:** Location changes but scene_items carry forward. Narrator YAML lists items from a previous room.
- **Fix:** When `delta.location_change` differs from current, clear `self.scene_items` (or filter to explicit transfers).
- [ ] Fixed

### 32. `EffectTracker` and 469-line `mechanics/effects.py` is dead
- **File:** `dnd_bot/game/mechanics/effects.py` (entire file)
- **Issue:** Zero importers. Live effect system uses `models.combat.CombatEffect` in `CombatManager.process_*_effects`. Duplicate `EffectProcessResult` dataclass too.
- **Fix:** Delete the file. Migrate any useful factory functions into `models/combat.py`.
- [ ] Fixed

### 33. `SceneEntityRegistry.get_by_name` bidirectional substring is false-positive prone
- **File:** `dnd_bot/game/scene/registry.py:131-142`
- **Issue:** `if name_lower in entity.name.lower() or entity.name.lower() in name_lower:` — "Captain Jones" matches any "Captain" or "Jones". Non-deterministic iteration on ambiguity.
- **Fix:** Tiered match: exact → exact alias → startswith → contains (longest first, with length floor).
- [ ] Fixed

### 34. `SPELL_CONDITION_MAP` applies Prone for non-grovel Command variants
- **File:** `dnd_bot/game/combat/coordinator.py:52`
- **Issue:** `"command": (Condition.PRONE, 1, False)` — Command has 6 variants; only Grovel applies Prone.
- **Fix:** Remove the `command` entry or branch on the spell variant.
- [ ] Fixed

### 35. `_execute_spell` silently fails on character cache miss
- **File:** `dnd_bot/game/combat/coordinator.py:809-816`
- **Issue:** Direct `_character_cache.get()` with no async loader fallback. Reaction spells / readied actions outside `start_turn` will silently fail.
- **Fix:** `character = await self._get_character(caster.character_id) if caster.character_id else None`.
- [ ] Fixed

### 36. KG load + ChromaDB sync is O(N) on every session start
- **File:** `dnd_bot/game/session.py:222-251, 423-431`, `game/knowledge/graph.py:46-70`
- **Issue:** Full campaign load on every session start, then indexable entities re-pushed to ChromaDB. Linear in campaign age.
- **Fix:** Pickle nx.DiGraph + `_entities` keyed by campaign, refresh-on-mutate. Gate ChromaDB sync by `entity.updated_at`.
- [ ] Fixed

### 37. Advantage/disadvantage silently dropped for non-d20 rolls
- **File:** `dnd_bot/game/mechanics/dice.py:115-118`
- **Issue:** Flag only honored for `num_dice == 1 and die_size == 20`. Caller passing `advantage=True` for `2d20` or damage reroll gets silent ignore.
- **Fix:** Raise `ValueError` or log warning if flag set but notation doesn't support it.
- [ ] Fixed

### Voice/Immersion

### 38. Quote regex can swallow paragraphs across unbalanced quotes
- **File:** `dnd_bot/immersion/prose_parser.py:22-25` (and dup `dialogue_attributor.py:54`)
- **Issue:** `[“"]\s*(.+?)\s*[”"]` with `re.DOTALL`. Missing close quote → next opening `"` pairs across paragraphs; everything between becomes one "dialogue" segment.
- **Fix:** Drop `DOTALL`, or reject quotes whose inner text exceeds N chars or contains `\n\n`.
- [ ] Fixed

### 39. Pass-2 attribution uses stale `attributed_ids`
- **File:** `dnd_bot/immersion/prose_parser.py:218-247`
- **Issue:** Snapshot taken once before loop; subsequent iterations see stale data. Also `segments.index(seg)` is O(N²) and wrong when two segments compare equal by value.
- **Fix:** `enumerate(segments)` for real index; recompute or update `attributed_ids` per iteration.
- [ ] Fixed

### 40. Fish Speech servers leak on bot shutdown
- **File:** `dnd_bot/voice/fish_manager.py:66-70, 122-168`
- **Issue:** Detached `cmd start` Popens not tracked; no shutdown hook. API servers + VRAM stay allocated after bot exits. Repeated restarts pile up zombies.
- **Fix:** Track Popens in a module list; `atexit` (or asyncio shutdown) → `taskkill /f /pid` or hit a shutdown endpoint.
- [ ] Fixed

### 41. Voice catalog never picks up removed/edited entries
- **File:** `dnd_bot/data/repositories/immersion_repo.py:83-113, 158-183`
- **Issue:** Reseeds only when `json_count > db_count` with `INSERT OR IGNORE`. Renames/removals invisible. `_catalog_seeded` set BEFORE seed runs — crash mid-seed permanently disables re-seeding.
- **Fix:** `INSERT OR REPLACE` per row (or content-hash check). Set `_catalog_seeded = True` AFTER seed succeeds.
- [ ] Fixed

### 42. `tts_assembler` adds silence between same-speaker split quotes
- **File:** `dnd_bot/immersion/tts_assembler.py:185-193`
- **Issue:** `elif audio_chunks:` branch inserts `PARAGRAPH_SILENCE` between every consecutive same-speaker dialogue pair, including split quotes `"Look out," he yelled, "trap!"` — cadence broken.
- **Fix:** Track `last_segment_type` too; only insert paragraph silence if a NARRATION segment intervened (or use 50ms gap for split quotes).
- [ ] Fixed

### 43. `asr_factory.get_asr()` + Riva/OpenAI/Deepgram providers are dead
- **File:** `dnd_bot/voice/asr_factory.py`, `voice/asr.py`, `voice/asr_openai.py`, `voice/asr_deepgram.py`
- **Issue:** Zero callers. LiveKit transport uses `nvidia.STT` directly; web uses browser `webkitSpeechRecognition`. `deepgram_api_key` setting and `DEEPGRAM_API_KEY` env unused.
- **Fix:** Delete `asr_factory.py` + the three providers (and `_reset_asr`), OR wire `get_asr()` into the not-yet-existing `/api/asr` endpoint.
- [ ] Fixed

### 44. Profile reset doesn't clear assembler TTS cache
- **File:** `dnd_bot/voice/tts_factory.py:110-113`, `immersion/tts_assembler.py:52`
- **Issue:** `_reset_tts()` clears `tts_factory._tts_instance` but not `tts_assembler._tts_cache`. Profile switch keeps stale per-voice instances.
- **Fix:** Add `_reset_tts_cache()` in tts_assembler; call from `_reset_tts()`.
- [ ] Fixed

### 45. `voice/api.py` owns the whole game API, not just voice
- **File:** `dnd_bot/voice/api.py` (entire 884-line file)
- **Issue:** Only ~60 lines are voice-specific (`/api/tts`, `/api/token`). All campaign/character/SRD/session endpoints live here, mislocated under `voice/`.
- **Fix:** Move to `dnd_bot/web/api.py` (matching `WEB_DIR` static); keep just `/api/tts` under `voice/`.
- [ ] Fixed

### 46. `image_comfyui.py` provider not registered in factory
- **File:** `dnd_bot/immersion/image_factory.py:25-48`, `immersion/image_comfyui.py`
- **Issue:** 208-line ComfyUIImageProvider with Flux and SDXL workflows; zero importers. Setting `image_provider: comfyui` raises `ValueError`.
- **Fix:** Add `elif provider == "comfyui": ...` to factory + surface `comfyui_url` setting, OR delete the file.
- [ ] Fixed

### Bot frontend / Web

### 47. Combat cog double-click leaks views
- **File:** `dnd_bot/bot/cogs/combat.py:1426-1542`
- **Issue:** Running `/combat turn` twice creates two `CombatActionView` for same combatant; both register callbacks → race on click; older view timeout never properly cleans up.
- **Fix:** Track active turn view per `channel_id`; error second call or stop+delete prior view's message before creating new.
- [ ] Fixed

### 48. Inline `onclick` handlers + unsanitized IDs interpolated into HTML
- **File:** `web/index.html:26-65, 96-99, 119-128, 145, 159, 205-219` + `web/app.js:52-57, 126-135`
- **Issue:** `id="cmp-${c.id}"`, `onclick="pickRace('${r.index}')"` — none go through `esc()`. Any HTML metacharacter in an ID is XSS. Also blocks any future CSP.
- **Fix:** Use `addEventListener` + `dataset.id`. At minimum run `esc_attr`/`esc` on the IDs.
- [ ] Fixed

### 49. `esc_attr` produces malformed HTML inside double-quoted attributes
- **File:** `web/app.js:53, 127, 839`
- **Issue:** Returns JSON with `"` → `&quot;`, inlined into `onclick="pickCampaign(${esc_attr(c)})"`. Browser decodes `&quot;` back to `"` before JS parses, but apostrophes in strings break the JS literal — character "D'rana" click silently no-ops.
- **Fix:** Move payload to `data-*` attributes via `esc_attr`-encoded JSON; read in delegated listener with `JSON.parse(el.dataset.payload)`.
- [ ] Fixed

### 50. `endGame()` sends no session key
- **File:** `web/app.js:610-618`
- **Issue:** `fetch('/api/game/end', {method: 'POST'})` empty body — server has no `session_key` to identify which session. Either ends nothing or ends everything.
- **Fix:** Send `body: JSON.stringify({ session_key: S.sessionKey })`.
- [ ] Fixed

### 51. Streaming narration can exceed Discord 2000-char message limit
- **File:** `dnd_bot/bot/frontends/discord_text.py:263-283`
- **Issue:** `_handle_narrative_token` truncates display at 4000 but keeps growing `self._stream_buf`. Plain-message edits hit 2000-char Discord cap and silently raise `HTTPException` (bare `except: pass`).
- **Fix:** Move streaming prose into an embed (`description` accepts 4096), or cap `_stream_buf` to 1900 before edit.
- [ ] Fixed

### 52. All combat/spell/save/check/rest commands default `campaign_id="default"`
- **File:** `dnd_bot/bot/cogs/actions.py:172, 274, 442`, `combat.py:142, 800`, `spells.py:362, 502, 587, 644, 698, 745`, `rest.py:38, 140, 244, 380`
- **Issue:** Real campaign IDs are UUIDs. `get_by_user_and_campaign(user, "default")` always returns `None`. Players see "You don't have a character" when they do.
- **Fix:** Remove the option; resolve via `_get_campaign_id`/`get_active_campaign_id(ctx.guild_id)`.
- [x] Fixed 2026-05-27 — all 15 `campaign_id: discord.Option(..., default="default")` blocks removed across 4 cogs. Each command now resolves via `get_active_campaign_id(ctx.guild_id)` with a "no active campaign" error response. Initiative roll degrades gracefully to modifier=0 when no campaign is active. Dead `campaign_id="default"` parameter also removed from `_sync_player_characters` helper.

### 53. `_get_immersion_settings` ignores DB writes from `/immersion` cog
- **File:** `dnd_bot/bot/frontends/discord_text.py:285-317` + `bot/cogs/immersion.py`
- **Issue:** `/immersion tts/images/frequency/...` persists to DB, but the frontend reads only `get_profile().immersion`. Users' settings are saved but never take effect.
- **Fix:** Read from `get_immersion_repo().get_or_create_guild_settings(guild_id)`; profile is default fallback.
- [ ] Fixed

### 54. `/combat attack`/`opportunity`/`ready` don't sync player HP to DB
- **File:** `dnd_bot/bot/cogs/combat.py:594-658, 1224-1266, 1268-1353`
- **Issue:** `_sync_player_characters` only called in `damage`/`heal`/`end`/`turn`. Legacy `/combat attack` damages players, never writes back. Character sheet stale after combat.
- **Fix:** Call `_sync_player_characters` after `manager.apply_damage` for player targets.
- [ ] Fixed

### 55. Character refetched on every XP/skill/save command
- **File:** `dnd_bot/bot/cogs/character.py:849, 891, 934, 1028, 1112, 1179, 1325` (and similar across cogs)
- **Issue:** Full character (abilities, slots, hit dice, conditions, equipment, spells) hydrated for one-field reads. With 4-5 players × 10 commands/session: hundreds of unnecessary loads.
- **Fix:** Add per-user TTL cache (5-10s) invalidated on `update()`, or narrow methods like `get_xp_only(user_id, campaign_id)`.
- [x] Fixed 2026-05-27 — module-level `_get_cache` in `character_repo.py` keyed by `(user_id, campaign_id)`, 5s TTL, returns a `model_copy(deep=True)` on hit so caller mutations don't poison cache. Every mutating method (`create`, `update`, `update_hp`, `update_death_saves`, `update_spell_slot`, `add_condition`, `remove_condition`, `update_description`, `update_portrait`, `update_voice_id`) calls `_invalidate_character_cache()`. 2 new integration tests verify deep-copy and invalidation.

### Data / Models / Config

### 56. Character `update()` not wrapped in transaction
- **File:** `dnd_bot/data/repositories/character_repo.py:298-391`
- **Issue:** 5 multi-statement workflows (main, slots, conditions delete+insert, spells) with single `commit()` at end. Mid-method exception leaves shared connection dirty for next caller.
- **Fix:** Wrap in `async with await db.transaction():` like `create()` does.
- [ ] Fixed

### 57. Stale migration defaults for immersion settings
- **File:** `migrations/004_immersion_features.sql:11-12`
- **Issue:** Defaults `'inworld'` / `'Diego'`. App defaults are `kokoro` / `af_heart`. Existing guild rows keep stale non-empty values → no fallback triggers.
- **Fix:** Migration `UPDATE guild_immersion_settings SET narrator_tts_provider='kokoro', narrator_tts_voice='af_heart' WHERE narrator_tts_provider='inworld'`; update schema default.
- [ ] Fixed

### 58. `add_item` returns wrong-ID item on stack
- **File:** `dnd_bot/data/repositories/inventory_repo.py:24-56`
- **Issue:** When stack matches, updates existing row + returns existing item — but caller passed fresh UUID. Caller's stored ID now wrong.
- **Fix:** Document the contract clearly and have callers always use the return value; or stop auto-stacking and let callers stack explicitly.
- [ ] Fixed

### 59. Duplicate `DeathSaves` model
- **File:** `dnd_bot/models/combat.py:65-92` vs `models/character.py:185-212`
- **Issue:** Two definitions. `__init__.py` exports only character version; `Combatant` uses combat-local one.
- **Fix:** Delete the one in `combat.py`, import from `character.py`.
- [ ] Fixed

### 60. Character `proficiencies` field dead
- **File:** `dnd_bot/models/character.py:289`
- **Issue:** Never read or written. DB has `proficiency_type` column with 'tool'/'weapon'/'armor' values being thrown away (`character_repo.py:458-465`).
- **Fix:** Either wire it into `_row_to_character` for tool/weapon/armor profs, or delete the field + `CharacterProficiency` class.
- [ ] Fixed

### 61. Repository singletons disregard injected `Database`
- **File:** `dnd_bot/data/repositories/*.py`
- **Issue:** Module-level `_repo = Repo()` initialized with no db arg. Test setup with custom db loses to import-order. DI silently broken.
- **Fix:** Delete the singleton (instantiation cost is zero), `return Repo()`.
- [ ] Fixed

### 62. `tests/integration/` empty
- **File:** `tests/integration/__init__.py` (only file)
- **Issue:** User prefers integration tests over mocks. No DB-roundtrip tests would have caught P0 #1.
- **Fix:** Add `test_character_repo_roundtrip.py`: create char with all fields populated, save, load, assert equal. Add similar for NPC, Campaign, InventoryItem.
- [x] Partial 2026-05-27 — `test_character_repo_roundtrip.py` added with 5 tests (full round-trip, get-by-user, null-immersion, missing-id, migration 005 idempotency). NPC/Campaign/InventoryItem round-trips still to do.

### 63. Voice catalog seed flag set before seeding runs
- **File:** `dnd_bot/data/repositories/immersion_repo.py:90-92`
- **Issue:** `_catalog_seeded = True` set before `seed_voice_catalog()`. Mid-seed crash → empty catalog for rest of process. Log line counts JSON length not actual inserts.
- **Fix:** Set flag after seed succeeds. Return cumulative `cursor.rowcount`.
- [ ] Fixed

### 64. `test_brain_benchmark.py` bypasses `__init__` via `__new__`
- **File:** `test_brain_benchmark.py:766-769`
- **Issue:** `DMOrchestrator.__new__(...)` then manually sets fields. New orchestrator dependencies silently missed. Tightly couples benchmark to private internals.
- **Fix:** Add `DMOrchestrator.for_benchmark(client)` classmethod or restructure benchmark to drive triage via public entry point.
- [ ] Fixed

---

## P2 — Code health (cleanup, naming, minor improvements)

### LLM

### 65. Two character-resolution code paths drift
- **File:** `dnd_bot/llm/orchestrator.py:1571-1574` vs `:4131-4175`
- **Fix:** Call `_resolve_character_by_name` from `_get_character_capabilities`; bail on None.
- [ ] Fixed

### 66. Inconsistent enum vs string entity-type comparisons
- **File:** `dnd_bot/llm/orchestrator.py:3322-3323` vs `:3608-3609`
- **Fix:** Pick one style (enums preferred); convert.
- [ ] Fixed

### 67. `Effect.apply_damage`/`apply_healing`/`add_condition` are no-op stubs
- **File:** `dnd_bot/llm/effects.py:888-919`
- **Fix:** Wire to `_execute_update_hp` or remove from `EffectType` + validator.
- [ ] Fixed

### 68. Brittle YAML phase extraction
- **File:** `dnd_bot/llm/brains/base.py:232`
- **Issue:** `context.world_state_yaml.split("phase:")[1].split("\n")[0]` breaks if "phase:" appears in any other field.
- **Fix:** Parse YAML properly, or pass `world_state.phase` through `BrainContext` as a structured field.
- [ ] Fixed

### 69. Empty `llm/tools/` and `llm/prompts/` packages
- **File:** `dnd_bot/llm/tools/__init__.py`, `llm/prompts/__init__.py`
- **Fix:** Delete both, or populate (move `narrator_tools.py` → `tools/narrator.py`).
- [ ] Fixed

### Game

### 70. Voice/web sessions use `channel_id=0` sentinel end-to-end
- **File:** `dnd_bot/voice/api.py:421, 487, 564`, `game/session.py:262-268`
- **Issue:** Persists to `session_repo`; per-channel lookups ambiguous. (Companion to P0 #8.)
- **Fix:** Make `channel_id: Optional[int] = None` end-to-end; use `session_key` everywhere.
- [ ] Fixed

### 71. `MessageBuffer.preserve_recent` unused
- **File:** `dnd_bot/memory/blocks.py:99-104`
- **Fix:** Either wire into `add()`'s overflow logic or drop the param.
- [ ] Fixed

### 72. `world_state.npc_updates` dead-NPC guard drops valid resurrections
- **File:** `dnd_bot/game/world_state.py:298-303`
- **Issue:** `if not existing.alive and update.alive is not True:` correct, but next-line `if update.disposition or update.location or update.notes:` rejects before `update.alive=True` applies.
- **Fix:** Apply `update.alive` first; gate other-field changes only if `existing.alive is False AND update.alive is not True`.
- [ ] Fixed

### 73. `world_state._max_recent_events` private-attr semantics wrong in Pydantic v2
- **File:** `dnd_bot/game/world_state.py:201-202`
- **Issue:** Leading-underscore class attrs become class-level not per-instance; intent of per-instance config doesn't work.
- **Fix:** Use `PrivateAttr(default=5)` or inline constants.
- [ ] Fixed

### 74. `removed_npcs` in StateDelta leaves removed NPCs queryable
- **File:** `dnd_bot/game/world_state.py:371-376`
- **Issue:** Clears `npc.location = ""` instead of deleting; `find_npc`/`_resolve_npc` still match them.
- **Fix:** Delete the npc, or set `removed: bool` flag and exclude from lookups.
- [ ] Fixed

### Voice/Immersion

### 75. Three WAV-builder copies drift
- **File:** `dnd_bot/voice/asr_openai.py:58-77`, `asr_deepgram.py:69-81`, `voice/api.py:854-870`
- **Fix:** Extract `dnd_bot/voice/_wav.py` with `pcm_to_wav(...)` using stdlib `wave`; import from all three.
- [ ] Fixed

### 76. `RivaTTS.synthesize_stream_async` is fake-streaming and dead
- **File:** `dnd_bot/voice/tts.py:101-113`
- **Fix:** Delete (and verify `TTSSentenceQueue` above isn't imported outside `frontend.py`).
- [ ] Fixed

### 77. TTS provider signatures inconsistent (emotion arg)
- **File:** `dnd_bot/voice/tts.py:52` + 5 other providers, `tts_assembler._synthesize_segment:107`
- **Issue:** `if provider in ("fish", "inworld")` special-cases emotion. Adding emotion-capable provider requires editing assembler.
- **Fix:** Define `TTSProtocol` with `synthesize(text, emotion=None)`; have all providers accept (and ignore) emotion.
- [ ] Fixed

### 78. `_QUOTE_PATTERN` duplicated in two modules with subtly different filters
- **File:** `dnd_bot/immersion/prose_parser.py:22-25`, `dialogue_attributor.py:54-56`
- **Issue:** `len(strip) > 2` vs `len > 2`. Attribution maps 1-indexed quote numbers between the two. Divergence → silent misalignment.
- **Fix:** Move to `dnd_bot/immersion/_quotes.py` with shared `find_quotes(text)` helper.
- [ ] Fixed

### 79. `VoiceFrontend._session` stored but never used
- **File:** `dnd_bot/voice/frontend.py:147, 159`
- **Fix:** Drop param + TYPE_CHECKING import; update caller.
- [ ] Fixed

### 80. Hardcoded narrator defaults override profile config
- **File:** `dnd_bot/immersion/voice_resolver.py:37-41`
- **Issue:** `"kokoro"` / `"af_heart"` hardcoded; profile's `narrator_tts_provider` ignored when `guild_settings is None` (voice/web path).
- **Fix:** Fall back to `get_profile().immersion.narrator_tts_provider` instead of module constants.
- [ ] Fixed

### Bot/Web

### 81. Two parallel `_creation_states` dicts across cogs
- **File:** `dnd_bot/bot/cogs/campaign.py:45-46`, `character.py:45`
- **Issue:** Plus unused `_creation_states_lock` in campaign.py (author knew about concurrency but didn't wire it).
- **Fix:** Move to `game/character/creation_states.py`, import from both cogs.
- [ ] Fixed

### 82. Mechanics embed builder duplicated
- **File:** `dnd_bot/bot/cogs/game.py:64-127` and `bot/frontends/discord_text.py:32-115`
- **Fix:** Extract to `bot/embeds/mechanics_embed.py`.
- [ ] Fixed

### 83. Dead imports across cogs
- **File:** `character.py:20-23, 38`, `discord_text.py:22, 508` (dup `build_combat_end_embed`), `game.py:751` (`startswith("/")` on slash commands is useless)
- **Fix:** Drop the unused names.
- [ ] Fixed

### 84. `_handle_combat_end` is a TODO stub
- **File:** `dnd_bot/bot/frontends/discord_text.py:506-511`
- **Fix:** Include `combat` object in `COMBAT_END` event payload, call `build_combat_end_embed`.
- [ ] Fixed

### Data / Config

### 85. `players_json` unused in `save_session`
- **File:** `dnd_bot/data/repositories/session_repo.py:34`
- **Fix:** Remove param + update callers.
- [ ] Fixed

### 86. Three entry points duplicate sys.path/env bootstrap
- **File:** `dnd_bot/main.py:10-11`, `run_voice.py:36-40`, `test_brain_benchmark.py:21`
- **Fix:** Single `dnd_bot/_bootstrap.py` helper.
- [ ] Fixed

### 87. `datetime.utcnow()` deprecated in Python 3.12+
- **File:** `dnd_bot/models/{character,campaign,npc,inventory,combat}.py` (12 sites), `data/repositories/{campaign,inventory,npc,transaction}_repo.py` (5 sites), `bot/cogs/admin.py:53`
- **Fix:** `datetime.now(timezone.utc)` everywhere.
- [ ] Fixed

### 88. SRD loader silently drops categories on missing `index`
- **File:** `dnd_bot/data/srd/loader.py:73`
- **Fix:** Log at WARNING with offending item; fall back to `slug`/`name` or skip non-conforming entries instead of dropping the whole category.
- [ ] Fixed

### 89. Voice catalog dedup uses count comparison
- **File:** `dnd_bot/data/repositories/immersion_repo.py:105`
- **Fix:** Set diff of voice_ids between DB and JSON; delete missing, insert new.
- [ ] Fixed

### 90. `transaction_log` missing index on `(operation_type, target_id)`
- **File:** `migrations/001_initial_schema.sql:264-272`, `data/repositories/transaction_repo.py:151-188`
- **Fix:** New migration: `CREATE INDEX idx_transaction_op_target ON transaction_log(operation_type, target_id, applied_at DESC);`
- [ ] Fixed

### 91. `npc_repo.get_at_location`/`get_by_name` use `LIKE %x%` without index
- **File:** `dnd_bot/data/repositories/npc_repo.py:58-67, 98-107`
- **Issue:** Narrator's frequent "who is at this location?" queries grow O(N) with campaign NPC count.
- **Fix:** In-memory cache `{campaign_id: {name_lower: npc_id}}` populated on session start (NPC writes are rare).
- [ ] Fixed

### 92. `RulesBrain.process` / `RULES_TOOLS` schema dead (companion to #26)
- See P1 #26.
- [ ] Tracked

### 93. Per-turn brain mutation tier-temperature lost
- **File:** `dnd_bot/llm/orchestrator.py:2144`
- **Issue:** `self.narrator.client = client` switches client but `NarratorBrain.__init__` set `temperature` once at start.
- **Fix:** `_select_narrator_client_for_turn` returns `(client, temperature)` tuple.
- [ ] Fixed

### 96. Brain benchmark report crashes on Windows console encoding
- **File:** `test_brain_benchmark.py:918` (`print_report`)
- **Issue:** Discovered while running the TRIAGE benchmark to verify #95. `print_report` uses Unicode box-drawing chars (`─`, `✓`) but Windows' default console is cp1252 — `print_report` dies with `UnicodeEncodeError` AFTER all cases run, so you lose the accuracy summary every time on Windows.
- **Fix:** Reconfigure stdout/stderr to UTF-8 at module top.
- [x] Fixed 2026-05-27 — added `sys.stdout/stderr.reconfigure(encoding="utf-8")` guard at the top of `test_brain_benchmark.py`. Report now renders. Verified: gemma4:e2b TRIAGE = 92% action-type (24/26), 92% roll-decision, 50% skill-selection, 0 JSON failures, 1.1s/case avg.

---

## Summary

| Section | Count | Done |
|---------|-------|------|
| P0 Critical | 20 | 18 |
| P1 Important | 46 | 5 |
| P2 Code health | 30 | 1 |
| Partial | 1 | 0 |
| **Total** | **96** | **24** |

**Done so far (2026-05-27):** P0s #1, #2, #3, #4, #5, #8, #9, #10, #11, #12, #13, #14, #15, #16, #17, #18, #19, #94. P1s #20, #21, #52, #55, #95. P2 #96. Partial: #62 (character round-trip with 8 tests). Tests: 491 passing (+33 from session start).

**Verification (2026-05-27):** #95 confirmed live via a tool-bearing probe to gemma4:e2b — `tool_path=native`, `sdk_tool_calls=1`, `hermes_tool_calls=0`, clean `roll_dice(2d6+3)` call with no Hermes block (prompt_tokens=134). Brain benchmark (TRIAGE, gemma4:e2b) passed at 92% routing / 0 JSON failures — confirms no regression from the session's changes. **Note:** triage uses JSON-schema mode (native /api/chat path), NOT the tools path, so the benchmark validates the brain end-to-end but does not itself exercise #95 — the probe did that.

**Remaining P0s (2):** #6 + #7 — coupled architectural cluster. Currently latent because the global lock keeps shared orchestrator state safe; tackling either alone breaks more than it fixes. Defer until you actively need concurrent multi-campaign throughput.

**Quick wins (low effort, high impact):**
- #1, #2 — fix the character schema/migration bug (data corruption now)
- #4 — one-line tuple swap (purchase broken)
- #10 — one-character fix (`and` instead of `or`)
- #13 — `is not None` check (Schnell broken)
- #14 — drop OpenRouter key for OpenAI calls
- #16 — delete duplicate cog-load loop
- #19 — delete 8 orphan methods
- #21 — Anthropic prompt caching (recurring cost win)

**Recommended order:**
1. P0 #1, #2 first — every existing character is reading scrambled fields right now.
2. P0 #4 — broken player feature.
3. P0 #11, #12, #14, #15 — voice/TTS reliability cluster.
4. P0 #6, #7, #8 — multi-session safety (architectural; tackle as one slice).
5. Then P1 by area, starting with the cluster you're working on next.

**False positives / known-tracked items** (do NOT re-investigate):
- All items in `AUDIT_IMMERSION_2026_04.md` sections 1-6 already triaged there.
- The 7 web-parity items (4.1–4.7) are tracked as features, not bugs.
