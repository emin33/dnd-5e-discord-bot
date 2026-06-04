# Data-Flow Audit — 29 May 2026

A **trace-driven** cross-store pass by 5 parallel Opus 4.8 sub-agents, each owning ONE end-to-end
flow (not a file slice). Goal: the data-flow bugs the two prior layer/slice-scoped passes
(`AUDIT_FULL_2026_05.md`, 96 findings; `AUDIT_2026_05_28.md`, 58 findings) structurally couldn't
see — state dropped, double-written, or silently diverging as it moves between the orchestrator,
WorldState, knowledge graph, ChromaDB, scene registry, combat, memory, and the DB.

The just-fixed player/world-state **persistence cluster (N1/N2/N3/R3)** was verified intact by direct
re-read (effects.py:1154+, manager.py:147/159, coordinator.py:848/1016, orchestrator.py:2341) and is
NOT re-reported. **But this audit found that the cluster fix does not hold end-to-end** — see DF-1.

Flows: (1) combat state machine, (2) entity coherence across 4 stores, (3) narrator context
assembly, (4) session lifecycle/resume, (5) orchestrator turn lifecycle + state-ownership map.
Synthesis deduped overlap, re-ranked globally, and **the top P0s were re-verified by the synthesizer
reading the actual code** (cited inline). Raw sub-agent lists recoverable from this session's
transcript (agent IDs a16db69cefea7c7d9, ac4938b831c784fb1, a405f02ff4b94949b, ae30ea0b5dcfab5ee,
a34742d9d98213b23).

---

## §0. The unified cross-store state model (the 5 maps, merged)

Every finding below is a consequence of one fact: **two logical data — "a PC's mutable state" and
"an NPC's identity" — are each scattered across many stores with no single authority and no single
write seam.**

### Model A — A player character's mutable state (HP, temp-HP, conditions, slots, concentration, death-saves)

| # | Where it lives | Written by | Read by | Role |
|---|---|---|---|---|
| 1 | **Combatant** (`models/combat.py`) | combat apply_damage/heal/death-save/effects | combat UI, `## Combat` narrator block | **combat truth** |
| 2 | **`session.players[uid].character`** (the "live" object) | loaded ONCE at join (session.py:99) / recover (411); **never refreshed** | `sync_player`→WorldState `<party>`/`<acting_character>` (session.py:559); **`_sync_session_characters` persists it at session end (session.py:349)** | **what the narrator sees AND what session-end writes — and it is stale** |
| 3 | **`coordinator._character_cache`** | `_execute_spell` (slots, concentration only) | concentration checks | partial combat-cast cache |
| 4 | **fresh `get_by_id`** in `_execute_update_player` (effects.py:1196) & combat `_sync_player_characters` (combat.py:63) | N1 + combat sync mutate & `update()` THIS object | — discarded after persist | **the object actually persisted mid-session** |
| 5 | **#55 read-cache** (5 s TTL) | repo | repo readers | cache |
| 6 | **DB row** | (4) mid-session, then (2) clobbers at end | repo | nominal truth |

**The nexus:** object **(2)** is simultaneously the narrator's source and the session-end persist
source, yet **nothing ever copies combat (1) or N1 (4) mutations into it.** So the narrator reads
stale HP, and graceful session-end overwrites the DB with stale HP. (This is the "three diverging
Character instances" the prior audit deferred as *Option B* — now shown to cause active data loss.)

### Model B — An NPC's identity & state (name, location, alive, disposition, description, inventory)

| Store | Key for the SAME npc | Learns: introduce | dedup | move | rename | **die** |
|---|---|---|---|---|---|---|
| Scene registry | `SceneEntity.id` (fresh UUID #A, by-name) | ✅ | ❌ | ❌(no loc field) | ❌ | ❌ |
| npc DB table | `NPC.id` (fresh UUID #B) | ✅ | ❌ | ❌(`update_location` 0 callers) | ❌ | ❌(`mark_dead` 0 callers) |
| **WorldState** | `NPCState.id` (UUID #C) | ✅ | ✅ | ✅ | ⚠ extractor-only | ✅ |
| KG (NetworkX+SQLite) | **`slug(name)` on tool path / UUID #C on delta path** | ⚠ split | ❌(N9) | ⚠ | ✅ KG-only | ❌(no UPDATE_ENTITY handler) |
| ChromaDB | `entity_<slug|UUID>` (inherits KG) | ⚠ split | ❌ | ❌ | ⚠ re-add under slug | ❌(stale "alive" vector) |

**The nexus:** the narrator-supplied id is discarded; each store mints its own key; the tool path
keys KG/Chroma by **slug(name)** while the delta path and the matcher use the **UUID** — so cross-store
joins silently miss, and every lifecycle event after *introduce* updates only **WorldState**, leaving
2-4 stores stale. Death reaches **only** WorldState; the npc DB row stays `alive` and **resurrects the
NPC on the next session start**.

### Model C — What survives a restart (session lifecycle)

| Store | Persisted? | Restored by `recover_sessions`? | Correct? |
|---|---|---|---|
| NetworkX KG | ✅ SQLite write-through | ✅ `load()` (session.py:430) | ✅ |
| DB Character | ✅ | ✅ | partial (only to last sync; combat lost) |
| **WorldState** | ❌ nowhere | ❌ built with no `world_state` arg | **total loss** |
| **Combat** | ❌ (tables exist, never written) | ❌ | **total loss mid-combat** |
| **ChromaDB** | derived | ❌ sync skipped (N10) | empty until next fresh start |
| **Scene registry roster** | partial | ❌ NPC preload skipped | empty |
| **Active-campaign map** | ❌ module dict | ❌ `set_active_campaign` not called; `guild_id=0` | **session unreachable (zombie)** |
| `session_snapshot` table + `create_snapshot`/`get_latest_snapshot` | — | — | **dead code (0 callers)** |

---

## §1. The three root generators

Most findings collapse into three roots. Fixing the root dissolves the cluster.

- **ROOT-1 — No single session-owned Character + no single persist seam.** (Model A.) Generates
  DF-1, DF-2, DF-8, DF-9, DF-12, and the combat-cog sync gaps. **This is the deferred "Option B."
  This audit's evidence (active session-end clobber that *undoes the N1/N3 fix*) is the trigger the
  prior audit named for pulling B forward.**
- **ROOT-2 — No canonical cross-store entity id on the tool path (slug vs UUID).** (Model B.)
  Generates DF-3, DF-4, DF-10, DF-13, DF-14, DF-19, DF-23. **One surgical fix** (make `_effect_add_npc`
  and `_effect_ref_entity` reuse the WorldState NPCState UUID as the KG `node_id`, mirroring
  `_handle_new_npc`) collapses most of it. Independent of #6/#7+B.
- **ROOT-3 — Live world is never serialized; `recover_sessions` is a thin subset of `start_session`.**
  (Model C.) Generates DF-5, DF-6, DF-7, DF-16, DF-17, DF-22. The `combat`/`combatant`/`session_snapshot`
  tables already exist (migration 001) — this is migration/code drift, exactly the in-scope kind.

---

## §2. Counts

| Tier | NEW | Confirmed-prior re-touched | Total in this doc |
|------|-----|----------------------------|-------------------|
| P0   | 10  | 0                          | 10 |
| P1   | 14  | 4 (N9, C4, C16, C20-siblings) | 18 |
| P2   | 9   | 5 (N10, C10, C14, C15, C33) | 14 |
| **Total** | **33** | **9** | **42** |

No regressions of the `[x]`-fixed items (N1/N2/N3/R3, #1/#3/#9/#17/#52/#55/#94 sampled-intact).
**One fix-defeating interaction:** session-end clobber (DF-1) silently overwrites the values N1/N3
just-correctly-persisted — the fix is intact in code but does not survive a graceful session end.

---

## §3. Global punch list (ranked)

### P0 — data loss / silent corruption

#### DF-1 (P0, NEW, ROOT-1) — Session-end character sync clobbers all mid-session persistence with the stale live Character — undoes N1/N3
- File: `game/session.py:349` (`_sync_session_characters` → `char_repo.update(player.character)`) vs `effects.py:1196` & `bot/cogs/combat.py:63` (both persist a *fresh* `get_by_id`)
- Type: data-loss   Confidence: **high (synthesizer-verified)**
- Issue: `session.players[].character` is loaded once and never refreshed (session.py:99/411); N1's out-of-combat persist and combat's `_sync_player_characters` mutate *different* freshly-fetched objects, so at `end_session` `update(player.character)` writes pre-change HP/slots/conditions over everything persisted during the session — the **live object (2)** and the **DB (6)** disagree and the stale one wins. Mid-session the DB is correct; a graceful shutdown reverts it (a crash, ironically, preserves it).
- Fix: after N1/combat persist, copy the mutated fields back onto `session.players[uid].character` (or have session-end re-fetch). This is the Option-B "one session-owned Character + one persist seam" made concrete.

#### DF-2 (P0, NEW, ROOT-1) — The primary `/game` combat path persists HP/death-saves/conditions NOWHERE
- File: `bot/cogs/game.py` turn closures (`_show_player_turn_ui`/`_auto_run_npc_turns`); `_sync_player_characters` is called from `bot/cogs/combat.py` only (8 sites) — **0 calls in game.py** (synthesizer-confirmed via grep)
- Type: data-loss   Confidence: high
- Issue: the auto-from-narrative `/game` encounter mutates the **Combatant** for every wound/KO/condition but never calls any sync, so the **DB Character** never receives them. The `/combat turn` cog was fixed; this parallel (and primary) driver was not.
- Fix: extract combat.py's `_sync_player_characters` into a shared helper and call it from game.py after each action, on turn-end, and before `manager.end_combat()`.

#### DF-3 (P0, NEW, ROOT-2) — Tool-introduced NPCs get a `slug(name)` KG/Chroma id while WorldState/matcher use the UUID — graph fragments, KG context + vector recall silently miss
- File: `game/knowledge/bridge.py:550` (`_effect_add_npc`: `npc_id = slugify(npc_name)`) vs `:259` (`_handle_new_npc`: `npc_id = npc.id`, comment calls it "the cross-layer identity anchor"); matcher seeds the UUID (`matcher.py:97`)
- Type: divergence / coherence   Confidence: **high (synthesizer-verified)**
- Issue: the **KG/ChromaDB** key the tool-path NPC by slug while **WorldState** (and the BFS seed) key it by UUID, so the matcher never reaches the slug node — every narrator-tool-created NPC gets zero graph context and zero narrative recall, and a duplicate node appears if the extractor delta also fires. The sibling method one screen away documents the exact invariant this one breaks.
- Fix: in `_effect_add_npc`, resolve the WorldState NPCState by name and reuse its `.id` as the KG `node_id` (mirror `_handle_new_npc`); this also makes N9's `slugify(entity_id)` a pass-through for UUIDs.

#### DF-4 (P0, NEW, ROOT-2) — NPC death never propagates past WorldState; dead NPCs resurrect next session
- File: `game/knowledge/bridge.py` `convert_effects` has **no UPDATE_ENTITY case**; `npc_repo.mark_dead` (npc_repo.py:144) has **0 callers**; reload via `get_alive_by_campaign` (session.py:277)
- Type: data-loss / divergence   Confidence: **high (synthesizer-verified mark_dead has 0 callers)**
- Issue: `update_entity(status="dead")` sets `alive=False` in **WorldState** only; the **KG node**, **ChromaDB vector**, and **npc DB `is_alive`** all stay alive, so `get_alive_by_campaign` reloads the corpse into the scene registry on the next start and vector-match can still surface it.
- Fix: add an UPDATE_ENTITY handler to `convert_effects` emitting `UpdateNode(alive=false)` + Chroma re-index, and call `npc_repo.mark_dead` when status resolves to dead.

#### DF-5 (P0, NEW, ROOT-3) — WorldState is never persisted and never restored: every restart resets the world to empty
- File: `game/session.py:220` (fresh `WorldState()`) vs `:400-406` (recover builds `GameSession` with **no `world_state`**); no write path in `session_repo.py` (synthesizer-verified)
- Type: data-loss   Confidence: high
- Issue: current_location, scene_items, established_facts, recent_events, and the live NPC roster (promoted names/aliases/dispositions/inventory) live only in the in-memory **WorldState**; nothing serializes it and `recover_sessions` doesn't even instantiate one — after a restart the KG entities survive but the *world view* (who/where/what-happened) is gone.
- Fix: serialize `WorldState.model_dump()` into the already-existing `session_snapshot` table each turn / on `end_session`; rehydrate in `recover_sessions`.

#### DF-6 (P0, NEW, ROOT-3) — Combat is 100% in-memory; mid-combat restart loses HP/initiative/round and the `combat`/`combatant`/`session_snapshot` tables are dead code
- File: `game/combat/manager.py` `_active_combats` module dict; `orchestrator.py` combat-start never persists; tables in `migrations/001_initial_schema.sql`; `session_repo.create_snapshot`/`get_latest_snapshot` **0 callers** (synthesizer-verified)
- Type: data-loss / divergence   Confidence: high
- Issue: a full combat schema exists but is never written; `save_session` is called once (at start) so `active_combat_id` stays NULL and `state` stays `exploration` — a mid-combat restart loses all combatant state and the session resumes as `ACTIVE`, not `COMBAT`.
- Fix: persist Combat+Combatants on start/each round and `save_session(state="combat", active_combat_id=…)`; reload in `recover_sessions` when `active_combat_id` is set. (Or delete the dead tables/methods if combat is intentionally ephemeral.)

#### DF-7 (P0, NEW, ROOT-3) — Recovered sessions are unreachable: active-campaign map not rebuilt, `guild_id=0`
- File: `game/session.py:403` (`guild_id=0`) + no `set_active_campaign` in `recover_sessions`; sole caller is `/campaign` (campaign.py:998) (synthesizer-verified)
- Type: data-loss (orphaned session)   Confidence: high
- Issue: every slash cog routes via `get_active_campaign_id(ctx.guild_id)`, but recover never calls `set_active_campaign` and stores `guild_id=0`, so a recovered session sits in `_sessions` that no command can reach — a zombie until someone re-runs `/campaign`.
- Fix: resolve the campaign's real `guild_id` from its row and call `set_active_campaign` in `recover_sessions`; populate `session.guild_id`.

#### DF-8 (P0, NEW) — Player concentration is never checked when the PC is damaged on another combatant's turn
- File: `game/combat/coordinator.py:1081-1100` (`_is_concentrating` reads `_character_cache` with no async load; cache populated only for the active turn-taker)
- Type: coherence / data-loss   Confidence: high
- Issue: when a caster is the *target* of an attack/AoE on someone else's turn, the cache miss returns `False`, so `_check_concentration` is skipped — the **Combatant** takes damage but **Character.concentration_spell_id** is never tested and concentration silently persists forever.
- Fix: make `_is_concentrating`/`_check_concentration` use `await self._get_character(...)` (async load on miss), like `_execute_attack` does for the attacker.

#### DF-9 (P0, NEW, extends N3) — Breaking concentration clears Combatant effects but not Character/DB concentration (asymmetric N3)
- File: `game/combat/coordinator.py:1117` (`break_concentration`) vs `manager.py:1020`; N3 persists concentration on *cast* (coordinator.py:1016) but nothing un-persists on *break*
- Type: divergence   Confidence: high
- Issue: on a failed CON save, `break_concentration` removes the concentration CombatEffects, but neither cached **Character.concentration_spell_id** nor the **DB** (`update_concentration`) is cleared — the effects store and the Character disagree, and the next cast's "break existing" branch thinks the caster is still concentrating. N3's persist is one-directional.
- Fix: in `_check_concentration` failure, also set `character.concentration_spell_id = None` and `await repo.update_concentration(character.id, None)`.

#### DF-10 (P0, NEW, ROOT-2) — Narrator `ref_entity`/`update_entity` never reach the scene registry (UUID vs name lookup)
- File: `llm/effects.py:1022` & `:1081` (`get_by_name(entity_id)` where `entity_id` is the roster UUID)
- Type: coherence / data-loss   Confidence: high
- Issue: the roster YAML emits `id:<UUID>`; the narrator echoes it as `entity_id`; `get_by_name` substring-matches against SceneEntity *names*, which a UUID never matches — so the **scene registry** never receives the disposition/status/description updates that **WorldState** does, and registry hostility/combat-trigger state drifts from the narrator's intent.
- Fix: resolve by id first (`get_by_id(entity_id) or get_by_name(entity_id)`) and stamp the WorldState UUID onto the SceneEntity at creation.

### P1 — real divergence, fix soon

#### DF-11 (P1, NEW, ROOT-1) — Narrator's `<party>`/`<acting_character>` HP is stale and, during combat, contradicts the live `## Combat` block
- File: `game/session.py:559` (`sync_player` reads stale live Character) vs `:719` (combat_context reads live `combatant.hp_current`)
- Type: divergence   Confidence: high
- Issue: the same PC appears at 20/20 in `<world_state> party:` and 4/20 in `## Combat` in one prompt; the narrator is fed self-contradictory HP. Same ROOT-1 object as DF-1; the *visible* symptom is the narrator describing a full-HP PC who just took damage.
- Fix: DF-1's fix (write mutations back to the live Character) resolves this too.

#### DF-12 (P1, NEW) — Idempotency-duplicate effects re-run the WorldState mirror and re-bridge the KG (double-write)
- File: `llm/orchestrator.py:3518-3530` (guards on `result.success`, not `result.was_duplicate`); duplicate path effects.py:676
- Type: double-write   Confidence: high
- Issue: a retried turn (same `session_key:turn-N` key) returns `was_duplicate=True` without re-running the executor, but the orchestrator still calls `_sync_effect_to_world_state` (re-appends transfer ledger, **re-mints NPC-inventory entries**) and re-appends to `_last_executed_effects`, which Step 4b re-bridges into KG/Chroma. The executor is idempotent; its mirror is not. (P0 risk if your transport retries often — duplicate item minting is silent corruption.)
- Fix: gate the block on `if result.success and not result.was_duplicate:`.

#### DF-13 (P1, NEW, ROOT-2) — `ref_entity` name-promotion updates the KG node but not WorldState's `NPCState.name`
- File: `game/knowledge/graph.py:195` (`promote_entity_name`) vs no corresponding WorldState write on the ref_entity path
- Type: divergence   Confidence: high
- Issue: after promotion the **KG** says "Captain Vex" while **WorldState `NPCState.name`** stays "the hooded figure" (WorldState renames only via the separate extractor `new_name` path). The roster (from WorldState) and the KG context block show two canonical names for one entity.
- Fix: when applying a `NamePromotion`, also map the slug back to the UUID, update `NPCState.name`, and push the old name to `.aliases`.

#### DF-14 (P1, PRIOR=N9, ROOT-2) — `_effect_ref_entity` slugifies a UUID, targeting a nonexistent KG node (confirmed still open)
- File: `game/knowledge/bridge.py:679` (`node_id = slugify(entity_id)`)
- Type: divergence   Confidence: high
- Issue: `entity_id` is the NPCState UUID; `slugify` mangles it into neither the UUID node nor the slug node, so the alias promotion is a silent no-op — **KG never learns the alias WorldState recorded.** Dissolved by ROOT-2's fix.
- Fix: pass `entity_id` through unchanged; let `promote_entity_name` reject genuine unknowns.

#### DF-15 (P1, NEW) — `established_facts` are append-only and never retracted; contradicted facts persist forever in the narrator YAML
- File: `game/world_state.py:387` (apply_delta only appends `new_facts`); `StateDelta` (world_state.py:150) has no retraction field
- Type: coherence / data-loss (false world truth)   Confidence: high
- Issue: when the world changes (bridge destroyed, NPC dies), the old fact stays in `facts:` beside the new one, and the system prompt tells the narrator facts "must not be contradicted." Distinct from C14 (size/cost) — this is *contradiction*. Compounds with C14's unboundedness.
- Fix: add `remove_facts`/`superseded_facts` to `StateDelta` + extractor retraction, or key facts by subject and replace on update.

#### DF-16 (P1, NEW, ROOT-3) — Scene-registry NPC roster not repopulated on recover; narrator loses "who is in the room"
- File: `game/session.py:271-292` (start preloads alive NPCs) vs `:397-433` (recover skips it)
- Type: data-loss / coherence   Confidence: high
- Issue: `start_session` seeds the registry from `get_alive_by_campaign`; recover doesn't, so a recovered session's `get_triage_context`/`get_narrator_roster` is empty and `_build_context` feeds an empty scene block. One of four sibling omissions in recover (Chroma, WorldState, scene roster, active-campaign).
- Fix: factor start_session's post-load init into shared helpers invoked by both paths.

#### DF-17 (P1, NEW, ROOT-3) — Two idempotency mechanisms diverge across restart (volatile narrator vs durable tool)
- File: `llm/orchestrator.py:573,3446` (in-mem `_BoundedKeySet`) vs `:4273-4299` (durable `transaction_log`)
- Type: divergence   Confidence: high
- Issue: narrator-emitted effects dedup against an in-memory set whose key `session_key:turn-{world_state.turn}` can't even be rebuilt after restart (no world_state → uuid fallback), while the structured tool path dedups against the durable `transaction_log`. Same logical "apply an effect" has inconsistent cross-restart guarantees.
- Fix: route the narrator effect path through `transaction_log` too (the durable ledger is the correct model — see Option-B note).

#### DF-18 (P1, NEW) — Scene-registry roster never location-filtered or cleared on move; narrator told departed NPCs are "AUTHORITATIVE present"
- File: `game/scene/registry.py:290,343` (no location filter) consumed at `game/session.py:735,739`; registry never cleared on location_change
- Type: divergence   Confidence: high
- Issue: `<current_scene>` injects the registry roster ("AUTHORITATIVE", keyed by *slug*) with no location concept; after Tavern→Forest, `<world_state> npcs_here` correctly shows Forest NPCs but the registry still lists the Tavern NPCs as present — and tells the narrator to `ref_entity` them by slug while WorldState says UUID. Distinct store from C10 (scene_items) / C16 (name-resolvable).
- Fix: clear/location-filter the scene registry on `location_change`; emit one roster keyed on the WorldState UUID.

#### DF-19 (P1, NEW, ROOT-2) — npc DB `location` is written from the scene description and never updated
- File: `game/scene/registry.py:432,441,451` (`location = self._scene_description[:100]`); `npc_repo.update_location` **0 callers** (synthesizer-verified)
- Type: divergence   Confidence: high
- Issue: `sync_to_npc_repo` stores a 100-char slice of the scene *description* into the npc row's `location`, and `update_location` is never called, so the **npc DB location** permanently disagrees with **WorldState `NPCState.location`**; `get_at_location` (used to repopulate scenes) matches garbage.
- Fix: pass the real `current_location` into `sync_to_npc_repo` and write it; call `update_location` on moves.

#### DF-20 (P1, NEW) — Combat-cog commands `death_save`/`stabilize`/`next`/`temp_hp`/`set_hp` mutate persistent state, never sync (C20 siblings)
- File: `bot/cogs/combat.py:861` (death_save), `:962` (stabilize), `:493` (next→ongoing-damage effects), `:1001` (temp_hp), `:1041` (set_hp)
- Type: data-loss   Confidence: high
- Issue: each mutates the **Combatant** (death-save state machine, ongoing AoE damage, manual HP) with no `_sync_player_characters`, so the **DB Character** diverges — the death-save and effect-processing facets C20 (attack-only) did not name.
- Fix: call `_sync_player_characters(manager)` after each when the target is a player.

#### DF-21 (P1, NEW) — `_sync_player_characters` runs BEFORE `end_turn` applies end-of-turn ongoing damage (ordering)
- File: `bot/cogs/combat.py:1528` (sync) then `coordinator.py:203` (`process_end_of_turn_effects`)
- Type: ordering   Confidence: high
- Issue: even on the "good" `/combat turn` path, the DB sync happens, then end-of-turn ongoing damage/expiry hits the **Combatant**, so that final delta lands in the **DB** only on the next sync — and is lost entirely if combat ends on this turn.
- Fix: move `_sync_player_characters` to *after* `await coordinator.end_turn(current)`.

#### DF-22 (P1, NEW) — `_scratchpad`/`_scratchpad_turn` leak across sessions on the singleton
- File: `llm/orchestrator.py:580-582,671-679`; read at `:759,770,2643`; not cleared by `set_session(None)`
- Type: coherence   Confidence: high
- Issue: the narrator scratchpad (tensions, NPC moods) and the `_scratchpad_turn` counter live on the process-wide orchestrator and survive session switches, so **session B's narrator is injected with session A's notes** and shares one turn counter. NOT dissolved by #6/#7+B unless the scratchpad is explicitly moved into the session.
- Fix: key the scratchpad by `session_key` (or move it onto the session); clear on session switch.

#### DF-23 (P1, NEW, ROOT-2 / extends C4) — `<past_narration>` recall silently returns empty due to name-vs-UUID chunk tags
- File: `llm/orchestrator.py:1146,1228` (chunk tagged with raw `npc_name`) vs `:1074` recall filtered by UUID seeds (`vector_store.py:459`)
- Type: divergence / continuity   Confidence: med
- Issue: `add_narrative_chunk` tags chunks with raw NPC *names* (the ADD_NPC branch of `_narrator_ref_ids`) while recall filters by UUID seeds, so a chunk tagged "Fred" never intersects Fred's UUID seed — `<past_narration>` comes back empty and the narrator forgets how it described that NPC. The narrative-recall facet of C4.
- Fix: resolve `_narrator_ref_ids` ADD_NPC entries to canonical node UUIDs before tagging (store only node-ids).

#### DF-24 (P1, NEW) — Streaming callback ignored on skill-check and purchase narration paths
- File: `llm/orchestrator.py:2825` (`_narrate_outcome`) & `:2472` (`_narrate_mechanical_result`) always call `chat`; only `_narrate_action:2691` honors `_on_narrative_token`
- Type: coherence   Confidence: high
- Issue: a turn producing a skill-check resolution or a purchase/sell result never streams even when the frontend wired streaming — players get the whole block at once for those action types. N2 fixed the followup tool-set, not the streaming-invocation parity.
- Fix: add the same `if self._on_narrative_token and hasattr(client,"chat_stream")` branch to both methods.

### P2 — latent / health (most state-ownership smells are P2 behind the global lock)

- **DF-25 (P2, NEW, ROOT-1)** — `CombatTurnManager` (`bot/views/combat_actions.py:899-1023`) runs full turns with zero persistence; live API surface, currently not the primary driver. Thread a persist callback.
- **DF-26 (P2, NEW, ROOT-3)** — Voice combat (`voice/frontend.py:207`) has no persist seam and depends on `game/combat/turn_loop.py:run_combat_loop`, which is **dead (0 callers)**. Wire a seam or delete (corroborates C33).
- **DF-27 (P2, NEW)** — `sync_to_character` (`manager.py:758`) derives `expires_round` from the in-memory round counter that is discarded at combat end, so out-of-combat timed conditions are meaningless against the next combat's numbering (mitigated by add_player's past-round guard, so they silently drop). Use wall-clock for out-of-combat durations or document combat-scoping.
- **DF-28 (P2, NEW, extends C14)** — Two parallel unbounded fact stores (`memory/blocks.py` `pinned_facts` + WorldState `established_facts`), and session.py:641 copies pinned→WS each turn, so the narrator sees the same facts twice in `<memory>` and `<world_state>`. Pick one canonical store; cap with a ring buffer + `set`.
- **DF-29 (P2, NEW, ROOT-2)** — `_sync_npcs_to_registry` (`orchestrator.py:3316`) creates SceneEntities with no `npc_id`/UUID link, so every cross-store join for delta-path NPCs depends on fuzzy `get_by_name`. Stamp the shared id.
- **DF-30 (P2, NEW, extends C15)** — An entity touched by both the delta path (Step 3.6c) and the effect path (Step 4b) is `add_entity_description`-re-embedded **twice in one turn** (orchestrator.py:1207 & 1353). Gate on a description hash.
- **DF-31 (P2, NEW)** — Bookend phase is parsed by `world_state_yaml.split("phase:")[1]` (`brains/base.py:232`); any other `phase:` substring (a fact/NPC note) mis-selects rule injection. Pass `world_state.phase` as a typed `BrainContext` field.
- **DF-32 (P2, NEW)** — `_last_narrator_routing`/token-stat `self._*` fields (orchestrator.py:1124,1411) are read in finalize but written only inside narrate; an early-return turn logs the *previous* turn's routing/tokens. Reset per-turn or carry on the turn record.
- **DF-33 (P2, NEW)** — Streaming path narrates with NO `tools=` passed (orchestrator.py:2691); prose is composed blind to schemas and tool capture is deferred to the followup, so what the prose *describes* and what the followup can *reconstruct* can diverge. Stream with tools where supported, or document the reliance on followup+extractors.

Confirmed-prior also re-touched (not re-counted): **N10** (recover skips Chroma sync), **C10** (scene_items not cleared), **C14** (established_facts unbounded), **C15** (re-index O(N)/turn), **C16** (departed NPCs resolvable), **C33** (voice path no immersion), **C4/C5** (extractor hint mixing / matcher rebuilt) — all verified still open.

---

## §4. State-ownership map (the #6/#7 + Option-B blueprint)

From the orchestrator trace. All TURN-SCOPED rows are latent behind `session.py:582 _processing_lock`
(one global lock serializes every turn of every session — the deferred #7).

**TURN-SCOPED state wrongly on the singleton (the #6 corruption surface, dissolved by a per-turn `TurnContext`):**
| `self._field` | holds | written | read | verdict |
|---|---|---|---|---|
| `self.narrator.client` | this turn's tier client | orch:2196 | 2331,2472,2693,2825,1119 | racy w/o lock — headline #6 site (C9) |
| `self._on_narrative_token` | this turn's stream callback | 767 | 2691,2697 | leaks across turns |
| `self._effect_executor.acting_character_id` | acting PC for update_player | 3452 | effects.py:1146 | racy — defeats N1's "never wrong PC" guard |
| `self._last_executed_effects` | effects to KG-bridge | 3434/3530 | 1308,1317 | racy — could bridge another turn's effects |
| `self._last_narrator_prose` | prose for dedup judge | 1117 | 3477 | racy |
| `self._last_narrator_routing` / `_last_*_tokens` | telemetry | 2200,2370 | 1124,1411 | leaks across turns (DF-32) |

**SESSION-SCOPED on the singleton (dissolved by making session/registry method args):**
| `self._current_session` | which session every step mutates | session.py:585, cleared 676 | ~40 sites | leaks across sessions w/o lock — the lock is load-bearing exactly here |
| `self._scene_registry` | this session's registry | 586, cleared 677 | 614,897,1130,3302… | same |
| `self._scratchpad`/`_scratchpad_turn` | narrator continuity | 580,671 | 759,770,2643 | **leaks across sessions ALWAYS (DF-22) — NOT cleared on switch; needs own fix** |

**Correctly process-global:** `_applied_effects` (namespaced keys), the stateless brain/extractor services
(modulo the `narrator.client` swap; `self.rules`/`self._nli_validator` are dead — C6/C8).

**What the #6/#7 + Option-B refactor dissolves vs. needs its own fix:**
- **Dissolved by Option B** (one session-owned Character + one persist seam): DF-1, DF-2, DF-11, DF-9 (partly), and the combat-cog sync gaps DF-20/DF-21 (if the seam is central). **This audit is the trigger to pull B forward — DF-1 is active data loss, not latent.**
- **Dissolved by #6/#7** (per-turn context + client-as-argument): every TURN-SCOPED row above; `_current_session`/`_scene_registry`. Latent today behind the lock.
- **Need their own fix regardless:** ROOT-2 (DF-3/4/10/13/14/19/23/29 — one surgical bridge.py id change), ROOT-3 (DF-5/6/7/16/17 — serialize WorldState+combat, recover parity), DF-22 (scratchpad keying), DF-15 (fact retraction), DF-12 (was_duplicate guard), DF-18 (registry location-clear), DF-8 (concentration async load).

---

## §5b. Refactor progress / remaining TODO (updated 2026-05-29)

Single-authority refactor (Option B) is being done in tested increments. Status:
- [x] **ROOT-2 core** — `_effect_add_npc` keys KG nodes on the WorldState UUID (DF-3). Remaining ROOT-2: `_effect_ref_entity`/registry resolution by id (DF-10) + UPDATE_ENTITY→KG handler + `mark_dead` wiring (DF-4) — folded into **Stage C** below (the roster hands the narrator a name-slug, so this needs the canonical-id decision, not a one-liner).
- [x] **DF-1** — `_sync_session_characters` no longer clobbers with the stale live object.
- [x] **Stage A.1** — narrated effects (`_execute_update_player`) mutate the session-owned `Character`.
- [x] **Stage A.2** — combat (coordinator reads + `_sync_player_characters` writes) on the same session-owned `Character`.
- [x] **DF-2** — `/game` combat persists after player actions AND NPC turns (`coordinator.persist_player_characters`).
- [ ] **Stage C — NPC identity end-to-end** (DF-10 + ROOT-2 remainder): pick one canonical NPC id, make the roster, ref-resolution, scene-registry, KG, and ChromaDB all use it. Medium, user-facing, independent of concurrency. **Next.**
- [ ] **Stage B — per-session state/lock** (#6/#7): the ~97 `self._last_*` orchestrator refs → a TurnContext; move the global `_processing_lock` per-session. Biggest/riskiest; NOT urgent (single-server is fine behind the global lock). Deferred.
- [ ] **ROOT-3** — serialize WorldState+combat to the dead tables; `recover_sessions` reuse `start_session` init (DF-5/6/7/16). Independent of B/C.

All increments above are committed on branch `audit-and-single-authority-refactor` (501 tests green) except DF-2 (done, uncommitted at time of writing).

## §5. Recommended sequencing

1. **ROOT-2 first — it's one surgical fix for ~8 findings.** Make `_effect_add_npc`/`_effect_ref_entity`
   reuse the WorldState NPCState UUID as the KG node_id (DF-3/14), add the UPDATE_ENTITY→KG handler +
   `mark_dead` call (DF-4), resolve registry by id (DF-10). Cheap, high-leverage, independent.
2. **DF-1 + DF-2 + the combat-cog sync gaps — and seriously consider pulling Option B forward now.**
   The clobber actively undoes the persistence cluster just fixed; the prior audit named "another
   persist-bug found" as B's trigger and this audit found several. At minimum, ship the
   write-back-to-live-Character band-aid (DF-1) + the game.py sync (DF-2) immediately.
3. **ROOT-3 — serialize WorldState + combat into the existing dead tables; make `recover_sessions`
   call the same init helpers as `start_session`** (DF-5/6/7/16, N10). Restart currently silently
   drops the world.
4. **Combat correctness:** DF-8 (concentration on off-turn damage), DF-9 (concentration-break persist).
5. **DF-12 (one-line `was_duplicate` guard), DF-15 (fact retraction), DF-18, DF-22.**
6. **#6/#7 + the rest of Option B** as the deliberate ~1-2 day refactor, using §4 as the blueprint.
   Do the durable-`transaction_log` unification (DF-17) here.
7. P2 opportunistically when in-file.

## How this was produced
5 parallel Opus 4.8 sub-agents, one per end-to-end flow, each given the prior audit docs, told to
verify-not-trust the `[x]` items and trace concrete operations across every store. Synthesis merged
the 5 state-flow maps into the unified model (§0), grouped findings under three root generators,
re-ranked globally, and re-verified the 10 P0s by direct code reads (cited "synthesizer-verified").
