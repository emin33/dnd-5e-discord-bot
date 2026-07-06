"""Single-writer WorldStateStore (REFACTOR_PLAN Step 4).

The write authority over one :class:`WorldState`. The three write paths —
the narrator-effect sync chain (moved here verbatim from
``DMOrchestrator._sync_effect_to_world_state``), the StateDelta extractor
pipeline (:meth:`apply_delta`), and the session-layer turn bookkeeping —
funnel through this one seam instead of mutating the state object from
three modules. Step 5 slots the dedup pass inside :meth:`apply_delta`;
nothing else should ever need to know HOW a change lands.

Import note: :class:`ProposedEffect`/:class:`EffectType` come from
``llm.effects``, which is a data-only DTO module at import time (the
executor's game imports are lazy) — this is the same models-grade edge the
coordinator already takes for NarrationSpec; the eventual ``protocols.py``
boundary owns relocating these DTOs.
"""

from typing import TYPE_CHECKING, Iterable, Optional

import structlog

from .world_state import NPCState, NPCUpdate, StateDelta, WorldState
from ..llm.effects import EffectType, ProposedEffect

if TYPE_CHECKING:
    from ..models import Character

logger = structlog.get_logger()


class WorldStateStore:
    """Write authority over one :class:`WorldState` instance.

    Thin and stateless beyond the wrapped reference — sessions derive one
    per access (``GameSession.world_store``), so reassigning
    ``session.world_state`` can never orphan a stale wrapper.
    """

    def __init__(self, state: WorldState) -> None:
        self._state = state

    @property
    def state(self) -> WorldState:
        """The wrapped state — transitional READ access until the read-only
        view lands. Do not mutate through this; every write goes through
        the apply methods below."""
        return self._state

    # ── Session-layer bookkeeping seams ───────────────────────────────────

    def begin_turn(self, characters: Iterable["Character"]) -> None:
        """Turn bookkeeping: advance the counter, refresh party snapshots.

        Moved from ``process_message``'s inline turn-start block — the
        snapshot fields mirror what the narrator's ``<party>`` block reads.
        """
        self._state.increment_turn()
        for character in characters:
            conditions = [
                c.condition.value for c in character.conditions
            ] if character.conditions else []
            self._state.sync_player(
                name=character.name,
                hp=character.hp.current,
                max_hp=character.hp.maximum,
                conditions=conditions,
                concentration=character.concentration_spell_id or "",
            )

    def reconcile_phase(self, in_combat: bool) -> None:
        """Align the narrative phase with the session's combat mode.

        One method serves both the ModeMachine push/pop (enter/exit combat)
        and ``process_message``'s per-turn reconcile — the OTHER phase
        writer is the delta extractor, and its narrative phases (dialogue,
        rest, …) are deliberately preserved outside combat: only a literal
        "combat" phase resets to exploration.
        """
        if in_combat:
            if self._state.phase != "combat":
                self._state.phase = "combat"
        elif self._state.phase == "combat":
            self._state.phase = "exploration"

    def add_established_fact(self, fact: str) -> None:
        """Record a pinned fact once (the memory→world-state fact sync)."""
        if fact and fact not in self._state.established_facts:
            self._state.established_facts.append(fact)

    # ── The extractor pipeline's apply seam ──────────────────────────────

    async def apply_delta(
        self, delta: StateDelta, *, narrator_prose: str = ""
    ) -> list[str]:
        """Dedup → validate → write: the extractor pipeline's apply seam.

        The dedup pass runs HERE, inside the write pipeline — never as an
        event or a coordinator method (plan anti-re-flag rule; Step 5).
        ``narrator_prose`` gives the brain judge the turn's prose for its
        paraphrase decision; with no proposed NPCs or an empty roster the
        judge is never consulted.
        """
        if delta.new_npcs and self._state.npcs:
            delta = await self._dedup_delta(delta, narrator_prose)
        return self._state.apply_delta(delta)

    async def _dedup_delta(
        self, delta: StateDelta, narrator_prose: str
    ) -> StateDelta:
        """Run each ``delta.new_npcs`` entry through the brain dedup judge.

        Mirrors :meth:`dedup_effect` (the narrator-side ADD_NPC rewrite),
        but operates on the state-extractor's proposed ``new_npcs`` before
        they land. On high-confidence rewrite the entry is dropped from
        ``new_npcs`` and an ``NPCUpdate(id=target_id, add_aliases=[…])``
        is appended so the write records the alias against the existing
        entity.

        Default safe: any judge error / parse failure / unknown target id
        keeps the original ``new_npcs`` entry. False negatives (missed
        dedup) recover next turn when the registry has more recency
        signal; false positives (wrongly merging two distinct characters)
        do not, so we bias to keep.
        """
        world_state = self._state

        try:
            from ..llm.extractors.dedup_judge import get_dedup_judge
        except Exception as e:
            logger.warning("extractor_dedup_judge_import_failed", error=str(e), exc_info=True)
            return delta

        judge = get_dedup_judge()
        surviving: list = []
        appended_updates: list = []

        for proposed in delta.new_npcs:
            try:
                decision = await judge.judge_add_npc(
                    proposed_name=proposed.name or "",
                    proposed_description=proposed.description or "",
                    narrator_prose=narrator_prose,
                    existing_npcs=list(world_state.npcs.values()),
                    current_turn=world_state.turn,
                )
            except Exception as e:
                logger.warning("extractor_dedup_judge_call_exception", error=str(e), exc_info=True)
                surviving.append(proposed)
                continue

            if not decision.is_rewrite:
                surviving.append(proposed)
                continue

            target_id = decision.target_id
            if target_id not in world_state.npcs:
                logger.warning(
                    "extractor_dedup_target_not_in_world_state",
                    target_id=target_id,
                )
                surviving.append(proposed)
                continue

            alias = (decision.alias or proposed.name or "").strip()
            update = NPCUpdate(
                id=target_id,
                add_aliases=[alias] if alias else None,
            )
            appended_updates.append(update)

            logger.info(
                "extractor_dedup_rewrite_applied",
                proposed_name=proposed.name,
                target_id=target_id,
                alias=alias,
            )

        delta.new_npcs = surviving
        if appended_updates:
            delta.npc_updates = list(delta.npc_updates) + appended_updates
        return delta

    async def dedup_effect(
        self, effect: ProposedEffect, narrator_prose: str = ""
    ) -> ProposedEffect:
        """Run the brain dedup judge on an ADD_NPC effect.

        The pre-execution step of the effect write pipeline (whose tail is
        :meth:`apply_effect`): if the judge confidently identifies the
        proposed NPC as one already in the roster (paraphrase drift), the
        effect is rewritten to a REF_ENTITY pointing at the existing id and
        the paraphrased name is stashed as an alias — a world-state write
        that previously lived on the orchestrator (exactly the sub-object
        bypass class the Step-4 review flagged as its watch item).

        Self-gating: anything that isn't an ADD_NPC with a name against a
        non-empty roster passes through untouched, judge never consulted.
        Default safe: returns the original on any judge error / parse
        failure / "accept" decision.
        """
        world_state = self._state

        if (
            effect.effect_type != EffectType.ADD_NPC
            or not effect.npc_name
            or not world_state.npcs
        ):
            return effect

        try:
            from ..llm.extractors.dedup_judge import get_dedup_judge
        except Exception as e:
            logger.warning("dedup_judge_import_failed", error=str(e), exc_info=True)
            return effect

        judge = get_dedup_judge()
        try:
            decision = await judge.judge_add_npc(
                proposed_name=effect.npc_name or "",
                proposed_description=effect.npc_description or "",
                narrator_prose=narrator_prose,
                existing_npcs=list(world_state.npcs.values()),
                current_turn=world_state.turn,
            )
        except Exception as e:
            logger.warning("dedup_judge_call_exception", error=str(e), exc_info=True)
            return effect

        if not decision.is_rewrite:
            return effect

        # Rewrite ADD_NPC → REF_ENTITY pointing at the existing id.
        # Also accumulate the paraphrased name as an alias on the
        # existing NPCState so future paraphrases match more easily.
        target_id = decision.target_id
        existing = world_state.npcs.get(target_id)
        if existing is None:
            # Judge proposed an id that doesn't exist — be safe and accept the original
            logger.warning(
                "dedup_judge_target_not_in_world_state",
                target_id=target_id,
            )
            return effect

        if decision.alias and decision.alias != existing.name and decision.alias not in existing.aliases:
            existing.aliases.append(decision.alias)

        logger.info(
            "dedup_rewrite_applied",
            original_name=effect.npc_name,
            target_id=target_id,
            existing_name=existing.name,
            alias=decision.alias,
        )

        # Build the rewritten REF_ENTITY effect — preserve idempotency-relevant
        # context (no idempotency key change; that's tied to tool-call index).
        return ProposedEffect(
            effect_type=EffectType.REF_ENTITY,
            ref_entity_id=target_id,
            ref_alias_used=decision.alias,
            # Preserve any dialogue tracking the narrator added on the
            # original add_npc call — those still belong to this entity.
            dialogue_indices=list(effect.dialogue_indices),
            dialogue_emotions=list(effect.dialogue_emotions),
        )

    # ── The narrator-effect sync seam ─────────────────────────────────────

    def apply_effect(self, effect: ProposedEffect) -> None:
        """Sync a successfully executed effect into WorldState.

        This is the critical bridge: effects execute mechanically via the
        effect system, and here we record them into WorldState so the
        narrator sees them in the YAML snapshot next turn. Moved verbatim
        from ``DMOrchestrator._sync_effect_to_world_state`` (Step 4); each
        branch's exact diff is pinned in tests/unit/test_world_state_sync.py.
        """
        world_state = self._state

        etype = effect.effect_type

        if etype == EffectType.SPAWN_OBJECT:
            obj_id = effect.object_name or "unknown_item"
            desc = effect.object_description or effect.object_name or "an object"
            world_state.spawn_item(obj_id, desc)
            world_state.record_transfer(f"{desc} appeared in the scene")

        elif etype == EffectType.TRANSFER_ITEM:
            item = effect.item_name or "an item"
            src = effect.from_entity or "somewhere"
            dst = effect.to_entity or "someone"
            # If player picked up from scene, remove from scene items
            if src.startswith("scene"):
                world_state.remove_item(effect.object_name or effect.item_name or "")
            world_state.record_transfer(f"{item} moved from {src} to {dst}")

        elif etype == EffectType.GRANT_CURRENCY:
            parts = []
            if effect.gold: parts.append(f"{effect.gold}gp")
            if effect.silver: parts.append(f"{effect.silver}sp")
            if effect.copper: parts.append(f"{effect.copper}cp")
            if effect.platinum: parts.append(f"{effect.platinum}pp")
            if effect.electrum: parts.append(f"{effect.electrum}ep")
            amount = ", ".join(parts) if parts else "currency"
            src = effect.source or "someone"
            dst = effect.target or "player"
            world_state.record_transfer(f"{src} gave {amount} to {dst}")

        # No APPLY_DAMAGE branch: its executor now fails honestly (it never
        # mutated HP), so this sync — which runs only on executor success —
        # can never see one. APPLY_HEALING / ADD_CONDITION / REMOVE_CONDITION
        # were deleted outright (no producer on any path); player-side HP and
        # conditions flow through UPDATE_PLAYER below.

        elif etype == EffectType.ADD_NPC:
            npc_name = effect.npc_name or "Unknown"
            # Mint a new NPCState only if we don't already have a matching
            # one (by name or alias). Dedup by id is enforced inside
            # apply_delta; here at the effect-sync we use name lookup as
            # the cheap pre-check (the brain dedup judge runs upstream of
            # this to catch paraphrases).
            existing = world_state._find_npc(npc_name)
            if existing is None:
                npc = NPCState(
                    name=npc_name,
                    location=world_state.current_location,
                    disposition=effect.npc_disposition or "neutral",
                    description=effect.npc_description or "",
                    last_seen_turn=world_state.turn,
                )
                world_state.npcs[npc.id] = npc

        elif etype == EffectType.CONSUME_RESOURCE:
            resource = effect.resource_name or effect.item_name or "a resource"
            world_state.record_transfer(f"Consumed {effect.quantity}x {resource}")

        elif etype == EffectType.SET_FLAG:
            if effect.flag_name:
                world_state.global_flags[effect.flag_name] = effect.flag_value

        elif etype == EffectType.REMOVE_ENTITY:
            # Remove from scene items if present. Keys are object names but
            # the target may arrive in the roster's [id: slug] dialect, so
            # exact-pop alone missed 'rusty-key' vs 'Rusty Key' — compare
            # slugified too (final review).
            if effect.target:
                from .knowledge.models import slugify
                target_slug = slugify(effect.target)
                for key in list(world_state.scene_items):
                    if key == effect.target or (
                        target_slug and slugify(key) == target_slug
                    ):
                        world_state.remove_item(key)

        elif etype == EffectType.CHANGE_LOCATION:
            # Narrator-authoritative location change: overrides whatever the
            # state extractor may have produced for this turn. The state
            # extractor still runs (as fallback for cases where the narrator
            # didn't tool-call), but if both fired the narrator wins because
            # this sync runs AFTER the extractor's apply_delta.
            new_loc = (effect.location_name or "").strip()
            if new_loc:
                if world_state.current_location != new_loc:
                    # Track previous location as a connected one (it's reachable)
                    if (
                        world_state.current_location
                        and world_state.current_location not in world_state.connected_locations
                    ):
                        world_state.connected_locations.append(world_state.current_location)
                world_state.current_location = new_loc
                if effect.location_description:
                    world_state.location_description = effect.location_description
                world_state.record_transfer(f"party arrived at {new_loc}")

        elif etype == EffectType.REF_ENTITY:
            # Narrator referenced an existing roster entity — bump recency
            # so relevance-based roster selection (last_seen_turn window)
            # keeps this entity surfaced. Lightweight; no other state change.
            ref_id = (effect.ref_entity_id or "").strip()
            if ref_id:
                npc_state = world_state.npcs.get(ref_id) or world_state._find_npc(ref_id)
                if npc_state is not None:
                    npc_state.last_seen_turn = world_state.turn
                    # If the prose used a different alias than the canonical
                    # name, accumulate it. Helps future paraphrase resolution.
                    alias = (effect.ref_alias_used or "").strip()
                    if alias and alias != npc_state.name and alias not in npc_state.aliases:
                        npc_state.aliases.append(alias)

        elif etype == EffectType.UPDATE_ENTITY:
            # Narrator-authoritative entity update: mirrors the SceneEntity
            # mutations that _execute_update_entity already applied so the
            # WorldState YAML the narrator sees next turn matches.
            entity_id = (effect.update_entity_id or "").strip()
            if entity_id:
                npc_state = world_state.npcs.get(entity_id) or world_state._find_npc(entity_id)
                if npc_state is not None:
                    # Bump recency on any update too
                    npc_state.last_seen_turn = world_state.turn
                    if effect.update_disposition is not None:
                        npc_state.disposition = effect.update_disposition.lower()
                    if effect.update_status is not None:
                        # WorldState uses .alive bool — translate status
                        status = effect.update_status.lower()
                        if status in ("dead",):
                            npc_state.alive = False
                        elif status in ("alive", "wounded", "unconscious", "fled", "captured"):
                            # Keep alive=True but record status in notes for narrator visibility
                            if status != "alive":
                                npc_state.notes = (
                                    (npc_state.notes + " " if npc_state.notes else "")
                                    + f"[{status}]"
                                ).strip()
                    if effect.update_importance is not None:
                        npc_state.important = bool(effect.update_importance)
                    if effect.update_description_addition:
                        addition = effect.update_description_addition.strip()
                        if addition and addition not in (npc_state.description or ""):
                            npc_state.description = (
                                (npc_state.description + " " if npc_state.description else "")
                                + addition
                            ).strip()
                    # NPC inventory deltas — adds and removes apply directly
                    # to the NPCState.inventory list so the narrator sees
                    # them in the YAML next turn.
                    if effect.update_add_items:
                        for item in effect.update_add_items:
                            item_norm = item.strip()
                            if item_norm and item_norm not in npc_state.inventory:
                                npc_state.inventory.append(item_norm)
                    if effect.update_remove_items:
                        for item in effect.update_remove_items:
                            item_norm = item.strip().lower()
                            # Remove case-insensitively
                            npc_state.inventory = [
                                i for i in npc_state.inventory
                                if i.strip().lower() != item_norm
                            ]

        elif etype == EffectType.UPDATE_PLAYER:
            # Consolidated player-state mutation. The Character object lives
            # in the session; we update what we can on the WorldState side
            # (player snapshot + transfer log) and rely on the orchestrator's
            # downstream wiring (inventory_repo, character_repo) for the
            # mechanical mutations.
            log_parts: list[str] = []
            if effect.player_item_grant:
                names = [e.get("name", "") for e in effect.player_item_grant if e.get("name")]
                if names:
                    log_parts.append(f"player gained: {', '.join(names)}")
                # Also: if any grant has source='npc:...', mirror as removal
                # from that NPC's inventory.
                for entry in effect.player_item_grant:
                    src = (entry.get("source") or "").strip()
                    if src.startswith("npc:"):
                        npc_id = src.split(":", 1)[1]
                        npc_state = world_state.npcs.get(npc_id) or world_state._find_npc(npc_id)
                        if npc_state is not None:
                            item_norm = entry.get("name", "").strip().lower()
                            npc_state.inventory = [
                                i for i in npc_state.inventory
                                if i.strip().lower() != item_norm
                            ]
            if effect.player_item_remove:
                names = [e.get("name", "") for e in effect.player_item_remove if e.get("name")]
                if names:
                    log_parts.append(f"player lost: {', '.join(names)}")
                # If any remove has destination='npc:...', mirror as add to
                # that NPC's inventory (this is how "I give the relic to the
                # innkeeper" sticks 20 turns later).
                for entry in effect.player_item_remove:
                    dst = (entry.get("destination") or "").strip()
                    if dst.startswith("npc:"):
                        npc_id = dst.split(":", 1)[1]
                        npc_state = world_state.npcs.get(npc_id) or world_state._find_npc(npc_id)
                        if npc_state is not None:
                            item = entry.get("name", "").strip()
                            if item and item not in npc_state.inventory:
                                npc_state.inventory.append(item)
            if effect.player_currency_delta:
                log_parts.append(f"currency: {effect.player_currency_delta}")
            if effect.player_hp_delta is not None:
                sign = "+" if effect.player_hp_delta > 0 else ""
                log_parts.append(
                    f"HP {sign}{effect.player_hp_delta}"
                    + (f" ({effect.player_damage_type})" if effect.player_damage_type else "")
                )
            if effect.player_add_conditions:
                log_parts.append(f"conditions+: {effect.player_add_conditions}")
            if effect.player_remove_conditions:
                log_parts.append(f"conditions-: {effect.player_remove_conditions}")
            if log_parts:
                world_state.record_transfer(" | ".join(log_parts))
