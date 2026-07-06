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

from typing import Optional

import structlog

from .world_state import NPCState, StateDelta, WorldState
from ..llm.effects import EffectType, ProposedEffect

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

    # ── The extractor pipeline's apply seam ──────────────────────────────

    def apply_delta(self, delta: StateDelta) -> list[str]:
        """Validate and apply a StateDelta; returns rejections.

        The Step-5 dedup pass runs HERE (inside the write pipeline), never
        as an event or a coordinator method (plan anti-re-flag rule).
        """
        return self._state.apply_delta(delta)

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
