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
import uuid

import structlog

from .world_state import NPCState, NPCUpdate, StateDelta, WorldState
from .identity import (
    is_generic_npc_label,
    locations_equivalent,
    resolve_unique_identity,
)
from ..llm.effects import EffectType, ProposedEffect

if TYPE_CHECKING:
    from ..models import Character
    from ..models.npc import SceneEntity
    from .scene.registry import SceneEntityRegistry

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

    # ── Persistence format (ROOT-3) ───────────────────────────────────────
    #
    # The store owns HOW a WorldState becomes bytes and back; the session
    # layer owns WHEN (per-turn) and WHERE (session_snapshot via the
    # session repo). Nothing else may know the wire shape.

    def to_snapshot(self) -> dict:
        """Serialize the wrapped state to a JSON-safe dict."""
        return self._state.model_dump(mode="json")

    @staticmethod
    def state_from_snapshot(data: dict) -> WorldState:
        """Rebuild a WorldState from :meth:`to_snapshot` output.

        Raises ``pydantic.ValidationError`` on a payload that doesn't
        validate — recovery treats that session as unrecoverable rather
        than resuming from a half-parsed world.
        """
        return WorldState.model_validate(data)

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
        """Record a pinned fact once (the memory→world-state fact sync).

        A fact the supersession seam retired stays retired — without this
        check the per-turn sync re-adds it from memory every turn.
        """
        if not fact or fact in self._state.established_facts:
            return
        if any(
            entry.get("fact") == fact
            for entry in self._state.superseded_facts
        ):
            return
        self._state.established_facts.append(fact)

    # ── The extractor pipeline's apply seam ──────────────────────────────

    async def apply_delta(
        self,
        delta: StateDelta,
        *,
        narrator_prose: str = "",
        scene_registry: Optional["SceneEntityRegistry"] = None,
    ) -> list[str]:
        """Dedup → validate → write: the extractor pipeline's apply seam.

        The dedup pass runs HERE, inside the write pipeline — never as an
        event or a coordinator method (plan anti-re-flag rule; Step 5).
        ``narrator_prose`` gives the brain judge the turn's prose for its
        paraphrase decision; with no proposed NPCs or an empty roster the
        judge is never consulted. ``scene_registry`` supplies the scene
        layer's identity merges — the naming-promotion seam: when a
        generic-labeled person's revealed proper name was already merged
        onto their SceneEntity, a proposed new_npc under that name is
        rewritten onto the existing WorldState NPC instead of fragmenting
        into a parallel identity.
        """
        if delta.new_npcs and self._state.npcs:
            delta = await self._dedup_delta(
                delta, narrator_prose, scene_registry=scene_registry
            )
        if delta.new_facts and self._state.established_facts:
            await self._supersede_conflicting_facts(delta)
        return self._state.apply_delta(delta)

    async def _supersede_conflicting_facts(self, delta: StateDelta) -> None:
        """Retire established facts a new fact makes untrue (fail-open).

        Anchor-word overlap gates candidates; the brain judge decides;
        anything uncertain keeps both facts. Retired facts move to
        ``superseded_facts`` with provenance so the ledger stays honest
        without feeding the narrator both sides of a contradiction.
        """
        from .fact_supersession import (
            FactSupersessionJudge,
            candidate_indices,
        )

        judge: Optional[FactSupersessionJudge] = None
        for new_fact in delta.new_facts:
            new_fact = (new_fact or "").strip()
            if not new_fact or new_fact in self._state.established_facts:
                continue
            indices = candidate_indices(
                new_fact, self._state.established_facts
            )
            if not indices:
                continue
            candidates = [
                self._state.established_facts[index] for index in indices
            ]
            if judge is None:
                try:
                    judge = FactSupersessionJudge()
                except Exception as e:
                    logger.warning(
                        "fact_supersession_judge_init_failed",
                        error=str(e), exc_info=True,
                    )
                    return
            retired = await judge.judge(new_fact, candidates)
            for fact in retired:
                if self._state.retire_fact(fact, superseded_by=new_fact):
                    logger.info(
                        "fact_superseded",
                        retired=fact[:120],
                        superseded_by=new_fact[:120],
                        turn=self._state.turn,
                    )

    # ── Scene-registry identity consult (cross-store naming promotion) ────

    def _scene_registry_npcs(
        self, scene_registry: Optional["SceneEntityRegistry"]
    ) -> list["SceneEntity"]:
        """NPC-typed SceneEntities, or [] when no registry is wired."""
        if scene_registry is None:
            return []
        try:
            entities = scene_registry.get_all()
        except Exception as e:
            logger.warning("scene_registry_read_failed", error=str(e), exc_info=True)
            return []
        return [
            entity
            for entity in entities
            if getattr(getattr(entity, "entity_type", None), "value", "") == "npc"
        ]

    def _resolve_npc_via_scene_registry(
        self,
        name: str,
        scene_registry: Optional["SceneEntityRegistry"],
    ) -> Optional[NPCState]:
        """Map a proposed NPC name to an existing world NPC through the
        scene registry's identity merges.

        The registry is where a newly revealed proper name is first merged
        onto a generic-labeled person ("the older woman" gains alias
        "Orris"). WorldState hasn't heard yet, so the world-roster
        deterministic check misses it and the judge sees two
        distinct-looking names. Resolution stays at the codebase's exact
        identity-key bar (``resolve_unique_identity``) on the registry
        side, then crosses stores via the canonical ``npc_id`` link (name
        keys as fallback). Abstains on anything ambiguous.
        """
        if not (name or "").strip():
            return None
        entities = self._scene_registry_npcs(scene_registry)
        if not entities:
            return None
        match = resolve_unique_identity(name, entities)
        if match is None:
            return None
        npc = self._state.npcs.get(getattr(match, "npc_id", None) or "")
        if npc is None:
            npc = resolve_unique_identity(
                getattr(match, "name", "") or "", self._state.npcs.values()
            )
        return npc

    def _scene_alias_evidence(
        self, scene_registry: Optional["SceneEntityRegistry"]
    ) -> Optional[list[dict]]:
        """The registry's alias map, as evidence rows for the dedup judge."""
        rows: list[dict] = []
        for entity in self._scene_registry_npcs(scene_registry):
            aliases = [a for a in (getattr(entity, "aliases", None) or []) if a]
            if not aliases:
                continue
            row: dict = {"name": entity.name, "aliases": aliases}
            npc_id = getattr(entity, "npc_id", None)
            if npc_id and npc_id in self._state.npcs:
                row["world_npc_id"] = npc_id
            rows.append(row)
        return rows or None

    async def _dedup_delta(
        self,
        delta: StateDelta,
        narrator_prose: str,
        scene_registry: Optional["SceneEntityRegistry"] = None,
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
        alias_evidence = self._scene_alias_evidence(scene_registry)

        for proposed in delta.new_npcs:
            deterministic = resolve_unique_identity(
                proposed.name or "",
                world_state.npcs.values(),
            )
            if deterministic is not None:
                alias = (proposed.name or "").strip()
                appended_updates.append(NPCUpdate(
                    id=deterministic.id,
                    add_aliases=(
                        [alias]
                        if alias
                        and alias.casefold() != deterministic.name.casefold()
                        else None
                    ),
                ))
                logger.info(
                    "extractor_dedup_deterministic_rewrite",
                    proposed_name=proposed.name,
                    target_id=deterministic.id,
                )
                continue

            # Scene-registry identity consult (naming promotion): the
            # registry already merged this name onto an existing entity —
            # the extractor's "new" NPC is that person's revealed proper
            # name, not a new identity. Promote the proper name over a
            # generic label; otherwise record it as an alias.
            registry_npc = self._resolve_npc_via_scene_registry(
                proposed.name or "", scene_registry
            )
            if registry_npc is not None:
                alias = (proposed.name or "").strip()
                promote = bool(
                    alias
                    and not is_generic_npc_label(alias)
                    and is_generic_npc_label(registry_npc.name)
                )
                appended_updates.append(NPCUpdate(
                    id=registry_npc.id,
                    new_name=alias if promote else None,
                    add_aliases=(
                        [alias]
                        if alias
                        and not promote
                        and alias.casefold() != registry_npc.name.casefold()
                        else None
                    ),
                ))
                logger.info(
                    "extractor_dedup_scene_registry_rewrite",
                    proposed_name=proposed.name,
                    target_id=registry_npc.id,
                    existing_name=registry_npc.name,
                    name_promoted=promote,
                )
                continue

            try:
                decision = await judge.judge_add_npc(
                    proposed_name=proposed.name or "",
                    proposed_description=proposed.description or "",
                    narrator_prose=narrator_prose,
                    existing_npcs=list(world_state.npcs.values()),
                    current_turn=world_state.turn,
                    scene_alias_map=alias_evidence,
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
        self,
        effect: ProposedEffect,
        narrator_prose: str = "",
        *,
        scene_registry: Optional["SceneEntityRegistry"] = None,
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

        deterministic = resolve_unique_identity(
            effect.npc_name,
            world_state.npcs.values(),
        )
        if deterministic is not None:
            # The paraphrased name reaches the roster as an alias through
            # apply_effect's REF_ENTITY leg — which runs only after the
            # rewritten effect validates and executes. Appending here would
            # graft an alias the validator may still reject (e.g. another
            # NPC's canonical name), an unreceipted write nothing undoes.
            alias = (effect.npc_name or "").strip()
            logger.info(
                "dedup_deterministic_rewrite",
                original_name=effect.npc_name,
                target_id=deterministic.id,
                existing_name=deterministic.name,
            )
            return ProposedEffect(
                effect_type=EffectType.REF_ENTITY,
                ref_entity_id=deterministic.id,
                ref_alias_used=(
                    alias if alias.casefold() != deterministic.name.casefold() else None
                ),
                dialogue_indices=list(effect.dialogue_indices),
                dialogue_emotions=list(effect.dialogue_emotions),
            )

        # Scene-registry identity consult (naming promotion): mirrors the
        # extractor path in _dedup_delta. The rewritten REF_ENTITY carries
        # the proper name as ``ref_alias_used``, so the existing
        # NamePromotion machinery (DeltaBridge._effect_ref_entity) promotes
        # the KG node's name through the one seam that already owns it.
        registry_npc = self._resolve_npc_via_scene_registry(
            effect.npc_name, scene_registry
        )
        if registry_npc is not None:
            # Same contract as the deterministic arm: alias accumulation
            # belongs to apply_effect, after validation.
            alias = (effect.npc_name or "").strip()
            logger.info(
                "dedup_scene_registry_rewrite",
                original_name=effect.npc_name,
                target_id=registry_npc.id,
                existing_name=registry_npc.name,
            )
            return ProposedEffect(
                effect_type=EffectType.REF_ENTITY,
                ref_entity_id=registry_npc.id,
                ref_alias_used=(
                    alias if alias.casefold() != registry_npc.name.casefold() else None
                ),
                dialogue_indices=list(effect.dialogue_indices),
                dialogue_emotions=list(effect.dialogue_emotions),
            )

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
                scene_alias_map=self._scene_alias_evidence(scene_registry),
            )
        except Exception as e:
            logger.warning("dedup_judge_call_exception", error=str(e), exc_info=True)
            return effect

        if not decision.is_rewrite:
            return effect

        # Rewrite ADD_NPC → REF_ENTITY pointing at the existing id. The
        # paraphrased name rides on ref_alias_used; apply_effect
        # accumulates it as an alias once the rewrite survives validation.
        target_id = decision.target_id
        existing = world_state.npcs.get(target_id)
        if existing is None:
            # Judge proposed an id that doesn't exist — be safe and accept the original
            logger.warning(
                "dedup_judge_target_not_in_world_state",
                target_id=target_id,
            )
            return effect

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

    # ── Canonical NPC identity seam (Stage C) ─────────────────────────────

    # How many known residents an arrival may restore. The scene roster is
    # prompt budget, not a census: a busy market must not crowd out the
    # narrator's actual context.
    MAX_HYDRATED_RESIDENTS = 6

    # Statuses that take an NPC off stage while leaving them alive. The
    # sanctioned channel for these is update_entity(status=...), which
    # deliberately does NOT clear the residency edge (remove_entity is
    # forbidden for them), so residency alone cannot mean "still here".
    _OFFSTAGE_STATUSES = frozenset({"dead", "fled", "captured"})

    def hydrate_residents(
        self,
        residents: list,
        *,
        dead_facts: list[NPCState] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Restore a location's known inhabitants into the scene roster.

        :meth:`rescope_scene` only DROPS — leaving a tavern evicts its
        non-important regulars and returning never brought them back, so the
        narrator met an empty room and either ignored the barkeep or invented
        a new one (which returning-NPC re-anchoring then had to reconcile
        after the fact). ``residents`` are the graph's durable record for the
        destination; this applies the scene policy and owns the mutation.

        Two invariants outrank scene continuity: the dead stay dead (a stale
        residency edge must never resurrect anyone), and the roster is
        capped. Returns the names restored.
        """
        from .identity import identity_keys

        destination = (self._state.current_location or "").strip()
        if not destination:
            return []
        cap = self.MAX_HYDRATED_RESIDENTS if limit is None else limit
        buried = list(dead_facts or [])
        buried_ids = {str(getattr(f, "id", "") or "") for f in buried}
        buried_keys: set[str] = set()
        for fact in buried:
            buried_keys |= set(identity_keys(getattr(fact, "name", "") or ""))
            for alias in getattr(fact, "aliases", []) or []:
                buried_keys |= set(identity_keys(alias))
        restored: list[str] = []

        for entity in residents:
            if len(restored) >= cap:
                break
            if getattr(getattr(entity, "entity_type", None), "value", "") != "npc":
                continue
            name = str(getattr(entity, "name", "") or "").strip()
            node_id = str(getattr(entity, "node_id", "") or "")
            if not name or not node_id:
                continue
            # Already on stage, under this id or any spelling of the name.
            if node_id in self._state.npcs or self._state._find_npc(name):
                continue
            properties = getattr(entity, "properties", {}) or {}
            if str(properties.get("alive", "true")).lower() == "false":
                continue
            # Off stage but alive (fled, captured): residency is stale by
            # design, since the status channel never clears the edge.
            if str(properties.get("status", "") or "").strip().lower() in (
                self._OFFSTAGE_STATUSES
            ):
                continue
            # The dead roster fails CLOSED. resolve_unique_identity abstains
            # when two buried NPCs share a name ("Cultist"), and reading that
            # abstention as "not dead" would let hydration resurrect one —
            # so ANY id or identity-key match blocks, ambiguous or not.
            if node_id in buried_ids:
                continue
            candidate_keys = set(identity_keys(name))
            for alias in getattr(entity, "aliases", []) or []:
                candidate_keys |= set(identity_keys(alias))
            if candidate_keys & buried_keys:
                continue

            self._state.npcs[node_id] = NPCState(
                id=node_id,
                name=name,
                aliases=list(getattr(entity, "aliases", []) or []),
                location=destination,
                description=str(properties.get("description", "") or ""),
                disposition=str(properties.get("disposition", "neutral") or "neutral"),
                alive=True,
                last_seen_turn=self._state.turn,
                # Preserve a surviving status marker so a hydrated NPC does
                # not silently shed state the story established.
                notes=(
                    f"[{properties['status']}]"
                    if str(properties.get("status", "") or "").strip()
                    else ""
                ),
            )
            restored.append(name)

        if restored:
            logger.info(
                "scene_hydrated_from_knowledge",
                location=destination,
                restored=restored,
                available=len(residents),
            )
        return restored

    def ensure_npc(
        self,
        name: str,
        *,
        disposition: str = "neutral",
        description: str = "",
        canonical_id: str | None = None,
    ) -> NPCState:
        """Find-or-mint the canonical NPCState for ``name`` and return it.

        The ONE place a tool-path NPCState is minted (both the narrator's
        add_npc executor and :meth:`apply_effect`'s ADD_NPC branch route
        here), so the WorldState UUID it returns is the shared cross-store
        key: the executor stamps it onto the SceneEntity's ``npc_id``, the
        KG bridge keys its node on it, and the DB row later adopts it.
        Find is by name/alias/slug via ``_find_npc`` — a paraphrase already
        collapses to a REF_ENTITY upstream (the dedup judge), so a genuine
        second call for the same NPC returns the existing one, never a twin.
        """
        existing = self._state._find_npc(name)
        if existing is not None:
            return existing
        if canonical_id and canonical_id in self._state.npcs:
            return self._state.npcs[canonical_id]
        npc = NPCState(
            id=canonical_id or str(uuid.uuid4()),
            name=name,
            location=self._state.current_location,
            disposition=disposition or "neutral",
            description=description or "",
            last_seen_turn=self._state.turn,
        )
        self._state.npcs[npc.id] = npc
        return npc

    @staticmethod
    def _append_non_npc_description(
        world_state: WorldState,
        entity_id: str,
        addition: str,
    ) -> None:
        """Append a narrator description to a scene item or the current place.

        The executor's ``update_entity`` world-reference fallback resolves
        these targets, so this is the writer that makes its
        ``description_appended`` receipt true. Slug-tolerant like the
        resolver, and idempotent: a restated addition is a no-op.

        Precedence MUST mirror ``resolve_world_reference`` (current location,
        then connected locations, then scene items): if the two disagree
        about which target an id names — a scene item sharing its name with
        a location — the executor's receipt and this writer desynchronize
        (a write the receipt withheld, or a receipt re-claimed forever
        because the dedup baseline never gains the text).
        """
        from .knowledge.models import slugify

        text = (addition or "").strip()
        if not text:
            return
        query_slug = slugify(entity_id)

        location = (world_state.current_location or "").strip()
        if location and query_slug == slugify(location):
            existing = world_state.location_description or ""
            if text not in existing:
                world_state.location_description = (
                    (existing + " " if existing else "") + text
                ).strip()
            return

        for known_location in world_state.connected_locations:
            if query_slug and query_slug == slugify(known_location):
                # The resolver classifies this id as a (non-current)
                # location, which has no description storage; the executor
                # withheld the receipt, so writing a same-named scene item
                # here would be a write the receipt denies.
                return

        for name in world_state.scene_items:
            if entity_id == name or (query_slug and query_slug == slugify(name)):
                existing = world_state.scene_items[name] or ""
                if text not in existing:
                    world_state.scene_items[name] = (
                        (existing + " " if existing else "") + text
                    ).strip()
                return

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
            # Find-or-mint through the one identity seam (Stage C). The
            # add_npc executor already minted+stamped this NPCState (so the
            # SceneEntity's npc_id points at it); here we resolve the same
            # one by name and no-op. When the executor path didn't run
            # (e.g. a sessionless effect apply), this is the mint. Dedup by
            # paraphrase is the brain judge's job upstream of both.
            self.ensure_npc(
                name=effect.npc_name or "Unknown",
                disposition=effect.npc_disposition or "neutral",
                description=effect.npc_description or "",
            )

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
                previous_location = world_state.current_location
                moved = bool(
                    previous_location
                    and not locations_equivalent(previous_location, new_loc)
                )
                # Track the origin as reachable — equivalence, not raw
                # string compare: when the extractor already applied this
                # move ('the tavern') and the tool restates it ('Tavern'),
                # a raw compare appended the current location back as a
                # phantom self-edge under the variant spelling.
                if moved and not any(
                    locations_equivalent(previous_location, known)
                    for known in world_state.connected_locations
                ):
                    world_state.connected_locations.append(previous_location)
                world_state.current_location = new_loc
                if effect.location_description:
                    world_state.location_description = effect.location_description
                world_state.record_transfer(f"party arrived at {new_loc}")
                # DF-18: a real move drops the old scene's transient
                # contents (scene_items + non-important roster NPCs not at
                # the new location). Effects later in the same batch land
                # AFTER this, so a spawn_object/add_npc for the new scene
                # survives; a restated same location (the extractor already
                # applied this move earlier in the turn) never re-rescopes.
                if moved:
                    world_state.rescope_scene()

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
                    # Casefold membership: the extractor's add_aliases writes
                    # the same list earlier in the turn, and 'Old Bram' /
                    # 'old Bram' must not fan out into duplicates.
                    alias = (effect.ref_alias_used or "").strip()
                    if (
                        alias
                        and alias.casefold() != npc_state.name.casefold()
                        and alias.casefold()
                        not in (a.casefold() for a in npc_state.aliases)
                    ):
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
                            # Keep alive=True but record status in notes for
                            # narrator visibility. Membership-checked: a
                            # restated status (every turn the narrator
                            # mentions the wound) must not pile up markers
                            # until they crowd out the real note.
                            marker = f"[{status}]"
                            if status != "alive" and marker not in (npc_state.notes or ""):
                                npc_state.notes = (
                                    (npc_state.notes + " " if npc_state.notes else "")
                                    + marker
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
                            # Casefold membership — the extractor's
                            # add_inventory writes this list earlier in the
                            # turn ('brass key' vs 'Brass Key').
                            if item_norm and item_norm.casefold() not in (
                                i.strip().casefold() for i in npc_state.inventory
                            ):
                                npc_state.inventory.append(item_norm)
                    if effect.update_remove_items:
                        for item in effect.update_remove_items:
                            item_norm = item.strip().lower()
                            # Remove case-insensitively
                            npc_state.inventory = [
                                i for i in npc_state.inventory
                                if i.strip().lower() != item_norm
                            ]
                elif effect.update_description_addition:
                    # Non-NPC targets reachable through the executor's world
                    # reference (a scene item, the current location). Without
                    # this branch the executor reported description_appended
                    # with nothing behind it (post-merge review, seam 3).
                    self._append_non_npc_description(
                        world_state,
                        entity_id,
                        effect.update_description_addition,
                    )

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
                            if item and item.casefold() not in (
                                i.strip().casefold() for i in npc_state.inventory
                            ):
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
