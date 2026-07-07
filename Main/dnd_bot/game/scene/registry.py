"""Scene Entity Registry - tracks who/what is in the current scene."""

from datetime import datetime, timedelta
from typing import Optional
import structlog

from ...models.npc import (
    SceneEntity,
    EntityType,
    Disposition,
    HostilityEvent,
    NPC,
)
from ...data.repositories.npc_repo import get_npc_repo
from ..knowledge.models import slugify

logger = structlog.get_logger()


# Hostility thresholds
HOSTILITY_CALM = 25
HOSTILITY_AGITATED = 50
HOSTILITY_THREATENING = 75
HOSTILITY_COMBAT = 85  # Auto-combat trigger


class SceneEntityRegistry:
    """
    Tracks entities currently present in the scene.

    This is an in-memory registry that:
    - Holds all NPCs, creatures, and objects currently in the scene
    - Tracks hostility levels for potential combatants
    - Syncs named NPCs to persistent storage
    - Provides context for triage decisions
    """

    def __init__(self, campaign_id: str, channel_id: int):
        self.campaign_id = campaign_id
        self.channel_id = channel_id
        self._entities: dict[str, SceneEntity] = {}
        self._hostility_log: list[HostilityEvent] = []
        self._scene_description: str = ""
        self._last_update: datetime = datetime.utcnow()

    # ==================== Entity Management ====================

    def register_entity(self, entity: SceneEntity) -> SceneEntity:
        """
        Register an entity in the scene.

        If an entity with the same name exists, updates it instead.
        """
        # Check for existing entity by name or alias
        existing = self.get_by_name(entity.name)
        if existing:
            # Update existing entity
            existing.last_mentioned_at = datetime.utcnow()
            existing.mention_count += 1
            if entity.description and entity.description != existing.description:
                existing.description = entity.description
            if entity.disposition != Disposition.NEUTRAL:
                existing.disposition = entity.disposition
            if entity.monster_index and not existing.monster_index:
                existing.monster_index = entity.monster_index
            # Adopt the canonical NPC id link (Stage C): preload order is
            # not guaranteed — an extractor-minted entity (npc_id=None) can
            # register before the DB/world preload of the same NPC, and the
            # by-name upsert would otherwise swallow the incoming link and
            # leave the row dark. First non-empty link wins; never clobber
            # an existing one (that would repoint a resolved entity).
            if entity.npc_id and not existing.npc_id:
                existing.npc_id = entity.npc_id
            # Merge aliases
            existing_aliases = set(a.lower() for a in (existing.aliases or []))
            for alias in (entity.aliases or []):
                if alias.lower() not in existing_aliases:
                    existing.aliases.append(alias)
            # If the new entity has a different name, add it as an alias
            if entity.name.lower() != existing.name.lower():
                if entity.name.lower() not in existing_aliases:
                    existing.aliases.append(entity.name)
            logger.debug(
                "entity_updated",
                name=existing.name,
                mentions=existing.mention_count,
                aliases=existing.aliases,
            )
            return existing

        # Auto-resolve monster_index for creatures/hostile NPCs from SRD
        if not entity.monster_index and entity.entity_type in (EntityType.CREATURE, EntityType.NPC):
            if entity.disposition in (Disposition.HOSTILE, Disposition.UNFRIENDLY) or entity.entity_type == EntityType.CREATURE:
                try:
                    from ...data.srd import get_srd
                    srd = get_srd()
                    match = srd.fuzzy_match_monster(entity.name)
                    if match:
                        entity.monster_index = match["index"]
                        logger.info(
                            "entity_auto_matched_srd",
                            name=entity.name,
                            monster_index=match["index"],
                            monster_name=match.get("name"),
                        )
                except Exception as e:
                    logger.debug("entity_srd_match_failed", name=entity.name, error=str(e), exc_info=True)

        # Register new entity
        self._entities[entity.id] = entity
        logger.info(
            "entity_registered",
            id=entity.id,
            name=entity.name,
            type=entity.entity_type.value,
            disposition=entity.disposition.value,
            hostility=entity.hostility_score,
            monster_index=entity.monster_index,
        )
        return entity

    def get_by_id(self, entity_id: str) -> Optional[SceneEntity]:
        """Get entity by ID."""
        return self._entities.get(entity_id)

    def get_by_name(self, name: str) -> Optional[SceneEntity]:
        """Get entity by name, alias, or the canonical NPC id.

        Also accepts the roster id dialect (final review): the narrator
        context lists entities as ``[id: slug]`` — ``slugify(name)``,
        hyphenated — which the substring check cannot bridge to spaced
        names ('old-bram' vs 'Old Bram'). Slugified equality on the name
        and aliases covers it.

        And the canonical id (Stage C): when a caller passes the WorldState
        NPCState UUID — the dedup rewrite echoes it as ``ref_entity_id``,
        and it is the shared cross-store key — match it against the
        entity's ``npc_id`` link. Exact, checked first so it wins.
        """
        name_lower = name.lower()
        query_slug = slugify(name)
        for entity in self._entities.values():
            # Canonical id link (Stage C) — authoritative, exact.
            if name and entity.npc_id and name == entity.npc_id:
                return entity
            # Check main name
            if name_lower in entity.name.lower() or entity.name.lower() in name_lower:
                return entity
            if query_slug and query_slug == slugify(entity.name):
                return entity
            # Check aliases
            for alias in (entity.aliases or []):
                if name_lower in alias.lower() or alias.lower() in name_lower:
                    return entity
                if query_slug and query_slug == slugify(alias):
                    return entity
        return None

    def get_all(self) -> list[SceneEntity]:
        """Get all entities in the scene."""
        return list(self._entities.values())

    def get_all_entities(self) -> list[SceneEntity]:
        """Alias for get_all() - returns all entities in the scene."""
        return self.get_all()

    def get_by_type(self, entity_type: EntityType) -> list[SceneEntity]:
        """Get entities by type."""
        return [e for e in self._entities.values() if e.entity_type == entity_type]

    def get_hostiles(self) -> list[SceneEntity]:
        """Get all hostile entities (disposition = HOSTILE or high hostility)."""
        return [
            e for e in self._entities.values()
            if e.disposition == Disposition.HOSTILE
            or e.hostility_score >= HOSTILITY_THREATENING
        ]

    def get_potential_targets(self) -> list[SceneEntity]:
        """Get all entities that could be combat targets (NPCs and creatures)."""
        return [
            e for e in self._entities.values()
            if e.entity_type in (EntityType.NPC, EntityType.CREATURE)
        ]

    def remove_entity(self, entity_id: str) -> Optional[SceneEntity]:
        """Remove an entity from the scene."""
        entity = self._entities.pop(entity_id, None)
        if entity:
            logger.info("entity_removed", id=entity_id, name=entity.name)
        return entity

    def remove_by_name(self, name: str) -> Optional[SceneEntity]:
        """Remove an entity by name."""
        entity = self.get_by_name(name)
        if entity:
            return self.remove_entity(entity.id)
        return None

    def clear(self) -> None:
        """Clear all entities (scene change)."""
        count = len(self._entities)
        self._entities.clear()
        self._hostility_log.clear()
        self._scene_description = ""
        logger.info("scene_cleared", entities_removed=count)

    # ==================== Hostility Management ====================

    def escalate_hostility(
        self,
        entity_id: str,
        delta: int,
        reason: str,
    ) -> tuple[int, bool]:
        """
        Escalate hostility for an entity.

        Returns:
            (new_hostility_score, triggered_combat)
        """
        entity = self.get_by_id(entity_id)
        if not entity:
            return (0, False)

        old_score = entity.hostility_score
        entity.hostility_score = max(0, min(100, entity.hostility_score + delta))

        # Log the event
        event = HostilityEvent(
            entity_id=entity_id,
            delta=delta,
            reason=reason,
        )
        self._hostility_log.append(event)
        entity.hostility_events.append(f"{reason} ({'+' if delta > 0 else ''}{delta})")

        # Check for disposition change
        if (
            entity.hostility_score >= HOSTILITY_COMBAT
            and entity.disposition != Disposition.HOSTILE
        ):
            entity.disposition = Disposition.HOSTILE
            logger.warning(
                "entity_became_hostile",
                name=entity.name,
                hostility=entity.hostility_score,
                reason=reason,
            )

        # Check combat trigger
        triggered_combat = (
            entity.hostility_score >= HOSTILITY_COMBAT
            and old_score < HOSTILITY_COMBAT
        )

        logger.info(
            "hostility_changed",
            entity=entity.name,
            old_score=old_score,
            new_score=entity.hostility_score,
            delta=delta,
            reason=reason,
            triggered_combat=triggered_combat,
        )

        return (entity.hostility_score, triggered_combat)

    def escalate_by_name(
        self,
        name: str,
        delta: int,
        reason: str,
    ) -> tuple[int, bool]:
        """Escalate hostility by entity name."""
        entity = self.get_by_name(name)
        if entity:
            return self.escalate_hostility(entity.id, delta, reason)
        return (0, False)

    def de_escalate_hostility(
        self,
        entity_id: str,
        delta: int,
        reason: str,
    ) -> int:
        """De-escalate hostility. Delta should be positive (will be negated)."""
        return self.escalate_hostility(entity_id, -abs(delta), reason)[0]

    def check_combat_threshold(self) -> list[SceneEntity]:
        """
        Check if any entities have crossed the combat threshold.

        Returns entities that should trigger combat.
        """
        return [
            e for e in self._entities.values()
            if e.hostility_score >= HOSTILITY_COMBAT
            and e.entity_type in (EntityType.NPC, EntityType.CREATURE)
            and e.disposition != Disposition.ALLIED  # Allies don't attack party
        ]

    # ==================== Context Building ====================

    def get_triage_context(self) -> str:
        """
        Build context string for triage.

        This is injected into the triage prompt so the Rules Brain
        knows who/what is present in the scene.
        """
        if not self._entities:
            return ""

        lines = ["## Scene Entities"]

        # Group by type
        npcs = self.get_by_type(EntityType.NPC)
        creatures = self.get_by_type(EntityType.CREATURE)
        objects = self.get_by_type(EntityType.OBJECT)

        if npcs:
            lines.append("\n### NPCs Present:")
            for e in npcs:
                status = self._disposition_indicator(e)
                desc = e.description[:80] + "..." if len(e.description) > 80 else e.description
                lines.append(f"- **{e.name}**{status}: {desc}")

        if creatures:
            lines.append("\n### Creatures Present:")
            for e in creatures:
                status = self._disposition_indicator(e)
                desc = e.description[:80] + "..." if len(e.description) > 80 else e.description
                lines.append(f"- **{e.name}**{status}: {desc}")

        if objects:
            lines.append("\n### Notable Objects:")
            for e in objects:
                desc = e.description[:80] + "..." if len(e.description) > 80 else e.description
                lines.append(f"- **{e.name}**: {desc}")

        return "\n".join(lines)

    def _disposition_indicator(self, entity: SceneEntity) -> str:
        """Get a visual indicator for disposition."""
        if entity.disposition == Disposition.HOSTILE or entity.hostility_score >= HOSTILITY_COMBAT:
            return " [HOSTILE]"
        elif entity.hostility_score >= HOSTILITY_THREATENING:
            return " [THREATENING]"
        elif entity.hostility_score >= HOSTILITY_AGITATED:
            return " [AGITATED]"
        elif entity.disposition == Disposition.FRIENDLY:
            return " [friendly]"
        elif entity.disposition == Disposition.ALLIED:
            return " [allied]"
        return ""

    def get_narrator_roster(self) -> str:
        """Build an authoritative NPC/entity roster for the narrator.

        This is injected into narrator context so it knows EXACTLY who is
        present and what their canonical names are. Prevents the narrator
        from inventing or confusing NPC names.
        """
        npcs = self.get_by_type(EntityType.NPC)
        creatures = self.get_by_type(EntityType.CREATURE)
        objects = self.get_by_type(EntityType.OBJECT)

        if not npcs and not creatures and not objects:
            return ""

        lines = [
            "## NPC & Entity Roster (AUTHORITATIVE)",
            "Use ref_entity <id> in INTENTS for every roster entity you mention in prose.",
        ]

        if npcs:
            for e in npcs:
                slug = slugify(e.name)
                disp = e.disposition.value if e.disposition else "neutral"
                desc = e.description[:120] if e.description else "No description"
                lines.append(f"- **{e.name}** [id: {slug}] ({disp}): {desc}")

        if creatures:
            for e in creatures:
                slug = slugify(e.name)
                desc = e.description[:80] if e.description else ""
                lines.append(f"- **{e.name}** [id: {slug}] (creature): {desc}")

        if objects:
            for e in objects:
                slug = slugify(e.name)
                desc = e.description[:80] if e.description else ""
                lines.append(f"- {e.name} [id: {slug}]: {desc}")

        return "\n".join(lines)

    def get_scene_summary(self) -> str:
        """Get a brief summary for the scene memory block."""
        entities = self.get_all()
        if not entities:
            return self._scene_description

        entity_parts = []
        for e in entities[:5]:
            status = self._disposition_indicator(e).strip()
            if status:
                entity_parts.append(f"{e.name} {status}")
            else:
                entity_parts.append(e.name)

        if len(entities) > 5:
            entity_parts.append(f"and {len(entities) - 5} others")

        entities_str = ", ".join(entity_parts)

        if self._scene_description:
            return f"{self._scene_description}\n\nPresent: {entities_str}"
        return f"Present: {entities_str}"

    def set_scene_description(self, description: str) -> None:
        """Set the base scene description."""
        self._scene_description = description
        self._last_update = datetime.utcnow()

    def get_scene_description(self) -> str:
        """Get the current scene description."""
        return self._scene_description

    # ==================== Persistence ====================

    async def sync_to_npc_repo(self, current_location: Optional[str] = None) -> int:
        """Sync named NPCs to persistent storage.

        Canonical id is authoritative (Stage C): a SceneEntity carrying an
        ``npc_id`` that has no DB row yet — a world-minted NPC whose id
        never reached the DB — is CREATED under that id, so the row, the KG
        node, and the WorldState NPCState all share one key (previously the
        ``if npc:`` update branch silently dropped it and the NPC never
        became durable). Death propagates: a SceneEntity ``status == 'dead'``
        writes ``is_alive = False`` so ``get_alive_by_campaign`` stops
        resurrecting the corpse next session (DF-4, DB side). Never
        resurrects — status other than 'dead' leaves ``is_alive`` alone.

        Location is world-authoritative (DF-19): written only when the
        caller supplies a real ``current_location``. The old
        ``scene_description[:100]`` slice was narration prose, not a place,
        and permanently disagreed with WorldState; when the world doesn't
        know a location the DB column is left untouched rather than clobbered.

        Returns count of NPCs synced.
        """
        repo = await get_npc_repo()
        synced = 0

        for entity in self.get_by_type(EntityType.NPC):
            is_dead = bool(entity.status) and entity.status.lower() == "dead"
            try:
                npc = None
                if entity.npc_id:
                    npc = await repo.get_by_id(entity.npc_id)
                elif entity.name:
                    # Legacy by-name adoption for entities minted before the
                    # canonical id was stamped onto them.
                    npc = await repo.get_by_exact_name(self.campaign_id, entity.name)
                    if npc is not None:
                        entity.npc_id = npc.id

                if npc is not None:
                    # Update the existing row.
                    if current_location:
                        npc.location = current_location
                    npc.last_seen_at = datetime.utcnow()
                    if is_dead:
                        npc.is_alive = False
                    await repo.update(npc)
                    synced += 1
                else:
                    # Create — preserving the canonical id when we have one,
                    # so a world-minted NPC's row adopts the shared key.
                    npc_kwargs: dict = dict(
                        campaign_id=self.campaign_id,
                        name=entity.name,
                        description=entity.description,
                        location=current_location,
                        monster_index=entity.monster_index,
                        base_disposition=entity.disposition,
                        is_alive=not is_dead,
                    )
                    if entity.npc_id:
                        npc_kwargs["id"] = entity.npc_id
                    npc = NPC(**npc_kwargs)
                    await repo.create(npc)
                    entity.npc_id = npc.id
                    synced += 1
            except Exception as e:
                # Uniform persist-failure taxonomy: "persist_failed" with an
                # entity key, like every other missed write.
                logger.error(
                    "persist_failed",
                    entity="npc",
                    campaign_id=self.campaign_id,
                    name=entity.name,
                    error=str(e),
                    exc_info=True,
                )

        if synced > 0:
            logger.info("npcs_synced", count=synced)
        return synced

    async def load_npcs_at_location(self, location: str) -> int:
        """
        Load NPCs at a location from the database.

        Returns count of NPCs loaded.
        """
        repo = await get_npc_repo()
        npcs = await repo.get_at_location(self.campaign_id, location)

        loaded = 0
        for npc in npcs:
            # Check if already in scene
            if self.get_by_name(npc.name):
                continue

            entity = SceneEntity(
                name=npc.name,
                entity_type=EntityType.NPC,
                description=npc.description,
                npc_id=npc.id,
                monster_index=npc.monster_index,
                disposition=npc.base_disposition,
                hostility_score=self._disposition_to_hostility(npc.base_disposition),
            )
            self.register_entity(entity)
            loaded += 1

        if loaded > 0:
            logger.info("npcs_loaded_from_location", location=location, count=loaded)
        return loaded

    def _disposition_to_hostility(self, disposition: Disposition) -> int:
        """Convert disposition to initial hostility score."""
        mapping = {
            Disposition.HOSTILE: 80,
            Disposition.UNFRIENDLY: 40,
            Disposition.NEUTRAL: 0,
            Disposition.FRIENDLY: 0,
            Disposition.ALLIED: 0,
        }
        return mapping.get(disposition, 0)

    # ==================== Cleanup ====================

    def prune_stale_entities(self, max_age_minutes: int = 30) -> int:
        """
        Remove entities that haven't been mentioned recently.

        Keeps NPCs longer than objects/creatures.
        Returns count removed.
        """
        threshold = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        npc_threshold = datetime.utcnow() - timedelta(minutes=max_age_minutes * 2)

        stale_ids = []
        for entity in self._entities.values():
            entity_threshold = npc_threshold if entity.entity_type == EntityType.NPC else threshold
            if entity.last_mentioned_at < entity_threshold:
                stale_ids.append(entity.id)

        for entity_id in stale_ids:
            self.remove_entity(entity_id)

        if stale_ids:
            logger.info("stale_entities_pruned", count=len(stale_ids))

        return len(stale_ids)

    def get_entity_count(self) -> int:
        """Get the number of entities in the scene."""
        return len(self._entities)

    def has_entities(self) -> bool:
        """Check if there are any entities in the scene."""
        return len(self._entities) > 0


# Registry instances by session_key (audit #8: was keyed by `int channel_id`,
# which collided across voice + web sessions because they all set
# `channel_id=0`. Now keyed by the per-session string identifier so each
# session gets its own registry).
_registries: dict[str, SceneEntityRegistry] = {}


def get_scene_registry(campaign_id: str, session_key: str) -> SceneEntityRegistry:
    """Get or create scene registry for a session.

    `session_key` must uniquely identify the session (e.g.
    ``"discord:{channel_id}"`` for Discord, ``"voice:{uuid}"`` for voice,
    ``"web:{uuid}"`` for web). Two sessions with the same key share state —
    that's a bug if you didn't intend it.
    """
    if session_key not in _registries:
        # `channel_id` on SceneEntityRegistry was historically used for debug
        # logging. Pass 0 here for non-Discord contexts; the registry doesn't
        # use it for keying.
        _registries[session_key] = SceneEntityRegistry(campaign_id, 0)
    return _registries[session_key]


def clear_scene_registry(session_key: str) -> None:
    """Clear scene registry for a session."""
    if session_key in _registries:
        _registries[session_key].clear()
        del _registries[session_key]
