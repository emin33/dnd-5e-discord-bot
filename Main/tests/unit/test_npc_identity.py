"""Stage-C net: pin the NPC identity flows across the five stores.

One logical NPC lives in five stores — WorldState (``NPCState.id`` UUID),
scene registry (``SceneEntity.id`` UUID + optional ``npc_id`` DB link),
npc DB (``NPC.id`` UUID), KG (``node_id``), ChromaDB (``entity_<node_id>``).
Stage C makes the WorldState NPCState UUID the ONE canonical id: stamped
on every SceneEntity (``npc_id``), used as the npc DB row id at first
persist, already the KG node id (ROOT-2 core) and the Chroma vector id.

Pinned FIRST (per the plan's rule). Tests marked PINNED-BROKEN assert
today's broken behavior with a ``→ flips`` arrow describing the fix;
unmarked tests pin behavior that must survive the change.

Audit corrections established while pinning (verify-don't-trust):
- DF-10's "UUID passed to get_by_name" is stale: the roster hands the
  narrator ``[id: slugify(name)]`` slugs, and both ``get_by_name`` and
  the store's ``_find_npc`` chains resolve most of that dialect. The
  surviving misses are narrower and pinned below.
- DF-14's mechanism ("slugify mangles the UUID") is stale: ``slugify``
  is an identity function for lowercase uuid4 strings, so a UUID ref
  promotes fine. The REAL miss is the common case — the narrator echoes
  the roster SLUG, and ``_effect_ref_entity`` hands that slug straight
  to a KG whose NPC nodes are UUID-keyed (ROOT-2 flipped the node ids
  but not this resolver).
"""

from types import SimpleNamespace

import pytest

from dnd_bot.game.knowledge.bridge import DeltaBridge, NamePromotion
from dnd_bot.game.knowledge.models import UpdateNode, slugify
from dnd_bot.game.scene.registry import SceneEntityRegistry
from dnd_bot.game.session import GameSession
from dnd_bot.game.world_state import NPCState, NPCUpdate, StateDelta, WorldState
from dnd_bot.game.world_store import WorldStateStore
from dnd_bot.llm.effects import EffectExecutor, EffectType, ProposedEffect
from dnd_bot.models.npc import NPC, Disposition, EntityType, SceneEntity


@pytest.fixture
def world() -> WorldState:
    ws = WorldState(current_location="Tavern")
    ws.turn = 7
    return ws


@pytest.fixture
def store(world) -> WorldStateStore:
    return WorldStateStore(world)


@pytest.fixture
def registry() -> SceneEntityRegistry:
    return SceneEntityRegistry(campaign_id="camp", channel_id=0)


def _world_npc(world: WorldState, name: str, **fields) -> NPCState:
    npc = NPCState(name=name, location="Tavern", **fields)
    world.npcs[npc.id] = npc
    return npc


def _scene_npc(registry: SceneEntityRegistry, name: str, **fields) -> SceneEntity:
    return registry.register_entity(SceneEntity(
        name=name,
        entity_type=EntityType.NPC,
        disposition=Disposition.NEUTRAL,  # neutral skips the SRD auto-match
        **fields,
    ))


# ─────────────────────────────────────────────────────────────────────────
# A. Resolution — every cross-store lookup must accept the canonical id
# ─────────────────────────────────────────────────────────────────────────


class TestRegistryResolution:
    def test_get_by_name_resolves_slug_and_alias(self, registry):
        """Survives: the narrator's roster-slug dialect keeps resolving."""
        entity = _scene_npc(registry, "Old Bram", aliases=["the innkeeper"])
        assert registry.get_by_name("old-bram") is entity
        assert registry.get_by_name("Old Bram") is entity
        assert registry.get_by_name("the innkeeper") is entity

    def test_get_by_name_resolves_canonical_id(self, registry):
        """FLIPPED (was PINNED-BROKEN): canonical-id lookup resolves.

        The dedup judge rewrites a paraphrased ADD_NPC to
        REF_ENTITY(ref_entity_id=<NPCState UUID>); the executor hands that
        UUID to get_by_name, which now matches it against the entity's
        npc_id link (Stage C) instead of silently missing.
        """
        entity = _scene_npc(
            registry, "Old Bram", npc_id="11111111-2222-4333-8444-555555555555"
        )
        assert registry.get_by_name("11111111-2222-4333-8444-555555555555") is entity

    def test_register_entity_name_upsert_adopts_incoming_npc_id(self, registry):
        """FLIPPED (was PINNED-BROKEN): the upsert adopts the incoming npc_id.

        Preload order dependence: an extractor-minted entity (npc_id=None)
        registered before the DB preload would otherwise swallow the DB
        entity's link — now the merged survivor adopts it, keeping the row
        reachable. First non-empty link wins.
        """
        bare = _scene_npc(registry, "Old Bram")
        merged = registry.register_entity(SceneEntity(
            name="Old Bram",
            entity_type=EntityType.NPC,
            npc_id="npc-row-1",
        ))
        assert merged is bare
        assert bare.npc_id == "npc-row-1"


class TestWorldStateResolution:
    def test_find_npc_resolves_id_name_alias(self, world):
        """Survives: the documented id → name → alias chain."""
        npc = _world_npc(world, "Old Bram", aliases=["the innkeeper"])
        assert world._find_npc(npc.id) is npc
        assert world._find_npc("old bram") is npc
        assert world._find_npc("THE INNKEEPER") is npc

    def test_find_npc_resolves_multiword_roster_slug(self, world):
        """FLIPPED (was PINNED-BROKEN): slug-equality bridges the dialect.

        The roster says '[id: old-bram]'; exact-lowercase compare can't
        reach 'Old Bram', so every multi-word NPC missed. Slug-equality on
        names/aliases closes it.
        """
        npc = _world_npc(world, "Old Bram")
        assert world._find_npc("old-bram") is npc

    def test_store_update_entity_by_slug_reaches_world(self, store, world):
        """FLIPPED (was PINNED-BROKEN): alive flips False through the slug ref.

        The narrator kills 'Old Bram' via update_entity(entity_id=
        'old-bram', status='dead'): the world roster the narrator's
        next-turn YAML reads now agrees with the prose that buried them.
        """
        npc = _world_npc(world, "Old Bram")
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="old-bram",
            update_status="dead",
        ))
        assert npc.alive is False

    def test_store_ref_entity_by_slug_bumps_recency(self, store, world):
        """FLIPPED (was PINNED-BROKEN): the slug ref bumps last_seen_turn."""
        npc = _world_npc(world, "Old Bram")
        npc.last_seen_turn = 2
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.REF_ENTITY,
            ref_entity_id="old-bram",
            ref_alias_used="the innkeeper",
        ))
        assert npc.last_seen_turn == 7
        assert npc.aliases == ["the innkeeper"]


class TestExecutorResolution:
    async def test_ref_entity_with_canonical_uuid_resolves_registry(self, registry):
        """FLIPPED (was PINNED-BROKEN): found_in_scene True via the npc_id link.

        The dedup-rewrite shape end-to-end: REF_ENTITY carrying the
        WorldState UUID reaches the executor, whose registry lookup now
        matches the entity by its npc_id link.
        """
        canonical = "11111111-2222-4333-8444-555555555555"
        _scene_npc(registry, "Old Bram", npc_id=canonical)
        executor = EffectExecutor(scene_registry=registry)
        result = await executor.execute(ProposedEffect(
            effect_type=EffectType.REF_ENTITY,
            ref_entity_id=canonical,
            ref_alias_used="the innkeeper",
        ))
        assert result.success is True
        assert result.details["found_in_scene"] is True


# ─────────────────────────────────────────────────────────────────────────
# B. Stamp — every SceneEntity mint must carry the canonical id
# ─────────────────────────────────────────────────────────────────────────


class TestDeltaPathStamp:
    def test_sync_npcs_to_registry_stamps_canonical_id(self, world, registry):
        """FLIPPED (was PINNED-BROKEN): SceneEntity.npc_id == NPCState.id (DF-29).

        The delta path now stamps the canonical id on the SceneEntity it
        mints, so every later cross-store join resolves by id, not fuzzy
        name.
        """
        from dnd_bot.llm.orchestrator import DMOrchestrator

        delta = StateDelta(new_npcs=[NPCState(name="Grit")])
        world.apply_delta(delta)  # production order: registry sync sees applied ids
        npc_state = world._find_npc("Grit")
        assert npc_state is not None

        duck = SimpleNamespace(_scene_registry=registry)
        DMOrchestrator._sync_npcs_to_registry(duck, delta, world)

        entity = registry.get_by_name("Grit")
        assert entity is not None
        assert entity.npc_id == npc_state.id

    def test_sync_npcs_to_registry_marks_extractor_death(self, world, registry):
        """Extractor-channel death (NPCUpdate.alive=False) marks the
        SceneEntity dead so sync_to_npc_repo carries it to the DB (DF-4).

        The extractor reports a kill as alive=False, not a status='dead'
        tool call. That reached WorldState + the KG but not the SceneEntity,
        so the DB row stayed alive and the NPC resurrected on the next fresh
        session — the review's confirmed gap.
        """
        from dnd_bot.llm.orchestrator import DMOrchestrator

        npc = _world_npc(world, "Old Bram")
        registry.register_entity(SceneEntity(
            name="Old Bram", entity_type=EntityType.NPC, npc_id=npc.id,
        ))
        delta = StateDelta(npc_updates=[NPCUpdate(name="Old Bram", alive=False)])
        duck = SimpleNamespace(_scene_registry=registry)
        DMOrchestrator._sync_npcs_to_registry(duck, delta, world)

        assert registry.get_by_name("Old Bram").status == "dead"


class TestToolPathStamp:
    async def test_execute_add_npc_stamps_canonical_id(self, registry, world):
        """FLIPPED (was PINNED-BROKEN): the executor mints/links the canonical id.

        _execute_add_npc now mints the NPCState through the store's identity
        seam and stamps its id on the SceneEntity — one id for the NPC, not
        two unrelated UUIDs in one turn. apply_effect's later ADD_NPC branch
        resolves the same NPCState by name and no-ops (no twin).
        """
        session = SimpleNamespace(world_store=WorldStateStore(world))
        executor = EffectExecutor(scene_registry=registry, session=session)
        result = await executor.execute(ProposedEffect(
            effect_type=EffectType.ADD_NPC,
            npc_name="Grit",
            npc_description="a gruff porter",
            npc_disposition="neutral",
        ))
        assert result.success is True
        assert len(world.npcs) == 1
        npc_state = next(iter(world.npcs.values()))
        assert npc_state.name == "Grit"
        entity = registry.get_by_name("Grit")
        assert entity.npc_id == npc_state.id

    async def test_add_npc_executor_then_apply_effect_no_twin(self, registry, world):
        """The two ADD_NPC seams (executor pre-mint + apply_effect sync)
        resolve the SAME NPCState — the second is a find, never a twin."""
        store = WorldStateStore(world)
        session = SimpleNamespace(world_store=store)
        executor = EffectExecutor(scene_registry=registry, session=session)
        effect = ProposedEffect(
            effect_type=EffectType.ADD_NPC, npc_name="Grit",
            npc_description="a gruff porter", npc_disposition="neutral",
        )
        await executor.execute(effect)
        # Discriminating (was passing pre-fix for the wrong reason): the
        # executor ALONE now mints the NPCState. Pre-fix it minted nothing,
        # so this would be 0 — this line catches a regression to the old
        # "executor mints nothing, apply_effect mints once" behavior.
        assert len(world.npcs) == 1
        minted = next(iter(world.npcs.values()))

        store.apply_effect(effect)  # the orchestrator's next step
        # apply_effect resolved the SAME object, never a twin.
        assert len(world.npcs) == 1
        assert next(iter(world.npcs.values())) is minted


# ─────────────────────────────────────────────────────────────────────────
# C. DB persistence — sync_to_npc_repo and the canonical row id
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_repo(monkeypatch):
    from tests.session_fakes import FakeNpcRepo

    repo = FakeNpcRepo()

    async def _get_repo():
        return repo

    monkeypatch.setattr(
        "dnd_bot.game.scene.registry.get_npc_repo", _get_repo
    )
    return repo


class TestSyncToNpcRepo:
    async def test_npc_id_with_no_row_creates_under_canonical_id(self, registry, fake_repo):
        """FLIPPED (was PINNED-BROKEN): the row is created WITH the canonical id.

        A world-minted NPC whose canonical id never reached the DB used to
        hit the update branch, get_by_id None, and drop silently. Now it is
        created under that same id, so the row, KG node and NPCState share
        one key.
        """
        canonical = "11111111-2222-4333-8444-555555555555"
        _scene_npc(registry, "Grit", npc_id=canonical)
        synced = await registry.sync_to_npc_repo()
        assert synced == 1
        assert len(fake_repo.created) == 1
        assert fake_repo.created[0].id == canonical

    async def test_update_leaves_location_when_world_unknown(self, registry, fake_repo):
        """FLIPPED (was PINNED-BROKEN): no world location ⇒ no clobber (DF-19).

        The old sync overwrote the row's location with a 100-char slice of
        the scene DESCRIPTION (prose, not a place). Now, with no
        current_location supplied, the column is left untouched.
        """
        row = NPC(id="npc-row-1", campaign_id="camp", name="Grit", location="Docks")
        fake_repo.npcs.append(row)
        _scene_npc(registry, "Grit", npc_id="npc-row-1")
        registry.set_scene_description("A dim tavern thick with pipe smoke")

        await registry.sync_to_npc_repo()

        assert len(fake_repo.updated) == 1
        assert fake_repo.updated[0].location == "Docks"

    async def test_update_writes_world_location_when_provided(self, registry, fake_repo):
        """The world-authoritative half of DF-19: a supplied current_location
        is what the row records — a real place, from WorldState."""
        row = NPC(id="npc-row-1", campaign_id="camp", name="Grit", location="Docks")
        fake_repo.npcs.append(row)
        _scene_npc(registry, "Grit", npc_id="npc-row-1")
        registry.set_scene_description("A dim tavern thick with pipe smoke")

        await registry.sync_to_npc_repo(current_location="the Rusty Anchor")

        assert fake_repo.updated[0].location == "the Rusty Anchor"

    async def test_dead_in_scene_reaches_db(self, registry, fake_repo):
        """FLIPPED (was PINNED-BROKEN): death reaches the row (DF-4 DB side).

        The narrator killed this NPC (SceneEntity.status='dead'); the sync
        now writes is_alive=False so get_alive_by_campaign stops resurrecting
        the corpse next session.
        """
        row = NPC(id="npc-row-1", campaign_id="camp", name="Grit")
        fake_repo.npcs.append(row)
        entity = _scene_npc(registry, "Grit", npc_id="npc-row-1")
        entity.status = "dead"

        await registry.sync_to_npc_repo()

        assert len(fake_repo.updated) == 1
        assert fake_repo.updated[0].is_alive is False

    async def test_create_under_canonical_id_carries_death(self, registry, fake_repo):
        """A never-persisted dead NPC is created already is_alive=False —
        it must not spring back to life on its first write."""
        entity = _scene_npc(registry, "Grit", npc_id="npc-canon-1")
        entity.status = "dead"

        await registry.sync_to_npc_repo()

        assert len(fake_repo.created) == 1
        assert fake_repo.created[0].id == "npc-canon-1"
        assert fake_repo.created[0].is_alive is False

    async def test_name_match_adopts_existing_row_id(self, registry, fake_repo):
        """Survives: the legacy by-name adoption branch keeps working."""
        row = NPC(id="legacy-row", campaign_id="camp", name="Grit")
        fake_repo.npcs.append(row)
        entity = _scene_npc(registry, "Grit")

        synced = await registry.sync_to_npc_repo()

        assert synced == 1
        assert entity.npc_id == "legacy-row"
        assert fake_repo.updated and fake_repo.updated[0] is row

    async def test_unlinked_entity_creates_row_and_backlinks(self, registry, fake_repo):
        """Survives: a link-less NPC still gets a fresh row + backlink."""
        entity = _scene_npc(registry, "Grit")

        synced = await registry.sync_to_npc_repo()

        assert synced == 1
        assert len(fake_repo.created) == 1
        assert entity.npc_id == fake_repo.created[0].id


# ─────────────────────────────────────────────────────────────────────────
# D. Preload/seed — session start & recovery converge the id (F4)
# ─────────────────────────────────────────────────────────────────────────


def _session(channel_id: int, world: WorldState) -> GameSession:
    session = GameSession(
        id="stage-c-session",
        channel_id=channel_id,
        guild_id=1,
        campaign_id="camp",
    )
    session.world_state = world
    return session


class TestPreloadSeed:
    async def test_preload_seeds_registry_not_world_state(
        self, manager, fake_npc_repo, registry_cleanup, unique_channel_id, world
    ):
        """The DB roster seeds the REGISTRY (stamped with the canonical id),
        NOT WorldState. Seeding world_state.npcs with the whole campaign
        roster flooded four scene-scoped consumers (review regressions), so
        the world stays empty until an NPC is actually referenced — the DB
        row durably preserves the id regardless.
        """
        fake_npc_repo.npcs = [
            NPC(id="npc-7", campaign_id="camp", name="Mira", location="Docks"),
        ]
        session = _session(unique_channel_id, world)
        registry_cleanup.append(session.session_key)

        await manager._preload_scene_npcs(session)

        # World untouched; registry seeded with the canonical id link.
        assert world.npcs == {}
        from dnd_bot.game.scene.registry import get_scene_registry
        registry = get_scene_registry("camp", session.session_key)
        entity = registry.get_by_name("Mira")
        assert entity is not None
        assert entity.npc_id == "npc-7"

    async def test_preload_skips_world_dead_npc(
        self, manager, fake_npc_repo, registry_cleanup, unique_channel_id, world
    ):
        """FLIPPED (was PINNED-BROKEN): a world-dead NPC is skipped.

        Crash-after-death: the snapshot knows the NPC died but the DB row
        is still alive (death reaches the DB only at graceful end). Preload
        now trusts the authoritative WorldState and leaves the corpse out
        of the roster.
        """
        dead = NPCState(id="npc-7", name="Grokk", alive=False)
        world.npcs[dead.id] = dead
        fake_npc_repo.npcs = [NPC(id="npc-7", campaign_id="camp", name="Grokk")]
        session = _session(unique_channel_id, world)
        registry_cleanup.append(session.session_key)

        await manager._preload_scene_npcs(session)

        from dnd_bot.game.scene.registry import get_scene_registry
        registry = get_scene_registry("camp", session.session_key)
        assert registry.get_by_name("Grokk") is None

    async def test_preload_seeds_registry_from_world_only_npc(
        self, manager, fake_npc_repo, registry_cleanup, unique_channel_id, world
    ):
        """FLIPPED (was PINNED-BROKEN): snapshot-only NPCs seed the registry (F4).

        An extractor-minted NPC lives only in the recovered WorldState (the
        per-turn DB sync is dead; the crash beat end_session). Preload now
        seeds the registry from those too, keyed on the same id — the
        narrator roster remembers who the world remembers.
        """
        whisper = NPCState(id="ws-1", name="Whisper", alive=True)
        world.npcs[whisper.id] = whisper
        fake_npc_repo.npcs = []
        session = _session(unique_channel_id, world)
        registry_cleanup.append(session.session_key)

        await manager._preload_scene_npcs(session)

        from dnd_bot.game.scene.registry import get_scene_registry
        registry = get_scene_registry("camp", session.session_key)
        entity = registry.get_by_name("Whisper")
        assert entity is not None
        assert entity.npc_id == "ws-1"


# ─────────────────────────────────────────────────────────────────────────
# E. KG bridge — death and promotion resolve by the canonical id
# ─────────────────────────────────────────────────────────────────────────


class TestBridgeLifecycle:
    def test_update_entity_produces_kg_update_node(self, world):
        """FLIPPED (was PINNED-BROKEN): UPDATE_ENTITY emits UpdateNode (DF-4 KG side).

        update_entity(status='dead') now reaches the KG node (and, via the
        orchestrator re-index step, the Chroma vector), so death stops
        dead-ending in WorldState.
        """
        npc = _world_npc(world, "Old Bram")
        bridge = DeltaBridge("camp")
        ops, promotions = bridge.convert_effects(
            [ProposedEffect(
                effect_type=EffectType.UPDATE_ENTITY,
                update_entity_id=npc.id,
                update_status="dead",
            )],
            world,
        )
        assert len(ops) == 1
        assert isinstance(ops[0], UpdateNode)
        assert ops[0].node_id == npc.id
        assert ops[0].properties.get("alive") == "false"
        assert ops[0].properties.get("status") == "dead"
        assert promotions == []

    def test_update_entity_resolves_roster_slug_to_uuid_node(self, world):
        """The roster-slug dialect resolves to the UUID node, not a slug id."""
        npc = _world_npc(world, "the hooded figure")
        bridge = DeltaBridge("camp")
        ops, _ = bridge.convert_effects(
            [ProposedEffect(
                effect_type=EffectType.UPDATE_ENTITY,
                update_entity_id="the-hooded-figure",
                update_disposition="hostile",
            )],
            world,
        )
        assert len(ops) == 1
        assert ops[0].node_id == npc.id
        assert ops[0].properties.get("disposition") == "hostile"

    def test_ref_entity_slug_promotion_resolves_uuid_node(self, world):
        """FLIPPED (was PINNED-BROKEN): the promotion resolves to the UUID node.

        The narrator refs the roster slug ('the-hooded-figure') with a
        proper-name alias; _effect_ref_entity now resolves the slug to the
        UUID-keyed NPC node (ROOT-2) instead of promoting a nonexistent
        slug node (DF-13/14).
        """
        npc = _world_npc(world, "the hooded figure")
        bridge = DeltaBridge("camp")
        promo = bridge._effect_ref_entity(
            ProposedEffect(
                effect_type=EffectType.REF_ENTITY,
                ref_entity_id="the-hooded-figure",
                ref_alias_used="Captain Vex",
            ),
            set(),
            world,
        )
        assert promo == NamePromotion(node_id=npc.id, new_name="Captain Vex")

    def test_ref_entity_uuid_promotion_already_resolves(self, world):
        """Survives (audit correction): slugify is identity for uuid4
        strings, so a UUID ref already targets the right node — DF-14's
        'mangled UUID' mechanism was stale. Now it resolves via the world
        lookup directly."""
        npc = _world_npc(world, "the hooded figure")
        assert slugify(npc.id) == npc.id  # the identity fact itself
        bridge = DeltaBridge("camp")
        promo = bridge._effect_ref_entity(
            ProposedEffect(
                effect_type=EffectType.REF_ENTITY,
                ref_entity_id=npc.id,
                ref_alias_used="Captain Vex",
            ),
            set(),
            world,
        )
        assert promo == NamePromotion(node_id=npc.id, new_name="Captain Vex")

    def test_remove_entity_resolves_slug_to_uuid_node(self, world):
        """Survives: the sibling resolver the fixes will mirror."""
        npc = _world_npc(world, "Old Bram")
        bridge = DeltaBridge("camp")
        ops = bridge._effect_remove_entity(
            ProposedEffect(effect_type=EffectType.REMOVE_ENTITY, target="old-bram"),
            world,
        )
        update_ops = [op for op in ops if isinstance(op, UpdateNode)]
        assert update_ops and update_ops[0].node_id == npc.id
