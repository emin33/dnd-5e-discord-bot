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
from dnd_bot.game.world_state import NPCState, StateDelta, WorldState
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

    def test_get_by_name_misses_canonical_id(self, registry):
        """PINNED-BROKEN → flips: canonical-id lookup resolves.

        The dedup judge rewrites a paraphrased ADD_NPC to
        REF_ENTITY(ref_entity_id=<NPCState UUID>); the executor hands that
        UUID to get_by_name, which only knows names/aliases/slugs — the
        rewritten ref silently misses the scene registry.
        """
        _scene_npc(registry, "Old Bram", npc_id="11111111-2222-4333-8444-555555555555")
        assert registry.get_by_name("11111111-2222-4333-8444-555555555555") is None

    def test_register_entity_name_upsert_drops_incoming_npc_id(self, registry):
        """PINNED-BROKEN → flips: the upsert adopts the incoming npc_id.

        Preload order dependence: an extractor-minted entity (npc_id=None)
        registered before the DB preload swallows the DB entity's link —
        the merged survivor keeps npc_id=None and the row goes dark.
        """
        bare = _scene_npc(registry, "Old Bram")
        merged = registry.register_entity(SceneEntity(
            name="Old Bram",
            entity_type=EntityType.NPC,
            npc_id="npc-row-1",
        ))
        assert merged is bare
        assert bare.npc_id is None  # → flips: == "npc-row-1"


class TestWorldStateResolution:
    def test_find_npc_resolves_id_name_alias(self, world):
        """Survives: the documented id → name → alias chain."""
        npc = _world_npc(world, "Old Bram", aliases=["the innkeeper"])
        assert world._find_npc(npc.id) is npc
        assert world._find_npc("old bram") is npc
        assert world._find_npc("THE INNKEEPER") is npc

    def test_find_npc_misses_multiword_roster_slug(self, world):
        """PINNED-BROKEN → flips: slug-equality bridges the roster dialect.

        The roster says '[id: old-bram]'; _find_npc compares exact
        lowercased names, and 'old-bram' != 'old bram'. Single-word names
        resolve by accident; every multi-word NPC misses.
        """
        _world_npc(world, "Old Bram")
        assert world._find_npc("old-bram") is None

    def test_store_update_entity_by_slug_misses_world(self, store, world):
        """PINNED-BROKEN → flips: alive flips False through the slug ref.

        The narrator kills 'Old Bram' via update_entity(entity_id=
        'old-bram', status='dead'): the SceneEntity gets status='dead'
        (executor resolves slugs) but THIS sync — the one the narrator's
        next-turn YAML reads — misses, so the world roster shows a living
        NPC the prose just buried.
        """
        npc = _world_npc(world, "Old Bram")
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="old-bram",
            update_status="dead",
        ))
        assert npc.alive is True  # → flips: False

    def test_store_ref_entity_by_slug_misses_recency(self, store, world):
        """PINNED-BROKEN → flips: the slug ref bumps last_seen_turn."""
        npc = _world_npc(world, "Old Bram")
        npc.last_seen_turn = 2
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.REF_ENTITY,
            ref_entity_id="old-bram",
            ref_alias_used="the innkeeper",
        ))
        assert npc.last_seen_turn == 2  # → flips: 7
        assert npc.aliases == []        # → flips: ["the innkeeper"]


class TestExecutorResolution:
    async def test_ref_entity_with_canonical_uuid_misses_registry(self, registry):
        """PINNED-BROKEN → flips: found_in_scene True via the npc_id link.

        The dedup-rewrite shape end-to-end: REF_ENTITY carrying the
        WorldState UUID reaches the executor, whose registry lookup only
        speaks the name dialect.
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
        assert result.details["found_in_scene"] is False  # → flips: True


# ─────────────────────────────────────────────────────────────────────────
# B. Stamp — every SceneEntity mint must carry the canonical id
# ─────────────────────────────────────────────────────────────────────────


class TestDeltaPathStamp:
    def test_sync_npcs_to_registry_mints_unlinked_entity(self, world, registry):
        """PINNED-BROKEN → flips: SceneEntity.npc_id == NPCState.id (DF-29).

        The delta path registers extractor-minted NPCs with no id link, so
        every later cross-store join for them rides fuzzy get_by_name.
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
        assert entity.npc_id is None  # → flips: == npc_state.id


class TestToolPathStamp:
    async def test_execute_add_npc_leaves_ids_unlinked(self, registry, world):
        """PINNED-BROKEN → flips: the executor mints/links the canonical id.

        Today _execute_add_npc registers a SceneEntity with npc_id=None and
        never touches the world; apply_effect later mints an UNRELATED
        NPCState UUID. Two fresh ids for one NPC in one turn.
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
        entity = registry.get_by_name("Grit")
        assert entity.npc_id is None   # → flips: == the minted NPCState.id
        assert world.npcs == {}        # → flips: NPCState pre-minted via the store


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
    async def test_npc_id_with_no_row_persists_nothing(self, registry, fake_repo):
        """PINNED-BROKEN → flips: the row is created WITH the canonical id.

        A world-minted NPC whose canonical id has never reached the DB hits
        the `if entity.npc_id:` branch, get_by_id returns None, and the
        sync silently drops it — the NPC never becomes durable.
        """
        _scene_npc(registry, "Grit", npc_id="11111111-2222-4333-8444-555555555555")
        synced = await registry.sync_to_npc_repo()
        assert synced == 0            # → flips: 1
        assert fake_repo.created == []  # → flips: one row, id == the canonical id

    async def test_update_clobbers_location_with_scene_slice(self, registry, fake_repo):
        """PINNED-BROKEN → flips: location is world-authoritative (DF-19).

        The row's location is overwritten with a 100-char slice of the
        scene DESCRIPTION — prose, not a place. After the fix the sync
        writes the NPC's world location, and leaves the column alone when
        the world doesn't know one.
        """
        row = NPC(id="npc-row-1", campaign_id="camp", name="Grit", location="Docks")
        fake_repo.npcs.append(row)
        _scene_npc(registry, "Grit", npc_id="npc-row-1")
        registry.set_scene_description("A dim tavern thick with pipe smoke")

        await registry.sync_to_npc_repo()

        assert len(fake_repo.updated) == 1
        assert fake_repo.updated[0].location == (
            "A dim tavern thick with pipe smoke"
        )  # → flips: "Docks" stays (no world knowledge ⇒ no clobber)

    async def test_dead_in_scene_stays_alive_in_db(self, registry, fake_repo):
        """PINNED-BROKEN → flips: death reaches the row (DF-4 DB side).

        The narrator killed this NPC (SceneEntity.status='dead', world
        alive=False) but the sync never writes is_alive, so
        get_alive_by_campaign resurrects the corpse next session.
        """
        row = NPC(id="npc-row-1", campaign_id="camp", name="Grit")
        fake_repo.npcs.append(row)
        entity = _scene_npc(registry, "Grit", npc_id="npc-row-1")
        entity.status = "dead"

        await registry.sync_to_npc_repo()

        assert len(fake_repo.updated) == 1
        assert fake_repo.updated[0].is_alive is True  # → flips: False

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
    async def test_preload_does_not_seed_world_state(
        self, manager, fake_npc_repo, registry_cleanup, unique_channel_id, world
    ):
        """PINNED-BROKEN → flips: DB rows seed NPCState with id == NPC.id.

        Today a fresh session's WorldState starts empty, so the first
        mention of a returning NPC mints a NEW NPCState UUID — the KG node
        keyed on last session's UUID orphans, and identity fragments per
        session. (test_session_lifecycle pins the same emptiness at the
        start_session level; both flip together.)
        """
        fake_npc_repo.npcs = [
            NPC(id="npc-7", campaign_id="camp", name="Mira", location="Docks"),
        ]
        session = _session(unique_channel_id, world)
        registry_cleanup.append(session.session_key)

        await manager._preload_scene_npcs(session)

        assert world.npcs == {}  # → flips: {"npc-7": NPCState(id="npc-7", name="Mira", …)}

    async def test_preload_resurrects_world_dead_npc(
        self, manager, fake_npc_repo, registry_cleanup, unique_channel_id, world
    ):
        """PINNED-BROKEN → flips: a world-dead NPC is skipped.

        Crash-after-death: the snapshot knows the NPC died but the DB row
        is still alive (death only reaches the DB at graceful end), so
        recovery's preload marches the corpse back into the roster.
        """
        dead = NPCState(id="npc-7", name="Grokk", alive=False)
        world.npcs[dead.id] = dead
        fake_npc_repo.npcs = [NPC(id="npc-7", campaign_id="camp", name="Grokk")]
        session = _session(unique_channel_id, world)
        registry_cleanup.append(session.session_key)

        await manager._preload_scene_npcs(session)

        from dnd_bot.game.scene.registry import get_scene_registry
        registry = get_scene_registry("camp", session.session_key)
        assert registry.get_by_name("Grokk") is not None  # → flips: None

    async def test_preload_skips_world_only_npc(
        self, manager, fake_npc_repo, registry_cleanup, unique_channel_id, world
    ):
        """PINNED-BROKEN → flips: snapshot-only NPCs seed the registry (F4).

        An extractor-minted NPC lives in the recovered WorldState but not
        the npc DB (per-turn DB sync is dead; the crash beat end_session).
        Recovery's preload reads only the DB, so the narrator roster
        forgets someone the world remembers.
        """
        whisper = NPCState(id="ws-1", name="Whisper", alive=True)
        world.npcs[whisper.id] = whisper
        fake_npc_repo.npcs = []
        session = _session(unique_channel_id, world)
        registry_cleanup.append(session.session_key)

        await manager._preload_scene_npcs(session)

        from dnd_bot.game.scene.registry import get_scene_registry
        registry = get_scene_registry("camp", session.session_key)
        assert registry.get_by_name("Whisper") is None  # → flips: entity with npc_id == "ws-1"


# ─────────────────────────────────────────────────────────────────────────
# E. KG bridge — death and promotion resolve by the canonical id
# ─────────────────────────────────────────────────────────────────────────


class TestBridgeLifecycle:
    def test_update_entity_produces_no_kg_ops(self, world):
        """PINNED-BROKEN → flips: UPDATE_ENTITY emits UpdateNode (DF-4 KG side).

        update_entity(status='dead') flips WorldState.alive but
        convert_effects has no UPDATE_ENTITY case — the KG node (and its
        Chroma vector's name/description) never learn about the death.
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
        assert ops == []  # → flips: [UpdateNode(node_id=npc.id, …alive=false…)]
        assert promotions == []

    def test_ref_entity_slug_promotion_targets_wrong_node(self, world):
        """PINNED-BROKEN → flips: the promotion resolves to the UUID node.

        The narrator refs the roster slug ('the-hooded-figure') with a
        proper-name alias; _effect_ref_entity passes the slug through as
        the node_id, but NPC nodes are UUID-keyed (ROOT-2) — the
        promotion targets a node that doesn't exist and silently no-ops.
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
        )
        assert promo == NamePromotion(
            node_id="the-hooded-figure", new_name="Captain Vex"
        )  # → flips: node_id == npc.id

    def test_ref_entity_uuid_promotion_already_resolves(self, world):
        """Survives (audit correction): slugify is identity for uuid4
        strings, so a UUID ref already targets the right node — DF-14's
        'mangled UUID' mechanism was stale."""
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
