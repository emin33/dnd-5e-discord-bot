"""`_handle_start`: the two legs of the start that only the cog performs.

Nothing imported `dnd_bot.bot.cogs.campaign` before this file. The suite ran
for a long time on an interpreter that could not even import it, so a mutation
to any cog survived every test in the run -- 1645 lines with no coverage at
all, holding the only production path that seeds an authored world.

Two behaviours are pinned here because both were review findings, both live in
the cog rather than in the library beneath it, and both fail *silently*:

1. **Intent is spent only on a confirmed checkpoint.** The chosen sourcebook is
   the campaign's single retry marker. Clearing it when the install returned
   left a window where the campaign was bound, nothing was snapshotted, and
   nothing recorded that a book was owed -- recovery ends a snapshotless
   session and the next start sees no pending book. The marker must survive a
   snapshot that did not land.

2. **Opening effects execute AND sync.** Executing alone puts an effect in the
   scene registry only; world state -- and therefore the snapshot taken moments
   later -- never learns about it. The turn pipeline has validate/execute/sync
   and this path was missing the third leg.

The real validator, executor, world store, session, graph, repository and
database are used throughout. Only the genuinely external edges are faked:
Discord itself, the narrator and memory (network), and the persistence
*result*, which is the variable under test in the first group.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dnd_bot.data.repositories.campaign_repo import CampaignRepository
from dnd_bot.data.repositories.sourcebook_repo import SourcebookRepository
from dnd_bot.llm.effects import EffectType, ProposedEffect
from dnd_bot.models import Campaign

from tests.integration.test_sourcebook_canon_repo import make_db, rich_book

CAMPAIGN_ID = "camp"
CHANNEL_ID = 4242


# ── fakes for the edges only ────────────────────────────────────────────────


class _Followup:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, content: str = "", **kwargs: Any) -> None:
        self.sent.append({"content": content, **kwargs})


class _Response:
    """The lobby button already deferred, so this must never be used.

    `response.send_message` after a defer raises InteractionResponded, which
    the surrounding try/except would swallow -- taking the opening narration
    with it. A previous review found exactly that, so it is pinned as an
    explosion rather than a comment.
    """

    async def send_message(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "_handle_start used interaction.response.send_message; the lobby "
            "button already deferred, so this raises InteractionResponded and "
            "the opening narration never runs"
        )


class _Guild:
    def __init__(self, member: Any) -> None:
        self._member = member

    def get_member(self, _user_id: int) -> Any:
        return self._member


class _Member:
    display_name = "Tester"


class _Interaction:
    def __init__(self) -> None:
        self.followup = _Followup()
        self.response = _Response()
        self.channel_id = CHANNEL_ID
        self.guild_id = 1
        self.guild = _Guild(_Member())
        self.channel = object()


class _SessionManager:
    """Real session in, controllable persistence result out.

    ``snapshot_result`` is the whole point of the first group of tests: True,
    False and "raise" are three different worlds for the retry marker.
    """

    def __init__(self, session: Any, snapshot_result: Any = True) -> None:
        self._session = session
        self.snapshot_result = snapshot_result
        self.snapshot_calls = 0
        self.joined: list[int] = []

    def has_active_session(self, _channel_id: int) -> bool:
        return False

    async def start_session(self, **_kwargs: Any) -> Any:
        return self._session

    async def join_session(self, user_id: int, **_kwargs: Any) -> None:
        self.joined.append(user_id)

    async def _persist_world_snapshot(self, _session: Any) -> bool:
        self.snapshot_calls += 1
        if self.snapshot_result == "raise":
            raise RuntimeError("disk went away")
        return self.snapshot_result


class _Narrator:
    def __init__(self, prose: str = "The rain keeps on.", effects: list | None = None):
        self.prose = prose
        self.effects = effects or []
        self.authored_scene_seen: str | None = None

    async def generate_opening(self, authored_scene: str = "", **_kwargs: Any):
        self.authored_scene_seen = authored_scene
        return self.prose, self.effects


class _Memory:
    def __init__(self) -> None:
        self.scene: str | None = None

    def update_scene(self, text: str) -> None:
        self.scene = text

    async def add_dm_response(self, **_kwargs: Any) -> None:
        return None


class _Immersion:
    tts_enabled = False
    image_enabled = False


class _Profile:
    immersion = _Immersion()


class _LogRecorder:
    """Captures the cog's own warnings.

    `_handle_start` wraps the whole narrator/effects section in a broad
    try/except that logs and moves on. Without reading that log, a test whose
    fakes are subtly wrong passes by asserting on a world nothing ever
    touched. Every test here asserts the swallow did not fire.
    """

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def warning(self, event: str, **_kw: Any) -> None:
        self.warnings.append(event)

    def info(self, event: str, **_kw: Any) -> None:
        self.infos.append(event)

    def debug(self, event: str, **_kw: Any) -> None:
        pass


# ── the rig ─────────────────────────────────────────────────────────────────


@pytest.fixture
async def rig(tmp_path: Path, monkeypatch):
    """A started campaign, wired to real machinery with faked edges."""
    import dnd_bot.bot.cogs.campaign as cog_module
    import dnd_bot.config as config_module
    import dnd_bot.game.scene.registry as registry_module
    import dnd_bot.game.session as session_module
    import dnd_bot.immersion.voice_assigner as voice_module
    import dnd_bot.llm.brains.narrator as narrator_module
    import dnd_bot.memory as memory_module
    from dnd_bot.game.knowledge.graph import KnowledgeGraph
    from dnd_bot.game.scene.registry import SceneEntityRegistry
    from dnd_bot.game.session import GameSession
    from dnd_bot.game.world_state import WorldState
    from dnd_bot.models import (
        AbilityScores, Character, HitDice, HitPoints,
    )
    from tests.unit.test_scene_hydration import _MemoryRepo

    db = await make_db(tmp_path, "coglifecycle.db")
    books = SourcebookRepository(db=db)
    campaigns = CampaignRepository(db=db)

    session = GameSession(
        id="s", channel_id=CHANNEL_ID, guild_id=1, campaign_id=CAMPAIGN_ID,
    )
    session.world_state = WorldState()
    session.knowledge_graph = KnowledgeGraph(
        campaign_id=CAMPAIGN_ID, repository=_MemoryRepo(),
    )
    await session.knowledge_graph.load()

    character = Character(
        discord_user_id=7,
        campaign_id=CAMPAIGN_ID,
        name="Elara",
        race_index="elf",
        class_index="wizard",
        level=1,
        abilities=AbilityScores(
            strength=8, dexterity=14, constitution=13,
            intelligence=16, wisdom=12, charisma=10,
        ),
        hp=HitPoints(maximum=7, current=7),
        hit_dice=HitDice(die_type=6, total=1, remaining=1),
        armor_class=12,
        speed=30,
        initiative_bonus=2,
    )

    class _CharRepo:
        async def get_all_by_campaign(self, _cid: str) -> list:
            return [character]

    manager = _SessionManager(session)
    narrator = _Narrator()
    recorder = _LogRecorder()
    registry = SceneEntityRegistry(
        campaign_id=CAMPAIGN_ID, channel_id=CHANNEL_ID,
    )

    async def _char_repo() -> Any:
        return _CharRepo()

    async def _campaign_repo() -> Any:
        return campaigns

    async def _sourcebook_repo() -> Any:
        return books

    async def _memory_manager(_cid: str) -> Any:
        return _Memory()

    async def _no_voice(**_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(cog_module, "get_character_repo", _char_repo)
    monkeypatch.setattr(cog_module, "get_campaign_repo", _campaign_repo)
    monkeypatch.setattr(cog_module, "logger", recorder)
    monkeypatch.setattr(session_module, "get_session_manager", lambda: manager)
    monkeypatch.setattr(narrator_module, "get_narrator", lambda: narrator)
    monkeypatch.setattr(memory_module, "get_memory_manager", _memory_manager)
    monkeypatch.setattr(config_module, "get_profile", lambda: _Profile())
    monkeypatch.setattr(voice_module, "assign_voice", _no_voice)
    monkeypatch.setattr(
        registry_module, "get_scene_registry", lambda *_a, **_k: registry,
    )
    monkeypatch.setattr(
        "dnd_bot.data.repositories.sourcebook_repo.get_sourcebook_repo",
        _sourcebook_repo,
    )
    # The cog's module-level intent cache is a global dict; never let one test
    # hand its choice to the next.
    monkeypatch.setattr(cog_module, "_campaign_sourcebook", {})

    cog = cog_module.CampaignCog.__new__(cog_module.CampaignCog)
    campaign = Campaign(
        id=CAMPAIGN_ID, guild_id=1, name="Camp", dm_user_id=1,
        world_setting="A wet town.",
    )

    class _Rig:
        pass

    rig = _Rig()
    rig.cog = cog
    rig.campaign = campaign
    rig.session = session
    rig.books = books
    rig.campaigns = campaigns
    rig.manager = manager
    rig.narrator = narrator
    rig.registry = registry
    rig.recorder = recorder
    rig.interaction = _Interaction()
    rig.cog_module = cog_module

    async def _install_a_book() -> str:
        key = (await books.import_book(rich_book())).sourcebook_key
        await campaigns.set_pending_sourcebook(CAMPAIGN_ID, key)
        return key

    rig.install_a_book = _install_a_book

    async def _start() -> None:
        await cog._handle_start(rig.interaction, campaign)

    rig.start = _start

    try:
        yield rig
    finally:
        await db.disconnect()


def _assert_nothing_was_swallowed(rig) -> None:
    """The broad try/except must not be what made the test pass."""
    assert "failed_to_generate_opening" not in rig.recorder.warnings, (
        "the opening section raised and was swallowed -- this test asserted "
        "on a world the code never reached"
    )


# ── 1. intent is spent only on a confirmed checkpoint ───────────────────────


class TestSourcebookIntentLifecycle:
    @pytest.mark.asyncio
    async def test_a_durable_snapshot_spends_the_retry_marker(self, rig):
        key = await rig.install_a_book()
        rig.cog_module._campaign_sourcebook[CAMPAIGN_ID] = key
        rig.manager.snapshot_result = True

        await rig.start()

        _assert_nothing_was_swallowed(rig)
        # It really did install: the book's room and cast are in the session.
        assert rig.session.world_state.current_location == "Copper Finch"
        assert await rig.books.sourcebook_keys_for_campaign(CAMPAIGN_ID) == [key]
        # ...and only now is the marker spent, in BOTH places it lives.
        assert CAMPAIGN_ID not in rig.cog_module._campaign_sourcebook
        reloaded = await rig.campaigns.get_by_id(CAMPAIGN_ID)
        assert reloaded.pending_sourcebook_key in (None, "")

    @pytest.mark.asyncio
    async def test_a_snapshot_that_did_not_land_keeps_the_retry_marker(self, rig):
        """The finding, stated as a test.

        Bound but not durable is the one state that must stay recoverable.
        Recovery ends a snapshotless session, so if the marker were spent
        here the next start would see no pending book and the party would
        wake up in an improvised world with the book silently gone.
        """
        key = await rig.install_a_book()
        rig.cog_module._campaign_sourcebook[CAMPAIGN_ID] = key
        rig.manager.snapshot_result = False

        await rig.start()

        _assert_nothing_was_swallowed(rig)
        assert rig.manager.snapshot_calls >= 1, "the snapshot was never attempted"
        assert rig.cog_module._campaign_sourcebook[CAMPAIGN_ID] == key
        reloaded = await rig.campaigns.get_by_id(CAMPAIGN_ID)
        assert reloaded.pending_sourcebook_key == key
        assert "seeded_world_not_durable_intent_kept" in rig.recorder.warnings

    @pytest.mark.asyncio
    async def test_a_raising_snapshot_keeps_the_marker_and_does_not_fail_the_start(
        self, rig,
    ):
        """A throwing persist is the same fact as a False one.

        `_snapshot` swallows the exception deliberately -- the party is
        already playing -- but "logged the failure and moved on" must not be
        mistaken for "durable".
        """
        key = await rig.install_a_book()
        rig.cog_module._campaign_sourcebook[CAMPAIGN_ID] = key
        rig.manager.snapshot_result = "raise"

        await rig.start()  # must not propagate

        assert rig.cog_module._campaign_sourcebook[CAMPAIGN_ID] == key
        reloaded = await rig.campaigns.get_by_id(CAMPAIGN_ID)
        assert reloaded.pending_sourcebook_key == key
        assert "seeded_world_snapshot_failed" in rig.recorder.warnings
        # The start still finished: the party got their embed.
        assert any("embed" in sent for sent in rig.interaction.followup.sent)

    @pytest.mark.asyncio
    async def test_a_campaign_with_no_book_never_touches_the_marker(self, rig):
        """Negative control.

        Without this, every assertion above would also pass against a
        `_handle_start` that simply never wrote intent at all.
        """
        await rig.start()

        _assert_nothing_was_swallowed(rig)
        assert rig.manager.snapshot_calls == 0
        assert await rig.books.sourcebook_keys_for_campaign(CAMPAIGN_ID) == []
        assert rig.narrator.authored_scene_seen == ""

    @pytest.mark.asyncio
    async def test_intent_surviving_only_in_the_database_still_installs(self, rig):
        """A restart during the lobby loses the in-memory choice.

        `pending_sourcebook_key` is what makes that recoverable, so the
        install must be reachable from the row alone.
        """
        key = await rig.install_a_book()
        assert CAMPAIGN_ID not in rig.cog_module._campaign_sourcebook

        await rig.start()

        _assert_nothing_was_swallowed(rig)
        assert await rig.books.sourcebook_keys_for_campaign(CAMPAIGN_ID) == [key]
        assert rig.session.world_state.current_location == "Copper Finch"


# ── 2. opening effects: validate, execute, AND sync ─────────────────────────


class TestOpeningEffectsReachTheWorld:
    @pytest.mark.asyncio
    async def test_a_spawned_object_reaches_world_state_not_just_the_registry(
        self, rig,
    ):
        """The third leg.

        The registry is what the narrator sees *this* turn; world state is
        what the snapshot -- taken moments later -- persists. Executing
        without syncing put the lantern in the first and not the second, so
        it vanished on recovery. Asserting on the registry alone would pass
        either way, which is why the discriminating assertion is on
        `scene_items`.
        """
        rig.narrator.effects = [ProposedEffect(
            effect_type=EffectType.SPAWN_OBJECT,
            object_name="brass lantern",
            object_description="A dented brass lantern, still warm.",
        )]

        await rig.start()

        _assert_nothing_was_swallowed(rig)
        assert "brass lantern" in rig.session.world_state.scene_items
        # And the snapshot that persists it happened after the effects ran.
        assert rig.manager.snapshot_calls >= 1

    @pytest.mark.asyncio
    async def test_an_effect_the_validator_rejects_never_reaches_the_world(self, rig):
        """Validated, like every other effect path.

        A sourcebook puts its cast on the roster before the narrator writes a
        word, so its first paragraph is precisely where an `add_npc` for
        someone already standing there comes from. The roster is NOT the
        discriminating observable -- the executor resolves such a name to the
        existing NPC and mints no twin, so a roster assertion would pass
        whether or not validation ran. The scene registry is: execution
        registers a SceneEntity, rejection does not.
        """
        key = await rig.install_a_book()
        rig.cog_module._campaign_sourcebook[CAMPAIGN_ID] = key
        rig.narrator.effects = [ProposedEffect(
            effect_type=EffectType.ADD_NPC,
            npc_name="Mara Venn",
            npc_description="A sharp-eyed woman in a charcoal coat.",
        )]

        await rig.start()

        _assert_nothing_was_swallowed(rig)
        assert "opening_effect_rejected" in rig.recorder.warnings
        assert [e.name for e in rig.registry.get_all()] == []

    @pytest.mark.asyncio
    async def test_a_valid_npc_is_executed_and_registered(self, rig):
        """Positive control for the rejection test above.

        Same path, same fixtures, a name the book does not already own. If
        this did not register an entity, the rejection test would be proving
        only that the effect path is broken.
        """
        key = await rig.install_a_book()
        rig.cog_module._campaign_sourcebook[CAMPAIGN_ID] = key
        rig.narrator.effects = [ProposedEffect(
            effect_type=EffectType.ADD_NPC,
            npc_name="Ilse Fenn",
            npc_description="A rain-soaked courier shaking out her hood.",
        )]

        await rig.start()

        _assert_nothing_was_swallowed(rig)
        assert "opening_effect_rejected" not in rig.recorder.warnings
        assert [e.name for e in rig.registry.get_all()] == ["Ilse Fenn"]
        # Executed WITH the session, so the entity is linked to a world-state
        # NPC rather than being a registry-only orphan.
        registered = rig.registry.get_all()[0]
        assert registered.npc_id
        assert registered.npc_id in rig.session.world_state.npcs

    @pytest.mark.asyncio
    async def test_no_effects_means_no_effect_snapshot(self, rig):
        """Negative control: an improvised opening does no extra work."""
        rig.narrator.effects = []

        await rig.start()

        _assert_nothing_was_swallowed(rig)
        assert rig.manager.snapshot_calls == 0
        assert rig.registry.get_all() == []


# ── 3. the interaction itself ───────────────────────────────────────────────


class TestInteractionLifecycle:
    @pytest.mark.asyncio
    async def test_the_start_answers_through_followup_only(self, rig):
        """`_Response.send_message` raises if touched -- see its docstring."""
        await rig.start()

        _assert_nothing_was_swallowed(rig)
        assert rig.interaction.followup.sent, "the DM was told nothing at all"

    @pytest.mark.asyncio
    async def test_a_campaign_with_no_characters_stops_before_starting(self, rig):
        class _Empty:
            async def get_all_by_campaign(self, _cid: str) -> list:
                return []

        async def _empty_repo() -> Any:
            return _Empty()

        rig.cog_module.get_character_repo = _empty_repo
        try:
            await rig.start()
        finally:
            pass

        assert rig.manager.joined == []
        assert rig.manager.snapshot_calls == 0
        assert "No players have joined yet" in rig.interaction.followup.sent[0]["content"]
