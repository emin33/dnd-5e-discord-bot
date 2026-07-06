"""Combat encounter construction — game-domain logic (REFACTOR_PLAN Step 3).

Moved from ``llm/orchestrator.py`` (``_trigger_combat`` + its CR / group /
SRD-guess helpers): encounter building — SRD lookup, CR capping,
plural-group spawning, surprise flags, dual registration, initiative —
lived in the LLM layer (AUDIT_QUALITY_2026_06_09, Architecture P2), and its
tail carried the orphaned note "State transition should be handled by
session manager".

The state transition now IS handled here: :func:`start_encounter` pushes
combat mode via ``session.enter_combat_mode`` — making this the single
combat-entry decision point all three entry signals funnel through (the
triage attack branch, the entity-extractor's ``combat_initiated`` flag, and
the narrator's ``start_combat`` tool).
"""

import re
from typing import Optional, TYPE_CHECKING

import structlog

from .manager import CombatManager, get_combat_for_channel, set_combat_for_channel
from ...data.srd import get_srd
from ...models.npc import Disposition, EntityType, SceneEntity

if TYPE_CHECKING:
    from ..scene.registry import SceneEntityRegistry
    from ..session import GameSession

logger = structlog.get_logger()


def gather_scene_hostiles(scene_registry: "SceneEntityRegistry") -> list[SceneEntity]:
    """Hostile/unfriendly NPCs + creatures currently in the scene.

    The participant draft for narrative-driven combat entry (the extractor's
    ``combat_initiated`` flag and the narrator's ``start_combat`` signal).
    """
    return [
        e for e in scene_registry.get_all_entities()
        if e.disposition in (Disposition.HOSTILE, Disposition.UNFRIENDLY)
        and e.entity_type in (EntityType.NPC, EntityType.CREATURE)
    ]


def guess_monster_index(entity_name: str) -> Optional[str]:
    """Try to find an SRD monster index from a narrative entity name.

    Uses keyword matching against SRD monster names.
    e.g., 'Hooded Figure' might match 'bandit', 'Ash-clad Intruder' → 'cult-fanatic'
    """
    srd = get_srd()

    name_lower = entity_name.lower()

    # Try direct SRD lookup first (e.g., "goblin" → "goblin")
    simple_index = name_lower.replace(" ", "-").replace("'", "")
    monster = srd.get_monster(simple_index)
    if monster:
        return simple_index

    # Try each word from the name as an SRD index
    for word in name_lower.split():
        if len(word) > 3:
            monster = srd.get_monster(word)
            if monster:
                return word

    # Common narrative-to-SRD fallbacks
    fallbacks = {
        "guard": "guard", "soldier": "guard", "watchman": "guard",
        "thug": "thug", "brute": "thug", "enforcer": "thug",
        "bandit": "bandit", "robber": "bandit", "brigand": "bandit",
        "assassin": "assassin", "killer": "assassin",
        "mage": "mage", "wizard": "mage", "sorcerer": "mage", "spellcaster": "mage",
        "cultist": "cultist", "fanatic": "cult-fanatic", "zealot": "cult-fanatic",
        "knight": "knight", "champion": "knight", "paladin": "knight",
        "priest": "priest", "cleric": "priest", "healer": "priest",
        "spy": "spy", "rogue": "spy", "scout": "scout",
        "wolf": "wolf", "bear": "brown-bear", "rat": "giant-rat",
        "skeleton": "skeleton", "zombie": "zombie", "ghoul": "ghoul",
        "figure": "bandit", "stranger": "bandit", "intruder": "bandit",
    }
    for word in name_lower.split():
        if word in fallbacks:
            return fallbacks[word]

    return None


def _detect_group_count(name: str) -> int:
    """Detect if an entity name represents a group and return the count.

    Returns 1 for singular entities, >1 for groups.
    Examples: "Goblins" → 3, "Three Bandits" → 3, "Goblin" → 1
    """
    name_lower = name.lower().strip()

    # Check for explicit number words
    number_words = {
        "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "pair": 2, "couple": 2, "trio": 3,
    }
    for word, count in number_words.items():
        if word in name_lower:
            return count

    # Check for digit prefix: "3 goblins"
    digit_match = re.match(r'^(\d+)\s+', name_lower)
    if digit_match:
        return min(int(digit_match.group(1)), 6)  # Cap at 6

    # Check for simple plural (ends in 's' but not 'ss')
    # Common D&D creature names that are plural: goblins, bandits, wolves, skeletons
    if name_lower.endswith('s') and not name_lower.endswith('ss'):
        return 3  # Default group size for unnamed plurals

    return 1


def _singularize_name(name: str) -> str:
    """Convert a plural group name to singular for individual combatants."""
    # Strip number prefixes: "Three Goblins" → "Goblins", "3 Bandits" → "Bandits"
    name = re.sub(r'^(?:two|three|four|five|six|pair|couple|trio|\d+)\s+', '', name, flags=re.IGNORECASE).strip()

    # Basic singularize: "Goblins" → "Goblin", "Wolves" → "Wolf"
    if name.lower().endswith('ves'):
        return name[:-3] + 'f'  # wolves → wolf
    if name.lower().endswith('ies'):
        return name[:-3] + 'y'  # harpies → harpy
    if name.lower().endswith('s') and not name.lower().endswith('ss'):
        return name[:-1]

    return name


def _max_cr_for_party(session: "GameSession") -> float:
    """Get the maximum CR monster appropriate for the current party."""
    levels = []
    for player in session.players.values():
        if player.character:
            levels.append(player.character.level)

    if not levels:
        return 1

    avg_level = sum(levels) / len(levels)
    party_size = len(levels)

    # Rough CR budget: avg_level for a full party of 4, scaled down for fewer
    # Solo player: max CR ~= level * 0.5
    # 2 players: max CR ~= level * 0.75
    # 3-4 players: max CR ~= level
    scale = min(1.0, party_size / 4)
    return max(0.5, avg_level * scale)


def start_encounter(
    session: Optional["GameSession"],
    hostile_entities: list[SceneEntity],
    player_initiated: bool = False,
) -> bool:
    """Build, register, and start a combat encounter — then push combat mode.

    The single combat-entry decision point (Step 3). Idempotent: an already
    registered combat for the channel is adopted (mode re-pushed onto it)
    instead of duplicated.

    Args:
        session: the live GameSession; None (e.g. voice paths that never
            bound one) means no encounter can be built.
        hostile_entities: entities to add as enemy combatants.
        player_initiated: if True, enemies are surprised (player ambush).

    Returns True when combat is running for this session's channel.
    """
    if not session:
        logger.warning("cannot_trigger_combat_no_session")
        return False

    # Check if combat already exists (idempotent - don't create duplicates)
    existing_combat = get_combat_for_channel(session.channel_id)
    if existing_combat:
        logger.info(
            "combat_already_exists",
            channel_id=session.channel_id,
            combat_id=existing_combat.combat.id,
        )
        # Adopt it: heals a session whose combat was registered by another
        # path (e.g. a /combat cog) without the mode ever flipping.
        session.enter_combat_mode(existing_combat)
        return True

    # Get entity names for logging
    hostile_names = [e.name for e in hostile_entities]

    logger.warning(
        "auto_triggering_combat",
        hostile_count=len(hostile_entities),
        hostiles=hostile_names,
        player_initiated=player_initiated,
    )

    try:
        # Create combat encounter
        description = f"Combat erupts with {', '.join(hostile_names)}!"
        if player_initiated:
            description = f"Your surprise attack catches them off guard! {description}"

        combat = CombatManager.create_encounter(
            session_id=session.id,
            channel_id=session.channel_id,
            name="Combat",
            description=description,
        )

        # Add player combatants
        for player in session.players.values():
            if player.character:
                combat.add_player(player.character)

        # Add hostile entities as combatants
        for entity in hostile_entities:
            combatant = None
            monster_index = entity.monster_index

            # If no monster_index, try fuzzy SRD lookup by name
            if not monster_index:
                monster_index = guess_monster_index(entity.name)

            # Detect group entities (plural names like "Goblins", "Three Bandits")
            # and spawn multiple individual combatants
            count = _detect_group_count(entity.name)
            if count > 1:
                # Singular name for individual combatants
                singular = _singularize_name(entity.name)
                for i in range(count):
                    individual_name = f"{singular} {i + 1}" if count > 1 else singular
                    ind_combatant = None
                    if monster_index:
                        try:
                            ind_combatant = combat.add_monster(monster_index, name=individual_name)
                        except Exception:
                            pass
                    if not ind_combatant:
                        ind_combatant = combat.add_custom_combatant(
                            name=individual_name,
                            hp=entity.hp_estimate or 20,
                            ac=entity.ac_estimate or 12,
                            is_player=False,
                        )
                    if ind_combatant and player_initiated:
                        ind_combatant.is_surprised = True
                continue  # Skip the single-combatant path below

            if monster_index:
                # CR cap: don't spawn monsters too strong for the party
                max_cr = _max_cr_for_party(session)
                try:
                    srd = get_srd()
                    monster_data = srd.get_monster(monster_index)
                    if monster_data:
                        cr = monster_data.get("challenge_rating", 0)
                        if cr > max_cr:
                            # Downgrade to a weaker variant
                            logger.info(
                                "monster_cr_capped",
                                monster=monster_index,
                                cr=cr,
                                max_cr=max_cr,
                                entity=entity.name,
                            )
                            # Use the base type (e.g., bandit-captain → bandit)
                            base_index = monster_index.split("-")[0] if "-" in monster_index else None
                            if base_index and srd.get_monster(base_index):
                                monster_index = base_index
                    combatant = combat.add_monster(monster_index, name=entity.name)
                except Exception as e:
                    logger.warning(
                        "monster_not_found_using_custom",
                        monster_index=monster_index,
                        error=str(e),
                        exc_info=True,
                    )

            # Fallback: create custom combatant with reasonable defaults
            if not combatant:
                combatant = combat.add_custom_combatant(
                    name=entity.name,
                    hp=entity.hp_estimate or 20,
                    ac=entity.ac_estimate or 12,
                    is_player=False,
                )

            # SURPRISE: If player initiated, enemies are surprised
            if combatant and player_initiated:
                combatant.is_surprised = True
                logger.info("combatant_surprised", combatant=combatant.name)

        # Roll initiative immediately so combat is ready
        combat.roll_all_initiative()
        combat.start_combat()

        # Only a fully started encounter is published: registration and the
        # mode push happen AFTER initiative/start, so an exception above
        # leaves NO half-state anywhere (the pre-move code registered and
        # stored the manager first — a failure stranded a SETUP-state combat
        # in the registry that the adopt branch would then resurrect).
        set_combat_for_channel(session.channel_id, combat)

        # Push combat mode: state, session.combat_manager, and
        # world_state.phase flip in one owned place (the ModeMachine push
        # that used to be process_message's inline branch).
        session.enter_combat_mode(combat)

        logger.info(
            "combat_auto_created",
            combatant_count=len(combat.combat.combatants),
            hostile_count=len(hostile_entities),
            enemies_surprised=player_initiated,
        )

        return True

    except Exception as e:
        logger.error("combat_trigger_failed", error=str(e), exc_info=True)
        return False
