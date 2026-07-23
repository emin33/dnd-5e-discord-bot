"""
Test Harness - Bypasses Discord to test the full game pipeline.

Usage:
    python test_harness.py                    # Interactive mode
    python test_harness.py --auto scenario1   # Run a predefined scenario
    python test_harness.py --action "I attack the goblin"  # Single action

Uses the real orchestration and memory stack with run-unique temporary SQLite
and Chroma storage. Only Discord message transport is skipped.
"""

import asyncio
import gc
import json
import re
import shutil
import sys
import os
import tempfile
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ensure we're in the right directory for .env loading
os.chdir(Path(__file__).parent)

# Must import after chdir so pydantic-settings finds .env
import uuid
from dnd_bot.config import get_settings, get_profile
from dnd_bot.game.session import get_session_manager, GameSessionManager
from dnd_bot.data.repositories import get_character_repo, get_campaign_repo
from dnd_bot.data.database import close_database, get_database
from dnd_bot.models.character import Character, HitPoints, HitDice, DeathSaves, SpellSlots, Skill
from dnd_bot.models.common import AbilityScore
from dnd_bot.models.character import AbilityScores
from dnd_bot.models.campaign import Campaign
from dnd_bot.llm.orchestrator import DMResponse


# -------------------- Output Formatting --------------------

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}\n")


def print_section(label: str, content: str, color: str = Colors.GREEN):
    print(f"{Colors.BOLD}{color}[{label}]{Colors.RESET} {content}")


def print_response(response: DMResponse, elapsed: float):
    """Pretty-print a DMResponse with all details."""
    print()

    # Mechanics
    if response.mechanical_result:
        m = response.mechanical_result
        action_type = m.get("action_type", "unknown")
        success = m.get("success")
        skill = m.get("skill", "")
        dc = m.get("dc", "?")
        roll = m.get("roll", "?")

        status = f"{Colors.GREEN}Success{Colors.RESET}" if success else f"{Colors.RED}Failure{Colors.RESET}"
        print_section("MECHANICS", f"{action_type}: {skill} | Roll: {roll} vs DC {dc} → {status}", Colors.BLUE)

    # Dice rolls
    if response.dice_rolls:
        for dr in response.dice_rolls:
            print_section("DICE", f"{dr}", Colors.YELLOW)

    # Narration
    if response.narrative:
        print_section("NARRATION", "", Colors.MAGENTA)
        # Word-wrap narration at 80 chars
        words = response.narrative.split()
        line = "  "
        for word in words:
            if len(line) + len(word) + 1 > 80:
                print(line)
                line = "  "
            line += word + " "
        if line.strip():
            print(line)

    # Combat — in automated tests, auto-resolve since real combat
    # uses Discord buttons/panels that the LLM player can't interact with.
    if response.combat_triggered:
        print_section("COMBAT", "TRIGGERED → auto-resolved for test", Colors.YELLOW)

    # Tool calls
    if response.tool_calls_made:
        print_section("TOOLS", f"{len(response.tool_calls_made)} tool calls made", Colors.DIM)

    # Timing
    print(f"\n{Colors.DIM}  [{elapsed:.1f}s]{Colors.RESET}")


# -------------------- Issue Detection --------------------

@dataclass
class Issue:
    severity: str  # "ERROR", "WARNING", "INFO"
    category: str  # "unnecessary_roll", "high_dc", "no_combat", "chinese_chars", etc.
    description: str
    context: dict = field(default_factory=dict)


_ATTACK_ACTION_RE = re.compile(
    r"\b(?:attack|strike|hit|stab|slash|shoot|fight|kill)\b",
    re.IGNORECASE,
)


def _looks_like_attack_action(action: str) -> bool:
    """Return whether an action contains a standalone attack verb."""
    return bool(_ATTACK_ACTION_RE.search(action))


def detect_issues(
    action: str,
    response: DMResponse,
    turn_number: int,
) -> list[Issue]:
    """Analyze a response for potential issues."""
    issues = []

    # Check for CJK characters in narration
    if response.narrative:
        cjk = re.findall(r'[\u4e00-\u9fff]+', response.narrative)
        if cjk:
            issues.append(Issue(
                severity="ERROR",
                category="chinese_chars",
                description=f"CJK characters in narration: {''.join(cjk)}",
                context={"chars": cjk},
            ))

    # Check for unnecessary rolls on trivial actions
    trivial_actions = [
        "look around", "look at", "examine", "check my inventory",
        "sit down", "stand up", "walk to", "go to", "enter",
        "talk to", "say", "ask", "tell", "greet",
    ]
    if response.mechanical_result:
        m = response.mechanical_result
        action_lower = action.lower()
        if m.get("action_type") == "skill_check":
            for trivial in trivial_actions:
                if trivial in action_lower:
                    # Check if DC is unreasonably high for the action
                    dc = m.get("dc", 0)
                    if dc > 10:
                        issues.append(Issue(
                            severity="WARNING",
                            category="unnecessary_roll",
                            description=f"Rolled {m.get('skill', '?')} DC {dc} for trivial action: '{action}'",
                            context={"skill": m.get("skill"), "dc": dc},
                        ))
                    break

    # Check for unreasonable DCs
    if response.mechanical_result:
        dc = response.mechanical_result.get("dc", 0)
        if dc and dc > 25:
            issues.append(Issue(
                severity="WARNING",
                category="high_dc",
                description=f"DC {dc} seems very high for a level 1 adventure",
                context={"dc": dc},
            ))
        if dc and dc < 5 and dc > 0:
            issues.append(Issue(
                severity="INFO",
                category="low_dc",
                description=f"DC {dc} is trivially easy - why roll?",
                context={"dc": dc},
            ))

    # Check for attack actions that didn't trigger combat
    if _looks_like_attack_action(action):
        if not response.combat_triggered and not (response.mechanical_result and response.mechanical_result.get("action_type") == "attack"):
            issues.append(Issue(
                severity="ERROR",
                category="no_combat",
                description=f"Attack action '{action}' did not trigger combat",
            ))

    # Check for empty narration
    if not response.narrative or len(response.narrative.strip()) < 20:
        issues.append(Issue(
            severity="ERROR",
            category="empty_narration",
            description="Narration is empty or too short",
        ))

    # Check for truncated narration
    if response.narrative and len(response.narrative.strip()) < 50:
        issues.append(Issue(
            severity="WARNING",
            category="short_narration",
            description=f"Narration is only {len(response.narrative)} chars - possibly truncated",
        ))

    return issues


def print_issues(issues: list[Issue]):
    """Print detected issues."""
    if not issues:
        print(f"  {Colors.GREEN}No issues detected{Colors.RESET}")
        return

    for issue in issues:
        color = Colors.RED if issue.severity == "ERROR" else Colors.YELLOW if issue.severity == "WARNING" else Colors.DIM
        print(f"  {color}[{issue.severity}] {issue.category}: {issue.description}{Colors.RESET}")


# -------------------- Test Session --------------------

class TestSession:
    """Manages a test game session with fresh disposable test data."""

    COMBAT_POLICIES = {"simulate_victory", "fail"}

    def __init__(
        self,
        combat_policy: str = "simulate_victory",
        world_setting: Optional[str] = None,
        isolated_storage: bool = True,
    ):
        if combat_policy not in self.COMBAT_POLICIES:
            choices = ", ".join(sorted(self.COMBAT_POLICIES))
            raise ValueError(f"Unknown combat policy '{combat_policy}'; choose {choices}")
        self.manager: Optional[GameSessionManager] = None
        self.channel_id = 999999  # Fake channel
        self.guild_id = 999999    # Fake guild
        self.user_id = 888888     # Fake user
        self.user_name = "TestPlayer"
        self.campaign_id: Optional[str] = None
        self.character: Optional[Character] = None
        self.turn_number = 0
        self.all_issues: list[Issue] = []
        self.action_log: list[dict] = []
        self._owns_data = False  # True if we created the campaign/character
        self.combat_policy = combat_policy
        self.world_setting = world_setting
        self.combat_interventions: list[dict] = []
        self.continuity_interventions: list[dict] = []
        self.last_combat_intervention: Optional[dict] = None
        self.isolated_storage = isolated_storage
        self.storage_root: Optional[Path] = None
        self._storage_temp = None
        self._original_storage_paths: Optional[tuple[str, str]] = None
        self._cleaned_up = False

    async def setup(self, campaign_id: Optional[str] = None, character_id: Optional[str] = None):
        """Initialize the test session with fresh test data."""
        if self.isolated_storage:
            if campaign_id or character_id:
                raise ValueError(
                    "Existing campaign/character IDs require isolated_storage=False"
                )
            await self._configure_isolated_storage()

        await get_database()
        self.manager = get_session_manager()

        camp_repo = await get_campaign_repo()
        char_repo = await get_character_repo()

        if campaign_id and character_id:
            # Use existing data
            self.campaign_id = campaign_id
            self.character = await char_repo.get_by_id(character_id)
        else:
            # Create fresh test data
            self._owns_data = True
            campaign, character = await self._create_test_data(camp_repo, char_repo)
            self.campaign_id = campaign.id
            self.character = character

        if not self.character:
            print(f"{Colors.RED}Failed to create/load character!{Colors.RESET}")
            return False

        print_section("CAMPAIGN", f"test_harness ({self.campaign_id[:8]}...)")
        print_section("CHARACTER", f"{self.character.name} - L{self.character.level} {self.character.race_index} {self.character.class_index}")

        # Start session
        session = await self.manager.start_session(
            channel_id=self.channel_id,
            guild_id=self.guild_id,
            campaign_id=self.campaign_id,
        )
        print_section("SESSION", f"Started: {session.id}")

        # The Discord campaign flow puts its setting into the opening scene,
        # but this transport-free harness does not generate that opening. Seed
        # the same always-on memory block so scenario-specific test campaigns
        # reach the production narrator context instead of silently falling
        # back to CoreMemory's generic fantasy world.
        setting = self.world_setting
        if not setting:
            campaign = await camp_repo.get_by_id(self.campaign_id)
            setting = campaign.world_setting if campaign else None
        if setting:
            from dnd_bot.memory import get_memory_manager
            memory = await get_memory_manager(self.campaign_id)
            memory.set_world_setting(setting)

        # Join player
        await self.manager.join_session(
            channel_id=self.channel_id,
            user_id=self.user_id,
            user_name=self.user_name,
            character=self.character,
        )
        print_section("PLAYER", f"Joined as {self.character.name}")
        return True

    async def _configure_isolated_storage(self) -> None:
        """Point SQLite and Chroma at a run-unique temporary namespace."""
        if self._storage_temp is not None:
            return

        await close_database()
        from dnd_bot.memory import reset_vector_store
        reset_vector_store()

        self._storage_temp = tempfile.TemporaryDirectory(prefix="dnd-longform-")
        self.storage_root = Path(self._storage_temp.name).resolve()
        settings = get_settings()
        self._original_storage_paths = (
            settings.database_path,
            settings.chroma_persist_path,
        )
        settings.database_path = str(self.storage_root / "dnd_bot.db")
        settings.chroma_persist_path = str(self.storage_root / "chroma")
        print_section("STORAGE", f"isolated: {self.storage_root}", Colors.DIM)

    async def _create_test_data(self, camp_repo, char_repo) -> tuple:
        """Create a fresh campaign and character for testing."""
        # Create campaign
        campaign = Campaign(
            id=str(uuid.uuid4()),
            guild_id=self.guild_id,
            name="test_harness",
            description="Automated test session - safe to delete",
            dm_user_id=self.user_id,
            world_setting=self.world_setting or (
                "A bustling medieval city with taverns, markets, dark alleys, "
                "and a looming threat from goblins in the nearby forest."
            ),
        )
        await camp_repo.create(campaign)

        # Create a Level 1 Elf Ranger (same as quick-join template)
        abilities = AbilityScores(
            strength=12, dexterity=17, constitution=13,
            intelligence=10, wisdom=14, charisma=8,
        )
        con_mod = (abilities.constitution - 10) // 2
        character = Character(
            id=str(uuid.uuid4()),
            discord_user_id=self.user_id,
            campaign_id=campaign.id,
            name="Kael Windrunner",
            race_index="elf",
            class_index="ranger",
            level=1,
            experience=0,
            abilities=abilities,
            armor_class=10 + abilities.dex_mod,  # 13 unarmored
            speed=30,
            initiative_bonus=abilities.dex_mod,
            hp=HitPoints(maximum=10 + con_mod, current=10 + con_mod, temporary=0),
            hit_dice=HitDice(die_type=10, total=1, remaining=1),
            death_saves=DeathSaves(),
            spellcasting_ability=AbilityScore.WISDOM,
            spell_slots=SpellSlots(),
            saving_throw_proficiencies=[AbilityScore.STRENGTH, AbilityScore.DEXTERITY],
            skill_proficiencies=[Skill.PERCEPTION, Skill.STEALTH, Skill.SURVIVAL],
        )
        await char_repo.create(character)

        # Assign starting equipment
        try:
            from dnd_bot.game.character.starting_equipment import assign_starting_equipment
            await assign_starting_equipment(character.id, character.class_index)
            print_section("EQUIPMENT", "Starting gear assigned")
        except Exception as e:
            print_section("EQUIPMENT", f"Skipped: {e}", Colors.YELLOW)

        return campaign, character

    async def send_action(self, action: str) -> Optional[DMResponse]:
        """Send a player action and get the response."""
        self.turn_number += 1
        self.last_combat_intervention = None
        print(f"\n{Colors.BOLD}{'-'*60}{Colors.RESET}")
        print(f"{Colors.BOLD}  Turn {self.turn_number}: {Colors.CYAN}\"{action}\"{Colors.RESET}")
        print(f"{Colors.BOLD}{'-'*60}{Colors.RESET}")

        start = time.time()
        try:
            response = await asyncio.wait_for(
                self.manager.process_message(
                    channel_id=self.channel_id,
                    user_id=self.user_id,
                    user_name=self.user_name,
                    content=action,
                ),
                timeout=180,  # 180s max per action (triage + narrate + extract = 3 LLM calls)
            )
        except asyncio.TimeoutError:
            elapsed = time.time() - start
            print(f"\n  {Colors.RED}TIMEOUT after {elapsed:.0f}s{Colors.RESET}")
            self.all_issues.append(Issue(
                severity="ERROR",
                category="timeout",
                description=f"Action timed out after {elapsed:.0f}s: '{action}'",
                context={"action": action, "turn": self.turn_number},
            ))
            self.action_log.append({
                "turn": self.turn_number,
                "action": action,
                "error": "timeout",
                "elapsed": elapsed,
            })
            return None
        except Exception as e:
            elapsed = time.time() - start
            print(f"\n  {Colors.RED}EXCEPTION: {type(e).__name__}: {e}{Colors.RESET}")
            self.all_issues.append(Issue(
                severity="ERROR",
                category="exception",
                description=f"{type(e).__name__}: {e}",
                context={"action": action, "turn": self.turn_number},
            ))
            self.action_log.append({
                "turn": self.turn_number,
                "action": action,
                "error": str(e),
                "elapsed": elapsed,
            })
            return None

        elapsed = time.time() - start

        if response:
            print_response(response, elapsed)

            # Detect issues
            issues = detect_issues(action, response, self.turn_number)
            print_issues(issues)
            self.all_issues.extend(issues)

            # Memory diagnostics
            memory_state = self._get_memory_diagnostics()
            if memory_state:
                print(f"  {Colors.DIM}[MEM] buffer={memory_state['buffer_size']}, "
                      f"overflow={memory_state['overflow_size']}, "
                      f"summary={'yes' if memory_state['has_summary'] else 'no'} "
                      f"({memory_state['summary_len']}ch), "
                      f"scratchpad={memory_state['scratchpad_entries']}"
                      f"{Colors.RESET}")

            # Auto-resolve combat for automated tests — real combat uses
            # Discord buttons/panels that the LLM player can't interact with.
            if response.combat_triggered:
                self.last_combat_intervention = await self._resolve_test_combat()
                self.combat_interventions.append(self.last_combat_intervention)

            # A narrative soak is not a death-save simulator. Noncombat tool
            # damage can also reduce the synthetic player to 0 HP and make PGI
            # block every remaining creative/memory turn. Under the same
            # simulate_victory policy, restore agency after recording the
            # consequence so the test can continue stressing the intended
            # systems. The fail policy remains untouched and exposes the stop.
            continuity_intervention = await self._restore_test_player_agency()
            if continuity_intervention:
                self.continuity_interventions.append(continuity_intervention)

            self.action_log.append({
                "turn": self.turn_number,
                "action": action,
                "narrative_len": len(response.narrative) if response.narrative else 0,
                "mechanics": response.mechanical_result,
                "combat": response.combat_triggered,
                "combat_intervention": self.last_combat_intervention,
                "continuity_intervention": continuity_intervention,
                "memory": memory_state,
                "issues": [{"severity": i.severity, "category": i.category, "desc": i.description} for i in issues],
                "elapsed": elapsed,
            })
        else:
            print(f"\n  {Colors.RED}No response returned{Colors.RESET}")
            self.all_issues.append(Issue(
                severity="ERROR",
                category="no_response",
                description=f"No response for action: '{action}'",
            ))

        # Save incremental log after every action (survives crashes)
        self._save_log()

        return response

    async def _restore_test_player_agency(self) -> Optional[dict]:
        """Heal a 0-HP synthetic player so a longform soak can keep running."""
        if self.combat_policy != "simulate_victory" or self.manager is None:
            return None
        game_session = self.manager.get_session(self.channel_id)
        if game_session is None:
            return None
        character = game_session.get_player_character(self.user_id)
        if character is None or character.hp.current > 0:
            return None

        hp_before = character.hp.current
        character.hp.current = character.hp.maximum
        character.hp.temporary = 0
        character.death_saves.reset()
        self.character = character

        repo = await get_character_repo()
        await repo.update_hp(character.id, character.hp.current, character.hp.temporary)
        await repo.update_death_saves(
            character.id,
            character.death_saves.successes,
            character.death_saves.failures,
        )

        intervention = {
            "turn": self.turn_number,
            "policy": self.combat_policy,
            "outcome": "player_agency_restored",
            "reason": "synthetic player reached 0 HP during narrative soak",
            "hp_before": hp_before,
            "hp_after": character.hp.current,
        }
        print_section(
            "CONTINUITY",
            f"restored synthetic player from {hp_before} to {character.hp.current} HP",
            Colors.YELLOW,
        )
        return intervention

    async def _resolve_test_combat(self) -> dict:
        """Apply the harness combat policy without exercising combat UI.

        Both policies use the production teardown owner. ``simulate_victory``
        lets a narrative/memory scenario continue; ``fail`` also records a
        hard issue for suites that prohibit any combat trigger.
        """
        teardown_complete = await self.manager.end_combat(self.channel_id)
        intervention = {
            "turn": self.turn_number,
            "policy": self.combat_policy,
            "outcome": (
                "player_victory"
                if self.combat_policy == "simulate_victory"
                else "failed"
            ),
            "teardown_complete": teardown_complete,
        }

        if self.combat_policy == "fail" or not teardown_complete:
            description = (
                "Combat triggered while combat_policy=fail"
                if self.combat_policy == "fail"
                else "Combat triggered but canonical teardown found no encounter"
            )
            self.all_issues.append(Issue(
                severity="ERROR",
                category="combat_policy",
                description=description,
                context=intervention,
            ))

        print_section(
            "COMBAT",
            f"policy={self.combat_policy}, outcome={intervention['outcome']}, "
            f"teardown={'ok' if teardown_complete else 'missing'}",
            Colors.YELLOW,
        )
        return intervention

    def _get_memory_diagnostics(self) -> dict:
        """Get memory subsystem state for diagnostics."""
        try:
            from dnd_bot.memory.manager import _managers
            mgr = _managers.get(self.campaign_id)
            if not mgr:
                return {}

            # Scratchpad from orchestrator
            scratchpad_entries = 0
            try:
                orch = self.manager.orchestrator
                scratchpad_entries = len(orch._scratchpad) if hasattr(orch, '_scratchpad') else 0
            except Exception:
                pass

            return {
                "buffer_size": len(mgr.buffer._messages),
                "overflow_size": len(mgr.buffer._overflow_buffer),
                "has_summary": bool(mgr.buffer.running_summary),
                "summary_len": len(mgr.buffer.running_summary),
                "summary_preview": mgr.buffer.running_summary[:120] if mgr.buffer.running_summary else "",
                "pinned_facts": mgr.buffer.pinned_facts[:],
                "pinned_facts_count": len(mgr.buffer.pinned_facts),
                "message_count": mgr._message_count,
                "last_summary_at": mgr._last_summary_at,
                "is_in_combat": mgr._is_in_combat,
                "scratchpad_entries": scratchpad_entries,
            }
        except Exception:
            return {}

    def _save_log(self):
        """Save action log to disk incrementally."""
        try:
            log_path = Path("data/test_logs")
            log_path.mkdir(parents=True, exist_ok=True)
            log_file = log_path / "test_latest.json"
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(self.action_log, f, indent=2, default=str)
        except Exception:
            pass

    async def cleanup(self):
        """End session, print summary, and delete test data."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        if self.manager:
            try:
                await self.manager.end_session(self.channel_id)
            except Exception as e:
                print_section("CLEANUP", f"Session end warning: {e}", Colors.YELLOW)

        # Delete test data if we created it
        if self._owns_data and self.campaign_id:
            try:
                db = await get_database()
                # Campaign is the ownership root. Foreign-key cascades remove
                # sessions, characters, memory, and knowledge-graph rows.
                await db.execute("DELETE FROM campaign WHERE id = ?", (self.campaign_id,))
                await db.commit()

                residue_queries = {
                    "campaign": "SELECT COUNT(*) FROM campaign WHERE id = ?",
                    "sessions": "SELECT COUNT(*) FROM game_session WHERE campaign_id = ?",
                    "characters": "SELECT COUNT(*) FROM character WHERE campaign_id = ?",
                    "memory": "SELECT COUNT(*) FROM campaign_memory WHERE campaign_id = ?",
                    "kg_nodes": "SELECT COUNT(*) FROM kg_node WHERE campaign_id = ?",
                    "kg_edges": "SELECT COUNT(*) FROM kg_edge WHERE campaign_id = ?",
                }
                residue = {}
                for label, query in residue_queries.items():
                    row = await db.fetch_one(query, (self.campaign_id,))
                    count = int(row[0]) if row else 0
                    if count:
                        residue[label] = count
                if residue:
                    raise RuntimeError(f"SQLite cleanup residue: {residue}")

                from dnd_bot.memory import get_vector_store
                if not get_vector_store().delete_campaign(self.campaign_id):
                    raise RuntimeError("Chroma campaign collection cleanup failed")
                print_section("CLEANUP", "SQLite and vector data deleted", Colors.DIM)
            except Exception as e:
                print_section("CLEANUP", f"Warning: {e}", Colors.YELLOW)

        # Print memory final state
        mem_state = self._get_memory_diagnostics()
        if mem_state:
            print_header("MEMORY DIAGNOSTICS")
            print(f"  Messages processed: {mem_state.get('message_count', '?')}")
            print(f"  Buffer size: {mem_state.get('buffer_size', '?')}")
            print(f"  Overflow waiting: {mem_state.get('overflow_size', '?')}")
            print(f"  Running summary: {'YES' if mem_state.get('has_summary') else 'NO'} ({mem_state.get('summary_len', 0)} chars)")
            if mem_state.get('summary_preview'):
                print(f"  Summary preview: \"{mem_state['summary_preview']}...\"")
            print(f"  Scratchpad entries: {mem_state.get('scratchpad_entries', 0)}")

        # Print summary
        print_header("TEST SUMMARY")
        print(f"  Turns played: {self.turn_number}")
        print(f"  Total issues: {len(self.all_issues)}")

        errors = [i for i in self.all_issues if i.severity == "ERROR"]
        warnings = [i for i in self.all_issues if i.severity == "WARNING"]
        print(f"    {Colors.RED}Errors: {len(errors)}{Colors.RESET}")
        print(f"    {Colors.YELLOW}Warnings: {len(warnings)}{Colors.RESET}")

        if errors:
            print(f"\n  {Colors.RED}ERRORS:{Colors.RESET}")
            for e in errors:
                print(f"    - [{e.category}] {e.description}")

        if warnings:
            print(f"\n  {Colors.YELLOW}WARNINGS:{Colors.RESET}")
            for w in warnings:
                print(f"    - [{w.category}] {w.description}")

        # Save log
        log_path = Path("data/test_logs")
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / f"test_{int(time.time())}.json"
        with open(log_file, "w") as f:
            json.dump(self.action_log, f, indent=2, default=str)
        print(f"\n  Log saved: {log_file}")

        # Release process caches before removing the run-unique namespace.
        if self.campaign_id:
            from dnd_bot.memory import discard_memory_manager
            discard_memory_manager(self.campaign_id)
        await close_database()

        if self._storage_temp is not None:
            from dnd_bot.memory import reset_vector_store
            reset_vector_store()
            if self._original_storage_paths is not None:
                settings = get_settings()
                settings.database_path, settings.chroma_persist_path = (
                    self._original_storage_paths
                )
            try:
                self._storage_temp.cleanup()
            except Exception as e:
                # Chroma's embedded SQLite handle can finish releasing a beat
                # after system.stop() on Windows. Force collection and retry
                # only this run-owned temp namespace before reporting residue.
                storage_error = e
                storage_root = self.storage_root
                if storage_root is not None:
                    temp_root = Path(tempfile.gettempdir()).resolve()
                    resolved = storage_root.resolve()
                    is_owned_temp = (
                        resolved.parent == temp_root
                        and resolved.name.startswith("dnd-longform-")
                    )
                    if is_owned_temp:
                        for delay in (0.0, 0.05, 0.2):
                            gc.collect()
                            if delay:
                                await asyncio.sleep(delay)
                            try:
                                if resolved.exists():
                                    shutil.rmtree(resolved)
                                storage_error = None
                                break
                            except Exception as retry_error:
                                storage_error = retry_error
                if storage_error is not None:
                    print_section(
                        "CLEANUP",
                        f"Temporary storage warning: {storage_error}",
                        Colors.YELLOW,
                    )
            self._storage_temp = None


# -------------------- Predefined Scenarios --------------------

SCENARIOS = {
    "tavern_explore": [
        "I look around the tavern",
        "I walk over to the bartender and ask about rumors",
        "I examine the notice board on the wall",
        "I order a drink and sit down",
    ],
    "combat_flow": [
        "I walk down the dark alley",
        "I draw my weapon and attack the closest enemy",
        # After combat triggers, these would need combat UI - future enhancement
    ],
    "social_encounter": [
        "I approach the merchant and greet them",
        "I ask if they have any healing potions for sale",
        "I try to haggle for a better price",
        "I thank them and leave",
    ],
    "exploration": [
        "I search the room for traps",
        "I try to pick the lock on the chest",
        "I open the chest carefully",
        "I take what's inside and check for secret doors",
    ],
    "stress_test": [
        "I look around",
        "I sit down",
        "I stand up",
        "I walk forward",
        "I attack the nearest enemy",
        "I cast a fireball",
        "I check for traps",
        "I try to persuade the guard to let me through",
    ],
    # ---- Memory Continuity Test ----
    # Phase 1 plants named details (NPC name, item, location).
    # Phase 2 is filler that pushes Phase 1 out of the message buffer.
    # Phase 3 references Phase 1 details BY NAME - the narrator can
    # only know them if the compaction summary or scratchpad preserved them.
    "memory_continuity": [
        # Phase 1: Plant memorable details
        "I enter the tavern and introduce myself to the bartender",
        "I ask the bartender what his name is",
        "I ask if he has heard about any trouble in the nearby forest",
        "I order the house special and look for anything unusual",
        # Phase 2: Filler actions to push Phase 1 out of the 20-message window
        # (each turn = 2 messages: player + DM, so 10 turns ≈ 20 messages)
        "I finish my drink and step outside",
        "I look up at the sky and check the weather",
        "I walk down the main street of the village",
        "I stop at a market stall and browse the wares",
        "I ask the merchant about the price of rope",
        "I continue walking toward the town gate",
        "I examine the gate and the guards posted there",
        "I nod to the guards and walk through the gate",
        "I follow the path into the woods outside town",
        "I stop and listen to the sounds of the forest",
        # Phase 3: Reference Phase 1 details - narrator needs memory to respond well
        "I turn back toward the tavern to find the bartender I met earlier",
        "I ask the bartender to remind me what he said about the forest trouble",
    ],
}


# -------------------- Main --------------------

async def run_interactive(session: TestSession):
    """Interactive mode - type actions manually."""
    print_header("INTERACTIVE MODE")
    print("  Type player actions. Commands: /quit, /summary, /scenario <name>")
    print(f"  Available scenarios: {', '.join(SCENARIOS.keys())}")
    print()

    while True:
        try:
            action = input(f"{Colors.CYAN}> {Colors.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not action:
            continue
        if action == "/quit":
            break
        if action == "/summary":
            await session.cleanup()
            return  # cleanup already prints summary
        if action.startswith("/scenario "):
            name = action.split(" ", 1)[1].strip()
            if name in SCENARIOS:
                for a in SCENARIOS[name]:
                    await session.send_action(a)
            else:
                print(f"  Unknown scenario. Available: {', '.join(SCENARIOS.keys())}")
            continue

        await session.send_action(action)

    await session.cleanup()


async def run_scenario(session: TestSession, scenario_name: str):
    """Run a predefined scenario."""
    if scenario_name not in SCENARIOS:
        print(f"{Colors.RED}Unknown scenario: {scenario_name}{Colors.RESET}")
        print(f"Available: {', '.join(SCENARIOS.keys())}")
        return

    print_header(f"SCENARIO: {scenario_name}")
    actions = SCENARIOS[scenario_name]

    for action in actions:
        response = await session.send_action(action)
        if not response:
            print(f"\n  {Colors.RED}Scenario aborted due to error{Colors.RESET}")
            break
        # Small delay between actions to let state settle
        await asyncio.sleep(0.5)

    await session.cleanup()


async def run_single(session: TestSession, action: str):
    """Run a single action."""
    await session.send_action(action)
    await session.cleanup()


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="D&D Bot Test Harness")
    parser.add_argument("--auto", type=str, help="Run predefined scenario")
    parser.add_argument("--action", type=str, help="Run single action")
    args = parser.parse_args()

    # Tee all output to a log file
    import io
    log_file_path = Path("data/test_run.log")
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    class TeeWriter:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    log_fh = open(log_file_path, "w", encoding="utf-8")
    sys.stdout = TeeWriter(sys.__stdout__, log_fh)
    sys.stderr = TeeWriter(sys.__stderr__, log_fh)

    print_header("D&D 5e BOT TEST HARNESS")

    settings = get_settings()
    profile = get_profile()
    print_section("PROFILE", f"{profile.name}: narrator={profile.narrator.provider}/{profile.narrator.model}, brain={profile.brain.provider}/{profile.brain.model}")

    session = TestSession()
    success = await session.setup()
    if not success:
        return
    print_section("DB", str(settings.db_path))

    if args.auto:
        await run_scenario(session, args.auto)
    elif args.action:
        await run_single(session, args.action)
    else:
        await run_interactive(session)


if __name__ == "__main__":
    asyncio.run(main())
