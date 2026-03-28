"""Character repository for database operations."""

from typing import Optional
import json

from ...models import (
    AbilityScore,
    AbilityScores,
    Character,
    CharacterCondition,
    CharacterProficiency,
    Condition,
    DeathSaves,
    HitDice,
    HitPoints,
    Skill,
    SpellSlots,
)
from ..database import Database, get_database


class CharacterRepository:
    """Repository for Character database operations."""

    def __init__(self, db: Optional[Database] = None):
        self._db = db

    async def _get_db(self) -> Database:
        if self._db:
            return self._db
        return await get_database()

    async def create(self, character: Character) -> Character:
        """Create a new character in the database.

        Uses a savepoint transaction so all inserts succeed or none do.
        """
        db = await self._get_db()

        async with await db.transaction():
            await db.execute(
                """
                INSERT INTO character (
                    id, discord_user_id, campaign_id, name,
                    race_index, class_index, subclass_index, level, experience, background_index,
                    strength, dexterity, constitution, intelligence, wisdom, charisma,
                    armor_class, speed, initiative_bonus,
                    hp_max, hp_current, hp_temp,
                    hit_dice_type, hit_dice_total, hit_dice_remaining,
                    death_save_successes, death_save_failures,
                    spellcasting_ability, concentration_spell_id,
                    is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    character.id,
                    character.discord_user_id,
                    character.campaign_id,
                    character.name,
                    character.race_index,
                    character.class_index,
                    character.subclass_index,
                    character.level,
                    character.experience,
                    character.background_index,
                    character.abilities.strength,
                    character.abilities.dexterity,
                    character.abilities.constitution,
                    character.abilities.intelligence,
                    character.abilities.wisdom,
                    character.abilities.charisma,
                    character.armor_class,
                    character.speed,
                    character.initiative_bonus,
                    character.hp.maximum,
                    character.hp.current,
                    character.hp.temporary,
                    character.hit_dice.die_type,
                    character.hit_dice.total,
                    character.hit_dice.remaining,
                    character.death_saves.successes,
                    character.death_saves.failures,
                    character.spellcasting_ability.value if character.spellcasting_ability else None,
                    character.concentration_spell_id,
                    1 if character.is_active else 0,
                ),
            )

            # Insert proficiencies
            for skill in character.skill_proficiencies:
                await db.execute(
                    """
                    INSERT INTO character_proficiency (id, character_id, proficiency_index, proficiency_type, expertise)
                    VALUES (?, ?, ?, 'skill', ?)
                    """,
                    (
                        f"{character.id}-skill-{skill.value}",
                        character.id,
                        f"skill-{skill.value}",
                        1 if skill in character.skill_expertise else 0,
                    ),
                )

            for save in character.saving_throw_proficiencies:
                await db.execute(
                    """
                    INSERT INTO character_proficiency (id, character_id, proficiency_index, proficiency_type, expertise)
                    VALUES (?, ?, ?, 'save', 0)
                    """,
                    (
                        f"{character.id}-save-{save.value}",
                        character.id,
                        f"saving-throw-{save.value}",
                    ),
                )

            # Insert spell slots
            for level in range(1, 10):
                current, maximum = character.spell_slots.get_slots(level)
                if maximum > 0:
                    await db.execute(
                        """
                        INSERT INTO character_spell_slots (character_id, slot_level, slots_max, slots_current)
                        VALUES (?, ?, ?, ?)
                        """,
                        (character.id, level, maximum, current),
                    )

            # Create currency record
            await db.execute(
                """
                INSERT INTO character_currency (character_id, copper, silver, electrum, gold, platinum)
                VALUES (?, 0, 0, 0, 0, 0)
                """,
                (character.id,),
            )

        return character

    async def get_by_id(self, character_id: str) -> Optional[Character]:
        """Get a character by ID."""
        db = await self._get_db()

        row = await db.fetch_one(
            "SELECT * FROM character WHERE id = ?",
            (character_id,),
        )

        if not row:
            return None

        return await self._row_to_character(db, row)

    async def get_by_user_and_campaign(
        self, user_id: int, campaign_id: str
    ) -> Optional[Character]:
        """Get a character by user and campaign."""
        db = await self._get_db()

        row = await db.fetch_one(
            "SELECT * FROM character WHERE discord_user_id = ? AND campaign_id = ? AND is_active = 1",
            (user_id, campaign_id),
        )

        if not row:
            return None

        return await self._row_to_character(db, row)

    async def get_all_by_campaign(self, campaign_id: str) -> list[Character]:
        """Get all characters in a campaign."""
        db = await self._get_db()

        rows = await db.fetch_all(
            "SELECT * FROM character WHERE campaign_id = ? AND is_active = 1",
            (campaign_id,),
        )

        characters = []
        for row in rows:
            char = await self._row_to_character(db, row)
            characters.append(char)

        return characters

    async def get_all_by_user_in_guild(self, user_id: int, guild_id: int) -> list[Character]:
        """Get all characters owned by a user in a specific guild (across all campaigns)."""
        db = await self._get_db()

        rows = await db.fetch_all(
            """
            SELECT c.* FROM character c
            JOIN campaign camp ON c.campaign_id = camp.id
            WHERE c.discord_user_id = ? AND camp.guild_id = ? AND c.is_active = 1
            """,
            (user_id, guild_id),
        )

        characters = []
        for row in rows:
            char = await self._row_to_character(db, row)
            characters.append(char)

        return characters

    async def update_hp(
        self, character_id: str, current: int, temporary: int = 0
    ) -> bool:
        """Update character HP."""
        db = await self._get_db()

        await db.execute(
            "UPDATE character SET hp_current = ?, hp_temp = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (current, temporary, character_id),
        )
        await db.commit()
        return True

    async def update_death_saves(
        self, character_id: str, successes: int, failures: int
    ) -> bool:
        """Update death save counts."""
        db = await self._get_db()

        await db.execute(
            """
            UPDATE character
            SET death_save_successes = ?, death_save_failures = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (successes, failures, character_id),
        )
        await db.commit()
        return True

    async def update_spell_slot(
        self, character_id: str, slot_level: int, current: int
    ) -> bool:
        """Update a character's spell slot count for a given level.

        Clamps current to [0, slots_max] to prevent overfilling.
        """
        db = await self._get_db()

        # Clamp to valid range — never exceed maximum or go below 0
        current = max(0, current)
        await db.execute(
            """
            UPDATE character_spell_slots
            SET slots_current = MIN(?, slots_max)
            WHERE character_id = ? AND slot_level = ?
            """,
            (current, character_id, slot_level),
        )
        await db.commit()
        return True

    async def add_condition(
        self, character_id: str, condition: CharacterCondition
    ) -> bool:
        """Add a condition to a character."""
        db = await self._get_db()

        await db.execute(
            """
            INSERT INTO character_condition (id, character_id, condition_name, source, expires_round, expires_time, combat_id, stacks)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                condition.id,
                character_id,
                condition.condition.value,
                condition.source,
                condition.expires_round,
                condition.expires_time.isoformat() if condition.expires_time else None,
                condition.combat_id,
                condition.stacks,
            ),
        )
        await db.commit()
        return True

    async def remove_condition(self, character_id: str, condition_name: str) -> bool:
        """Remove a condition from a character."""
        db = await self._get_db()

        await db.execute(
            "DELETE FROM character_condition WHERE character_id = ? AND condition_name = ?",
            (character_id, condition_name),
        )
        await db.commit()
        return True

    async def update(self, character: Character) -> bool:
        """
        Update a character's full state in the database.

        This updates all mutable fields including HP, conditions, spell slots, etc.
        Used after combat to persist state changes.
        """
        db = await self._get_db()

        # Update main character record
        await db.execute(
            """
            UPDATE character SET
                hp_current = ?,
                hp_temp = ?,
                hit_dice_remaining = ?,
                death_save_successes = ?,
                death_save_failures = ?,
                concentration_spell_id = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                character.hp.current,
                character.hp.temporary,
                character.hit_dice.remaining,
                character.death_saves.successes,
                character.death_saves.failures,
                character.concentration_spell_id,
                character.id,
            ),
        )

        # Update spell slots
        for level in range(1, 10):
            current, maximum = character.spell_slots.get_slots(level)
            if maximum > 0:
                await db.execute(
                    """
                    UPDATE character_spell_slots
                    SET slots_current = ?
                    WHERE character_id = ? AND slot_level = ?
                    """,
                    (current, character.id, level),
                )

        # Sync conditions - delete all and re-insert
        await db.execute(
            "DELETE FROM character_condition WHERE character_id = ?",
            (character.id,),
        )
        for condition in character.conditions:
            await db.execute(
                """
                INSERT INTO character_condition (id, character_id, condition_name, source, expires_round, expires_time, combat_id, stacks)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    condition.id,
                    character.id,
                    condition.condition.value,
                    condition.source,
                    condition.expires_round,
                    condition.expires_time.isoformat() if condition.expires_time else None,
                    condition.combat_id,
                    condition.stacks,
                ),
            )

        # Sync prepared spells
        await db.execute(
            "UPDATE character_spell SET is_prepared = 0 WHERE character_id = ?",
            (character.id,),
        )
        for spell_index in character.prepared_spells:
            # Insert or update
            existing = await db.fetch_one(
                "SELECT 1 FROM character_spell WHERE character_id = ? AND spell_index = ?",
                (character.id, spell_index),
            )
            if existing:
                await db.execute(
                    "UPDATE character_spell SET is_prepared = 1 WHERE character_id = ? AND spell_index = ?",
                    (character.id, spell_index),
                )
            else:
                import uuid
                await db.execute(
                    "INSERT INTO character_spell (id, character_id, spell_index, is_prepared) VALUES (?, ?, ?, 1)",
                    (str(uuid.uuid4()), character.id, spell_index),
                )

        await db.commit()
        return True

    async def delete(self, character_id: str) -> bool:
        """Soft delete a character."""
        db = await self._get_db()

        await db.execute(
            "UPDATE character SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (character_id,),
        )
        await db.commit()
        return True

    async def _row_to_character(self, db: Database, row) -> Character:
        """Convert a database row to a Character model."""
        character_id = row[0]

        # Load proficiencies
        prof_rows = await db.fetch_all(
            "SELECT proficiency_index, proficiency_type, expertise FROM character_proficiency WHERE character_id = ?",
            (character_id,),
        )

        skill_proficiencies = []
        skill_expertise = []
        saving_throw_proficiencies = []

        skill_map = {f"skill-{s.value}": s for s in Skill}
        ability_map = {
            "saving-throw-str": AbilityScore.STRENGTH,
            "saving-throw-dex": AbilityScore.DEXTERITY,
            "saving-throw-con": AbilityScore.CONSTITUTION,
            "saving-throw-int": AbilityScore.INTELLIGENCE,
            "saving-throw-wis": AbilityScore.WISDOM,
            "saving-throw-cha": AbilityScore.CHARISMA,
        }

        for prof_row in prof_rows:
            prof_index, prof_type, expertise = prof_row
            if prof_type == "skill" and prof_index in skill_map:
                skill_proficiencies.append(skill_map[prof_index])
                if expertise:
                    skill_expertise.append(skill_map[prof_index])
            elif prof_type == "save" and prof_index in ability_map:
                saving_throw_proficiencies.append(ability_map[prof_index])

        # Load spell slots
        slot_rows = await db.fetch_all(
            "SELECT slot_level, slots_max, slots_current FROM character_spell_slots WHERE character_id = ?",
            (character_id,),
        )

        spell_slots = SpellSlots()
        for slot_row in slot_rows:
            level, max_slots, current_slots = slot_row
            setattr(spell_slots, f"level_{level}", (current_slots, max_slots))

        # Load conditions
        cond_rows = await db.fetch_all(
            "SELECT id, condition_name, source, expires_round, expires_time, combat_id, stacks FROM character_condition WHERE character_id = ?",
            (character_id,),
        )

        conditions = []
        for cond_row in cond_rows:
            cond_id, cond_name, source, expires_round, expires_time, combat_id, stacks = cond_row
            try:
                conditions.append(
                    CharacterCondition(
                        id=cond_id,
                        condition=Condition(cond_name),
                        source=source or "",
                        expires_round=expires_round,
                        combat_id=combat_id,
                        stacks=stacks,
                    )
                )
            except ValueError:
                pass  # Invalid condition name

        # Load spells
        spell_rows = await db.fetch_all(
            "SELECT spell_index, is_prepared FROM character_spell WHERE character_id = ?",
            (character_id,),
        )
        known_spells = [r[0] for r in spell_rows]
        prepared_spells = [r[0] for r in spell_rows if r[1]]

        # Parse spellcasting ability
        spellcasting_ability = None
        if row[27]:  # spellcasting_ability column
            ability_map = {
                "str": AbilityScore.STRENGTH,
                "dex": AbilityScore.DEXTERITY,
                "con": AbilityScore.CONSTITUTION,
                "int": AbilityScore.INTELLIGENCE,
                "wis": AbilityScore.WISDOM,
                "cha": AbilityScore.CHARISMA,
            }
            spellcasting_ability = ability_map.get(row[27])

        return Character(
            id=row[0],
            discord_user_id=row[1],
            campaign_id=row[2],
            name=row[3],
            race_index=row[4],
            class_index=row[5],
            subclass_index=row[6],
            level=row[7],
            experience=row[8],
            background_index=row[9],
            abilities=AbilityScores(
                strength=row[10],
                dexterity=row[11],
                constitution=row[12],
                intelligence=row[13],
                wisdom=row[14],
                charisma=row[15],
            ),
            armor_class=row[16],
            speed=row[17],
            initiative_bonus=row[18],
            hp=HitPoints(maximum=row[19], current=row[20], temporary=row[21]),
            hit_dice=HitDice(die_type=row[22], total=row[23], remaining=row[24]),
            death_saves=DeathSaves(successes=row[25], failures=row[26]),
            spellcasting_ability=spellcasting_ability,
            concentration_spell_id=row[28],
            spell_slots=spell_slots,
            known_spells=known_spells,
            prepared_spells=prepared_spells,
            saving_throw_proficiencies=saving_throw_proficiencies,
            skill_proficiencies=skill_proficiencies,
            skill_expertise=skill_expertise,
            conditions=conditions,
            is_active=bool(row[29]),
        )


# Global repository instance
_repo: Optional[CharacterRepository] = None


async def get_character_repo() -> CharacterRepository:
    """Get the global character repository."""
    global _repo
    if _repo is None:
        _repo = CharacterRepository()
    return _repo
