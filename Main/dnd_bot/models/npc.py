"""NPC and Scene Entity data models."""

from datetime import datetime
from enum import Enum
from typing import Optional
import uuid

from pydantic import BaseModel, Field

from .common import CampaignId, CharacterId


class Disposition(str, Enum):
    """NPC disposition toward the party."""
    HOSTILE = "hostile"       # Will attack on sight
    UNFRIENDLY = "unfriendly" # Dislikes party, may become hostile
    NEUTRAL = "neutral"       # Indifferent
    FRIENDLY = "friendly"     # Likes party
    ALLIED = "allied"         # Will fight alongside party


class EntityType(str, Enum):
    """Types of scene entities."""
    NPC = "npc"               # Named character
    CREATURE = "creature"     # Monster/beast (unnamed or named)
    OBJECT = "object"         # Interactable object
    ENVIRONMENTAL = "environmental"  # Environmental feature


class NPC(BaseModel):
    """
    A persistent NPC in the campaign.

    Matches the existing database schema in npc table.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    campaign_id: CampaignId
    name: str

    # Description and identity
    description: str = ""
    location: Optional[str] = None
    monster_index: Optional[str] = None  # SRD monster reference for combat stats

    # Disposition baseline (-100 to 100, stored as int in DB)
    base_disposition: Disposition = Disposition.NEUTRAL

    # Roleplay notes
    voice_notes: Optional[str] = None

    # Status
    is_alive: bool = True

    # Tracking
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_seen_at: Optional[datetime] = None


class NPCRelationship(BaseModel):
    """
    Relationship between an NPC and a player character.

    Matches the existing database schema in npc_relationship table.
    """

    npc_id: str
    character_id: CharacterId

    # Sentiment: -100 (hates) to +100 (loves)
    sentiment: int = Field(default=0, ge=-100, le=100)

    # Interaction tracking
    interaction_count: int = 0
    last_interaction: Optional[datetime] = None

    # Notes about relationship
    notes: str = ""


class SceneEntity(BaseModel):
    """
    An entity currently present in the scene (transient, in-memory).

    This tracks NPCs, creatures, and objects that the narrator has
    introduced in the current scene. It includes hostility tracking
    for potential combat triggers.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Identity
    name: str
    entity_type: EntityType
    description: str = ""
    aliases: list[str] = Field(default_factory=list)  # Alternate names ("the horse", "mare")

    # Reference to persistent data (if applicable)
    npc_id: Optional[str] = None      # Links to NPC model if persisted
    monster_index: Optional[str] = None  # SRD monster reference

    # Current state in scene
    disposition: Disposition = Disposition.NEUTRAL
    hostility_score: int = Field(default=0, ge=0, le=100)
    # Hostility thresholds:
    # 0-25: Calm
    # 26-50: Agitated
    # 51-75: Threatening
    # 76-84: Hostile
    # 85-100: Combat (auto-trigger threshold)

    # Combat-ready stats (if known or estimated)
    hp_estimate: Optional[int] = None
    ac_estimate: Optional[int] = None

    # Tracking
    introduced_at: datetime = Field(default_factory=datetime.utcnow)
    last_mentioned_at: datetime = Field(default_factory=datetime.utcnow)
    mention_count: int = 1

    # Actions that escalated hostility
    hostility_events: list[str] = Field(default_factory=list)

    def is_combat_ready(self) -> bool:
        """Check if entity has crossed combat threshold."""
        return self.hostility_score >= 85

    def get_hostility_status(self) -> str:
        """Get human-readable hostility status."""
        if self.hostility_score >= 85:
            return "COMBAT"
        elif self.hostility_score >= 76:
            return "HOSTILE"
        elif self.hostility_score >= 51:
            return "THREATENING"
        elif self.hostility_score >= 26:
            return "AGITATED"
        else:
            return "CALM"


class HostilityEvent(BaseModel):
    """An event that changed hostility."""

    entity_id: str
    delta: int  # Change in hostility (-30 to +30 typically)
    reason: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Hostility thresholds as constants
HOSTILITY_CALM = 25
HOSTILITY_AGITATED = 50
HOSTILITY_THREATENING = 75
HOSTILITY_HOSTILE = 84
HOSTILITY_COMBAT = 85  # Auto-combat trigger threshold


# Hostility delta guidelines
HOSTILITY_DELTAS = {
    # Escalation events
    "insult": 5,
    "threat": 10,
    "provocation": 10,
    "weapon_drawn": 15,
    "aggressive_action": 15,
    "attack": 30,  # Immediate combat

    # De-escalation events
    "apology": -5,
    "backing_away": -10,
    "diplomacy": -10,
    "gift": -10,
    "successful_persuasion": -15,
    "successful_intimidation": -15,
}
