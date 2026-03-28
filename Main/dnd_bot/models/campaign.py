"""Campaign and game session data models."""

from datetime import datetime
from typing import Optional
import uuid

from pydantic import BaseModel, Field

from .common import CampaignId, ChannelId, GameState, GuildId, UserId


class Campaign(BaseModel):
    """A D&D campaign."""

    id: CampaignId = Field(default_factory=lambda: str(uuid.uuid4()))
    guild_id: GuildId
    name: str = Field(min_length=1, max_length=100)
    description: Optional[str] = None

    # The DM (bot owner for this campaign)
    dm_user_id: UserId

    # World setting for narrative context
    world_setting: str = Field(
        default="A classic high fantasy world filled with magic, monsters, and adventure."
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_played_at: Optional[datetime] = None


class GameSession(BaseModel):
    """An active game session within a campaign."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    campaign_id: CampaignId
    channel_id: ChannelId
    session_number: int = Field(default=1, ge=1)
    state: GameState = Field(default=GameState.LOBBY)

    # Active combat (if any)
    active_combat_id: Optional[str] = None

    # Metadata
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None

    @property
    def is_active(self) -> bool:
        """Check if session is still active."""
        return self.ended_at is None and self.state != GameState.PAUSED

    @property
    def in_combat(self) -> bool:
        """Check if session is in combat."""
        return self.state == GameState.COMBAT and self.active_combat_id is not None


class SessionSnapshot(BaseModel):
    """A saved snapshot of game state for rollback."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    snapshot_type: str = "manual"  # 'auto', 'manual', 'pre_combat'
    game_state: dict  # Full serialized state
    created_at: datetime = Field(default_factory=datetime.utcnow)
