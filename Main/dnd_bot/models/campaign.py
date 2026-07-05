"""Campaign data models."""

from datetime import datetime
from typing import Optional
import uuid

from pydantic import BaseModel, Field

from .common import CampaignId, GuildId, UserId


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
