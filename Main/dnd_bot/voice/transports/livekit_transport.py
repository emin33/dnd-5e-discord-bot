"""LiveKit WebRTC transport for the D&D voice frontend.

Uses LiveKit Agents framework with NVIDIA Riva STT/TTS.
Instead of a generic LLM, routes player speech through the D&D game engine.

Architecture:
    Browser (WebRTC) <-> LiveKit Room <-> This Agent
                                          |-- Riva STT: audio -> text
                                          |-- Game Engine: text -> DMResponse
                                          |-- VoiceFrontend: events -> speech
                                          |-- Riva TTS: text -> audio
                                          `-> LiveKit: audio -> Browser

Usage:
    python -m dnd_bot.voice.transports.livekit_transport dev

Requires:
    - LiveKit server running (livekit-server --dev)
    - Riva ASR NIM on localhost:50051
    - Riva TTS NIM on localhost:50052
    - Game engine configured (profiles.yaml, database, etc.)
"""

from __future__ import annotations

import os
import asyncio
from typing import Optional

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import nvidia, silero
import structlog

from ..tts import RivaTTS
from ..frontend import VoiceFrontend
from ...game.session import get_session_manager, GameSession, SessionState
from ...llm.orchestrator import get_orchestrator

load_dotenv()

logger = structlog.get_logger()

# Config from environment
RIVA_ASR_URL = os.getenv("RIVA_ASR_URL", "localhost:50051")
RIVA_TTS_URL = os.getenv("RIVA_TTS_URL", "localhost:50052")
TTS_VOICE = os.getenv("TTS_VOICE", "Magpie-Multilingual.EN-US.Aria")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")


class HQRivaTTS(nvidia.TTS):
    """Riva TTS at 44.1kHz instead of plugin default 16kHz."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._opts.sample_rate = 44100


class DnDGameLLM:
    """Bridges the D&D game engine into LiveKit's LLM interface.

    Instead of calling an LLM API, feeds player speech through
    GameSessionManager.process_message() and streams the narrative
    response back as if it were LLM tokens.

    This is a simplified adapter for v1. It handles exploration mode
    well. Combat mode uses the VoiceFrontend's get_combat_action()
    which operates outside this LLM interface.
    """

    def __init__(
        self,
        session_key: str,
        user_id: int,
        user_name: str,
        voice_frontend: VoiceFrontend,
    ):
        self._session_key = session_key
        self._user_id = user_id
        self._user_name = user_name
        self._frontend = voice_frontend
        self._session_manager = get_session_manager()

    async def process_player_speech(self, transcription: str) -> Optional[str]:
        """Process a player's speech through the game engine.

        Returns the narrative response text (or None if no response).
        The VoiceFrontend handles mechanics/combat events via on_event().
        """
        session = self._session_manager.get_session_by_key(self._session_key)
        if not session:
            return "No active game session. Ask the DM to start a game."

        # During combat, route to combat action resolution
        if session.state == SessionState.COMBAT:
            player = session.get_player(self._user_id)
            player_name = player.character.name if player and player.character else self._user_name

            # Get the current turn context from coordinator
            from ...game.combat.coordinator import get_coordinator_by_key
            coordinator = get_coordinator_by_key(self._session_key)
            if coordinator:
                current = coordinator.manager.combat.get_current_combatant()
                if current and current.is_player:
                    turn_ctx = await coordinator.start_turn(current)
                    await self._frontend.resolve_voice_combat(
                        transcription, turn_ctx, player_name
                    )
                    return None  # Combat loop handles response

        # Exploration mode: process through normal pipeline
        response = await self._session_manager.process_message(
            channel_id=0,  # Not used when session_key lookup works
            user_id=self._user_id,
            user_name=self._user_name,
            content=transcription,
            frontend=self._frontend,
        )

        if response:
            return response.narrative
        return None


class DnDVoiceAgent(Agent):
    """LiveKit Agent that runs a D&D game session via voice."""

    def __init__(self):
        super().__init__(
            instructions=(
                "You are a Dungeon Master running a D&D 5th Edition game. "
                "Players speak their actions and you narrate the results. "
                "Keep narration dramatic but concise for voice delivery. "
                "Do not use markdown or formatting - everything is spoken aloud."
            )
        )


async def entrypoint(ctx: agents.JobContext):
    """LiveKit agent entrypoint - called when a participant joins the room."""
    await ctx.connect()

    # Extract session info from room metadata or use defaults
    room_name = ctx.room.name or "default"
    session_key = f"voice:{room_name}"

    logger.info(
        "voice_agent_connected",
        room=room_name,
        session_key=session_key,
    )

    # Initialize Riva TTS for the voice frontend
    tts = RivaTTS(server_url=RIVA_TTS_URL, voice=TTS_VOICE)

    # Create the LiveKit AgentSession with Riva STT/TTS
    session = AgentSession(
        stt=nvidia.STT(
            server=RIVA_ASR_URL,
            use_ssl=False,
            language_code="en-US",
            model="",
        ),
        tts=HQRivaTTS(
            server=RIVA_TTS_URL,
            use_ssl=False,
            voice=TTS_VOICE,
            language_code="en-US",
        ),
        vad=silero.VAD.load(),
        turn_handling={
            "interruption": {"mode": "vad"},
        },
    )

    # The speak function uses the LiveKit session's TTS
    async def speak(text: str) -> None:
        """Speak text through the LiveKit session's TTS pipeline."""
        await session.say(text)

    # Create voice frontend
    orchestrator = get_orchestrator()
    voice_frontend = VoiceFrontend(
        tts=tts,
        speak_fn=speak,
        orchestrator=orchestrator,
    )

    # Start the agent
    await session.start(
        room=ctx.room,
        agent=DnDVoiceAgent(),
    )

    # Greet the player
    await session.say(
        "Welcome, adventurer! A game session will begin when "
        "the Dungeon Master starts one. You can speak your actions "
        "and I will narrate the results."
    )

    logger.info("voice_agent_ready", session_key=session_key)


def main():
    """Run the LiveKit voice agent."""
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint),
    )


if __name__ == "__main__":
    main()
