"""LiveKit WebRTC transport for the D&D voice frontend.

Uses LiveKit Agents framework with NVIDIA Riva STT/TTS.

Architecture:
    Browser (WebRTC) <-> LiveKit Room <-> This Agent
                                          |-- Riva STT: audio -> text
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

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent
from livekit.plugins import nvidia, silero
import structlog

from ..frontend import VoiceFrontend

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
    voice_frontend = VoiceFrontend(speak_fn=speak)

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
