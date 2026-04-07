"""Entry point for the D&D 5e Voice Agent.

Starts the LiveKit voice agent worker that connects to a LiveKit room
and runs D&D sessions via voice (Riva ASR/TTS).

Prerequisites:
    1. Riva ASR + TTS containers running:
       docker compose -f docker-compose.voice.yml up -d

    2. LiveKit server running:
       livekit-server --dev

    3. Environment configured (.env):
       ACTIVE_PROFILE=voice_production  (or voice_hybrid)
       RIVA_ASR_URL=localhost:50051
       RIVA_TTS_URL=localhost:50052
       LIVEKIT_URL=ws://localhost:7880
       LIVEKIT_API_KEY=devkey
       LIVEKIT_API_SECRET=secret
       TTS_VOICE=Magpie-Multilingual.EN-US.Aria

Usage:
    # Download VAD models (first run only):
    python run_voice.py download-files

    # Start the voice agent:
    python run_voice.py dev

    # Connect from browser:
    #   Open web/index.html
    #   Generate token: livekit-cli create-token \\
    #       --api-key devkey --api-secret secret \\
    #       --join --room dnd-session --identity player1 \\
    #       --valid-for 24h
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dnd_bot.voice.transports.livekit_transport import main

if __name__ == "__main__":
    main()
