"""Entry point for the D&D 5e Voice Game.

Starts the FastAPI web server (game management + web UI) and
optionally the LiveKit voice agent worker.

Usage:
    # Start the web server (game management + UI):
    python run_voice.py

    # Start with a specific port:
    python run_voice.py --port 8080

    # Start the LiveKit voice agent worker (separate terminal):
    python run_voice.py --agent

Prerequisites:
    1. Riva ASR + TTS containers running:
       docker compose -f docker-compose.voice.yml up -d

    2. LiveKit server running:
       "C:\\Projects\\Voice Agent\\livekit-server.exe" --dev

    3. Environment configured (.env):
       ACTIVE_PROFILE=voice_gemma
       RIVA_ASR_URL=localhost:50051
       RIVA_TTS_URL=localhost:50052
       LIVEKIT_URL=ws://localhost:7880
       LIVEKIT_API_KEY=devkey
       LIVEKIT_API_SECRET=secret
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load env before anything else
from dotenv import load_dotenv
load_dotenv()


def run_server(port: int = 8000):
    """Run the FastAPI web server."""
    import uvicorn
    from dnd_bot.voice.api import app

    print(f"\n  D&D 5e Voice Game Server")
    print(f"  Open http://localhost:{port} in your browser")
    print(f"  API docs at http://localhost:{port}/docs\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def run_agent():
    """Run the LiveKit voice agent worker."""
    # LiveKit CLI parses sys.argv itself, so we need to replace
    # our --agent flag with 'dev' which is what LiveKit expects
    sys.argv = [sys.argv[0], "dev"]
    from dnd_bot.voice.transports.livekit_transport import main
    main()


if __name__ == "__main__":
    if "--agent" in sys.argv:
        run_agent()
    else:
        port = 8000
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        run_server(port)
