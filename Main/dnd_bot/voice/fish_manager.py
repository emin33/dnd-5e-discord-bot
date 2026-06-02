"""Fish Speech server manager: auto-starts and manages local Fish Speech instances.

On bot startup, if the active profile uses Fish Speech for TTS,
this manager ensures the configured number of instances are running.
Each instance runs in its own terminal window on consecutive ports.
"""

import subprocess
import time
from pathlib import Path
from typing import Optional

import requests
import structlog

logger = structlog.get_logger()

# Path to the Fish Speech S1-mini server codebase
# __file__ is dnd_bot/voice/fish_manager.py -> parent.parent = dnd_bot -> parent = Main -> parent = D&D 5e
_MAIN_DIR = Path(__file__).resolve().parent.parent.parent  # Main/
_PROJECT_DIR = _MAIN_DIR.parent  # D&D 5e/
_FISH_REPO = _PROJECT_DIR / "fish-speech-s1-mini"
_CHECKPOINT_DIR = _MAIN_DIR / "fish_checkpoints" / "openaudio-s1-mini"


def _health_check(port: int, timeout: float = 2.0) -> bool:
    """Check if a Fish Speech server is responding on the given port."""
    try:
        r = requests.get(f"http://localhost:{port}/", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _start_instance(port: int, device: str = "cuda") -> Optional[subprocess.Popen]:
    """Start a Fish Speech server in a new terminal window."""
    if not _FISH_REPO.exists():
        logger.warning("fish_repo_not_found", path=str(_FISH_REPO))
        return None

    llama_path = str(_CHECKPOINT_DIR)
    codec_path = str(_CHECKPOINT_DIR / "codec.pth")

    if not Path(codec_path).exists():
        logger.warning("fish_checkpoint_not_found", path=codec_path)
        return None

    logger.info("starting_fish_instance", port=port, repo=str(_FISH_REPO))

    # Write a temporary batch file to avoid shell escaping issues with & in paths
    import tempfile
    bat_content = (
        f'@echo off\n'
        f'title Fish Speech (port {port})\n'
        f'cd /d "{_FISH_REPO}"\n'
        f'python tools/api_server.py '
        f'--llama-checkpoint-path "{llama_path}" '
        f'--decoder-checkpoint-path "{codec_path}" '
        f'--device {device} --half '
        f'--listen 0.0.0.0:{port}\n'
        f'pause\n'
    )
    bat_path = Path(tempfile.gettempdir()) / f"fish_speech_{port}.bat"
    bat_path.write_text(bat_content)

    proc = subprocess.Popen(
        ["cmd", "/c", "start", "", str(bat_path)],
    )

    return proc


def ensure_fish_servers(
    base_port: int = 8080,
    num_instances: int = 1,
    device: str = "cuda",
    startup_timeout: float = 60.0,
) -> int:
    """Ensure the configured number of Fish Speech instances are running.

    Checks each port, starts any that aren't responding, waits for them
    to become healthy.

    Returns the number of healthy instances.
    """
    healthy = 0
    started = []

    for i in range(num_instances):
        port = base_port + i

        if _health_check(port):
            logger.info("fish_instance_already_running", port=port)
            healthy += 1
        else:
            proc = _start_instance(port, device)
            if proc:
                started.append(port)

    # Wait for started instances to become healthy
    if started:
        logger.info("waiting_for_fish_instances", ports=started)
        deadline = time.monotonic() + startup_timeout

        while started and time.monotonic() < deadline:
            time.sleep(3)
            still_waiting = []
            for port in started:
                if _health_check(port):
                    logger.info("fish_instance_ready", port=port)
                    healthy += 1
                else:
                    still_waiting.append(port)
            started = still_waiting

        if started:
            logger.warning("fish_instances_failed_to_start", ports=started)

    return healthy


def auto_start_fish_if_needed():
    """Check the active profile and auto-start Fish Speech servers if needed.

    Call this on bot startup.
    """
    try:
        from ..config import get_profile, get_settings

        profile = get_profile()
        settings = get_settings()

        # Check if any immersion provider uses Fish
        uses_fish = (
            profile.immersion.narrator_tts_provider == "fish"
            or profile.immersion.character_tts_provider == "fish"
        )

        if not uses_fish:
            return

        base_port = 8080
        try:
            # Parse port from fish_speech_url
            url = settings.fish_speech_url
            if ":" in url.rsplit("/", 1)[-1]:
                base_port = int(url.rsplit(":", 1)[1])
        except Exception:
            pass

        num_instances = max(1, settings.fish_speech_instances)

        logger.info(
            "fish_auto_start_check",
            base_port=base_port,
            num_instances=num_instances,
        )

        healthy = ensure_fish_servers(
            base_port=base_port,
            num_instances=num_instances,
            startup_timeout=90.0,
        )

        logger.info("fish_servers_ready", healthy=healthy, requested=num_instances)

    except Exception as e:
        logger.warning("fish_auto_start_failed", error=str(e))
