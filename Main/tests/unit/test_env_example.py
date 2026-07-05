"""Guards that keep Main/.env.example in lockstep with dnd_bot.config.Settings.

A stale key in .env.example crashes the bot at boot: Settings uses
extra='forbid', so any key in a copied .env that is not a Settings field
raises ValidationError (this happened with OLLAMA_MODEL; see
AUDIT_QUALITY_2026_06_09.md, Configuration P0). These tests fail instead.

Conventions assumed for .env.example:
- Active (uncommented) lines are required secrets:  KEY=placeholder
- Commented option lines show the default:          # KEY=default
- Prose comments never look like "# UPPER_SNAKE_KEY=..."
"""

import os
import re
from pathlib import Path

import pytest

from dnd_bot.config import Settings

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"

# "KEY=value" (an active line). Keys are UPPER_SNAKE by convention.
_ACTIVE_RE = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")
# "# KEY=value" or "#KEY=value" (a commented-out option line).
_COMMENTED_RE = re.compile(r"^#\s*([A-Z][A-Z0-9_]*)=(.*)$")


def _parse_example():
    """Return (active, commented) lists of (key, value) from .env.example."""
    active = []
    commented = []
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _ACTIVE_RE.match(line)
        if m:
            active.append((m.group(1), m.group(2)))
            continue
        m = _COMMENTED_RE.match(line)
        if m:
            commented.append((m.group(1), m.group(2)))
    return active, commented


@pytest.fixture
def isolated_settings_env(monkeypatch):
    """Strip Settings-related vars from the OS env so only the tmp .env counts.

    pydantic-settings reads OS env vars at higher priority than the dotenv
    file; without this, a developer's real DISCORD_BOT_TOKEN etc. would leak
    into the test. monkeypatch restores everything afterwards.
    """
    field_names = {name.lower() for name in Settings.model_fields}
    for key in list(os.environ):
        if key.lower() in field_names:
            monkeypatch.delenv(key, raising=False)


def test_env_example_boots_settings(tmp_path, isolated_settings_env):
    """A .env built from .env.example (all keys enabled) must construct Settings.

    Every commented option is uncommented with its documented default and
    every required secret gets a dummy value, so a single stale key in the
    example fails here (extra='forbid') instead of at bot startup.
    """
    active, commented = _parse_example()
    assert active, ".env.example has no uncommented required keys - parser broken?"
    assert commented, ".env.example has no commented option lines - parser broken?"

    lines = [f"{key}=dummy-{key.lower()}" for key, _ in active]
    lines += [f"{key}={value}" for key, value in commented]
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)  # must not raise

    # Proves the tmp file (not a default) supplied the required secret,
    # and that a documented default round-trips.
    assert settings.discord_bot_token == "dummy-discord_bot_token"
    assert settings.active_profile == "production"


def test_settings_forbids_unknown_keys():
    """Pin the mechanism these guards rely on: Settings rejects extras.

    If extra='forbid' is ever relaxed, stale .env keys stop failing at boot
    AND the lockstep tests above stop proving anything — fail here instead.
    """
    assert Settings.model_config.get("extra") == "forbid"


def test_env_example_keys_match_settings_fields():
    """Every key in .env.example is a real Settings field, and vice versa."""
    active, commented = _parse_example()
    field_names = set(Settings.model_fields)

    unknown_active = sorted(k for k, _ in active if k.lower() not in field_names)
    assert not unknown_active, (
        f"Uncommented keys in .env.example with no Settings field: {unknown_active}"
    )

    unknown_commented = sorted(k for k, _ in commented if k.lower() not in field_names)
    assert not unknown_commented, (
        f"Commented keys in .env.example with no Settings field: {unknown_commented}"
    )

    example_keys = {k.lower() for k, _ in active} | {k.lower() for k, _ in commented}
    missing = sorted(field_names - example_keys)
    assert not missing, f"Settings fields missing from .env.example: {missing}"
