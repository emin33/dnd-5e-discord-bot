"""Tests for voice-catalog seeding in ImmersionRepository (fix C3).

Pins three behaviors:
- A mid-seed failure does NOT permanently disable seeding (_catalog_seeded
  is only set after a successful pass).
- Seeding uses upsert semantics: edits to existing JSON entries propagate
  to existing DB rows (no count-based gate, no INSERT OR IGNORE staleness).
- seed_voice_catalog returns the number of rows actually inserted/updated,
  not the JSON length.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dnd_bot.data.database import Database
from dnd_bot.data.repositories.immersion_repo import ImmersionRepository

CATALOG = [
    {
        "voice_id": "v1",
        "name": "Alice",
        "provider": "kokoro",
        "gender": "female",
        "age": "young",
        "style_tags": ["warm"],
    },
    {
        "voice_id": "v2",
        "name": "Bob",
        "provider": "kokoro",
        "gender": "male",
        "age": "mature",
        "style_tags": [],
    },
]


@pytest.fixture
async def db(tmp_path: Path):
    db = Database(db_path=tmp_path / "test.db")
    await db.connect()
    yield db
    await db.disconnect()


@pytest.fixture
def catalog_file(tmp_path: Path) -> Path:
    path = tmp_path / "voice_catalog.json"
    path.write_text(json.dumps(CATALOG))
    return path


@pytest.fixture(autouse=True)
def reset_seeded_flag():
    ImmersionRepository._catalog_seeded = False
    yield
    ImmersionRepository._catalog_seeded = False


async def test_seed_returns_real_change_count(db: Database, catalog_file: Path) -> None:
    repo = ImmersionRepository(db=db)
    assert await repo.seed_voice_catalog(catalog_file) == 2
    # Second pass with identical JSON: nothing changes.
    assert await repo.seed_voice_catalog(catalog_file) == 0


async def test_json_edits_propagate_to_existing_rows(
    db: Database, catalog_file: Path
) -> None:
    repo = ImmersionRepository(db=db)
    await repo.seed_voice_catalog(catalog_file)

    edited = [dict(CATALOG[0], name="Alicia", provider="elevenlabs"), CATALOG[1]]
    catalog_file.write_text(json.dumps(edited))

    changed = await repo.seed_voice_catalog(catalog_file)
    assert changed == 1

    voice = await repo.get_voice_by_id("v1")
    assert voice is not None
    assert voice.name == "Alicia"
    assert voice.provider == "elevenlabs"


async def test_mid_seed_failure_does_not_disable_seeding(
    db: Database, catalog_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = ImmersionRepository(db=db)

    async def boom(path: Path) -> int:
        raise RuntimeError("seed failed")

    monkeypatch.setattr(repo, "seed_voice_catalog", boom)
    monkeypatch.setattr(
        "dnd_bot.data.repositories.immersion_repo.Path.exists", lambda self: True
    )
    with pytest.raises(RuntimeError):
        await repo._ensure_catalog_seeded()

    # Flag must NOT be latched by the failed attempt.
    assert ImmersionRepository._catalog_seeded is False

    # A later successful pass latches the flag.
    async def ok(path: Path) -> int:
        return await ImmersionRepository.seed_voice_catalog(repo, catalog_file)

    monkeypatch.setattr(repo, "seed_voice_catalog", ok)
    await repo._ensure_catalog_seeded()
    assert ImmersionRepository._catalog_seeded is True
    assert await repo.get_voice_by_id("v1") is not None
