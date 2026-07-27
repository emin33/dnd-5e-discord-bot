# D&D 5e Discord Bot

An LLM-powered Dungeon Master bot for Discord. Uses a local Ollama model to narrate adventures, adjudicate rules, and run combat encounters based on the D&D 5e SRD.

## Features

- **Campaign Management** - Create and join campaigns with a lobby system
- **Character Creation** - Guided character creation with SRD races, classes, and starting equipment
- **Combat System** - Turn-based combat with initiative, zones, conditions, and NPC AI
- **Spellcasting** - Spell slots, spell effects, and concentration tracking
- **Inventory** - Item management, equipment, and currency
- **Resting** - Short and long rest mechanics with hit dice and resource recovery
- **LLM Dungeon Master** - Narrates scenes, interprets player actions, and enforces rules via structured LLM calls
- **Campaign Memory** - Vector-store-backed memory so the DM remembers past events

## Architecture

```
Main/
  dnd_bot/
    bot/          # Discord interface (cogs, embeds, views)
    data/         # Database layer (SQLite + repositories)
    game/         # Game logic (combat, character, mechanics, scenes)
    llm/          # LLM integration (brains, extractors, orchestrator)
    memory/       # Campaign memory (vector store, context blocks)
    models/       # Pydantic data models
  migrations/     # SQL schema migrations
  tests/          # Unit and integration tests
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) running locally with `qwen3:32b`
- A Discord bot token
- The [5e-database](https://github.com/5e-bits/5e-database) SRD data (cloned alongside this repo)

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/emin33/dnd-5e-discord-bot.git
   cd dnd-5e-discord-bot/Main
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your Discord bot token and settings
   ```

5. **Pull the Ollama model**
   ```bash
   ollama pull qwen3:32b
   ```

6. **Run the bot**
   ```bash
   python -m dnd_bot.main
   ```

## Configuration

See `Main/.env.example` for the complete, annotated list of settings. Every
key there maps 1:1 to a field on `dnd_bot.config.Settings`, and unknown keys
in `.env` are rejected at boot, so don't add keys that aren't in that file.
`DISCORD_BOT_TOKEN` is the only required key. LLM provider/model selection
lives in `Main/config/profiles.yaml`, chosen via `ACTIVE_PROFILE`.

## Development

```bash
pip install -e ".[dev]"
```

Run the suite with the repo's own runner, from `Main/` (where Setup left you):

```
./run_tests.bat                                  # the whole suite
./run_tests.bat tests/unit/test_world_state.py   # one file
./run_typecheck.bat                              # mypy gate, must exit 0
```

From the repo root instead, prefix them with `Main/`. Under `cmd.exe`, spell the
separators as backslashes.

Do not run a bare `pytest`. This project needs Python >= 3.11 and py-cord, and an
interpreter with neither does not fail loudly -- it reports green while silently
collecting four tests fewer, the Discord cogs among them. `tests/conftest.py`
refuses such an environment outright rather than skipping past it, and
`run_tests.bat`, `run_typecheck.bat`, `run_longform.bat` and `run_reliability.bat`
all use `Main/venv`.

## License

This project uses the D&D 5e SRD under the [Open Gaming License](https://dnd.wizards.com/resources/systems-reference-document).
