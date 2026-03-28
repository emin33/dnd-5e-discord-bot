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

See `.env.example` for all available settings:

| Variable | Description | Default |
|---|---|---|
| `DISCORD_BOT_TOKEN` | Your Discord bot token | (required) |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Model to use | `qwen3:32b` |
| `NARRATOR_TEMPERATURE` | Creativity for narration | `0.75` |
| `RULES_TEMPERATURE` | Strictness for rules calls | `0.1` |
| `DATABASE_PATH` | SQLite database location | `data/dnd_bot.db` |
| `SRD_DATA_PATH` | Path to 5e-database SRD JSON | `../5e-database/src/2014` |

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

This project uses the D&D 5e SRD under the [Open Gaming License](https://dnd.wizards.com/resources/systems-reference-document).
