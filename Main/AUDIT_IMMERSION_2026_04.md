# Immersion & Modularity Audit — April 2026

Audit of the immersion system, web/discord parity, provider modularity, and image generation.
Chip away at these across sessions. Mark items `[x]` when done.

---

## 1. Bugs

### ~~1.1 Emotion parameter crashes OpenAI/ElevenLabs TTS~~
- **FALSE POSITIVE** — already guarded at `tts_assembler.py:99`: `if segment.emotion and provider in ("fish", "inworld")`

### ~~1.2 image_fal.py double-timeout crash~~
- **FALSE POSITIVE** — `raise` on line 85 prevents reaching `result.get()` with None

### ~~1.3 tts_inworld.py subprocess result not checked~~
- **FALSE POSITIVE** — `result.returncode` already checked at line 101

### 1.4 Misplaced docstring in immersion_repo
- **File:** `dnd_bot/data/repositories/immersion_repo.py:101`
- [x] Fixed — moved docstring before await

---

## 2. Race Conditions

### 2.1 TTS cache double-create
- **File:** `dnd_bot/immersion/tts_assembler.py` (`_tts_cache`)
- [x] Fixed — added `_tts_cache_lock` with lazy init

### ~~2.2 Fish instance counter unprotected~~
- **FALSE POSITIVE** — `_fish_instance_counter` was dead code. Removed.

### 2.3 Image factory singleton double-create
- **File:** `dnd_bot/immersion/image_factory.py`
- [x] Fixed — added `threading.Lock` with double-checked locking pattern

### 2.4 Catalog seeding race across repo instances
- **File:** `dnd_bot/data/repositories/immersion_repo.py`
- [x] Fixed — promoted `_catalog_seeded` to class-level flag, switched to INSERT OR IGNORE, removed redundant per-row SELECT check

---

## 3. Code Quality / Cleanup

### 3.1 Unused imports / dead code
- [x] `asyncio` removed from `dnd_bot/voice/tts_browser.py`
- [x] `struct` removed from `dnd_bot/immersion/tts_assembler.py`
- [x] Dead `_fish_instance_counter` removed from `dnd_bot/immersion/tts_assembler.py`

### 3.2 Late imports in hot paths
- **No action needed** — by design for optional dependencies. Python caches after first call (`sys.modules` dict lookup). Missing packages only error when used, not on module import.

### 3.3 Silent exception swallowing
- [x] `dnd_bot/immersion/voice_resolver.py:82` — added structured logging with voice_id and error
- [x] `dnd_bot/immersion/image_factory.py:69` — added logging on profile load failure

### 3.4 Private variable imported externally
- [x] Added `get_provider_name()` to `image_factory.py`, updated `image_coordinator.py` import

### 3.5 Hardcoded Fish Speech port offsets
- **File:** `dnd_bot/immersion/tts_assembler.py:77-80`
- [x] Fixed — port derives from configurable `fish_speech_url`, not hardcoded. Fixed misleading config comment, added debug log on port offset.

### 3.6 Sample rate fallback
- [x] Fixed — replaced silent `getattr` fallback with explicit check + debug log

---

## 4. Web App / Discord Parity

The web app has **none** of the immersion features. Full feature gap:

| Feature | Discord | Web |
|---------|---------|-----|
| Multi-voice narration | Full pipeline | Single voice |
| Prose parsing / dialogue | 461-line system | Missing |
| Character voice profiles | Voice catalog + auto-assign | No |
| Emotion in TTS | Fish/Inworld | No |
| Scene images | Full pipeline + frequency | No endpoint, no UI |
| Voice resolution | Per-segment routing | Global provider |
| Immersion settings | /immersion commands + DB | No settings UI |

### 4.1 Backend: narration audio endpoint
- Add `/api/narration-audio` that runs prose_parser -> voice_resolver -> tts_assembler
- Returns compiled multi-voice MP3 (or streams segments)
- [ ] Implemented

### 4.2 Backend: image generation endpoint
- Add `/api/scene-image` that runs image_coordinator
- Returns image URL or base64
- [ ] Implemented

### 4.3 Backend: immersion settings endpoints
- CRUD for narrator voice, character voice provider, image frequency, image enabled
- Store per-session or per-user
- [ ] Implemented

### 4.4 Frontend: replace sentence-level TTS
- Replace `speakRiva()` single-voice flow with call to `/api/narration-audio`
- Play compiled MP3 in browser
- [ ] Implemented

### 4.5 Frontend: scene image display
- Add image container in UI
- Fetch and display scene images from `/api/scene-image`
- [ ] Implemented

### 4.6 Frontend: immersion settings UI
- Voice selection, image toggle, frequency selector
- [ ] Implemented

### 4.7 WebFrontend upgrade
- **File:** `dnd_bot/voice/api.py:545-570`
- Current `WebFrontend` is a thin event collector. Needs to process immersion events (trigger TTS assembly, image generation) like `DiscordTextFrontend` does.
- [ ] Implemented

---

## 5. Provider Modularity

### 5.1 [HIGH] JSON schema silently ignored on Anthropic/Gemini
- **File:** `dnd_bot/llm/client.py` (AnthropicClient, GeminiClient)
- [x] Fixed — AnthropicClient now injects JSON enforcement system hint when `json_schema`/`json_mode` set. GeminiClient now sets `response_mime_type="application/json"` in GenerationConfig.

### 5.2 [HIGH] Unused format constants
- **File:** `dnd_bot/llm/orchestrator.py:71-83`
- [x] Resolved — removed dead `NARRATOR_FORMAT_XML`/`NARRATOR_FORMAT_TEXT` constants. Replaced stale "xml"/"text" format_type logging with actual provider name.

### 5.3 [MED] Streaming is Ollama-only
- **File:** `dnd_bot/llm/orchestrator.py:2470`
- [x] Fixed — by design (local = low latency benefit, streaming excludes tools). Added debug log when streaming activates. Cloud providers reliably produce tools+prose in one shot, so streaming less valuable.

### 5.4 [MED] Two-pass tool calling hardcoded to Ollama
- **File:** `dnd_bot/llm/orchestrator.py:2270`
- [x] Fixed — removed `isinstance(OllamaClient)` guard. Tool followup now runs for all providers when tools weren't called, consistent with `_narrate_action()`.

### 5.5 [MED] Groq-only fallback strategy
- **File:** `dnd_bot/llm/client.py:587-621`
- [x] Acknowledged — intentional design. Groq is a cloud relay to local models, so fallback to local Ollama is uniquely sensible. Other cloud providers (Anthropic, Gemini) have their own rate limit handling. Generic retry is a larger architecture effort.

### 5.6 [MED] tool_choice="required" not truly supported on Ollama
- **File:** `dnd_bot/llm/brains/rules.py:546, 589, 618, 654`
- [x] Acknowledged — system message hint is the best available approach for Ollama. Two-pass tool followup (5.4 fix) serves as the safety net when the hint is ignored.

### 5.7 [LOW] String-based type check for Anthropic
- **File:** `dnd_bot/llm/orchestrator.py:86-88`
- [x] Fixed — replaced `_is_anthropic_client` with `_get_client_provider` using `isinstance()`. Now imports `AnthropicClient` directly.

### ~~5.8 [LOW] Qwen-specific think tag stripping runs on all providers~~
- **FALSE POSITIVE** — `_clean_llm_content()` is only called from OllamaClient and GroqClient. Already correctly scoped.

### 5.9 [LOW] think=False passed everywhere but ignored by most providers
- [x] Acknowledged — harmless no-op on providers that don't support it. Not worth adding conditional logic.

---

## 6. Image Model Options

Current: **Flux** (high quality, high VRAM)

### 6.1 Add DreamShaper XL Lightning as budget option
- ~4-6GB VRAM, 2-4 steps, 5-15 seconds per image
- Fantasy/medieval fine-tuned — great D&D fit
- [x] Profile `haiku_immersive_dreamshaper` added (`lykon/dreamshaper-xl-lightning`, 4 steps, guidance 1.5)
- [ ] Tested on actual hardware

### 6.2 Add Flux Schnell as mid-tier option
- ~7-8GB VRAM, 4 steps, ~95% of Flux Dev quality
- Uses diffusers path (no ComfyUI GGUF needed for non-quantized Schnell)
- [x] Profile `haiku_immersive_schnell` added (`black-forest-labs/FLUX.1-schnell`, 4 steps, guidance 0.0)
- [ ] Tested on actual hardware
- **Note:** For even lower VRAM, Schnell Q5 GGUF via ComfyUI is an option — would need a GGUF workflow in image_comfyui.py

### 6.3 Update image_factory / config to support per-profile model settings
- [x] Added `image_model`, `image_steps`, `image_guidance` to `ImmersionConfig`
- [x] Updated profile loader to read new fields
- [x] Updated `image_factory.get_image_provider()` to read from profile first, fall back to settings

---

## Item Count

| Section | Total | Done |
|---------|-------|------|
| 1. Bugs | 1 | 1 |
| 2. Race Conditions | 3 | 3 |
| 3. Code Quality | 5 | 5 |
| 4. Web Parity | 7 | 0 |
| 5. Provider Modularity | 8 | 8 |
| 6. Image Models | 5 | 3 |
| **Total** | **29** | **20** |
