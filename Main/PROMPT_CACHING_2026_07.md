# PROMPT CACHING ANALYSIS — 2026-07-17

Scope: is provider-side prompt caching worth engineering for, what would it pay on the
narrator and brain call families, and what has to change to collect the money.
Provider facts verified against official docs on 2026-07-17 (sources at bottom).
Code facts cited as `file:line` against the current tree (post merge 5956942).

**This doc is analysis + recommendations only. No prompt reordering is being performed
in this pass.** The measurement plumbing (usage capture + harness roll-up) lands first;
reorder decisions happen after we have measured baselines in §8.

---

## 1. TLDR verdict

- **DeepSeek caching is automatic, free, and already flowing through our client** —
  `DeepSeekClient` maps `prompt_cache_hit_tokens` → `cache_read_tokens`
  (dnd_bot/llm/client.py:1953-1996). There is nothing to "turn on." The question is
  only how much of our prompt actually hits.
- **Today's narrator layout hits ~35-40% at best, near-zero in combat.** The stable
  prefix is persona (~800 tok) + tool schemas (~5.1k tok) ≈ 5.9k of a 14-15.5k prompt.
  The first mutating byte is `turn: N` — literally the first content line of the first
  user message (dnd_bot/game/world_state.py:541). Everything after it (~8.7k tok)
  re-prefills every turn. In combat it is worse: the combat block is appended *inside
  the system message* (dnd_bot/llm/brains/base.py:221-228), so every combat round
  rewrites message[0] and voids even the tool-schema prefix — hit rate collapses to
  ~5% (persona-only) for the rest of the fight.
- **A reorder (volatile state moved to the bottom bookend) raises steady-state hits to
  ~85-90%** and — because the bookend research the layout is built on says the message
  END is also a high-attention zone — it plausibly *helps* recency grounding rather
  than hurting it. It must still be guarded by an eval run before adoption (§6).
- **Money:** on `deepseek-v4-flash` the reorder saves ~$0.06 per 4-hour session
  ($0.108 → $0.046) — real but small. On `deepseek-v4-pro` it saves ~$0.19/session
  ($0.334 → $0.140) and turns pro from 3.1× flash's cost into ~3.0× — the reorder is
  what makes pro-tier narration economically comfortable. The larger practical win is
  **prefill latency**, which also applies to local Ollama runs where there is no bill
  at all.
- **Brain side: nothing to do today.** Groq has no caching for `qwen/qwen3-32b`
  (GPT-OSS models only), and Ollama's KV-prefix reuse is defeated by the per-turn
  interleave of four different system prompts. A future DeepSeek brain would cache
  60-75% of triage input for free.
- **First actionable step (this pass): capture usage everywhere.** Every provider
  client already returns token counts on `LLMResponse` (client.py:71-98) and the
  orchestrator throws all of it away except narrator tokens
  (dnd_bot/llm/orchestrator.py:1473-1495). Until brain/player usage is recorded, every
  number in §7 is an estimate.

---

## 2. How each provider's caching works (verified 2026-07-17)

### 2.1 DeepSeek (our narrator target)

Models: `deepseek-v4-flash` (284B/13B active) and `deepseek-v4-pro` (1.6T/49B active),
both 1M context, 384K max output, thinking default-on. Legacy `deepseek-chat`/
`deepseek-reasoner` names **deprecate 2026-07-24** (they currently alias v4-flash
non-thinking / thinking) — config/profiles.yaml should be checked for legacy names
before that date.

Pricing (per 1M tokens, USD, official pricing page):

| | v4-flash | v4-pro |
|---|---|---|
| Input, cache HIT | **$0.0028** | **$0.003625** |
| Input, cache MISS | **$0.14** | **$0.435** |
| Output | **$0.28** | **$0.87** |
| Concurrency | 2500 | 500 |

Hit price is 1/50 of miss on flash, ~1/120 on pro. No off-peak discounts exist
anymore (program ended 2025-09-05; the v4-pro 75% cut was made permanent 2026-05-22).
A mid-July "peak-hour pricing" rumor is unconfirmed — do not plan around it.

Mechanics (kv_cache guide, current):

- **Automatic, on-disk, default-on, no opt-out, no breakpoints.** `cache_control` is
  documented as *Ignored* everywhere on the Anthropic-compat endpoint; no equivalent
  knob exists on the OpenAI-compat endpoint we use.
- **The old 64-token-block partial matching is GONE.** Current rule (sliding-window
  attention): each cached prefix is an independent unit; a request hits only if it
  **fully matches a persisted prefix unit**. Units persist at (a) request boundaries —
  end-of-user-input AND end-of-model-output, (b) detected common prefixes across
  requests, (c) fixed intervals on long sequences.
- **Consequence:** request A+B then A+C → the A+C request gets **zero** hit, but the
  shared prefix A is then persisted, so A+D hits. Budget one cold request per
  divergence point. For our per-turn loop the divergence point is stable (end of the
  static prefix), so after ~2 warm-up turns every turn hits the shared unit.
- **TTL:** none fixed; unused entries clear "within a few hours to a few days" —
  comfortably covers a 4-hour session. Best-effort, no hit-rate guarantee.
- Usage accounting: `usage.prompt_cache_hit_tokens` + `usage.prompt_cache_miss_tokens`;
  `prompt_tokens` = hit + miss (documented identity). **Not** OpenAI's
  `prompt_tokens_details.cached_tokens`. Streaming: `stream_options.include_usage=true`
  adds a final usage chunk.
- **Docs are silent on tools/response_format in prefix identity.** Inferred (industry-
  standard chat-template behavior): tool schemas render into the prompt ahead of the
  conversation, so *any* byte change to the tools array invalidates from token 0. Our
  registry is deterministic and byte-stable per tier (dnd_bot/llm/tool_registry.py:
  66-102) — keep it that way. Thinking↔non-thinking mode switches likely split the
  cache namespace (dual templates); so does a model switch (flash↔pro tiering).

### 2.2 The others (comparison)

| | Groq | Gemini API (implicit) | Anthropic |
|---|---|---|---|
| Applies to our models? | **No** — caching exists only for GPT-OSS 20B/120B/safeguard; `qwen/qwen3-32b` gets nothing | Yes — auto on 2.5+ models | Yes (Sonnet 4.6 production narrator) |
| Cached-token price | 50% of input | 10% of input (90% off): Flash $0.03 vs $0.30, Pro $0.125 vs $1.25 | 10% of input on read |
| Write premium | none | none (explicit `cachedContent` adds storage $1.00-$4.50/1M tok/hr — not relevant to us) | 1.25× (5-min TTL) or 2× (1-hr TTL) |
| Min cacheable prefix | 128-1024 tok by model | **2,048 tok** (2.5 flash & pro) | **1,024 tok** Sonnet 4.6; 4,096 Haiku 4.5 — shorter prompts silently skip caching |
| TTL | 2h idle | minutes-scale, best-effort | 5m/1h, refreshed on read |
| Config | automatic | automatic | manual `cache_control` breakpoints (max 4) |
| Usage field | `usage.prompt_tokens_details.cached_tokens` | `usage_metadata.cached_content_token_count` | `cache_read_input_tokens` / `cache_creation_input_tokens` |

Notes that matter to us:

- **Gemini:** our deprecated `google-generativeai` SDK still gets implicit caching for
  free (it is server-side), but `GeminiClient` currently **discards**
  `cached_content_token_count` (client.py:1671-1678). The harness's Gemini player
  prompts are mostly per-turn-fresh and usually under the 2,048-token floor anyway.
- **Anthropic:** tools → system → messages render order means one breakpoint on the
  last system block caches tools + persona together. Break-even math: 5-min TTL pays
  for itself on the 2nd request (1.25 + 0.1 < 2×). Our client already maps both cache
  fields (client.py:1323-1342). If the production narrator profile (Sonnet 4.6) ever
  gets `cache_control` added, the same reorder logic in §6 applies — but unlike
  DeepSeek, Anthropic charges a write premium, so a prefix that mutates every turn
  would cost 1.25× *more* than no caching. The reorder is a precondition there, not
  an optimization.
- **Groq:** verdict for this project: **no caching, full price, no action possible.**

---

## 3. Position map of one narrator request

One narration turn = `NarrationStrategy.run` (dnd_bot/llm/narration.py:127) →
`build_bookend_messages` (dnd_bot/llm/brains/base.py:197-324) → spec prompt + tool
reminder appended (narration.py:153-155, orchestrator.py:2302-2351) → `client.chat`.
Context is rebuilt from scratch every turn (dnd_bot/game/session.py:1115-1209).
Tools sit after `messages` in the JSON body but render server-side into the
system-region token prefix, before the first user message.

| # | Segment | Source | Volatility | Est. tok |
|---|---------|--------|-----------|----------|
| 0 | model id | tier selection (orchestrator.py:2240-2252) | static — EXCEPT tiered profiles swap flash↔pro per turn → separate cache namespaces | — |
| 1 | SYSTEM: persona | narrator.py:21-72 | **STATIC** | ~800 |
| 1b | SYSTEM: combat block | base.py:221-228, appended inside msg[0] | **mutates every combat round → invalidates EVERYTHING incl. tools** | 0-150 |
| 2 | tools (rendered) | tool_registry, deterministic order (66-102) | **STATIC per tier** (core=3/core_plus=5/full=8) | ~5,100 |
| 3 | USER 1: `<world_state>` | world_state.py:532-662; **first key is `turn:`** | **MUTATES EVERY TURN, first content byte** | 600-1,500 |
| 4 | `<entity_relationships>` | KG, seeded from this turn's action (orchestrator.py:1114-1142) | mutates every turn (action-keyed) | 0-1,000 |
| 5 | `<past_narration>` | vector recall on this turn's seeds (1144-1162) | mutates every turn | 0-170 |
| 6 | `<party>` | session.py:1127-1138 | slow (until damage/condition) | 60-150 |
| 7 | `<current_scene>` | session.py:1167-1177 | slow-ish (state extractor churns NPCs) | 300-700 |
| 8 | `<active_quests>` | session.py:1163-1165 | slow | 50-150 |
| 9 | `<acting_character>` | session.py:1211-1246 | slow (HP/slots) | ~150 |
| 10 | `<relevant_rules>` | top-3 keyed on action (base.py:264-272) | mutates every turn (action-keyed) | 150-350 |
| 11 | ASSISTANT anchor | fixed string (base.py:276-279) | static | ~20 |
| 12 | USER 2: `<memory>` | manager.py:656-721; contains **per-turn RAG recall** (702-710) + cadence-stable summaries | RAG part every turn; rest on condense/compact cadence | 1,000-2,200 |
| 13 | ASSISTANT anchor | base.py:294-297 | static | ~15 |
| 14 | history window | limit=30 but verbatim buffer **slides its head every message** once full (blocks.py:162-168, size 8-12); final-msg cap exemption re-renders last msg as it ages (base.py:15-33) | **mutates every turn once buffer is full** | 1,500-3,000 |
| 15 | FINAL USER: action+reminder | base.py:307-322 | every turn (terminal — harmless) | 100-250 |
| 16 | spec prompt | rotating style hint (orchestrator.py:2568-2619) | every turn (terminal) | 80-150 |
| 17 | tool-reminder sys msg | ENTITY FACTS roster (orchestrator.py:2302-2351) | on roster change (terminal) | 150-400 |

Measured anchor (config/profiles.yaml:101-105): total 14-15.5k in, ~1,500 out.

**Deliberate design tension:** the bookend layout puts volatile state FIRST on purpose
("Lost in the Middle" — primacy zone, base.py:203-212). Cache-optimal order
(static → volatile) is the exact inverse. The same research says the END is the other
high-attention zone, which is what makes the §6 reorder viable rather than heretical.

---

## 4. Expected hit rate TODAY on DeepSeek (honest math)

- Stable prefix out of combat: segments 1+2 ≈ 5.9k of ~15k = **~39%** ceiling.
  Under DeepSeek's full-unit matching: turn 1 all-miss, turn 2 all-miss (divergent
  suffix → zero hit, common prefix persisted), turns 3+ hit the ~5.9k unit. So ~39%
  steady-state, minus two cold turns.
- In combat: the combat block mutates *inside message[0]* each round, so the shared
  prefix shrinks to the ~800-token persona head → **~5%**. (The task brief's worst
  case — "world-state mutating inside the system prompt at position ~0" — is not quite
  our layout out of combat, but it IS our layout in combat, and there the hit rate is
  indeed near zero.)
- Tiered profiles (`deepseek_tiered` etc., profiles.yaml:494-559) swap flash↔pro per
  turn → two disjoint cache namespaces, each needing its own warm-up; blended hit
  rate drops further on alternating-tier stretches.
- **Blended session estimate (70% explore / 30% combat): ~30-35%.** Use 35% as the
  planning number for "current layout, cache on."
- One accidental bright spot: `_tool_followup` (narration.py:253-267) resends the
  primary message stack as a strict prefix plus two messages — the primary request's
  end-of-input unit makes the followup leg hit at ~100% of the primary length already,
  today, with no changes. Tool-heavy turns are therefore cheaper than they look.
- State-extractor (brain) footnote: its user message opens with the same `turn: N`
  YAML (state_extractor.py:124-147), so its realistic ceiling is system-prompt-only
  (~1,245 of 1.8-3.5k ≈ 35-50%) — and that only matters if the brain ever moves off
  Groq/Ollama.

---

## 5. Cost tables

Assumptions: narrator-only traffic; primary call 15k in / 1.5k out per turn (end-game
realistic, profiles.yaml:101-105); tool-followup legs excluded (they are ~fully cached
on DeepSeek, see §4, so they add little). "No-cache" = hypothetical all-miss baseline
(what we'd pay if DeepSeek had no cache — useful as the denominator). Hit rates:
current = 35% blended, reordered = 85% (conservative end of the 85-90% estimate).

### 4-hour session, ~60 turns (0.9M in / 90k out)

| Scenario | v4-flash | v4-pro |
|---|---|---|
| Current layout, no cache (baseline) | $0.151 | $0.470 |
| Current layout, cache (~35% hit) | **$0.108** (-29%) | **$0.334** (-29%) |
| Reordered, cache (~85% hit) | **$0.046** (-69%) | **$0.140** (-70%) |

### Long-horizon suite run, 22 turns (avg 10.5k in / 1.2k out → 231k in / 26.4k out)

| Scenario | v4-flash | v4-pro |
|---|---|---|
| Current layout, no cache | $0.040 | $0.123 |
| Current layout, cache (~35%) | $0.029 | $0.089 |
| Reordered, cache (~85%) | $0.013 | $0.039 |

Reading:

- Flash is nearly free either way; the reorder saves ~$0.06 per 4-hour session. Not a
  reason to reorder by itself.
- Pro narration drops from ~$0.47 to ~$0.14/session — the reorder is what makes
  running pro-tier as the standard narrator (not just premium turns) defensible.
- The unpriced win: **prefill latency.** ~8.7k tokens re-prefilled per turn today vs
  ~1.5-2k after reorder. On DeepSeek that is time-to-first-token; on local Ollama
  (same KV-prefix logic, no bill) it is the difference the players actually feel.
- Suite runs are cheap in all scenarios (< $0.13) — cost is not a reason to avoid
  DeepSeek-profile long-horizon runs tonight.

---

## 6. Reordering plan — ranked, RECOMMENDATIONS ONLY (do not implement this pass)

Ranked by expected prefix gain ÷ implementation risk. Every item trades against the
bookend-attention rationale (base.py:203-212); the mitigation in all cases is that the
destination is the END bookend — the other high-attention zone — not the middle.
Gate any adoption on a before/after `test_eval.py` rubric run.

| # | Change | Gain | Risk | Notes |
|---|--------|------|------|-------|
| R1 | Move the per-turn action-keyed blocks — `<world_state>` (or its fast keys), `<entity_relationships>`, `<past_narration>`, `<relevant_rules>` — from USER 1 to the bottom bookend | **Largest single lever**: stable prefix 5.9k → 8-10k immediately; unlocks R3/R4 compounding to 85-90% | Medium: grounding-quality regression risk; the blocks land in the recency zone, which the bookend research itself rates highly, but this must be measured, not asserted | USER 1 keeps party/scene/quests/acting-character (multi-turn stable) |
| R2 | Split `WorldState.to_yaml()` into slow head (`location`, `npcs_here`, `facts`, quests) / fast tail (`turn`, `time_of_day`, `recent_events`, `recent_transfers`, `last_seen_turn`) | Small-medium; only matters if world_state stays in USER 1 (superseded by R1) | Low: pure serialization order | Cheapest partial win if R1 stalls |
| R3 | Advance the verbatim history window on condensation boundaries instead of per-message (blocks.py:162-168); cap ALL history messages uniformly (drop the final-message exemption, base.py:15-33) | Turns per-turn history invalidation into per-~2-exchange; required for the 85-90% figure | Low-medium: prompt grows slightly between boundaries; memory-behavior change needs the long-horizon suite as a guard | Complementary to R1 |
| R4 | Move the per-turn RAG recall (manager.py:702-710) out of the middle `<memory>` block to the bottom bookend | Makes `<memory>` cadence-stable (invalidates only on condense/compact) | Low | |
| R5 | Move the combat block out of the system message (base.py:221-228) into a late message | Preserves the full 5.9k persona+tools prefix during combat instead of ~800 | Low-medium: combat grounding placement | Biggest per-token win *during fights* |
| R6 | Pin narrator tier per scene/session instead of per turn on tiered profiles | Avoids namespace splitting and doubled warm-ups | Low | Policy change in orchestrator.py:2197-2276 |

R1+R3+R4 (+R5 for combat) ⇒ typical mid-session turn re-prefills only the terminal
~1.5-2k tokens: **~85-90% hit**. DeepSeek's one-cold-request-per-divergence-point rule
means each cadence event (condense, compact, scene change) costs one extra miss turn —
already haircut into the 85% planning number.

Also worth stating: this reorder is provider-neutral. It is the same change that would
make Anthropic `cache_control` on the production narrator economical (write premium
1.25× means a churning prefix actively loses money there) and that shortens Ollama
local prefill. Do it once, collect on three providers.

---

## 7. Brain-side caching verdict

Call families and cadence (per-turn): triage every turn (~3.4-4.5k in, 3k-token
byte-stable system prompt = 65-85% of the prompt); state extractor every narrated turn
(~1.8-3.5k in, 1,245-token stable system); dedup judge 0-2 on NPC turns; memory calls
amortized ~1.2/turn. All on the single brain client (client.py:2077, alias at 2194).

- **Groq (`qwen/qwen3-32b`) — no caching exists for this model.** Only GPT-OSS models
  are supported. Zero action available; every triage pays full $0.29/M input. (If the
  brain ever moves to `openai/gpt-oss-120b` on Groq, 50% cached discount applies
  automatically and the triage system prompt qualifies.)
- **Ollama local brains — KV-prefix reuse exists but is defeated** by the per-turn
  interleave of four different system prompts (triage → extract → judge → memory) on
  one model slot; each call re-prefills. Only fix would be server-side multi-slot
  cache retention — not worth chasing. Related pre-existing hazard, not caching:
  e2b/e4b profiles set brain `context_size: 4000` (profiles.yaml:115,305,409,654,726)
  while worst-case triage input is ~4.5k → Ollama silently truncates the prompt HEAD,
  i.e. the triage system prompt itself.
- **Hypothetical DeepSeek brain — would cache well with zero work.** DeepSeek persists
  each call family's prefix unit independently, so interleaving does not evict; expect
  ~60-75% of triage and ~35-50% of state-extract input at $0.0028/M. Caveats:
  `DeepSeekClient` degrades `json_schema` → `json_object` (client.py:1923-1924), so
  triage/extract would lean entirely on post-hoc pydantic validation; penalties are
  dropped when thinking is on (1903-1907).
- **Observability gap (fix in this pass):** brain-call usage is computed by every
  provider and then discarded — `process_action` logs only `_last_narrator_*`
  (orchestrator.py:1473-1495). GroqClient additionally discards everything beyond
  prompt/completion (client.py:1034-1042) and GeminiClient discards
  `cached_content_token_count` (1671-1678). Until the usage recorder lands, brain
  cost/caching claims are estimates.

---

## 8. Measured results (2026-07-17, current layout, no reorder)

Two 22-turn long-horizon runs on the instrumented harness (`test_long_horizon.py`,
emergent_callback scenario, Gemini-Flash player). Numbers are the harness's own
end-of-run report; artifacts under `data/long_horizon/`.

| Profile | Turns | Narrator in/out (tok) | Narrator cache-hit % | Brain in/out (tok) | Brain cache-hit % | Player in/out | Est. cost (USD) | p50/p95 turn | p50/p95 narrate | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek_groq | 22 | 376460/9289 | **64.6%** | 72463/15351 (groq) | 13.4% | 15590/234 | **$0.0573** | 12.4s/15.4s | 6.4s/9.0s | 35 Groq JSON failures (see below) |
| deepseek_v4_flash | 22 | 433367/7082 | **60.3%** | 158397/8916 (ollama gemma4:e2b) | 0.0% (local) | 16681/266 | **$0.0325** | 12.9s/23.9s | 6.2s/8.7s | 0 brain errors; brain free |

**The §4 prediction (~35-40% blended, out of combat) was too pessimistic.** Measured
blended narrator hit rate is **60-65%** across both runs. Per-call the distribution is
bimodal, visible in the raw `deepseek_response cache_hit_ratio` log line:

- **~98-99%** (≈10k cached / ~150 miss) — the dominant narrator call each turn. Nearly
  the entire prompt is served from cache. This is *higher* than the position-map's
  "5.9k stable prefix" ceiling implied, which means DeepSeek caches at block
  granularity and tolerates our largely append-only history better than a strict
  single-prefix model predicts.
- **~56-62%** (6016 cached / ~4k miss) — a second per-turn call; matches the §4
  prediction (persona+tools prefix caches, the rest re-prefills).
- **~6-7%** (≈700 cached / ~9.9k miss) — only on location-change turns (3 of 22),
  where the scene/world block rewrites the prefix early. This is the collapse §1
  predicted for combat, confirmed empirically — but for **location changes**, which
  the scene-rescope path (DF-18) makes a discrete high-churn event.

Takeaways for the reorder decision (§6):
1. The reorder's upside is smaller than §7's tables assumed, because the baseline is
   already ~62% not ~35%. Recompute the "current-with-cache" column from 62%, not 35%,
   before committing.
2. The remaining win is concentrated in the **location-change turns** (6-7% → would be
   ~85%+ if volatile scene state moved to the bottom bookend). Those are exactly the
   turns where the narrator most needs recency grounding anyway, so the reorder and the
   attention argument still align — just with a narrower, sharper target than "every
   turn."
3. **Cost is a non-issue confirmed**: a full 22-turn arc is **3-6 cents**. The local-brain
   profile (`deepseek_v4_flash`) is nearly half the cost of the Groq-brain one *and* had
   zero brain failures.

Two harness/reliability findings the runs surfaced (tracked separately, not caching):

- **Groq `qwen/qwen3-32b` is unreliable as a brain.** 35 `json_validate_failed` /
  "max completion tokens reached before generating a valid document" errors across the
  run; triage silently fell back to a default `roleplay` decision each time. Root cause:
  `client.py:985-991` skips `reasoning_format` when `json_schema`/`json_mode` is set, so
  qwen3's thinking still runs and consumes the 500-token brain budget before valid JSON
  is emitted — `think=False` is effectively ignored on the Groq JSON path. Local
  gemma4:e2b had 0 errors. **Recommendation: prefer a local brain, or fix the Groq
  thinking/token handling before using Groq for triage.**
- **Gemini-Flash seed pick truncated** (`'{"type":'`) at `max_tokens=400` because
  Gemini 2.5 Flash's default adaptive thinking drew from the same output budget — every
  recall verdict came back UNTRUSTED. Fixed in the harness (`pick_seed` → `max_tokens=2048`).

**Trusted recall verdict (post-fix run, deepseek_v4_flash, 22 turns):** seed
`A hidden door`, **PASS — 6/6 assertions**. The seed appeared in the explore phase,
the player referenced it at callback, the narrator recalled it, the KG surfaced it into
the narrator context, WorldState retained it through the entire filler phase, and a tool
fired for it during exploration. The architecture retains and re-surfaces arbitrary
established state across a full 22-turn arc. Cache-hit on that run: 60.9% ($0.0316).

Column-order rows for pasting into the table above:

| deepseek_v4_flash (post-fix, trusted) | 22 | 406196/9030 | 60.9% | 156599/9528 | 0.0% | 17136/388 | $0.0316 | 11.7s/17.0s | 5.6s/8.3s | recall PASS 6/6 |

---

## Sources

DeepSeek: api-docs.deepseek.com — /quick_start/pricing, /guides/kv_cache,
/api/create-chat-completion, /news/news260424, /guides/anthropic_api (cache_control
ignored), /guides/tool_calls + /guides/json_mode (silent on caching).
Groq: console.groq.com/docs/prompt-caching, /docs/models, groq.com GPT-OSS caching
announcement. Gemini: ai.google.dev/gemini-api/docs/generate-content/caching,
/docs/pricing, deprecated-generative-ai-python repo. Anthropic:
platform.claude.com/docs/en/build-with-claude/prompt-caching.md.
Code: current tree at C:/Projects/Discord Bots/D&D 5e/Main (line refs in text).
