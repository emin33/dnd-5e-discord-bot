/**
 * D&D 5e Voice Dungeon — Web Client
 *
 * Flow: Campaign select → Character select/create (wizard) → Game
 */

const API = window.location.origin;

/* ── State ─────────────────────────────────────────────────────────── */
const S = {
  campaign: null,        // { id, name, world_setting }
  character: null,       // full CharacterResponse
  sessionKey: null,
  room: null,
  connected: false,
  micOn: false,
  processing: false,
  ttsEnabled: true,      // Auto-speak narration via Riva TTS
  ttsQueue: [],          // Queue of texts to speak
  ttsBusy: false,
  // Wizard
  wizRace: null,
  wizClass: null,
  abilityMethod: 'standard_array',
};

let racesCache = null;
let classesCache = null;

/* ── Screen nav ────────────────────────────────────────────────────── */
function go(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

/* ================================================================
   CAMPAIGN SCREEN
   ================================================================ */

async function loadCampaigns() {
  const grid = document.getElementById('campaign-grid');
  try {
    const res = await fetch(`${API}/api/campaigns`);
    if (!res.ok) throw new Error();
    const list = await res.json();

    if (!list.length) {
      grid.innerHTML = '<div class="placeholder">No campaigns yet — create one to get started.</div>';
      return;
    }

    grid.innerHTML = list.map(c => `
      <div class="campaign-card" onclick="pickCampaign(${esc_attr(c)})" id="cmp-${c.id}">
        <div class="card-name">${esc(c.name)}</div>
        <div class="card-meta">${esc((c.description || c.world_setting || '').slice(0, 90))}</div>
      </div>
    `).join('');
  } catch {
    grid.innerHTML = '<div class="placeholder">Failed to load campaigns. Is the server running?</div>';
  }
}

function pickCampaign(c) {
  S.campaign = c;
  S.character = null;

  // Go to lobby
  document.getElementById('lobby-campaign-name').textContent = c.name;
  document.getElementById('lobby-campaign-setting').textContent = c.world_setting || '';
  go('screen-lobby');
  loadCharacters();
}

function goToCampaigns() {
  S.campaign = null;
  S.character = null;
  go('screen-campaigns');
  loadCampaigns();
}

/* Create campaign */
function toggleCreateCampaign() {
  document.getElementById('create-campaign-panel').classList.toggle('collapsed');
}

async function createCampaign() {
  const name = document.getElementById('inp-campaign-name').value.trim();
  const setting = document.getElementById('inp-campaign-setting').value.trim();
  if (!name) return;

  try {
    const res = await fetch(`${API}/api/campaigns`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, world_setting: setting || undefined }),
    });
    if (!res.ok) throw new Error(await res.text());

    document.getElementById('inp-campaign-name').value = '';
    document.getElementById('inp-campaign-setting').value = '';
    toggleCreateCampaign();
    await loadCampaigns();
  } catch (e) {
    alert('Failed: ' + e.message);
  }
}

/* ================================================================
   LOBBY SCREEN — Character selection
   ================================================================ */

async function loadCharacters() {
  const grid = document.getElementById('character-grid');
  grid.innerHTML = '<div class="placeholder">Loading...</div>';
  updateLaunch();

  try {
    const res = await fetch(`${API}/api/campaigns/${S.campaign.id}/characters`);
    const list = await res.json();

    if (!list.length) {
      grid.innerHTML = '<div class="placeholder">No characters in this campaign yet.</div>';
      return;
    }

    grid.innerHTML = list.map(c => `
      <div class="char-card" onclick="pickCharacter(${esc_attr(c)})" id="chr-${c.id}">
        <div class="card-name">${esc(c.name)}</div>
        <div class="card-meta">Level ${c.level} ${cap(c.race)} ${cap(c.class_name)}</div>
        <div>
          <span class="card-badge hp">HP ${c.hp_current}/${c.hp_max}</span>
          <span class="card-badge cls">AC ${c.ac}</span>
        </div>
      </div>
    `).join('');
  } catch {
    grid.innerHTML = '<div class="placeholder">Failed to load characters.</div>';
  }
}

function pickCharacter(c) {
  S.character = c;
  document.querySelectorAll('.char-card').forEach(el => el.classList.remove('selected'));
  const el = document.getElementById('chr-' + c.id);
  if (el) el.classList.add('selected');
  updateLaunch();
}

function updateLaunch() {
  document.getElementById('btn-launch').disabled = !S.character;
}

/* Quick join — template character */
async function quickJoin() {
  if (!S.campaign) return;
  const name = 'Elara Swiftbow';
  try {
    const res = await fetch(`${API}/api/characters/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        campaign_id: S.campaign.id,
        player_name: 'Player',
        name,
        race_index: 'elf',
        class_index: 'ranger',
        ability_method: 'standard_array',
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || 'Failed');
    const c = await res.json();
    await loadCharacters();
    pickCharacter(c);
  } catch (e) {
    alert('Quick join failed: ' + e.message);
  }
}

/* ================================================================
   CHARACTER WIZARD
   ================================================================ */

function startCharacterWizard() {
  S.wizRace = null;
  S.wizClass = null;
  S.abilityMethod = 'standard_array';
  document.getElementById('wiz-name').value = '';
  go('screen-wizard');
  wizGoTo(1);
  loadSRD();
}

function cancelWizard() {
  go('screen-lobby');
}

let wizCurrent = 1;

function wizGoTo(step) {
  wizCurrent = step;
  // Show/hide panels
  document.querySelectorAll('.wiz-panel').forEach(p => {
    p.style.display = p.dataset.panel == step ? '' : 'none';
  });
  // Update progress
  document.querySelectorAll('.wiz-step').forEach(s => {
    const n = +s.dataset.step;
    s.classList.toggle('active', n === step);
    s.classList.toggle('done', n < step);
  });
}

function wizNext(step) {
  // Validate current step
  if (wizCurrent === 1) {
    const name = document.getElementById('wiz-name').value.trim();
    if (!name || name.length < 2) { document.getElementById('wiz-name').focus(); return; }
  }
  if (wizCurrent === 2 && !S.wizRace) return;
  if (wizCurrent === 3 && !S.wizClass) return;

  if (step === 5) buildPreview();
  wizGoTo(step);
}

/* Load SRD data */
async function loadSRD() {
  if (racesCache && classesCache) {
    renderRaces();
    renderClasses();
    return;
  }
  try {
    const [rr, cr] = await Promise.all([
      fetch(`${API}/api/races`), fetch(`${API}/api/classes`),
    ]);
    racesCache = await rr.json();
    classesCache = await cr.json();
    renderRaces();
    renderClasses();
  } catch (e) {
    console.error('SRD load failed', e);
  }
}

function renderRaces() {
  const grid = document.getElementById('wiz-race-grid');
  grid.innerHTML = racesCache.map(r => {
    const bonuses = r.ability_bonuses.map(b => `+${b.bonus} ${b.ability.toUpperCase()}`).join(', ');
    return `
      <div class="pick-card" id="race-${r.index}" onclick="pickRace('${r.index}')">
        <div class="pick-name">${r.name}</div>
        <div class="pick-detail">${bonuses}<br>Speed ${r.speed}ft</div>
      </div>`;
  }).join('');
}

function renderClasses() {
  const grid = document.getElementById('wiz-class-grid');
  grid.innerHTML = classesCache.map(c => {
    const saves = c.saving_throws.map(s => s.toUpperCase()).join(', ');
    return `
      <div class="pick-card" id="cls-${c.index}" onclick="pickClass('${c.index}')">
        <div class="pick-name">${c.name}</div>
        <div class="pick-detail">Hit Die: d${c.hit_die}<br>Saves: ${saves}</div>
      </div>`;
  }).join('');
}

function pickRace(index) {
  S.wizRace = index;
  document.querySelectorAll('#wiz-race-grid .pick-card').forEach(c => c.classList.remove('selected'));
  document.getElementById('race-' + index).classList.add('selected');
  wizNext(3);
}

function pickClass(index) {
  S.wizClass = index;
  document.querySelectorAll('#wiz-class-grid .pick-card').forEach(c => c.classList.remove('selected'));
  document.getElementById('cls-' + index).classList.add('selected');
  wizNext(4);
}

function setAbilityMethod(m) {
  S.abilityMethod = m;
  document.querySelectorAll('.ability-toggle .btn-pill').forEach(b => {
    b.classList.toggle('active', b.dataset.method === m);
  });
  document.getElementById('ability-desc').innerHTML = m === 'standard_array'
    ? '15, 14, 13, 12, 10, 8 &mdash; auto-assigned optimally for your class.'
    : 'Roll 4d6 drop lowest, 6 times &mdash; auto-assigned for your class.';
}

function buildPreview() {
  const name = document.getElementById('wiz-name').value.trim();
  const race = racesCache?.find(r => r.index === S.wizRace);
  const cls  = classesCache?.find(c => c.index === S.wizClass);

  document.getElementById('wiz-preview').innerHTML = `
    <div class="preview-name">${esc(name)}</div>
    <div class="preview-sub">${race?.name || '?'} ${cls?.name || '?'} &mdash; Level 1</div>
    <div class="preview-stats">
      <div class="preview-stat"><div class="preview-stat-label">Hit Die</div><div class="preview-stat-val">d${cls?.hit_die || '?'}</div></div>
      <div class="preview-stat"><div class="preview-stat-label">Speed</div><div class="preview-stat-val">${race?.speed || 30}ft</div></div>
      <div class="preview-stat"><div class="preview-stat-label">Scores</div><div class="preview-stat-val">${S.abilityMethod === 'standard_array' ? 'Standard' : '4d6 Roll'}</div></div>
    </div>
  `;
}

async function confirmCharacter() {
  const name = document.getElementById('wiz-name').value.trim();
  const playerName = document.getElementById('inp-player-name')?.value.trim() || 'Player';

  try {
    const res = await fetch(`${API}/api/characters/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        campaign_id: S.campaign.id,
        player_name: playerName,
        name,
        race_index: S.wizRace,
        class_index: S.wizClass,
        ability_method: S.abilityMethod,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Creation failed');
    }
    const c = await res.json();

    go('screen-lobby');
    await loadCharacters();
    pickCharacter(c);
  } catch (e) {
    alert('Failed: ' + e.message);
  }
}

/* ================================================================
   LAUNCH GAME
   ================================================================ */

async function launchGame() {
  if (!S.campaign || !S.character) return;

  const btn = document.getElementById('btn-launch');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Starting...';

  const playerName = document.getElementById('inp-player-name').value.trim() || S.character.name;

  try {
    // 1. Start session
    const startRes = await fetch(`${API}/api/game/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        campaign_id: S.campaign.id,
        room_name: 'dnd-session',
        player_name: playerName,
      }),
    });
    if (!startRes.ok) {
      const err = await startRes.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to start');
    }
    const session = await startRes.json();
    S.sessionKey = session.session_key;

    // 2. Join with character
    const joinRes = await fetch(`${API}/api/game/join`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_key: S.sessionKey,
        player_name: playerName,
        character_id: S.character.id,
      }),
    });
    if (joinRes.ok) {
      const j = await joinRes.json();
      if (j.character) S.character = j.character;
    }

    // 3. Switch to game screen
    document.getElementById('game-tag-campaign').textContent = session.campaign_name;
    go('screen-game');
    updateSheet();

    // 4. Opening narration
    if (session.opening_narrative) {
      addNarr(session.opening_narrative, 'narrative');
      speak(session.opening_narrative);
    }

    document.getElementById('inp-action').focus();

    // 5. Try voice (non-blocking)
    tryVoice(playerName);

  } catch (e) {
    alert('Failed to start: ' + e.message);
    btn.disabled = false;
    btn.textContent = 'Begin Adventure';
  }
}

/* ================================================================
   GAME — Actions
   ================================================================ */

async function sendAction() {
  const inp = document.getElementById('inp-action');
  const text = inp.value.trim();
  if (!text || S.processing) return;

  inp.value = '';
  addNarr(text, 'player');
  S.processing = true;

  const btn = document.getElementById('btn-send');
  btn.disabled = true;

  try {
    const res = await fetch(`${API}/api/game/action`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_key: S.sessionKey,
        player_name: S.character?.name || 'Player',
        action: text,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed');
    }

    const data = await res.json();

    // Mechanics
    if (data.mechanical_result) {
      const m = fmtMech(data.mechanical_result);
      if (m) addNarr(m, 'mechanics');
    }

    // Dice rolls
    if (data.dice_rolls?.length) {
      for (const r of data.dice_rolls) {
        if (r.reason) addNarr(`${r.reason}: [${r.dice.join(', ')}] + ${r.modifier} = ${r.total}`, 'mechanics');
      }
    }

    // Narrative
    if (data.narrative) {
      addNarr(data.narrative, 'narrative');
      speak(data.narrative);
    }

    // Update character sheet
    if (data.character_state) updateSheetFromState(data.character_state);

    // Combat
    if (data.combat_triggered) {
      addNarr('Combat has begun!', 'system');
      showCombat(true);
    }

    // Events — skip types we already handled above to avoid duplicates
    if (data.events) {
      for (const ev of data.events) {
        if (ev.type === 'narrative_complete' || ev.type === 'mechanics_ready') continue;
        handleEvent(ev);
      }
    }

  } catch (e) {
    addNarr('Error: ' + e.message, 'system');
  } finally {
    S.processing = false;
    btn.disabled = false;
    inp.focus();
  }
}

function combatAction(type) {
  const map = {
    attack: 'I attack the nearest enemy',
    cast_spell: 'I cast a spell',
    dodge: 'I take the Dodge action',
    dash: 'I Dash',
    disengage: 'I Disengage',
    hide: 'I try to Hide',
    end_turn: 'I end my turn',
  };
  document.getElementById('inp-action').value = map[type] || type;
  sendAction();
}

/* ── Events ────────────────────────────────────────────────────────── */
function handleEvent(ev) {
  const d = ev.data || {};
  switch (ev.type) {
    case 'narrative_complete':
      if (d.narrative) addNarr(d.narrative, 'narrative');
      break;
    case 'mechanics_ready':
      { const m = fmtMech(d.mechanical_result || d); if (m) addNarr(m, 'mechanics'); }
      break;
    case 'combat_start':  showCombat(true);  addNarr('Combat begins!', 'system'); break;
    case 'combat_end':    showCombat(false); addNarr(d.victory ? 'Victory!' : 'Defeat...', 'system'); break;
    case 'turn_prompt':   showCombatActions(true);  if (d.combatants) renderCombatants(d.combatants); break;
    case 'action_result': showCombatActions(false); if (d.narrative) addNarr(d.narrative, 'narrative'); break;
    case 'turn_end':      showCombatActions(false); if (d.combatants) renderCombatants(d.combatants); break;
    case 'error':         addNarr(d.message || 'Error', 'system'); break;
  }
}

function fmtMech(mr) {
  if (!mr?.action_type) return null;
  const t = mr.action_type;
  if (t === 'attack') {
    const hit = mr.hit ? `Hit! ${mr.damage || 0} ${mr.damage_type || ''} damage` : 'Miss!';
    return `Attack: ${mr.attack_roll || '?'} vs AC ${mr.target_ac || '?'} — ${hit}`;
  }
  if (t === 'skill_check' || t === 'check' || t === 'ability_check') {
    return `${cap(mr.skill || mr.ability || 'Check')}: ${mr.roll || '?'} vs DC ${mr.dc || '?'} — ${mr.success ? 'Success!' : 'Failure'}`;
  }
  if (t === 'saving_throw') {
    return `${cap(mr.ability || '')} Save: ${mr.roll || '?'} vs DC ${mr.dc || '?'} — ${mr.success ? 'Success!' : 'Failure'}`;
  }
  if (t === 'spell') {
    let s = `Casting ${mr.spell_name || 'a spell'}`;
    if (mr.damage) s += ` — ${mr.damage} damage`;
    if (mr.healing) s += ` — ${mr.healing} healing`;
    return s;
  }
  return null;
}

/* ── Combat UI ─────────────────────────────────────────────────────── */
function showCombat(show) {
  document.getElementById('combat-tracker').classList.toggle('active', show);
}
function showCombatActions(show) {
  document.getElementById('combat-actions').style.display = show ? 'grid' : 'none';
}
function renderCombatants(list) {
  document.getElementById('combatant-list').innerHTML = list.map(c => {
    const pct = c.hp_max > 0 ? (c.hp_current / c.hp_max) * 100 : 0;
    const cls = pct > 50 ? 'hp-ok' : pct > 25 ? 'hp-mid' : 'hp-low';
    const type = c.is_player ? 'player' : 'npc';
    return `<div class="cbt ${type}${c.active ? ' active' : ''}">
      <span>${c.active ? '> ' : ''}${esc(c.name)}</span>
      <div style="display:flex;align-items:center;gap:6px">
        ${c.is_player ? `<span style="font-size:.7rem">${c.hp_current}/${c.hp_max}</span>` : ''}
        <div class="hp-bar"><div class="hp-fill ${cls}" style="width:${pct}%"></div></div>
      </div>
    </div>`;
  }).join('');
}

/* ── Character Sheet ───────────────────────────────────────────────── */
function updateSheet() {
  const c = S.character;
  if (!c) return;
  document.getElementById('sh-name').textContent = c.name;
  document.getElementById('sh-sub').textContent = `Level ${c.level} ${cap(c.race)} ${cap(c.class_name)}`;
  document.getElementById('sh-hp').textContent = `${c.hp_current}/${c.hp_max}`;
  document.getElementById('sh-ac').textContent = c.ac || '?';
  document.getElementById('sh-lvl').textContent = c.level;

  const pct = c.hp_max > 0 ? (c.hp_current / c.hp_max) * 100 : 0;
  const bar = document.getElementById('sh-hp-bar');
  bar.style.width = pct + '%';
  bar.style.background = pct > 50 ? 'var(--green)' : pct > 25 ? 'var(--orange)' : 'var(--red)';

  if (c.abilities) {
    const mod = v => { const m = Math.floor((v-10)/2); return `${v} (${m>=0?'+':''}${m})`; };
    document.getElementById('sh-str').textContent = mod(c.abilities.str);
    document.getElementById('sh-dex').textContent = mod(c.abilities.dex);
    document.getElementById('sh-con').textContent = mod(c.abilities.con);
    document.getElementById('sh-int').textContent = mod(c.abilities.int);
    document.getElementById('sh-wis').textContent = mod(c.abilities.wis);
    document.getElementById('sh-cha').textContent = mod(c.abilities.cha);
  }
}

function updateSheetFromState(st) {
  if (!st) return;
  if (S.character) {
    S.character.hp_current = st.hp_current;
    S.character.hp_max = st.hp_max;
    if (st.ac) S.character.ac = st.ac;
  }
  updateSheet();

  if (st.spell_slots && Object.keys(st.spell_slots).length) {
    const el = document.getElementById('sh-slots');
    el.innerHTML = Object.entries(st.spell_slots).map(([k,v]) =>
      `<div class="slot-group"><div class="slot-label">${k.replace('level_','L')}</div><div class="slot-val">${v.current}/${v.max}</div></div>`
    ).join('');
  }
}

/* ── End Game ──────────────────────────────────────────────────────── */
async function endGame() {
  if (!confirm('End this session?')) return;
  try { await fetch(`${API}/api/game/end`, { method: 'POST' }); } catch {}
  if (S.room) { try { await S.room.disconnect(); } catch {} S.room = null; }
  S.sessionKey = null;
  S.connected = false;
  go('screen-lobby');
  loadCharacters();
}

/* ================================================================
   TTS — Riva via /api/tts, fallback to browser speechSynthesis
   ================================================================ */

let _rivaTtsAvailable = null; // null=unknown, true/false after first try

function splitSentences(text) {
  return text.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(s => s.length > 0);
}

/** Clean text for TTS — Riva chokes on special chars, markdown, unicode punctuation. */
function cleanForTTS(text) {
  return text
    // Strip markdown formatting
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/_([^_]+)_/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    // Replace unicode punctuation that kills Riva
    .replace(/[\u2014\u2013]/g, ', ')       // em dash, en dash → comma
    .replace(/[\u2018\u2019]/g, "'")         // smart single quotes
    .replace(/[\u201C\u201D]/g, '"')         // smart double quotes
    .replace(/[\u2026]/g, '...')             // ellipsis char
    .replace(/[\u2022\u2023\u25E6]/g, ', ')  // bullets
    // Strip remaining non-ASCII that Riva can't handle
    .replace(/[^\x20-\x7E\n]/g, ' ')
    // Collapse multiple spaces/newlines
    .replace(/\s+/g, ' ')
    .trim();
}

async function speak(text) {
  if (!S.ttsEnabled || !text) return;

  const clean = cleanForTTS(text);
  const sentences = splitSentences(clean).filter(s => s.length > 1);
  if (!sentences.length) return;

  for (const s of sentences) S.ttsQueue.push(s);

  if (!S.ttsBusy) drainTtsQueue();
}

async function drainTtsQueue() {
  if (S.ttsBusy || !S.ttsQueue.length) return;
  S.ttsBusy = true;

  while (S.ttsQueue.length) {
    const sentence = S.ttsQueue.shift();

    try {
      if (_rivaTtsAvailable !== false) {
        const ok = await speakRiva(sentence);
        if (ok) { _rivaTtsAvailable = true; continue; }
        if (_rivaTtsAvailable === null) _rivaTtsAvailable = false;
      }
      await speakBrowser(sentence);
    } catch (e) {
      console.warn('TTS failed for sentence:', sentence.slice(0, 40), e);
    }
  }

  S.ttsBusy = false;
}

// Shared AudioContext for reliable playback
let _audioCtx = null;
function getAudioCtx() {
  if (!_audioCtx) _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (_audioCtx.state === 'suspended') _audioCtx.resume();
  return _audioCtx;
}

async function speakRiva(text) {
  try {
    const res = await fetch(`${API}/api/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
      signal: AbortSignal.timeout(20000),
    });
    if (!res.ok) return false;

    const arrayBuf = await res.arrayBuffer();
    if (arrayBuf.byteLength < 100) {
      console.warn('Riva returned empty audio for:', text.slice(0, 40));
      return false;
    }

    const ctx = getAudioCtx();
    const audioBuffer = await ctx.decodeAudioData(arrayBuf);

    // Play and wait for exact duration
    return new Promise(resolve => {
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      source.onended = () => resolve(true);
      source.start();
    });
  } catch (e) {
    console.warn('speakRiva error:', e);
    return false;
  }
}

function speakBrowser(text) {
  return new Promise(resolve => {
    if (!window.speechSynthesis) { resolve(); return; }
    const utt = new SpeechSynthesisUtterance(text);
    utt.rate = 1.0;
    utt.pitch = 0.9;
    utt.onend = resolve;
    utt.onerror = resolve;
    // Safety timeout
    const timeout = setTimeout(resolve, Math.max(8000, text.length * 80));
    utt.onend = () => { clearTimeout(timeout); resolve(); };
    utt.onerror = () => { clearTimeout(timeout); resolve(); };
    speechSynthesis.speak(utt);
  });
}

/* ================================================================
   ASR — Browser Web Speech API (mic button)
   ================================================================ */

let _recognition = null;

function tryVoice() {
  // Use browser's built-in speech recognition for mic input
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    setVoice('Speech recognition not supported in this browser');
    return;
  }
  setVoice('Mic ready — click to speak');
  setBadge(false);
}

function toggleMic() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) { setVoice('Speech recognition not available'); return; }

  if (S.micOn) {
    // Stop listening
    if (_recognition) _recognition.stop();
    S.micOn = false;
    updateMicBtn();
    setVoice('Mic off');
    return;
  }

  // Start listening
  _recognition = new SpeechRec();
  _recognition.continuous = false;
  _recognition.interimResults = true;
  _recognition.lang = 'en-US';

  _recognition.onstart = () => {
    S.micOn = true;
    updateMicBtn();
    setVoice('Listening...');
  };

  _recognition.onresult = (event) => {
    let transcript = '';
    let isFinal = false;
    for (const result of event.results) {
      transcript += result[0].transcript;
      if (result.isFinal) isFinal = true;
    }

    document.getElementById('voice-transcript').textContent = '"' + transcript + '"';

    if (isFinal && transcript.trim()) {
      document.getElementById('inp-action').value = transcript.trim();
      sendAction();
    }
  };

  _recognition.onend = () => {
    S.micOn = false;
    updateMicBtn();
    setVoice('Mic ready — click to speak');
  };

  _recognition.onerror = (e) => {
    console.warn('ASR error:', e.error);
    S.micOn = false;
    updateMicBtn();
    setVoice(e.error === 'not-allowed' ? 'Mic access denied' : 'Mic ready — click to speak');
  };

  _recognition.start();
}

function updateMicBtn() {
  document.getElementById('mic-btn').classList.toggle('on', S.micOn);
}

function setBadge(on) {
  const b = document.getElementById('game-badge');
  b.textContent = on ? 'Voice' : 'Text + Voice';
  b.classList.toggle('connected', on);
}
function setVoice(txt) { document.getElementById('voice-status').textContent = txt; }

/* ── Narration log ─────────────────────────────────────────────────── */
function addNarr(text, type) {
  const log = document.getElementById('narration-log');
  const d = document.createElement('div');
  d.className = 'narr narr-' + type;
  d.textContent = text;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
}

/* ── Helpers ────────────────────────────────────────────────────────── */
function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }
function esc_attr(obj) { return JSON.stringify(obj).replace(/'/g, "\\'").replace(/"/g, '&quot;'); }
function cap(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : ''; }

/* ── Init ──────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  go('screen-campaigns');
  loadCampaigns();
});
