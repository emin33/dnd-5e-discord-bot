/**
 * D&D 5e Voice Session - Web Client
 *
 * Connects to a LiveKit room for real-time voice interaction with
 * the D&D game engine. Players speak their actions and hear narration.
 * During combat, the UI shows action buttons alongside voice input.
 */

// LiveKit connection config
const LIVEKIT_URL = 'ws://localhost:7880';
const TOKEN_ENDPOINT = '/api/token'; // Token generation endpoint

let room = null;
let isConnected = false;
let isMicEnabled = false;

// --- UI Helpers ---

function addNarration(text, type = 'narrative') {
  const log = document.getElementById('narration-log');
  const entry = document.createElement('div');
  entry.className = `narration-entry narration-${type}`;
  entry.textContent = text;
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
}

function setStatus(status, text) {
  const badge = document.getElementById('connection-status');
  badge.className = `status-badge status-${status}`;
  badge.textContent = text;
}

function setVoiceStatus(text, className = '') {
  const el = document.getElementById('voice-status');
  el.textContent = text;
  el.className = `voice-status ${className}`;
}

function showCombatPanel(show) {
  const panel = document.getElementById('combat-panel');
  panel.classList.toggle('active', show);
}

function showActionButtons(show) {
  const buttons = document.getElementById('action-buttons');
  buttons.style.display = show ? 'grid' : 'none';
}

function updateCombatants(combatants) {
  const list = document.getElementById('combatant-list');
  list.innerHTML = '';

  combatants.forEach(c => {
    const div = document.createElement('div');
    const hpPercent = c.hp_max > 0 ? (c.hp_current / c.hp_max) * 100 : 0;
    const hpClass = hpPercent > 50 ? 'hp-healthy' : hpPercent > 25 ? 'hp-wounded' : 'hp-critical';
    const typeClass = c.is_player ? 'player' : 'npc';
    const activeClass = c.active ? ' active' : '';

    div.className = `combatant ${typeClass}${activeClass}`;
    div.innerHTML = `
      <span>${c.active ? '> ' : ''}${c.name}</span>
      <div style="display: flex; align-items: center; gap: 6px;">
        <span style="font-size: 0.75rem;">${c.is_player ? c.hp_current + '/' + c.hp_max : ''}</span>
        <div class="hp-bar">
          <div class="hp-fill ${hpClass}" style="width: ${hpPercent}%"></div>
        </div>
      </div>
    `;
    list.appendChild(div);
  });
}

// --- LiveKit Connection ---

async function connect() {
  setStatus('connecting', 'Connecting...');

  try {
    // For development, use a hardcoded token or request from server
    // In production, your server generates tokens via LiveKit Server SDK
    const token = await getToken();

    room = new LivekitClient.Room({
      adaptiveStream: true,
      dynacast: true,
    });

    // Handle room events
    room.on(LivekitClient.RoomEvent.Connected, () => {
      isConnected = true;
      setStatus('connected', 'Connected');
      setVoiceStatus('Connected - speak your actions');
      addNarration('Connected to voice session. Speak your actions to the DM.', 'system');
    });

    room.on(LivekitClient.RoomEvent.Disconnected, () => {
      isConnected = false;
      setStatus('disconnected', 'Disconnected');
      setVoiceStatus('Disconnected');
      addNarration('Disconnected from voice session.', 'system');
    });

    // Handle data messages from the agent (game events, combat updates)
    room.on(LivekitClient.RoomEvent.DataReceived, (data, participant) => {
      try {
        const message = JSON.parse(new TextDecoder().decode(data));
        handleGameEvent(message);
      } catch (e) {
        console.warn('Failed to parse data message:', e);
      }
    });

    // Handle transcription events (show what ASR heard)
    room.on(LivekitClient.RoomEvent.TranscriptionReceived, (segments, participant) => {
      const text = segments.map(s => s.text).join(' ');
      if (text && participant && participant.isLocal) {
        document.getElementById('transcription').textContent = `You: "${text}"`;
        addNarration(text, 'player');
      }
    });

    await room.connect(LIVEKIT_URL, token);

    // Enable microphone
    await room.localParticipant.setMicrophoneEnabled(true);
    isMicEnabled = true;
    updateMicUI();

  } catch (err) {
    console.error('Connection failed:', err);
    setStatus('disconnected', 'Connection Failed');
    addNarration(`Connection failed: ${err.message}`, 'system');
  }
}

async function getToken() {
  // For development: use LiveKit CLI to generate tokens, or
  // request from your backend. This is a placeholder.
  try {
    const response = await fetch(TOKEN_ENDPOINT);
    if (response.ok) {
      const data = await response.json();
      return data.token;
    }
  } catch (e) {
    // Fallback: prompt user for token
  }

  // Dev fallback: user pastes token
  const token = prompt(
    'Enter LiveKit room token:\n\n' +
    'Generate with: livekit-cli create-token \\\n' +
    '  --api-key devkey --api-secret secret \\\n' +
    '  --join --room dnd-session --identity player1 \\\n' +
    '  --valid-for 24h'
  );
  return token;
}

// --- Game Event Handling ---

function handleGameEvent(event) {
  switch (event.type) {
    case 'narrative_complete':
      addNarration(event.data.narrative, 'narrative');
      break;

    case 'mechanics_ready':
      const mechText = formatMechanics(event.data);
      if (mechText) addNarration(mechText, 'mechanics');
      break;

    case 'combat_start':
      showCombatPanel(true);
      addNarration('Combat begins!', 'system');
      break;

    case 'combat_end':
      showCombatPanel(false);
      showActionButtons(false);
      const endMsg = event.data.victory ? 'Victory!' : 'Defeat...';
      addNarration(endMsg, 'system');
      break;

    case 'turn_prompt':
      showActionButtons(true);
      if (event.data.combatants) {
        updateCombatants(event.data.combatants);
      }
      break;

    case 'action_result':
      showActionButtons(false);
      if (event.data.narrative) {
        addNarration(event.data.narrative, 'narrative');
      }
      break;

    case 'turn_end':
      showActionButtons(false);
      if (event.data.combatants) {
        updateCombatants(event.data.combatants);
      }
      break;

    case 'error':
      addNarration(event.data.message, 'system');
      break;

    default:
      console.log('Unknown game event:', event);
  }
}

function formatMechanics(data) {
  const mr = data.mechanical_result;
  if (!mr) return null;

  const at = mr.action_type;
  if (at === 'attack') {
    const hit = mr.hit ? `Hit! ${mr.damage} ${mr.damage_type || ''} damage` : 'Miss!';
    return `Attack: ${mr.attack_roll} vs AC ${mr.target_ac} - ${hit}`;
  }
  if (at === 'skill_check' || at === 'check') {
    const skill = mr.skill || mr.ability || 'Check';
    const result = mr.success ? 'Success' : 'Failure';
    return `${skill}: ${mr.roll} vs DC ${mr.dc} - ${result}`;
  }
  return null;
}

// --- Combat Actions ---

function sendAction(actionType) {
  if (!room || !isConnected) return;

  const message = JSON.stringify({
    type: 'combat_action',
    action_type: actionType,
  });

  room.localParticipant.publishData(
    new TextEncoder().encode(message),
    { reliable: true }
  );

  showActionButtons(false);
  addNarration(`Action: ${actionType.replace('_', ' ')}`, 'player');
}

// --- Mic Control ---

function toggleMic() {
  if (!isConnected) {
    connect();
    return;
  }

  isMicEnabled = !isMicEnabled;
  room.localParticipant.setMicrophoneEnabled(isMicEnabled);
  updateMicUI();
}

function updateMicUI() {
  const btn = document.getElementById('mic-btn');
  btn.classList.toggle('listening', isMicEnabled);

  if (isMicEnabled) {
    setVoiceStatus('Listening...', 'listening');
  } else {
    setVoiceStatus('Mic muted', '');
  }
}

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
  setVoiceStatus('Click microphone to connect');
});
