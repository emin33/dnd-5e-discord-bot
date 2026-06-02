"""Multi-voice TTS assembler: synthesizes segments in parallel and encodes to MP3.

Takes a list of NarrativeSegments (with voice_id/voice_provider resolved)
and produces a single MP3 file as a BytesIO buffer. Each segment is
synthesized with its assigned voice, resampled to a common rate, and
concatenated with brief silence between speaker changes.
"""

import asyncio
import io
import wave
from typing import Optional

import numpy as np
import structlog

from ..models.immersion import GuildImmersionSettings, NarrativeSegment, SegmentType
from ..voice.tts_factory import _create_tts

logger = structlog.get_logger()

# Common output sample rate (Hz)
OUTPUT_SAMPLE_RATE = 44100
# Silence duration between speaker changes (seconds)
SPEAKER_CHANGE_SILENCE = 0.3
# Silence duration between narration paragraphs (seconds)
PARAGRAPH_SILENCE = 0.15


def _make_silence(duration_seconds: float, sample_rate: int = OUTPUT_SAMPLE_RATE) -> np.ndarray:
    """Create a silence buffer."""
    return np.zeros(int(duration_seconds * sample_rate), dtype=np.int16)


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample audio from src_rate to dst_rate."""
    if src_rate == dst_rate:
        return audio
    try:
        from scipy.signal import resample as scipy_resample
        num_samples = int(len(audio) * dst_rate / src_rate)
        return scipy_resample(audio, num_samples).astype(np.int16)
    except ImportError:
        # Fallback: simple linear interpolation
        ratio = dst_rate / src_rate
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio.astype(np.float64)).astype(np.int16)


# Cache TTS instances per provider+voice to avoid re-creating on every segment
_tts_cache: dict[str, object] = {}
_tts_cache_lock: Optional[asyncio.Lock] = None


def _get_tts_cache_lock() -> asyncio.Lock:
    """Lazily create the cache lock (needs a running event loop)."""
    global _tts_cache_lock
    if _tts_cache_lock is None:
        _tts_cache_lock = asyncio.Lock()
    return _tts_cache_lock


async def _get_or_create_tts(provider: str, voice: str, fish_instance: int = 0) -> object:
    """Get a cached TTS instance or create one.

    For Fish Speech, fish_instance selects which server port to use
    (0 = 8080, 1 = 8081, etc.) for parallel generation.

    Guarded by `_tts_cache_lock` so concurrent assemble_audio() calls don't
    double-construct expensive providers (Kokoro, ElevenLabs HTTP client, etc.).
    Double-checked: fast path on cache hit, lock only on miss.
    """
    if provider == "fish" and fish_instance > 0:
        cache_key = f"{provider}:{voice}:port{fish_instance}"
    else:
        cache_key = f"{provider}:{voice}"

    # Fast path: no lock needed for a hit.
    if cache_key in _tts_cache:
        return _tts_cache[cache_key]

    async with _get_tts_cache_lock():
        # Re-check inside the lock — another coroutine may have populated it.
        if cache_key in _tts_cache:
            return _tts_cache[cache_key]

        tts_kwargs = {"voice": voice}
        if provider == "elevenlabs":
            from ..config import get_settings
            tts_kwargs["api_key"] = get_settings().elevenlabs_api_key
        if provider == "fish" and fish_instance > 0:
            from ..config import get_settings
            base_url = get_settings().fish_speech_url
            # Derive port from base URL + instance offset (e.g. :8080 → :8081)
            if ":" in base_url.rsplit("/", 1)[-1]:
                base, port_str = base_url.rsplit(":", 1)
                port = int(port_str) + fish_instance
                tts_kwargs["server_url"] = f"{base}:{port}"
                logger.debug("fish_instance_port", instance=fish_instance, port=port)
        _tts_cache[cache_key] = _create_tts(provider, **tts_kwargs)
        return _tts_cache[cache_key]


async def _synthesize_segment(
    segment: NarrativeSegment,
    settings: Optional[GuildImmersionSettings],
    fish_instance: int = 0,
) -> Optional[tuple[np.ndarray, int]]:
    """Synthesize a single segment. Returns (pcm_array, sample_rate) or None."""
    try:
        provider = segment.voice_provider or "openai"
        voice = segment.voice_id or "onyx"

        tts = await _get_or_create_tts(provider, voice, fish_instance=fish_instance if provider == "fish" else 0)

        # Synthesize -- pass emotion for providers that support it.
        loop = asyncio.get_event_loop()

        def _do_synth():
            if segment.emotion and provider in ("fish", "inworld"):
                return tts.synthesize(segment.text, emotion=segment.emotion)
            return tts.synthesize(segment.text)

        # Audit #12 (R2): this assembler fans segments out via asyncio.gather, so
        # multiple threads can hit the same in-process model (Kokoro's shared
        # KPipeline) concurrently. The single-voice api.py path serialized via
        # needs_lock()/get_tts_lock(); this path didn't. Serialize here too for
        # lock-requiring providers; HTTP providers stay concurrent.
        from ..voice.tts_factory import needs_lock, get_tts_lock
        if needs_lock(tts):
            async with get_tts_lock():
                audio = await loop.run_in_executor(None, _do_synth)
        else:
            audio = await loop.run_in_executor(None, _do_synth)

        if audio is None or len(audio) == 0:
            return None

        # Determine sample rate from provider
        if hasattr(tts, 'sample_rate'):
            sample_rate = tts.sample_rate
        else:
            sample_rate = OUTPUT_SAMPLE_RATE
            logger.debug("tts_sample_rate_fallback", provider=provider, voice=voice)

        return audio, sample_rate

    except Exception as e:
        logger.warning(
            "tts_segment_failed",
            voice_provider=segment.voice_provider,
            voice_id=segment.voice_id,
            error=str(e),
        )
        return None


async def assemble_audio(
    segments: list[NarrativeSegment],
    guild_settings: Optional[GuildImmersionSettings] = None,
) -> Optional[io.BytesIO]:
    """Synthesize all segments in parallel and assemble into a single MP3.

    Args:
        segments: NarrativeSegments with voice_id/voice_provider resolved.
        guild_settings: Guild settings for narrator voice config.

    Returns:
        BytesIO containing the MP3 file, or None on failure.
    """
    if not segments:
        return None

    # Determine number of Fish Speech instances for parallel generation
    fish_instances = 1
    try:
        from ..config import get_settings
        fish_instances = max(1, get_settings().fish_speech_instances)
    except Exception:
        pass

    # Assign Fish segments round-robin across instances
    fish_counter = 0
    tasks = []
    for seg in segments:
        instance = 0
        if (seg.voice_provider or "") == "fish" and fish_instances > 1:
            instance = fish_counter % fish_instances
            fish_counter += 1
        tasks.append(_synthesize_segment(seg, guild_settings, fish_instance=instance))
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect successful results and resample to common rate
    audio_chunks: list[np.ndarray] = []
    prev_speaker = None

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning("tts_gather_exception", index=i, error=str(result))
            continue
        if result is None:
            continue

        audio, sample_rate = result
        resampled = _resample(audio, sample_rate, OUTPUT_SAMPLE_RATE)

        # Add silence between speaker changes
        current_speaker = segments[i].speaker_entity_id or segments[i].segment_type.value
        if prev_speaker is not None and current_speaker != prev_speaker:
            audio_chunks.append(_make_silence(SPEAKER_CHANGE_SILENCE))
        elif audio_chunks:
            audio_chunks.append(_make_silence(PARAGRAPH_SILENCE))

        audio_chunks.append(resampled)
        prev_speaker = current_speaker

    if not audio_chunks:
        return None

    # Concatenate all audio
    combined = np.concatenate(audio_chunks)

    # Encode to MP3
    mp3_buffer = _encode_mp3(combined, OUTPUT_SAMPLE_RATE)
    if mp3_buffer:
        return mp3_buffer

    # Fallback to WAV if MP3 encoding fails
    return _encode_wav(combined, OUTPUT_SAMPLE_RATE)


def _encode_mp3(audio: np.ndarray, sample_rate: int) -> Optional[io.BytesIO]:
    """Encode int16 PCM to MP3 using lameenc."""
    try:
        import lameenc

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(128)
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_channels(1)
        encoder.set_quality(5)  # 0=best, 9=fastest

        mp3_data = encoder.encode(audio.tobytes())
        mp3_data += encoder.flush()

        buf = io.BytesIO(mp3_data)
        buf.seek(0)
        return buf

    except ImportError:
        logger.debug("lameenc_not_available_falling_back_to_wav")
        return None
    except Exception as e:
        logger.warning("mp3_encode_failed", error=str(e))
        return None


def _encode_wav(audio: np.ndarray, sample_rate: int) -> io.BytesIO:
    """Encode int16 PCM to WAV as a fallback."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    buf.seek(0)
    return buf
