"""Voice Sampler: Pull voice reference samples from YouTube for Fish Speech.

Full pipeline:
  1. Download audio from YouTube (yt-dlp)
  2. Vocal isolation -- strip music/SFX, keep speech only (demucs)
  3. Speaker diarization -- identify who speaks when (Deepgram)
  4. Dominant speaker extraction -- keep only the main voice
  5. Pick the best contiguous segment (10-30s)
  6. Transcribe the final clip (Deepgram)
  7. Save as Fish Speech reference (prompt.wav + prompt.lab)

Usage:
    python tools/voice_sampler.py "https://youtube.com/watch?v=..." --name gruff_dwarf
    python tools/voice_sampler.py "https://youtube.com/watch?v=..." --name wizard --start 30 --duration 120
    python tools/voice_sampler.py --list

The --start/--duration window is the SEARCH WINDOW -- the tool will find
the best 10-30s of the dominant speaker within that range.
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

# Load .env so DEEPGRAM_API_KEY etc. are available
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import numpy as np

DEFAULT_REFERENCES_DIR = Path(__file__).parent.parent / "fish_voices"

# Target clip length for Fish Speech reference
MIN_CLIP_SECONDS = 10
MAX_CLIP_SECONDS = 90
IDEAL_CLIP_SECONDS = 45


# ── Step 1: Download ────────────────────────────────────────────────────────

def download_audio(url: str, output_dir: Path) -> Path:
    """Download audio from YouTube as WAV."""
    print("  [1/6] Downloading audio...")
    out_template = str(output_dir / "source.%(ext)s")

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", out_template,
        "--no-playlist",
        "--quiet",
        "--js-runtimes", "node",
        "--remote-components", "ejs:github",
    ]
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")

    # Find output file (may be .wav directly or need conversion)
    for f in output_dir.glob("source.*"):
        if f.suffix == ".wav":
            return f
        # Convert non-wav to wav
        wav_path = output_dir / "source.wav"
        subprocess.run(
            ["ffmpeg", "-i", str(f), "-ar", "44100", "-ac", "1", "-y", str(wav_path)],
            capture_output=True,
        )
        f.unlink()
        return wav_path

    raise RuntimeError("No audio file found after download")


# ── Step 2: Vocal Isolation ─────────────────────────────────────────────────

def isolate_vocals(wav_path: Path, output_dir: Path) -> Path:
    """Strip music/SFX using demucs in-process, keep vocals only."""
    print("  [2/6] Isolating vocals (demucs)...")

    try:
        import torch
        import soundfile as sf
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        device = "cpu"
        if torch.cuda.is_available():
            try:
                t = torch.randn(8, 8, device="cuda")
                _ = t @ t
                del t
                device = "cuda"
            except Exception:
                pass
        print(f"    Device: {device}")

        # Load audio with soundfile (no torchcodec dependency)
        audio_np, sr = sf.read(str(wav_path), dtype="float32")
        if audio_np.ndim == 1:
            audio_np = np.stack([audio_np, audio_np], axis=1)  # mono -> stereo
        waveform = torch.from_numpy(audio_np.T)  # shape: [channels, samples]

        # Resample to 44100 if needed
        if sr != 44100:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, sr, 44100)
            sr = 44100

        # Ensure stereo (demucs expects 2 channels)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        # Load model
        model = get_model("htdemucs")
        if device == "cuda":
            model = model.cuda()

        # Run separation
        print("    Separating vocals from music...")
        with torch.no_grad():
            sources = apply_model(
                model,
                waveform.unsqueeze(0).to(device),
                device=device,
            )

        # sources shape: [1, num_sources, 2, samples]
        # htdemucs sources: drums, bass, other, vocals (index 3)
        vocals = sources[0, -1]  # Last source is vocals

        # Convert to mono
        vocals_mono = vocals.mean(dim=0, keepdim=True).cpu()

        # Save
        clean_path = output_dir / "vocals_clean.wav"
        sf.write(str(clean_path), vocals_mono.squeeze().numpy(), 44100)

        # Free GPU memory
        del model, sources, waveform
        if device == "cuda":
            torch.cuda.empty_cache()

        print("    Vocals isolated successfully")
        return clean_path

    except Exception as e:
        print(f"  WARNING: Demucs failed ({e}), using original audio")
        return wav_path


# ── Step 3: Extract Search Window ───────────────────────────────────────────

def extract_window(wav_path: Path, output_dir: Path, start: float, duration: float) -> Path:
    """Extract the search window from the full audio."""
    window_path = output_dir / "window.wav"
    cmd = [
        "ffmpeg", "-i", str(wav_path),
        "-ss", str(start), "-t", str(duration),
        "-ar", "44100", "-ac", "1", "-y",
        str(window_path),
    ]
    subprocess.run(cmd, capture_output=True)
    return window_path


# ── Step 4: Diarize + Find Dominant Speaker ─────────────────────────────────

async def diarize_audio(wav_path: Path) -> list[dict]:
    """Transcribe with speaker diarization using Deepgram REST API.

    Returns list of segments: {speaker: int, start: float, end: float, text: str}
    """
    print("  [3/6] Diarizing speakers (Deepgram)...")

    api_key = os.environ.get("DEEPGRAM_API_KEY", "")
    if not api_key:
        print("  WARNING: No DEEPGRAM_API_KEY -- skipping diarization")
        return []

    import requests

    with open(wav_path, "rb") as f:
        audio_data = f.read()

    response = await asyncio.to_thread(
        lambda: requests.post(
            "https://api.deepgram.com/v1/listen",
            params={
                "model": "nova-3",
                "language": "en",
                "punctuate": "true",
                "diarize": "true",
                "utterances": "true",
            },
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "audio/wav",
            },
            data=audio_data,
            timeout=60,
        )
    )

    if response.status_code != 200:
        print(f"  WARNING: Deepgram failed ({response.status_code}): {response.text[:200]}")
        return []

    data = response.json()
    utterances = data.get("results", {}).get("utterances", [])

    segments = []
    for utt in utterances:
        segments.append({
            "speaker": utt.get("speaker", 0),
            "start": utt.get("start", 0),
            "end": utt.get("end", 0),
            "text": utt.get("transcript", ""),
        })

    # Count speaker time
    speaker_time = {}
    for seg in segments:
        spk = seg["speaker"]
        speaker_time[spk] = speaker_time.get(spk, 0) + (seg["end"] - seg["start"])

    if speaker_time:
        for spk, total in sorted(speaker_time.items(), key=lambda x: -x[1]):
            print(f"    Speaker {spk}: {total:.1f}s")

    return segments


def find_dominant_speaker(segments: list[dict]) -> int:
    """Find the speaker with the most total speaking time."""
    speaker_time = {}
    for seg in segments:
        spk = seg["speaker"]
        speaker_time[spk] = speaker_time.get(spk, 0) + (seg["end"] - seg["start"])

    if not speaker_time:
        return 0

    dominant = max(speaker_time, key=speaker_time.get)
    total = speaker_time[dominant]
    print(f"  [4/6] Dominant speaker: {dominant} ({total:.1f}s total)")
    return dominant


# ── Step 5: Extract Best Contiguous Segment ─────────────────────────────────

def find_best_segment(
    segments: list[dict],
    dominant_speaker: int,
    min_sec: float = MIN_CLIP_SECONDS,
    max_sec: float = MAX_CLIP_SECONDS,
    ideal_sec: float = IDEAL_CLIP_SECONDS,
) -> tuple[float, float, str]:
    """Find the best contiguous run of the dominant speaker.

    Returns (start_time, end_time, combined_transcript).
    """
    # Filter to dominant speaker only
    dom_segs = [s for s in segments if s["speaker"] == dominant_speaker]
    if not dom_segs:
        return 0, ideal_sec, ""

    # Find runs of consecutive dominant speaker segments
    # (allowing small gaps < 2s between them)
    runs = []
    current_run = [dom_segs[0]]

    for seg in dom_segs[1:]:
        gap = seg["start"] - current_run[-1]["end"]
        if gap < 2.0:  # Allow up to 2s gap
            current_run.append(seg)
        else:
            runs.append(current_run)
            current_run = [seg]
    runs.append(current_run)

    # Score each run by total speech duration
    best_run = None
    best_duration = 0

    for run in runs:
        run_start = run[0]["start"]
        run_end = run[-1]["end"]
        duration = run_end - run_start

        if duration >= min_sec and duration > best_duration:
            best_run = run
            best_duration = duration

    if not best_run:
        # No run meets minimum -- take the longest one anyway
        best_run = max(runs, key=lambda r: r[-1]["end"] - r[0]["start"])

    start = best_run[0]["start"]
    end = best_run[-1]["end"]

    # Cap at max length
    if end - start > max_sec:
        end = start + max_sec

    transcript = " ".join(s["text"] for s in best_run if s["start"] < end)

    print(f"  [5/6] Best segment: {start:.1f}s - {end:.1f}s ({end - start:.1f}s)")
    return start, end, transcript


def splice_dominant_speaker(
    wav_path: Path,
    output_dir: Path,
    segments: list[dict],
    dominant_speaker: int,
) -> tuple[Path, str]:
    """Extract ONLY the dominant speaker's audio, removing all other speakers.

    Reads the WAV, keeps only the time ranges where the dominant speaker
    is talking, concatenates them with small silence gaps, and caps at
    MAX_CLIP_SECONDS.

    Returns (clip_path, transcript).
    """
    import soundfile as sf

    print(f"  [5/6] Splicing dominant speaker segments...")

    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono

    # Get all segments for the dominant speaker
    dom_segs = [s for s in segments if s["speaker"] == dominant_speaker]
    dom_segs.sort(key=lambda s: s["start"])

    if not dom_segs:
        # Fallback
        clip_path = output_dir / "prompt.wav"
        sf.write(str(clip_path), audio[:sr * IDEAL_CLIP_SECONDS], sr)
        return clip_path, ""

    # Extract each dominant segment and concatenate
    silence_gap = np.zeros(int(0.5 * sr), dtype=np.float32)  # 500ms gap between segments
    chunks = []
    texts = []
    total_duration = 0.0

    for seg in dom_segs:
        if total_duration >= MAX_CLIP_SECONDS:
            break

        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        seg_duration = seg["end"] - seg["start"]

        # Skip very short segments (< 0.5s) -- likely noise
        if seg_duration < 0.5:
            continue

        # Cap if adding this would exceed max
        if total_duration + seg_duration > MAX_CLIP_SECONDS:
            remaining = MAX_CLIP_SECONDS - total_duration
            end_sample = start_sample + int(remaining * sr)
            seg_duration = remaining

        chunk = audio[start_sample:end_sample]
        if len(chunk) > 0:
            if chunks:
                chunks.append(silence_gap)
            chunks.append(chunk)
            texts.append(seg["text"])
            total_duration += seg_duration

    if not chunks:
        clip_path = output_dir / "prompt.wav"
        sf.write(str(clip_path), audio[:sr * IDEAL_CLIP_SECONDS], sr)
        return clip_path, " ".join(s["text"] for s in dom_segs)

    spliced = np.concatenate(chunks)

    # Ensure minimum length
    if total_duration < MIN_CLIP_SECONDS:
        print(f"    WARNING: Only {total_duration:.1f}s of dominant speaker found (min {MIN_CLIP_SECONDS}s)")

    clip_path = output_dir / "prompt.wav"
    sf.write(str(clip_path), spliced, sr)
    transcript = " ".join(texts)

    print(f"    Spliced {len([c for c in chunks if len(c) > int(0.15 * sr)])} segments, {total_duration:.1f}s total")

    return clip_path, transcript


def extract_final_clip(
    wav_path: Path,
    output_dir: Path,
    start: float,
    end: float,
) -> Path:
    """Extract the final clip for the reference."""
    clip_path = output_dir / "prompt.wav"
    duration = end - start

    cmd = [
        "ffmpeg", "-i", str(wav_path),
        "-ss", str(start), "-t", str(duration),
        "-ar", "44100", "-ac", "1", "-y",
        str(clip_path),
    ]
    subprocess.run(cmd, capture_output=True)

    with wave.open(str(clip_path), 'rb') as wf:
        actual_dur = wf.getnframes() / wf.getframerate()
        print(f"  Final clip: {actual_dur:.1f}s at {wf.getframerate()}Hz mono")

    return clip_path


# ── Step 6: Save Reference ──────────────────────────────────────────────────

def save_reference(
    clip_path: Path,
    transcript: str,
    voice_name: str,
    references_dir: Path,
) -> Path:
    """Save clip + transcript in Fish Speech references format."""
    print(f"  [6/6] Saving reference...")

    voice_dir = references_dir / voice_name
    voice_dir.mkdir(parents=True, exist_ok=True)

    dest_wav = voice_dir / "prompt.wav"
    shutil.copy2(clip_path, dest_wav)

    dest_lab = voice_dir / "prompt.lab"
    dest_lab.write_text(transcript, encoding="utf-8")

    print(f"\n  Saved to: {voice_dir}")
    print(f"    prompt.wav ({dest_wav.stat().st_size // 1024}KB)")
    print(f"    prompt.lab ({len(transcript)} chars)")

    return voice_dir


# ── Fallback: No Diarization ────────────────────────────────────────────────

async def simple_transcribe(wav_path: Path) -> str:
    """Simple transcription without diarization."""
    import requests

    api_key = os.environ.get("DEEPGRAM_API_KEY", "")
    if not api_key:
        return input("  Enter transcript manually: ").strip()

    with open(wav_path, "rb") as f:
        audio_data = f.read()

    response = await asyncio.to_thread(
        lambda: requests.post(
            "https://api.deepgram.com/v1/listen",
            params={"model": "nova-3", "language": "en", "punctuate": "true"},
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "audio/wav",
            },
            data=audio_data,
            timeout=60,
        )
    )

    if response.status_code != 200:
        return input("  Deepgram failed. Enter transcript manually: ").strip()

    data = response.json()
    transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
    return transcript


# ── List Voices ─────────────────────────────────────────────────────────────

def list_voices(references_dir: Path):
    """List existing voice references."""
    if not references_dir.exists():
        print(f"No voices directory at {references_dir}")
        return

    voices = sorted(d.name for d in references_dir.iterdir() if d.is_dir())
    if not voices:
        print("No voices found.")
        return

    print(f"Voice references in {references_dir}:\n")
    for name in voices:
        voice_dir = references_dir / name
        has_wav = (voice_dir / "prompt.wav").exists()
        has_lab = (voice_dir / "prompt.lab").exists()

        status = "OK" if (has_wav and has_lab) else "INCOMPLETE"
        lab_preview = ""
        if has_lab:
            text = (voice_dir / "prompt.lab").read_text(encoding="utf-8")
            lab_preview = f' -- "{text[:60]}..."' if len(text) > 60 else f' -- "{text}"'

        wav_info = ""
        if has_wav:
            size_kb = (voice_dir / "prompt.wav").stat().st_size // 1024
            with wave.open(str(voice_dir / "prompt.wav"), 'rb') as wf:
                dur = wf.getnframes() / wf.getframerate()
            wav_info = f" ({dur:.0f}s, {size_kb}KB)"

        print(f"  [{status}] {name}{wav_info}{lab_preview}")


# ── Main Pipeline ───────────────────────────────────────────────────────────

async def process_voice(args):
    """Full pipeline: download -> isolate -> diarize -> extract -> save."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 1. Download
        source_wav = download_audio(args.url, tmpdir)

        # 2. Vocal isolation
        if not args.skip_isolation:
            clean_wav = isolate_vocals(source_wav, tmpdir)
        else:
            clean_wav = source_wav
            print("  [2/6] Skipping vocal isolation (--skip-isolation)")

        # 3. Extract search window
        window_wav = extract_window(clean_wav, tmpdir, args.start, args.duration)

        # 4-5. Diarize and find best segment
        segments = await diarize_audio(window_wav)

        if segments and len(set(s["speaker"] for s in segments)) > 1:
            dominant = find_dominant_speaker(segments)
            clip_path, transcript = splice_dominant_speaker(
                window_wav, tmpdir, segments, dominant
            )
        else:
            # Single speaker or no diarization -- use the whole window
            print("  [4/6] Single speaker detected (or diarization unavailable)")

            # Cap to ideal length
            with wave.open(str(window_wav), 'rb') as wf:
                total_dur = wf.getnframes() / wf.getframerate()

            if total_dur > MAX_CLIP_SECONDS:
                clip_path = extract_final_clip(window_wav, tmpdir, 0, IDEAL_CLIP_SECONDS)
            else:
                shutil.copy2(window_wav, tmpdir / "prompt.wav")
                clip_path = tmpdir / "prompt.wav"

            print(f"  [5/6] Using full window as clip")

            # Get transcript
            if segments:
                transcript = " ".join(s["text"] for s in segments)
            else:
                transcript = ""

        # Override transcript if manually provided
        if args.transcript:
            transcript = args.transcript
        elif not transcript:
            transcript = await simple_transcribe(clip_path)

        if not transcript:
            print("  ERROR: Transcript is required for Fish Speech.")
            return

        # 6. Save
        references_dir = Path(args.output)
        save_reference(clip_path, transcript, args.name, references_dir)

        # Also copy to Fish Speech server's references dir if it exists
        server_refs = Path(__file__).parent.parent.parent / "fish-speech-repo" / "references"
        if server_refs.exists() and server_refs != references_dir:
            server_voice_dir = server_refs / args.name
            if server_voice_dir.exists():
                shutil.rmtree(server_voice_dir)
            shutil.copytree(references_dir / args.name, server_voice_dir)
            print(f"  Also copied to server: {server_voice_dir}")

        print(f"\n  Use in Fish Speech with reference_id: {args.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Pull voice samples from YouTube for Fish Speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools\\voice_sampler.py "https://youtube.com/watch?v=xyz" --name gargamel
  python tools\\voice_sampler.py "https://youtube.com/watch?v=xyz" --name wizard --start 30 --duration 120
  python tools\\voice_sampler.py "https://youtube.com/watch?v=xyz" --name narrator --skip-isolation
  python tools\\voice_sampler.py --list
        """,
    )

    parser.add_argument("url", nargs="?", help="YouTube URL")
    parser.add_argument("--name", help="Voice name (becomes the reference_id)")
    parser.add_argument("--start", type=float, default=0, help="Search window start in seconds (default: 0)")
    parser.add_argument("--duration", type=float, default=120, help="Search window duration in seconds (default: 120)")
    parser.add_argument("--speaker", type=int, default=None, help="Force speaker ID (skip auto-dominant detection). Run once without this to see speaker breakdown.")
    parser.add_argument("--transcript", help="Manual transcript (skips Deepgram)")
    parser.add_argument("--skip-isolation", action="store_true", help="Skip demucs vocal isolation (for clean audio)")
    parser.add_argument("--output", default=str(DEFAULT_REFERENCES_DIR), help="References output directory")
    parser.add_argument("--list", action="store_true", help="List existing voice references")

    args = parser.parse_args()

    if args.list:
        list_voices(Path(args.output))
        return

    if not args.url:
        parser.error("URL is required (or use --list)")
    if not args.name:
        parser.error("--name is required")

    if not re.match(r'^[a-zA-Z0-9_-]+$', args.name):
        parser.error("Voice name must be alphanumeric with dashes/underscores only")

    print(f"\nVoice Sampler: {args.name}")
    print(f"  Source: {args.url}")
    print(f"  Search window: {args.start}s - {args.start + args.duration}s ({args.duration}s)")
    print()

    asyncio.run(process_voice(args))


if __name__ == "__main__":
    main()
