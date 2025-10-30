import io
import json
import os
import random
import tempfile
import numpy as np
from scipy.io.wavfile import write as write_wav
from scipy.signal import butter, lfilter
from synthesizer import Synthesizer, Waveform
import music21
from pathlib import Path
from typing import Dict, Any, Optional


# --------------------- JSON UTILITIES ---------------------

def parse_music_json(composition_text: Dict[str, Any]):
    if isinstance(composition_text, dict):
        return composition_text
    try:
        if isinstance(composition_text, str):
            return json.loads(composition_text)
    except:
        pass
    return {"instruments": [], "chords": [], "rhythm": "", "lyrics": "", "vocals_melody": []}


# --------------------- AUDIO BASICS ---------------------

def note_to_frequencies(note_list):
    freqs = []
    for note_str in note_list:
        try:
            n = music21.note.Note(note_str)
            freqs.append(n.pitch.frequency)
        except:
            continue
    return freqs


def generate_waveform_for_instrument(name: str):
    name = (name or "").lower()
    if "piano" in name: return Waveform.sine
    if "violin" in name or "strings" in name: return Waveform.sawtooth
    if "flute" in name: return Waveform.triangle
    if "vocals" in name or "choir" in name: return Waveform.sine
    if "guitar" in name or "bass" in name: return Waveform.square
    if "trumpet" in name or "sax" in name: return Waveform.square
    if "drum" in name or "percussion" in name: return Waveform.sawtooth
    if "synth" in name: return Waveform.square
    return Waveform.sine


def vary_melody(freqs, length=8):
    if not freqs:
        return []
    varied = []
    for _ in range(length):
        f = float(random.choice(freqs))
        shift = random.choice([0.5, 1, 2, 1.5, 0.75])
        varied.append(f * shift if random.random() > 0.6 else f)
    return varied


def lowpass_filter(data, cutoff=8000, fs=44100, order=5):
    if data is None or len(data) == 0:
        return data
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)


def apply_fade(audio, fade_len=2048):
    if audio is None or len(audio) < fade_len * 2:
        return audio
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    audio[:fade_len] *= fade_in
    audio[-fade_len:] *= fade_out
    return audio


def mix_audio_parts(parts):
    valid = [p for p in parts if isinstance(p, np.ndarray) and p.size > 0]
    if not valid:
        return np.array([])
    min_len = min(len(p) for p in valid)
    parts_trimmed = [p[:min_len] for p in valid]
    mix = np.mean(parts_trimmed, axis=0)
    return mix


# --------------------- MUSIC SYNTHESIS ---------------------

def generate_multi_instrument_audio(composition_text, duration=30, sample_rate=44100):
    data = parse_music_json(composition_text)
    audio_parts = []
    target_len = int(duration * sample_rate)

    for inst in data.get("instruments", []):
        name = inst.get("name", "Instrument")
        melody = inst.get("melody", "")
        notes = melody.split() if isinstance(melody, str) else melody
        freqs = note_to_frequencies(notes)
        if not freqs:
            continue

        waveform = generate_waveform_for_instrument(name)
        synth = Synthesizer(osc1_waveform=waveform, osc1_volume=0.7, use_osc2=False)
        part = np.array([], dtype=np.float32)
        elapsed = 0.0
        base_note_duration = max(0.125, duration / max(8, len(freqs)))

        while elapsed < duration:
            seq = vary_melody(freqs, length=8)
            waves = []
            for f in seq:
                try:
                    wave = synth.generate_constant_wave(float(f), base_note_duration)
                except Exception:
                    continue
                waves.append(np.array(wave, dtype=np.float32))
                elapsed += base_note_duration
                if elapsed >= duration:
                    break
            if waves:
                section = np.concatenate(waves)
                part = np.concatenate((part, section)) if part.size else section
            if len(part) >= target_len:
                break

        if part.size:
            audio_parts.append(part[:target_len])

    if not audio_parts:
        return b""

    mix = mix_audio_parts(audio_parts)
    if mix.size == 0 or np.max(np.abs(mix)) == 0:
        return b""

    mix = lowpass_filter(mix, fs=sample_rate)
    mix = apply_fade(mix, fade_len=min(2048, len(mix)//8))
    mix = mix / np.max(np.abs(mix))
    mix = np.clip(mix, -1.0, 1.0)

    buffer = io.BytesIO()
    write_wav(buffer, sample_rate, (mix * 32767).astype(np.int16))
    buffer.seek(0)
    return buffer.read()


# --------------------- MUSICGEN + TTS ---------------------

def _bytes_to_tempfile(b: bytes, suffix=".wav"):
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(b)
    tf.flush()
    tf.close()
    return tf.name


def generate_music_with_musicgen(prompt_text: str, seed_wav_bytes: Optional[bytes], duration: int = 30, model_name: str = "facebook/musicgen-medium"):
    try:
        from audiocraft.models import MusicGen
        model = MusicGen.get_pretrained(model_name)
        model.set_generation_params(duration=duration)
        if seed_wav_bytes:
            tmp = _bytes_to_tempfile(seed_wav_bytes, suffix=".wav")
            wavs = model.generate([prompt_text], audio_prompts=[tmp])
        else:
            wavs = model.generate([prompt_text])
        out = wavs[0]
        out = out.cpu().numpy() if hasattr(out, "cpu") else np.array(out)
        buffer = io.BytesIO()
        write_wav(buffer, 32000, (out * 32767).astype(np.int16))
        buffer.seek(0)
        return buffer.read()
    except Exception:
        try:
            comp = json.loads(prompt_text) if isinstance(prompt_text, str) and prompt_text.strip().startswith("{") else None
        except:
            comp = None
        if comp:
            return generate_multi_instrument_audio(comp, duration=duration, sample_rate=32000)
        return b""


def groq_tts_synthesize(text: str, voice: str = "alloy"):
    import requests
    key = os.getenv("GROQ_API_KEY")
    if not key or not text:
        return b""
    url = "https://api.groq.com/openai/v1/audio/speech"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": "gpt-voice-1", "voice": voice, "input": text}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        if r.status_code == 200:
            return r.content
    except:
        pass
    return b""


def synthesize_voice_if_available(composition_text, language: str = "English"):
    lyrics = ""
    if isinstance(composition_text, dict):
        lyrics = composition_text.get("lyrics", "")
    elif isinstance(composition_text, str):
        try:
            j = json.loads(composition_text)
            lyrics = j.get("lyrics", "")
        except:
            lyrics = ""
    if not lyrics:
        return b""

    audio = groq_tts_synthesize(lyrics)
    if audio:
        return audio

    key = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_LABS_KEY")
    if not key:
        return b""
    try:
        import requests
        url = "https://api.elevenlabs.io/v1/text-to-speech/default"
        headers = {"xi-api-key": key, "Content-Type": "application/json"}
        payload = {"text": lyrics, "voice": "alloy", "accept": "audio/wav"}
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code == 200:
            return resp.content
    except:
        pass
    return b""


# --------------------- AMBIENCE GENERATION ---------------------

def _load_sampled_ambience(folder: str, duration: int = 30, sample_rate: int = 32000):
    files = []
    if not os.path.isdir(folder):
        return b""
    for f in os.listdir(folder):
        if f.lower().endswith(".wav"):
            files.append(os.path.join(folder, f))
    if not files:
        return b""

    parts = []
    import soundfile as sf
    import librosa

    for p in files:
        try:
            data, sr = sf.read(p)
            if sr != sample_rate:
                data = librosa.resample(y=np.mean(data, axis=1) if data.ndim == 2 else data, orig_sr=sr, target_sr=sample_rate)
            parts.append(data)
        except:
            continue

    if not parts:
        return b""

    combined = np.concatenate(
        [p if len(p) >= duration * sample_rate else np.pad(p, (0, max(0, int(duration * sample_rate) - len(p)))) for p in parts],
        axis=0
    )
    combined = combined[:int(duration * sample_rate)]
    combined = combined / np.max(np.abs(combined))
    buf = io.BytesIO()
    write_wav(buf, sample_rate, (combined * 32767).astype(np.int16))
    buf.seek(0)
    return buf.read()


def _procedural_ambience(kind: str = "rain", duration: int = 30, sample_rate: int = 32000):
    tlen = int(duration * sample_rate)
    if kind == "rain":
        noise = np.random.randn(tlen)
        filtered = lowpass_filter(noise, cutoff=6000, fs=sample_rate, order=3)
        envelope = np.abs(np.sin(np.linspace(0, duration * np.pi, tlen))) ** 0.5
        out = filtered * envelope * 0.3
    elif kind == "wind":
        noise = np.random.randn(tlen)
        out = lowpass_filter(noise, cutoff=2000, fs=sample_rate, order=2) * 0.2
    elif kind == "cafe":
        noise = np.random.randn(tlen)
        out = lowpass_filter(noise, cutoff=4000, fs=sample_rate, order=3) * 0.12
    else:
        out = np.random.randn(tlen) * 0.05
    out = out / (np.max(np.abs(out)) + 1e-9)
    buf = io.BytesIO()
    write_wav(buf, sample_rate, (out * 32767).astype(np.int16))
    buf.seek(0)
    return buf.read()


def generate_ambience(mode: str = "sampled", kind: str = "rain", duration: int = 30, sample_rate: int = 32000):
    sampled_folder = os.path.join("assets", "ambient")
    if mode == "sampled":
        sampled = _load_sampled_ambience(sampled_folder, duration=duration, sample_rate=sample_rate)
        if sampled:
            return sampled
        return _procedural_ambience(kind=kind, duration=duration, sample_rate=sample_rate)
    return _procedural_ambience(kind=kind, duration=duration, sample_rate=sample_rate)


# --------------------- AUDIO MIXDOWN ---------------------

def combine_audio_tracks(instr_bytes: bytes, vocal_bytes: bytes, ambience_bytes: bytes = None, target_sr: int = 32000):
    if not instr_bytes and not vocal_bytes and not ambience_bytes:
        return b""

    import soundfile as sf
    import librosa

    def read_bytes(b):
        if not b:
            return None, None
        data, sr = sf.read(io.BytesIO(b))
        if data.ndim == 2:
            data = np.mean(data, axis=1)
        return data.astype(np.float32), sr

    instr_data, instr_sr = read_bytes(instr_bytes)
    vocal_data, vocal_sr = read_bytes(vocal_bytes)
    amb_data, amb_sr = read_bytes(ambience_bytes)

    if instr_data is None and vocal_data is not None:
        instr_data = np.zeros_like(vocal_data)
        instr_sr = vocal_sr
    if vocal_data is None and instr_data is not None:
        vocal_data = np.zeros_like(instr_data)
        vocal_sr = instr_sr
    if instr_data is None and amb_data is not None:
        instr_data = np.zeros_like(amb_data)
        instr_sr = amb_sr

    # ✅ librosa >= 0.10 compatible resampling
    if instr_sr != target_sr:
        instr_data = librosa.resample(y=instr_data, orig_sr=instr_sr, target_sr=target_sr)
    if vocal_sr != target_sr:
        vocal_data = librosa.resample(y=vocal_data, orig_sr=vocal_sr, target_sr=target_sr)
    if amb_data is not None and amb_sr != target_sr:
        amb_data = librosa.resample(y=amb_data, orig_sr=amb_sr, target_sr=target_sr)

    min_len = min([len(x) for x in [instr_data, vocal_data] if x is not None])
    instr = instr_data[:min_len]
    vocal = vocal_data[:min_len]
    mix = instr * 0.9 + vocal * 0.7

    if amb_data is not None:
        amb = amb_data[:min_len] if len(amb_data) >= min_len else np.pad(amb_data, (0, max(0, min_len - len(amb_data))))
        mix = mix + amb * 0.4

    maxv = np.max(np.abs(mix)) if np.max(np.abs(mix)) != 0 else 1.0
    mix = np.clip(mix / maxv, -1.0, 1.0)

    buf = io.BytesIO()
    write_wav(buf, target_sr, (mix * 32767).astype(np.int16))
    buf.seek(0)
    return buf.read()


# --------------------- SHEET MUSIC ---------------------

def musicxml_from_composition(composition_text):
    data = parse_music_json(composition_text)
    s = music21.stream.Score()
    part = music21.stream.Part()
    ts = music21.meter.TimeSignature('4/4')
    part.append(ts)

    tempo_bpm = 70
    if data.get("rhythm"):
        try:
            tempo_bpm = int(data.get("rhythm"))
        except:
            tempo_bpm = 70
    part.append(music21.tempo.MetronomeMark(number=tempo_bpm))

    for inst in data.get("instruments", []):
        melody = inst.get("melody", "")
        notes = melody.split() if isinstance(melody, str) else melody
        for n in notes:
            try:
                nn = music21.note.Note(n)
                part.append(nn)
            except:
                continue

    s.append(part)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".musicxml") as tf:
            temp_path = tf.name
        s.write('musicxml', fp=temp_path)
        with open(temp_path, 'rb') as f:
            xml_bytes = f.read()
        os.remove(temp_path)
        return xml_bytes
    except:
        return b""
