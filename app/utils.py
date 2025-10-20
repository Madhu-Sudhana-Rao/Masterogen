import music21
import numpy as np
import io
from scipy.io.wavfile import write as write_wav
from synthesizer import Synthesizer, Waveform
import json
import re
import random
from scipy.signal import butter, lfilter

def note_to_frequencies(note_list):
    freqs = []
    for note_str in note_list:
        try:
            note = music21.note.Note(note_str)
            freqs.append(note.pitch.frequency)
        except:
            continue
    return freqs

def generate_waveform_for_instrument(name):
    name = name.lower()
    if "piano" in name: return Waveform.sine
    if "violin" in name or "strings" in name: return Waveform.sawtooth
    if "flute" in name: return Waveform.triangle
    if "vocals" in name or "choir" in name: return Waveform.sine
    if "guitar" in name or "bass" in name: return Waveform.square
    if "trumpet" in name or "sax" in name: return Waveform.square
    if "drum" in name or "percussion" in name: return Waveform.sawtooth
    return Waveform.sine

def parse_music_json(composition_text):
    try:
        json_text = re.search(r'\{.*\}', composition_text, re.DOTALL).group(0)
        return json.loads(json_text.replace("'", '"'))
    except:
        return {"instruments": []}

def vary_melody(freqs):
    if not freqs:
        return []
    varied = []
    for f in freqs:
        shift = random.choice([0.5, 1, 2])
        varied.append(f * shift if random.random() > 0.7 else f)
    random.shuffle(varied)
    return varied

def lowpass_filter(data, cutoff=8000, fs=44100, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def apply_fade(audio, fade_len=2000):
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    audio[:fade_len] *= fade_in
    audio[-fade_len:] *= fade_out
    return audio

def generate_multi_instrument_audio(composition_text):
    data = parse_music_json(composition_text)
    audio_parts = []
    sample_rate = 44100
    for inst in data.get("instruments", []):
        name = inst.get("name", "Unknown")
        notes = inst.get("melody", "").split()
        freqs = note_to_frequencies(notes)
        if not freqs:
            continue
        waveform = generate_waveform_for_instrument(name)
        synth = Synthesizer(osc1_waveform=waveform, osc1_volume=0.8, use_osc2=False)
        note_duration = 0.5
        audio = np.array([])
        total_time = 0
        while total_time < 90:
            section = vary_melody(freqs)
            if not section:
                break
            part = np.concatenate([synth.generate_constant_wave(f, note_duration) for f in section])
            audio = np.concatenate((audio, part))
            total_time += len(section) * note_duration
        if len(audio) > 0:
            audio_parts.append(audio)
    if not audio_parts:
        return b""
    min_len = min(len(a) for a in audio_parts if len(a) > 0)
    mix = np.mean([a[:min_len] for a in audio_parts if len(a) > 0], axis=0)
    if len(mix) == 0 or np.max(np.abs(mix)) == 0:
        return b""
    mix = lowpass_filter(mix)
    mix = apply_fade(mix)
    mix = mix / np.max(np.abs(mix))
    buffer = io.BytesIO()
    write_wav(buffer, sample_rate, (mix * 32767).astype(np.int16))
    return buffer.getvalue()
