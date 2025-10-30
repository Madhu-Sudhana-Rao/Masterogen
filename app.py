import os
import json
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
from app.main import MusicLLM
from app.utils import (
    generate_multi_instrument_audio,
    musicxml_from_composition,
    synthesize_voice_if_available,
    generate_music_with_musicgen,
    combine_audio_tracks,
    generate_ambience
)

load_dotenv()

st.set_page_config(page_title="MAESTROGEN", layout="centered")
st.title("MAESTROGEN")
st.subheader("Realistic AI Music Composer — Groq + MusicGen with Ambience")
st.markdown("Provide a description and optional seed audio. Toggle Realistic Mode to use MusicGen (GPU recommended). Choose ambience type (sampled or procedural).")

input_mode = st.radio("Input mode", ["Text Prompt", "Upload Humming Audio", "Upload Seed Audio"])
music_input = ""
uploaded_seed = None
if input_mode == "Text Prompt":
    music_input = st.text_input("Describe the music you want to compose:")
elif input_mode == "Upload Humming Audio":
    audio_file = st.file_uploader("Upload a humming audio (wav/mp3)", type=["wav", "mp3", "m4a"])
    if audio_file:
        music_input = "user hummed melody attached"
        uploaded_seed = audio_file.read()
elif input_mode == "Upload Seed Audio":
    seed_file = st.file_uploader("Upload seed audio/stem (wav/mp3)", type=["wav", "mp3", "m4a"])
    if seed_file:
        music_input = "seed audio provided for conditioning"
        uploaded_seed = seed_file.read()

style = st.selectbox("Choose a style:", ["Romantic", "Sad", "Happy", "Jazz", "Cinematic", "Extreme"])
language = st.selectbox("Choose lyrics language:", ["English", "Hindi", "Telugu", "Spanish", "French", "None"])
duration = st.slider("Approximate duration (seconds)", 15, 120, 30, 5)

realistic_mode = st.checkbox("Use MusicGen (realistic audio) — GPU recommended", value=True)
ambience_mode = st.radio("Ambience source:", ["sampled", "procedural"])
ambience_kind = st.selectbox("Ambience kind:", ["rain", "wind", "cafe", "city", "none"])

if st.button("Generate Song"):
    if input_mode == "Text Prompt" and not music_input.strip():
        st.warning("Please enter a description.")
    else:
        generator = MusicLLM()
        with st.spinner("Generating composition JSON and lyrics..."):
            composition = generator.generate_music(music_input, style, language, duration)

        style_summary = composition.get("style_summary", "")
        lyrics = composition.get("lyrics", "")
        prompt_text = f"{style} {language} song about: {music_input}. {style_summary} Lyrics: {lyrics[:400]}"

        with st.spinner("Generating instrumental audio..."):
            instr_audio = b""
            if realistic_mode:
                instr_audio = generate_music_with_musicgen(prompt_text, uploaded_seed, duration=duration)
            if not instr_audio:
                instr_audio = generate_multi_instrument_audio(composition, duration=duration)

        with st.spinner("Synthesizing vocals..."):
            vocal_audio = synthesize_voice_if_available(composition, language)

        with st.spinner("Generating ambience..."):
            amb_audio = None
            if ambience_kind != "none":
                amb_audio = generate_ambience(mode=ambience_mode, kind=ambience_kind, duration=duration)

        with st.spinner("Mixing final track..."):
            final_audio = combine_audio_tracks(instr_audio, vocal_audio, amb_audio)

        if final_audio:
            st.audio(BytesIO(final_audio), format="audio/wav")
            st.success("Final mixed track ready.")
            st.download_button("Download Final Mix", data=BytesIO(final_audio), file_name="maestrogen_final_mix.wav", mime="audio/wav")
        else:
            st.error("Could not generate final audio.")

        if instr_audio:
            st.download_button("Download Instrumental", data=BytesIO(instr_audio), file_name="maestrogen_instrumental.wav", mime="audio/wav")
        if vocal_audio:
            st.download_button("Download Vocals", data=BytesIO(vocal_audio), file_name="maestrogen_vocals.wav", mime="audio/wav")

        sheet = musicxml_from_composition(composition)
        if sheet:
            st.download_button("Download Sheet (MusicXML)", data=BytesIO(sheet), file_name="maestrogen_sheet.musicxml", mime="application/xml")
        else:
            st.error("Sheet music could not be generated.")

        st.subheader("Lyrics")
        lyrics_text = composition.get("lyrics", "") if isinstance(composition, dict) else ""
        st.text_area("Generated Lyrics", value=lyrics_text, height=240)

        st.subheader("Composition JSON")
        st.json(composition)
