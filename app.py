import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
from app.main import MusicLLM
from app.utils import generate_multi_instrument_audio

load_dotenv()

st.set_page_config(page_title="MAESTROGEN", layout="centered")

st.title("MAESTROGEN")
st.subheader("An AI-Powered Dual Output System for Emotion-Aware Music and Sheet Composition using Text Prompts")
st.markdown("Generate AI Music by describing the style and scenario.")

music_input = st.text_input("Describe the music you want to compose:")
style = st.selectbox("Choose a style:", ["Sad", "Happy", "Jazz", "Romantic", "Extreme", "Cinematic"])

if st.button("Generate Music"):
    if not music_input.strip():
        st.warning("Please enter a music description before generating.")
    else:
        generator = MusicLLM()
        with st.spinner("Generating multi-instrument composition..."):
            composition = generator.generate_music(music_input, style)
            audio_data = generate_multi_instrument_audio(composition)

        if audio_data:
            st.audio(BytesIO(audio_data), format="audio/wav")
            st.success("Music generated successfully.")
            with st.expander("Composition Summary"):
                st.text(composition)
        else:
            st.error("No audio data was generated. Please try again.")
