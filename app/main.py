import os
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

class MusicLLM:
    def __init__(self, temperature=0.7):
        self.llm = ChatGroq(
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"
        )

    def generate_music(self, user_input, style):
        prompt = ChatPromptTemplate.from_template("""
You are a professional music composer AI.
Generate a multi-instrument composition for the described mood and scenario.

INPUT:
User Description: {input}
Style/Mood: {style}

OUTPUT FORMAT:
Return a structured JSON-like text with the following fields:
{{
  "instruments": [
    {{"name": "Piano", "melody": ""}},
    {{"name": "Violin", "melody": ""}},
    {{"name": "Flute", "melody": ""}},
    {{"name": "Vocals", "melody": ""}}
  ],
  "chords": "",
  "rhythm": "",
  "style_summary": ""
}}

RULES:
- Use instruments appropriate for the style.
- Provide realistic note ranges per instrument.
- Include at least 3 instruments.
- Romantic → Piano, Violin, Vocals
- Extreme →  Guitar, Drums, Bass
- Jazz → Saxophone, Piano, Trumpet, Bass
- Sad → Cello, Piano, Flute
- Happy → Ukulele, Guitar, Flute
- Cinematic → Strings, Choir, Percussion
Return only the JSON-like text, no explanations.
""")
        chain = prompt | self.llm
        return chain.invoke({"input": user_input, "style": style}).content.strip()
