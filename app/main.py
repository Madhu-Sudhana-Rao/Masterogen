import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

class MusicLLM:
    def __init__(self, temperature: float = 0.4):
        self.llm = ChatGroq(
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        )

    def generate_music(self, user_input: str, style: str, language: str = "English", duration: int = 30) -> dict:
        lines_expected = max(1, int(duration / 5))
        instr_defaults = ["Piano", "Violin", "Acoustic Guitar", "Drums"]

        # ✅ DOUBLE-BRACED JSON example
        prompt_text = f"""
You are MAESTROGEN — an advanced AI music composer that generates rich, emotionally intelligent songs with proper instrumental arrangement and singable lyrics.

INPUT:
Description: {{input}}
Style/Mood: {{style}}
Language: {{language}}
DurationSeconds: {{duration}}

TASK:
Compose a song for this situation that feels natural and emotionally deep.
Select instruments suitable for {{style}} and ensure a multi-layered composition (melody, harmony, rhythm).
Lyrics should be poetic, connected to the mood, and written in {{language}}.
Each line of lyrics should correspond roughly to one bar of music (~{lines_expected} lines).

OUTPUT (STRICT VALID JSON ONLY):
{{{{ 
  "instruments": [
    {{{{ "name":"{instr_defaults[0]}", "melody":["C4","E4","G4","A4","F4"] }}}},
    {{{{ "name":"{instr_defaults[1]}", "melody":["G3","A3","B3","C4","D4"] }}}},
    {{{{ "name":"{instr_defaults[2]}", "melody":["E4","D4","C4","B3","A3"] }}}},
    {{{{ "name":"{instr_defaults[3]}", "melody":["Kick","Snare","HiHat"] }}}}
  ],
  "chords": ["Cmaj7","Am7","Dm7","G7"],
  "rhythm": "90",
  "lyrics": "line1\\nline2\\nline3 ... (approx {lines_expected} lines)",
  "vocals_melody": ["E4","F4","G4","A4","B4","C5"],
  "style_summary": "short description of the musical approach and emotions"
}}}}

RULES:
- Use at least 4 instruments appropriate to the style.
- Lyrics should be singable and roughly {lines_expected} lines.
- vocals_melody should align with lyrical phrasing.
- Rhythm must be integer BPM.
Return ONLY valid JSON.
"""

        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | self.llm

        res = chain.invoke({
            "input": user_input or "",
            "style": style,
            "language": language,
            "duration": str(duration)
        })

        text = res.content.strip()

        # Attempt JSON parsing
        try:
            return json.loads(text)
        except:
            try:
                start = text.find("{")
                end = text.rfind("}") + 1
                return json.loads(text[start:end])
            except:
                # Fallback in case LLM output fails
                return {
                    "instruments": [
                        {"name": instr_defaults[0], "melody": ["C4", "E4", "G4", "A4"]},
                        {"name": instr_defaults[1], "melody": ["G3", "A3", "B3", "C4"]},
                        {"name": instr_defaults[2], "melody": ["E4", "D4", "C4", "B3"]},
                        {"name": instr_defaults[3], "melody": ["Kick", "Snare", "HiHat"]}
                    ],
                    "chords": ["Cmaj7", "Am7", "Dm7", "G7"],
                    "rhythm": "90",
                    "lyrics": "\n".join(["la"] * lines_expected),
                    "vocals_melody": ["E4", "F4", "G4", "A4"],
                    "style_summary": f"Fallback composition for {style}"
                }
