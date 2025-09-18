import os
from dotenv import load_dotenv
import assemblyai as aai

load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

transcriber = aai.Transcriber()

AUDIO_FILE = "dollar_drop_prediction.mp3"

print(" Transcription en cours...")
transcript = transcriber.transcribe(AUDIO_FILE)

print("\n--- Transcription ---\n")
print(transcript.text)
