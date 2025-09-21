import os
from pathlib import Path
from typing import Final

import assemblyai as aai
from dotenv import load_dotenv


def get_api_key() -> str:
    """Return the AssemblyAI API key from the environment."""

    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing ASSEMBLYAI_API_KEY environment variable. Set it before running the script."
        )
    return api_key


def main() -> None:
    load_dotenv()
    aai.settings.api_key = get_api_key()

    transcriber = aai.Transcriber()

    audio_file: Final[Path] = Path("dollar_drop_prediction.mp3")

    print(" Transcription en cours...")
    transcript = transcriber.transcribe(str(audio_file))

    print("\n--- Transcription ---\n")
    print(transcript.text)


if __name__ == "__main__":
    main()
