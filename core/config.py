"""
Central configuration for the whole project.

IMPORTANT: this module calls load_dotenv() BEFORE reading any environment
variable, and every other module imports its settings from here. That
guarantees the .env file is always loaded first, no matter which file
Python happens to import first.
"""

import os  # Read environment variables
from pathlib import Path  # Build file paths that work on any operating system

from dotenv import load_dotenv  # Load variables from the .env file

# Folder that contains core/ and utils/ - i.e. the VIDEO_AI project root.
# __file__ is core/config.py, so .parent is core/ and .parent.parent is VIDEO_AI/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_PATH = PROJECT_ROOT / ".env"  # Absolute path to the .env file

'''
Load the .env file using an absolute path, so it is found even if you start the app from a different folder.
⚠️ This line MUST stay ABOVE the settings below. That ordering is the whole point of this file - it is what fixes the bug.
# '''
load_dotenv(ENV_PATH)


# ─── Transcription settings ──────────────────────────────────────────────────
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # Whisper model size

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")  # Sarvam API key (no safe default)
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")  # Sarvam model name
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"  # API endpoint


# ─── LLM settings ────────────────────────────────────────────────────────────
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")  # Mistral API key
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")  # Mistral model name


def require_key(value, name: str) -> str:
    """Return the key if it exists, otherwise raise a clear, helpful error."""
    if not value:
        raise RuntimeError(
            f"{name} is missing.\n"
            f"Add this line to your .env file at {ENV_PATH}:\n"
            f"    {name}=your_key_here"
        )
    return value