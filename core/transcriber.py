import os  # Import os for file operations
import requests  # Import requests to make API calls to Sarvam AI
import whisper  # Import OpenAI Whisper for speech-to-text
from pydub import AudioSegment  # Import AudioSegment for splitting audio into smaller pieces

from core.config import (  # Import settings from the central config (this loads .env first)
    SARVAM_API_KEY,
    SARVAM_STT_MODEL,
    SARVAM_STT_TRANSLATE_URL,
    WHISPER_MODEL,
    require_key,
)

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25

_model = None  # store the whisper model so it loads only once


def load_model():

    global _model  # Access the global Whisper model variable

    if _model is None:  # Load the model only if it hasn't been loaded yet
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)  # Load the selected Whisper model
        print("Whisper model loaded.")
    return _model  # Return the loaded model


def transcribe_chunk_whisper(chunk_path: str, offset: float = 0.0) -> list:
    """
    Transcribe one chunk with Whisper and return a list of timed segments.

    Whisper already breaks speech into short segments and reports a start and
    end time for each one. The old code threw all of that away by reading only
    result["text"]. We keep it, and add the chunk's offset so the times refer
    to the whole video rather than to this chunk.
    """
    model = load_model()  # Get the Whisper model

    result = model.transcribe(chunk_path, task="transcribe")  # Convert audio chunk into text

    segments = []  # Collect one record per spoken segment

    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()

        if not text:  # Whisper sometimes emits empty segments during silence
            continue

        segments.append(
            {
                "start": offset + float(seg["start"]),  # Position in the whole video
                "end": offset + float(seg["end"]),
                "text": text,
            }
        )

    if not segments:  # Very short audio can come back with text but no segments
        whole = result.get("text", "").strip()
        if whole:
            segments.append({"start": offset, "end": offset, "text": whole})

    return segments  # Return the timed segments


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}  # Add API key to request header

    with open(piece_path, "rb") as f:  # open the audio file in binary mode
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}  # prepare audio file for upload
        data = {"model": SARVAM_STT_MODEL, "with_diarization": "false"}  # API request parameters
        response = requests.post(  # send audio to sarvam API
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:  # check whether the request was successful
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()  # Raise an exception if the request failed

    return response.json().get("transcript", "")  # return the transcript from the API


def transcribe_chunk_sarvam(chunk_path: str, offset: float = 0.0) -> list:
    """
    Transcribe one chunk with Sarvam and return a list of timed segments.

    Sarvam's sync API returns plain text with no timing information, so we
    cannot get segment-level times the way Whisper gives them. What we can do
    is use the piece boundaries we created ourselves: each 25-second piece
    becomes one segment. Timestamps are therefore accurate to about 25 seconds,
    which is still close enough to jump to the right place in a video.
    """
    require_key(SARVAM_API_KEY, "SARVAM_API_KEY")  # Fail early with a clear message if the key is missing

    audio = AudioSegment.from_wav(chunk_path)  # Load the WAV file
    piece_ms = SARVAM_PIECE_SECONDS * 1000  # convert piece duration into milliseconds

    segments = []  # Collect one record per piece
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms  # calculate total number of pieces

    for i, start_ms in enumerate(range(0, len(audio), piece_ms)):  # Loop through every audio piece
        piece = audio[start_ms: start_ms + piece_ms]  # Extract one audio piece
        piece_path = f"{chunk_path}_sv_{i}.wav"  # Create filename for the piece
        piece.export(piece_path, format="wav")  # Save the piece as a WAV file

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")  # show current processing status
            text = _send_to_sarvam(piece_path).strip()  # send piece to sarvam and read transcript
        finally:
            if os.path.exists(piece_path):  # Delete temporary audio piece after processing
                os.remove(piece_path)

        if text:  # Skip silent pieces
            piece_start = offset + (start_ms / 1000)  # Position in the whole video
            segments.append(
                {
                    "start": piece_start,
                    "end": piece_start + (len(piece) / 1000),
                    "text": text,
                }
            )

    return segments  # Return the timed segments


def transcribe_chunk(chunk_path: str, offset: float = 0.0, language: str = "english") -> list:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  → Whisper (local model, segment-level timestamps)
    - hinglish → Sarvam (translates to English, 25-second timestamps)
    """
    if language.lower() == "hinglish":  # Use sarvam for Hinglish
        return transcribe_chunk_sarvam(chunk_path, offset)
    return transcribe_chunk_whisper(chunk_path, offset)  # Use whisper for english


def transcribe_all(chunks: list, language: str = "english") -> list:
    """
    Transcribe every chunk and return one combined list of timed segments.

    "chunks" is a list of AudioChunk records from audio_processor, each holding
    a file path and the second at which that file starts in the video.
    """
    segments = []  # Store the complete list of timed segments

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"  # select engine name for display
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks, start=1):  # process every audio chunk

        print(f"Transcribing chunk {i}/{len(chunks)}...")

        segments.extend(transcribe_chunk(chunk.path, chunk.start, language))  # Add this chunk's segments

    print(f"Transcription complete — {len(segments)} timed segment(s).")

    return segments  # Return every timed segment, in order


def segments_to_text(segments: list) -> str:
    """
    Join timed segments back into one plain transcript.

    The summariser and extractor only need the words, so they use this. The
    vector store keeps the segments, because that is where the timing has to
    survive.
    """
    return " ".join(seg["text"] for seg in segments).strip()

'''
# ─── Notes ───────────────────────────────────────────────────────────────────
"transcriber.py" converts the processed audio chunks into text. 
> It uses local Whisper for English and Sarvam AI for Hinglish, splitting audio into smaller pieces when the Sarvam API requires it.
> It returns TIMED SEGMENTS - a list of {"start", "end", "text"} records whose times refer to the whole video, not to an individual chunk That is what lets the RAG engine cite a moment and the UI link to it. Use segments_to_text() when you just want the words.
> All settings (model names, API key, endpoint) come from core/config.py, which guarantees the .env file is loaded before any value is read.
> Audio chunks -> Speech-to-text -> timed segments -> vector store / summariser

'''
