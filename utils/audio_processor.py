import os  # Import os for file operations
from typing import NamedTuple  # Build a tiny record type with named fields

import yt_dlp  # Download videos or audio from a YouTube link
from pydub import AudioSegment  # AudioSegment can read audio, convert formats, trim and split audio

from core.config import PROJECT_ROOT  # Absolute path to the VIDEO_AI folder

from contextlib import contextmanager  # Lets us build a "with" block that always cleans up

# Absolute path, so audio always lands in the same place no matter which folder
# you start the app from.
DOWNLOAD_DIR = PROJECT_ROOT / "downloades"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)  # Create the folder once; do nothing if it exists

CHUNK_MINUTES = 10  # Default chunk length in minutes


class AudioChunk(NamedTuple):
    """
    One piece of audio, plus where that piece starts in the original video.

    The start time is the whole point. Whisper reports timestamps relative to
    the file you hand it, so a word 30 seconds into chunk 2 is really at
    10 minutes 30 seconds in the video. Without this offset every timestamp
    after the first chunk would be wrong.
    """

    path: str  # WAV file on disk
    start: float  # Seconds from the beginning of the video


def download_youtube_audio(url: str) -> str:
    """Download the audio track of a YouTube video and return the WAV file path."""
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {  # Options dictionary for yt-dlp
        "format": "bestaudio/best",  # Download the highest-quality audio available
        "outtmpl": output_path,  # %(title)s -> video title, %(ext)s -> file extension
        "restrictfilenames": True,  # Strip spaces/emojis/symbols that break Windows paths
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",  # After downloading, use FFmpeg to extract audio
                "preferredcodec": "wav",  # Convert to WAV, because speech-to-text works best with WAV
                "preferredquality": "192",  # Target audio quality in kbps
            }
        ],
        "quiet": True,  # Keep the terminal clean
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)  # Download the audio
        downloaded = ydl.prepare_filename(info)  # Path yt-dlp used BEFORE conversion

    # FFmpeg always writes a .wav file, whatever the original container was
    # (.webm, .m4a, .opus, .mp4 ...). splitext removes whatever extension is
    # actually there instead of guessing at specific ones.
    wav_path = os.path.splitext(downloaded)[0] + ".wav"

    if not os.path.exists(wav_path):  # Fail loudly instead of returning a bad path
        raise FileNotFoundError(
            f"Expected converted audio at {wav_path} but it was not created. "
            "Is FFmpeg installed and on your PATH?"
        )

    return wav_path  # Return the WAV file path


def convert_to_wav(input_path: str) -> str:
    """Convert any local audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"  # Name for the converted copy

    audio = AudioSegment.from_file(input_path)  # Load the original file
    audio = audio.set_channels(1).set_frame_rate(16000)  # Mono, 16 kHz - what speech models expect
    audio.export(output_path, format="wav")  # Save the processed audio as a WAV file

    return output_path  # Return the converted WAV file path


def chunk_audio(wav_path: str, chunk_minutes: int = CHUNK_MINUTES) -> list:
    """Split a WAV file into fixed-length chunks, each tagged with its start time."""
    # Load the WAV and normalise it to mono 16 kHz - exactly what speech-to-text
    # models expect. YouTube audio arrives as 48 kHz stereo, which is about 12x
    # larger for no accuracy benefit, so this shrinks the chunks a lot and makes
    # transcription faster.
    audio = AudioSegment.from_wav(wav_path).set_channels(1).set_frame_rate(16000)
    chunk_ms = chunk_minutes * 60 * 1000  # Convert chunk duration from minutes to milliseconds

    base = os.path.splitext(wav_path)[0]  # Drop the .wav so names stay readable
    chunks = []  # Store an AudioChunk for every generated piece

    for i, start_ms in enumerate(range(0, len(audio), chunk_ms)):  # Split into fixed-size chunks
        piece = audio[start_ms: start_ms + chunk_ms]  # Extract one chunk
        chunk_path = f"{base}_chunk_{i:03d}.wav"  # e.g. my_video_chunk_000.wav
        piece.export(chunk_path, format="wav")  # Save the chunk as a WAV file

        chunks.append(AudioChunk(path=chunk_path, start=start_ms / 1000))  # Record where it starts

    return chunks  # Return the list of AudioChunk records


def cleanup_files(paths) -> None:
    """Delete temporary files. Missing files are ignored, errors are only warned about."""
    removed = 0

    for path in paths:
        if not path:  # Skip None entries
            continue
        if os.path.exists(path):
            try:
                os.remove(path)
                removed += 1
            except OSError as e:  # File locked by another program, permissions, etc.
                print(f"⚠️  Could not delete {path}: {e}")

    if removed:
        print(f"Cleaned up {removed} temporary audio file(s).")


def _source_to_wav(source: str) -> str:
    """Turn either a URL or a local file path into one standard WAV file."""
    if source.startswith("http://") or source.startswith("https://"):  # Input is a URL
        print("Detected YouTube URL. Downloading audio...")
        return download_youtube_audio(source)

    print("Detected local file. Converting to WAV...")
    return convert_to_wav(source)  # Input is a local file


@contextmanager
def prepared_audio(source: str, chunk_minutes: int = CHUNK_MINUTES, keep_audio: bool = False):
    """
    Prepare audio chunks for transcription, then always delete them afterwards.

    Usage:
        with prepared_audio(source) as chunks:
            segments = transcribe_all(chunks, language)

    Yields a list of AudioChunk records, each carrying its start time in the
    video. The files are deleted when the "with" block ends - even if
    transcription raised an error - so temporary audio can never pile up.

    Set keep_audio=True while debugging if you want to inspect the WAV files.
    """
    wav_path = None  # The full WAV file (downloaded or converted)
    chunks = []  # The chunk records created from it

    try:
        wav_path = _source_to_wav(source)  # Download or convert

        print("Chunking audio...")
        chunks = chunk_audio(wav_path, chunk_minutes)  # Split into timed chunks
        print(f"Audio ready — {len(chunks)} chunk(s) created.")

        yield chunks  # Hand the chunks to the caller's "with" block

    finally:
        if keep_audio:
            print(f"keep_audio=True — leaving files in {DOWNLOAD_DIR}")
        else:
            # AudioChunk is a tuple, so pull the path out of each one
            cleanup_files([c.path for c in chunks] + [wav_path])


def process_input(source: str) -> list:
    """
    Old entry point: returns AudioChunk records WITHOUT deleting them afterwards.

    Kept so existing code keeps working. Prefer prepared_audio() instead,
    otherwise you are responsible for calling cleanup_files() yourself.
    """
    wav_path = _source_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")

    return chunks


'''
─── Notes ───────────────────────────────────────────────────────────────────
"audio_processor.py" handles the input audio. 
> It downloads audio from YouTube or converts a local file into a standard WAV, then splits that WAV into smaller chunks so the transcription model can process it efficiently. 
> Each chunk is returned as an AudioChunk(path, start). The start time lets transcriber.py convert Whisper's chunk-relative timestamps into real positions in the video, which is what makes clickable citations possible. 
> prepared_audio() is the recommended entry point: it is a context manager, so the temporary WAV files are deleted automatically when the "with" block finishes - even if an error happened inside it. 
YouTube URL -> Download or Convert -> WAV -> timed chunks -> transcriber.py -> delete
'''
