"""
Helpers for turning a position in the audio into something the user can click.

A timestamp is only useful if we can (a) show it in a readable form and
(b) build a YouTube link that jumps straight to that moment.
"""

from urllib.parse import parse_qs, urlparse  # Read the parts of a URL safely


def format_timestamp(seconds) -> str:
    """Turn 214.7 seconds into '3:34', or '1:03:34' for videos over an hour."""
    total = int(seconds or 0)  # Drop the fraction; nobody needs half a second

    hours, remainder = divmod(total, 3600)  # Whole hours, then what is left
    minutes, secs = divmod(remainder, 60)  # Whole minutes, then seconds

    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"  # 1:03:34
    return f"{minutes}:{secs:02d}"  # 3:34


def extract_video_id(source: str):
    """
    Pull the YouTube video ID out of any common YouTube URL shape.

    Returns None for local file paths. That is not an error - it just means we
    cannot build a clickable link, so the UI shows a plain timestamp instead.
    """
    if not source or not source.startswith(("http://", "https://")):
        return None  # A local file path has no video ID

    parsed = urlparse(source)
    host = parsed.netloc.lower().replace("www.", "")

    if host == "youtu.be":  # https://youtu.be/ry9SYnV3svc
        video_id = parsed.path.lstrip("/").split("/")[0]
        return video_id or None

    if "youtube.com" not in host:
        return None  # Some other site we cannot link into

    if parsed.path == "/watch":  # https://youtube.com/watch?v=ry9SYnV3svc
        return parse_qs(parsed.query).get("v", [None])[0]

    parts = [p for p in parsed.path.split("/") if p]  # Non-empty path pieces

    # https://youtube.com/shorts/ID , /embed/ID , /live/ID , /v/ID
    if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
        return parts[1]

    return None


def youtube_link(video_id: str, seconds) -> str:
    """Build a YouTube URL that starts playing at the given second."""
    return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds or 0)}s"

'''
─── Notes ───────────────────────────────────────────────────────────────────
> "timestamps.py" is a small shared helper module. 
> rag_engine.py uses format_timestamp() to label the passages it retrieves, and app.py uses extract_video_id() and youtube_link() to turn those labels into links that jump to the right moment in the video.
> seconds -> "3:34" -> https://www.youtube.com/watch?v=ID&t=214s
'''