"""Getting broadcast video into frames.

Three steps, each boring on purpose:

    media_videos()   the association's own video feed, which links one
                     YouTube video per broadcast set — no channel scraping
    download()       yt-dlp, capped to 1080p AVC for CPU-only decoding
    extract_frames() ffmpeg at a fixed sample rate into a work directory
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

import yt_dlp

_SITE = "https://www.pbatour.org"
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"


def media_videos(page: int = 1, page_size: int = 12) -> list[dict]:
    """Curated broadcast videos from the official feed: id, title, url."""
    url = f"{_SITE}/en/media/video/ajax/list?curPage={page}&pageRowSize={page_size}"
    request = urllib.request.Request(url, headers={"User-Agent": _UA, "Referer": f"{_SITE}/en/index"})
    with urllib.request.urlopen(request, timeout=25) as response:
        rows = (json.loads(response.read().decode("utf-8")).get("list")) or []
    out = []
    for row in rows:
        match = re.search(r"embed/([\w-]+)", row.get("VIDEO_URL", ""))
        if match:
            out.append({"video_id": match.group(1), "title": row.get("VIDEO_NAME") or row.get("TITLE") or ""})
    return out


def download(video_id: str, dest: Path, max_height: int = 1080) -> Path:
    """Download a YouTube video to ``dest`` (suffix .mp4), progressive AVC only."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    options = {
        "format": f"bv*[height<={max_height}][vcodec^=avc1]/bv*[height<={max_height}]",
        "outtmpl": str(dest),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    return dest


def extract_frames(video: Path, fps: float, out_dir: Path) -> list[Path]:
    """Sample the video at ``fps`` into ``out_dir``; returns frame paths in order."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH — install it (https://ffmpeg.org)")
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         "-vf", f"fps={fps}", "-q:v", "2", str(out_dir / "f_%06d.jpg")],
        check=True,
    )
    return sorted(out_dir.glob("f_*.jpg"))
