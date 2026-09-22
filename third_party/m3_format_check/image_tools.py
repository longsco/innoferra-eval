"""
M3 test helpers: any-size single-color PNG byte stream generator + preset video fixture paths

Design goals:
- Zero third-party dependencies (no Pillow), consistent with helpers.py:make_png_1x1 style
- Single-color PNG generation; IDAT uses zlib compression; a 4000x3000 PNG is only a few KB
- Video fixtures use preset file paths; fixed-resolution mp4 files live under fixtures/

helpers.py is left untouched; this file is standalone and imported by test_resolution_tier.py.
"""
import base64
import struct
import zlib
from pathlib import Path
from typing import Optional


# ---------- PNG generator ----------

def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Generate a single PNG chunk (mirrors the _chunk style in helpers.make_png_1x1)."""
    payload = chunk_type + data
    crc = struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    return struct.pack(">I", len(data)) + payload + crc


def make_png_bytes(width: int, height: int, r: int = 255, g: int = 0, b: int = 0) -> bytes:
    """
    Generate a single-color RGB PNG byte stream of width x height pixels.

    Implementation notes:
    - color type = 2 (RGB, same as make_png_1x1)
    - bit depth = 8
    - Each row starts with one filter byte (0x00 = None), followed by width RGB triplets
    - The full raw data goes through zlib.compress; single-color images compress extremely well — a 4000x3000 PNG is roughly 8-12KB
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid dimensions: {width}x{height}")

    header = b"\x89PNG\r\n\x1a\n"
    # IHDR: width, height, bit_depth=8, color_type=2 (RGB), compression=0, filter=0, interlace=0
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    pixel = bytes([r, g, b])
    row = b"\x00" + pixel * width  # filter byte + RGB pixels
    raw = row * height
    idat = zlib.compress(raw, level=9)

    return header + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def make_png_base64(width: int, height: int, r: int = 255, g: int = 0, b: int = 0) -> str:
    """data URL wrapper around make_png_bytes."""
    return "data:image/png;base64," + base64.b64encode(make_png_bytes(width, height, r, g, b)).decode()


# ---------- Video fixtures ----------

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def fixture_path(name: str) -> Path:
    """Absolute path of a file under fixtures/."""
    return FIXTURES_DIR / name


def fixture_video_base64(name: str, mime: str = "video/mp4") -> str:
    """Read a video file under fixtures/ and return base64 in data URL form."""
    p = fixture_path(name)
    if not p.exists():
        raise FileNotFoundError(
            f"fixture video not found: {p}\n"
            f"generate the fixture first (see fixtures/README.md)"
        )
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


# A handful of preset fixed-resolution fixtures (see fixtures/README.md for details)
# Filename → (width, height, purpose description)
VIDEO_FIXTURES = {
    "video_400x300.mp4":   (400, 300,   "low tier — no rescaling"),
    "video_640x480.mp4":   (640, 480,   "default tier — no rescaling"),
    "video_1280x720.mp4":  (1280, 720,  "high tier no rescaling / low tier rescaling"),
    "video_1920x1080.mp4": (1920, 1080, "default tier — rescaling"),
    "video_3840x2160.mp4": (3840, 2160, "high tier — rescaling"),
}


# COS backup links (see fixtures/README.md for details); the URL-series cases in test_resolution_tier.py go through this path
COS_VIDEO_BASE = "https://qa-tool-1315599187.cos.ap-shanghai.myqcloud.com/m3-test"


def fixture_video_url(name: str) -> str:
    """Return the COS direct link for a video file under fixtures/ (used when passing video_url.url directly as a URL)."""
    if name not in VIDEO_FIXTURES:
        raise KeyError(f"unknown fixture: {name}; known: {list(VIDEO_FIXTURES)}")
    return f"{COS_VIDEO_BASE}/{name}"
