#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Media Converter - audio <-> video, with live preview and trimming
==================================================================

Version : 1.0.0
Author  : Mohsen Ghamary Asl


A PyQt5 + pyqtgraph desktop application that converts **any video or audio file**
supported by FFmpeg into **any other audio or video format**:

* video  -> audio : extract / re-encode the sound track (MP3, M4A/AAC, AAC,
  Opus, OGG Vorbis, FLAC, WAV, AIFF, WMA, AC-3),
* audio  -> audio : convert between all of the above,
* video  -> video : remux or re-encode into MP4, MKV, WebM, MOV, AVI, MPG,
  FLV, WMV or TS.

Input formats: MP4, M4V, MOV, MKV, WebM, AVI, TS/M2TS, FLV, WMV, MPG, 3GP,
OGV, VOB, ... and MP3, M4A, AAC, WAV, FLAC, OGG, Opus, WMA, AIFF, AC3, AMR,
APE, MKA, WavPack, ... (anything FFmpeg can demux).

Highlights
----------
* The output-format list is context aware: a target is greyed out (with the
  reason in its tooltip) when the FFmpeg build lacks its encoder, when the
  source is *already* in that format, or when an audio-only file is asked to
  produce a video (and vice versa).
* Every conversion is previewed in a one-line summary before you start, e.g.
  "MP4 source (H264 video + AAC audio) -> MKV : H264 copy + AAC audio".
* "Original quality" preset stream-copies whatever can be copied (video and/or
  audio) and only re-encodes what the target container cannot carry.
* Reads the real properties of the source streams (codec, sample-rate,
  channels, bit-rate, duration, video resolution / frame-rate) with ffprobe.
* Values the target cannot store are converted automatically (e.g. >48 kHz and
  >2 channels for MP3, 48 kHz for Opus) and a notice is written to the log.
* Trim selection with the mouse on the wave-form or with millisecond-precise
  start/end input fields; it applies to audio *and* video conversions.
* Live progress (percentage, processed time and speed) parsed from
  ``ffmpeg -progress``, plus the possibility to cancel at any moment.
* The whole window is scrollable vertically, drag & drop accepts anything.
* Optional head-less command line mode (``python Media_Converter.py --nogui``)
  that uses exactly the same conversion engine as the GUI.

Requirements
------------
    pip install -r requirements.txt      # PyQt5, pyqtgraph, numpy,
                                         # imageio-ffmpeg (bundled FFmpeg binary)
    A system FFmpeg (with ffprobe/ffplay) is recommended; when it is missing
    the app falls back to the binary that ships with ``imageio-ffmpeg``.

Run
---
    python Media_Converter.py                                   # GUI
    python Media_Converter.py --nogui movie.mkv -f mp3 -o out.mp3
    python Media_Converter.py --nogui song.wav  -f flac -o out.flac
    python Media_Converter.py --nogui clip.mp4  -f mkv -o out.mkv
    python Media_Converter.py --nogui clip.mp4 -f mp4v -o out.mp4 --start 5.250 --end 30

Author : Mohsen Ghamary Asl
License: MIT (see LICENSE)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg

try:  # QtMultimedia is used for the "play the selected range" feature
    from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer

    MULTIMEDIA_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the local Qt installation
    QMediaContent = None  # type: ignore[assignment]
    QMediaPlayer = None  # type: ignore[assignment]
    MULTIMEDIA_AVAILABLE = False
from PyQt5.QtCore import (
    QElapsedTimer,
    QRect,
    QSize,
    QThread,
    QTimer,
    Qt,
    QUrl,
    pyqtSignal,
)
from PyQt5.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSlider,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

APP_NAME = "Media Converter"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Mohsen Ghamary Asl"
APP_LICENSE = "MIT"

#: Sample-rate used only to draw the wave-form (kept low so that even very
#: long videos are analysed in a fraction of a second).
PREVIEW_SAMPLE_RATE = 4000
#: Number of horizontal columns (min/max pairs) used for the wave-form plot.
PREVIEW_COLUMNS = 4000

#: Extensions offered in the file dialogs (FFmpeg handles far more than this -
#: the "All files" filter is always available).
VIDEO_EXTENSIONS = (
    ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".flv", ".ts", ".m2ts",
    ".wmv", ".mpg", ".mpeg", ".3gp", ".ogv", ".vob", ".rmvb", ".asf", ".mts",
)
AUDIO_EXTENSIONS = (
    ".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".oga", ".opus", ".wma",
    ".aiff", ".aif", ".alac", ".ac3", ".amr", ".ape", ".mka", ".mp2", ".wv",
    ".ra", ".au", ".caf", ".dsf", ".dff",
)
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS + AUDIO_EXTENSIONS




# ---------------------------------------------------------------------------
# Output formats (audio targets and video targets)
# ---------------------------------------------------------------------------

#: Preset keys shared by every lossy *audio* target, with their labels.
AUDIO_PRESETS: List[Tuple[str, str]] = [
    ("original", "Original quality (copy when possible, else source bit-rate)"),
    ("best", "Maximum quality"),
    ("vbr2", "Very high quality"),
    ("320", "320 kbps"),
    ("256", "256 kbps"),
    ("192", "192 kbps"),
    ("128", "128 kbps"),
]

#: Preset keys used by *video* targets (they control the video encoder).
VIDEO_PRESETS: List[Tuple[str, str]] = [
    ("original", "Original quality (stream copy when possible)"),
    ("best", "Maximum quality (large file)"),
    ("vbr2", "High quality"),
    ("320", "Balanced (recommended)"),
    ("128", "Small file"),
]


@dataclass(frozen=True)
class OutputFormat:
    """Description of one conversion target (either audio-only or video)."""

    key: str
    label: str
    extension: str
    kind: str                                 # "audio" or "video"

    # ---- audio settings (also used by video targets for their audio track)
    audio_codec: str = ""                     # FFmpeg encoder name
    audio_encoder: str = ""                   # encoder that must exist
    audio_extra: Tuple[str, ...] = ()
    #: Source audio codecs that can be stream-copied into this container.
    audio_copyable: Tuple[str, ...] = ()
    #: preset key -> encoder arguments (lossy audio targets)
    audio_quality: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    max_sample_rate: int = 0                  # 0 = unlimited
    max_channels: int = 0                     # 0 = unlimited
    force_sample_rate: int = 0
    bit_rate_cap: int = 320
    lossless: bool = False
    copy_with_trim: bool = True

    # ---- video settings (video targets only)
    video_codec: str = ""                     # FFmpeg encoder name
    video_encoder: str = ""
    video_extra: Tuple[str, ...] = ()         # e.g. -pix_fmt yuv420p
    video_copyable: Tuple[str, ...] = ()      # source video codecs that may be copied
    video_quality: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    video_short: str = ""                     # codec short name (H.264, VP9, ...)
    short: str = ""                           # display name (MP4, MKV, FLAC, ...)

    # ---- helpers ---------------------------------------------------------
    @property
    def is_video(self) -> bool:
        return self.kind == "video"

    @property
    def name(self) -> str:
        """Human friendly name of the target (``MP4``, ``FLAC``, ...)."""
        return self.short or self.key.upper()

    def audio_args_for(self, preset: str) -> Tuple[str, ...]:
        for key, args in self.audio_quality:
            if key == preset:
                return args
        for key, args in self.audio_quality:
            if key == "best":
                return args
        return ()

    def extra_muxer_args(self) -> Tuple[str, ...]:
        """Container specific flags (e.g. fast-start for MP4/MOV/M4A)."""
        if self.key in ("mp4v", "mov", "m4a"):
            return ("-movflags", "+faststart")
        return ()

    def video_args_for(self, preset: str) -> Tuple[str, ...]:
        for key, args in self.video_quality:
            if key == preset:
                return args
        for key, args in self.video_quality:
            if key == "vbr2":
                return args
        return ()


# --- reusable building blocks ---------------------------------------------

_X264_QUALITY = (
    ("best", ("-crf", "18", "-preset", "medium")),
    ("vbr2", ("-crf", "21", "-preset", "medium")),
    ("320", ("-crf", "23", "-preset", "medium")),
    ("128", ("-crf", "28", "-preset", "medium")),
)
_X264_ARGS = ("-pix_fmt", "yuv420p")

_VP9_QUALITY = (
    ("best", ("-crf", "28", "-b:v", "0", "-row-mt", "1")),
    ("vbr2", ("-crf", "32", "-b:v", "0", "-row-mt", "1")),
    ("320", ("-crf", "36", "-b:v", "0", "-row-mt", "1")),
    ("128", ("-crf", "42", "-b:v", "0", "-row-mt", "1")),
)
_VP9_ARGS = ("-pix_fmt", "yuv420p", "-deadline", "good", "-cpu-used", "2")

_MPEG4_ARGS = ("-vtag", "xvid", "-pix_fmt", "yuv420p")
_MPEG4_QUALITY = (
    ("best", ("-q:v", "2")),
    ("vbr2", ("-q:v", "3")),
    ("320", ("-q:v", "5")),
    ("128", ("-q:v", "8")),
)
_MPEG2_ARGS = ("-pix_fmt", "yuv420p")
_MPEG2_QUALITY = (
    ("best", ("-q:v", "2")),
    ("vbr2", ("-q:v", "4")),
    ("320", ("-q:v", "6")),
    ("128", ("-q:v", "10")),
)
_FLV1_ARGS = ("-pix_fmt", "yuv420p")
_FLV1_QUALITY = (
    ("best", ("-q:v", "2")),
    ("vbr2", ("-q:v", "4")),
    ("320", ("-q:v", "6")),
    ("128", ("-q:v", "9")),
)
_WMV2_ARGS = ("-pix_fmt", "yuv420p")
_WMV2_QUALITY = (
    ("best", ("-q:v", "2")),
    ("vbr2", ("-q:v", "4")),
    ("320", ("-q:v", "6")),
    ("128", ("-q:v", "9")),
)

_LAME_QUALITY = (
    ("best", ("-q:a", "0")),
    ("vbr2", ("-q:a", "2")),
    ("320", ("-b:a", "320k")),
    ("256", ("-b:a", "256k")),
    ("192", ("-b:a", "192k")),
    ("128", ("-b:a", "128k")),
)
_AAC_QUALITY = (
    ("best", ("-b:a", "288k")),
    ("vbr2", ("-b:a", "224k")),
    ("320", ("-b:a", "320k")),
    ("256", ("-b:a", "256k")),
    ("192", ("-b:a", "192k")),
    ("128", ("-b:a", "128k")),
)
_OPUS_QUALITY = (
    ("best", ("-b:a", "192k")),
    ("vbr2", ("-b:a", "160k")),
    ("320", ("-b:a", "160k")),
    ("256", ("-b:a", "160k")),
    ("192", ("-b:a", "128k")),
    ("128", ("-b:a", "96k")),
)
_VORBIS_QUALITY = (
    ("best", ("-q:a", "8")),
    ("vbr2", ("-q:a", "7")),
    ("320", ("-q:a", "6")),
    ("256", ("-q:a", "5")),
    ("192", ("-q:a", "4")),
    ("128", ("-q:a", "2")),
)
_WMAV2_QUALITY = (
    ("best", ("-b:a", "192k")),
    ("vbr2", ("-b:a", "160k")),
    ("320", ("-b:a", "160k")),
    ("256", ("-b:a", "192k")),
    ("192", ("-b:a", "128k")),
    ("128", ("-b:a", "96k")),
)
_AC3_QUALITY = (
    ("best", ("-b:a", "640k")),
    ("vbr2", ("-b:a", "448k")),
    ("320", ("-b:a", "384k")),
    ("256", ("-b:a", "320k")),
    ("192", ("-b:a", "256k")),
    ("128", ("-b:a", "192k")),
)


OUTPUT_FORMATS: Tuple[OutputFormat, ...] = (
    # ------------------------------- audio targets -------------------------
    OutputFormat(
        key="mp3", short="MP3", label="MP3  -  .mp3  (plays everywhere)", extension=".mp3", kind="audio",
        audio_codec="libmp3lame", audio_encoder="libmp3lame", audio_copyable=("mp3",),
        audio_quality=_LAME_QUALITY, max_sample_rate=48000, max_channels=2, bit_rate_cap=320,
    ),
    OutputFormat(
        key="m4a", short="M4A", label="M4A / AAC  -  .m4a  (Apple / mobile friendly)", extension=".m4a", kind="audio",
        audio_codec="aac", audio_encoder="aac", audio_copyable=("aac", "alac"),
        audio_quality=_AAC_QUALITY, max_sample_rate=96000, max_channels=8, bit_rate_cap=320,
    ),
    OutputFormat(
        key="aac", short="AAC", label="AAC  -  .aac  (raw ADTS stream)", extension=".aac", kind="audio",
        audio_codec="aac", audio_encoder="aac", audio_copyable=("aac",),
        audio_quality=_AAC_QUALITY, max_sample_rate=96000, max_channels=8, bit_rate_cap=320,
        copy_with_trim=False,
    ),
    OutputFormat(
        key="opus", short="OPUS", label="Opus  -  .opus  (best quality per kbps)", extension=".opus", kind="audio",
        audio_codec="libopus", audio_encoder="libopus", audio_copyable=("opus",),
        audio_quality=_OPUS_QUALITY, max_sample_rate=48000, max_channels=8,
        force_sample_rate=48000, bit_rate_cap=256, copy_with_trim=False,
    ),
    OutputFormat(
        key="ogg", short="OGG", label="OGG Vorbis  -  .ogg", extension=".ogg", kind="audio",
        audio_codec="libvorbis", audio_encoder="libvorbis", audio_copyable=("vorbis",),
        audio_quality=_VORBIS_QUALITY, max_sample_rate=48000, max_channels=8, bit_rate_cap=320,
        copy_with_trim=False,
    ),
    OutputFormat(
        key="flac", short="FLAC", label="FLAC  -  .flac  (lossless, compressed)", extension=".flac", kind="audio",
        audio_codec="flac", audio_encoder="flac", audio_copyable=("flac",),
        audio_extra=("-compression_level", "8"), max_channels=8, lossless=True,
        copy_with_trim=False,
    ),
    OutputFormat(
        key="wav", short="WAV", label="WAV  -  .wav  (lossless, uncompressed)", extension=".wav", kind="audio",
        audio_codec="pcm_s24le", audio_encoder="pcm_s24le",
        audio_copyable=("pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le", "pcm_u8"),
        max_channels=8, lossless=True,
    ),
    OutputFormat(
        key="aiff", short="AIFF", label="AIFF  -  .aiff  (lossless, Apple)", extension=".aiff", kind="audio",
        audio_codec="pcm_s16be", audio_encoder="pcm_s16be",
        audio_copyable=("pcm_s16be", "pcm_s24be", "pcm_s16le"), max_channels=8, lossless=True,
    ),
    OutputFormat(
        key="wma", short="WMA", label="WMA  -  .wma  (Windows Media)", extension=".wma", kind="audio",
        audio_codec="wmav2", audio_encoder="wmav2", audio_copyable=("wmav1", "wmav2"),
        audio_quality=_WMAV2_QUALITY, max_sample_rate=48000, max_channels=2, bit_rate_cap=192,
        copy_with_trim=False,
    ),
    OutputFormat(
        key="ac3", short="AC-3", label="AC-3  -  .ac3  (home cinema / DVD)", extension=".ac3", kind="audio",
        audio_codec="ac3", audio_encoder="ac3", audio_copyable=("ac3",),
        audio_quality=_AC3_QUALITY, max_sample_rate=48000, max_channels=6, bit_rate_cap=640,
        copy_with_trim=False,
    ),

    # ------------------------------- video targets -------------------------
    OutputFormat(
        key="mp4v", short="MP4", label="MP4 video  -  .mp4  (H.264 + AAC)", extension=".mp4", kind="video",
        video_codec="libx264", video_encoder="libx264", video_extra=_X264_ARGS,
        video_copyable=("h264", "hevc"), video_quality=_X264_QUALITY,
        audio_codec="aac", audio_encoder="aac", audio_copyable=("aac", "alac"),
        audio_quality=_AAC_QUALITY, max_sample_rate=96000, max_channels=8, bit_rate_cap=320,
        video_short="H.264",
    ),
    OutputFormat(
        key="mkv", short="MKV", label="MKV video  -  .mkv  (H.264 + AAC, flexible)", extension=".mkv", kind="video",
        video_codec="libx264", video_encoder="libx264", video_extra=_X264_ARGS,
        video_copyable=("h264", "hevc", "vp8", "vp9", "av1"),
        video_quality=_X264_QUALITY,
        audio_codec="aac", audio_encoder="aac",
        audio_copyable=("aac", "alac", "mp3", "ac3", "opus", "vorbis", "flac"),
        audio_quality=_AAC_QUALITY, max_sample_rate=96000, max_channels=8, bit_rate_cap=320,
        video_short="H.264",
    ),
    OutputFormat(
        key="webm", short="WebM", label="WebM video  -  .webm  (VP9 + Opus)", extension=".webm", kind="video",
        video_codec="libvpx-vp9", video_encoder="libvpx-vp9", video_extra=_VP9_ARGS,
        video_copyable=("vp8", "vp9"), video_quality=_VP9_QUALITY,
        audio_codec="libopus", audio_encoder="libopus", audio_copyable=("opus", "vorbis"),
        audio_quality=_OPUS_QUALITY, max_sample_rate=48000, max_channels=8,
        force_sample_rate=48000, bit_rate_cap=256,
        video_short="VP9",
    ),
    OutputFormat(
        key="mov", short="MOV", label="MOV video  -  .mov  (QuickTime, H.264 + AAC)", extension=".mov", kind="video",
        video_codec="libx264", video_encoder="libx264", video_extra=_X264_ARGS,
        video_copyable=("h264", "hevc"), video_quality=_X264_QUALITY,
        audio_codec="aac", audio_encoder="aac", audio_copyable=("aac", "alac", "mp3"),
        audio_quality=_AAC_QUALITY, max_sample_rate=96000, max_channels=8, bit_rate_cap=320,
        video_short="H.264",
    ),
    OutputFormat(
        key="avi", short="AVI", label="AVI video  -  .avi  (MPEG-4 + MP3)", extension=".avi", kind="video",
        video_codec="mpeg4", video_encoder="mpeg4", video_extra=_MPEG4_ARGS,
        video_copyable=("mpeg4",), video_quality=_MPEG4_QUALITY,
        audio_codec="libmp3lame", audio_encoder="libmp3lame", audio_copyable=("mp3",),
        audio_quality=_LAME_QUALITY, max_sample_rate=48000, max_channels=2, bit_rate_cap=320,
        video_short="MPEG-4",
    ),
    OutputFormat(
        key="mpg", short="MPG", label="MPG video  -  .mpg  (MPEG-2 + MP2)", extension=".mpg", kind="video",
        video_codec="mpeg2video", video_encoder="mpeg2video", video_extra=_MPEG2_ARGS,
        video_copyable=("mpeg2video",), video_quality=_MPEG2_QUALITY,
        audio_codec="mp2", audio_encoder="mp2", audio_copyable=("mp2",),
        audio_quality=(("best", ("-b:a", "384k")), ("vbr2", ("-b:a", "320k")),
                       ("320", ("-b:a", "256k")), ("128", ("-b:a", "192k"))),
        max_sample_rate=48000, max_channels=2, bit_rate_cap=384,
        video_short="MPEG-2",
    ),
    OutputFormat(
        key="flvv", short="FLV", label="FLV video  -  .flv  (Flash, H.263 + MP3)", extension=".flv", kind="video",
        video_codec="flv", video_encoder="flv", video_extra=_FLV1_ARGS,
        video_copyable=(), video_quality=_FLV1_QUALITY,
        audio_codec="libmp3lame", audio_encoder="libmp3lame", audio_copyable=("mp3",),
        audio_quality=_LAME_QUALITY, max_sample_rate=44100, max_channels=2, bit_rate_cap=320,
        video_short="FLV1",
    ),
    OutputFormat(
        key="wmv", short="WMV", label="WMV video  -  .wmv  (Windows Media)", extension=".wmv", kind="video",
        video_codec="wmv2", video_encoder="wmv2", video_extra=_WMV2_ARGS,
        video_copyable=(), video_quality=_WMV2_QUALITY,
        audio_codec="wmav2", audio_encoder="wmav2", audio_copyable=("wmav1", "wmav2"),
        audio_quality=_WMAV2_QUALITY, max_sample_rate=48000, max_channels=2, bit_rate_cap=192,
        video_short="WMV2",
    ),
    OutputFormat(
        key="tsv", short="TS", label="TS video  -  .ts  (broadcast stream, H.264 + AAC)", extension=".ts", kind="video",
        video_codec="libx264", video_encoder="libx264", video_extra=_X264_ARGS,
        video_copyable=("h264", "mpeg2video"), video_quality=_X264_QUALITY,
        audio_codec="aac", audio_encoder="aac", audio_copyable=("aac", "mp3", "ac3"),
        audio_quality=_AAC_QUALITY, max_sample_rate=96000, max_channels=8, bit_rate_cap=320,
        video_short="H.264",
    ),
)

FORMATS_BY_KEY = {fmt.key: fmt for fmt in OUTPUT_FORMATS}
FORMATS_BY_EXTENSION = {fmt.extension: fmt for fmt in OUTPUT_FORMATS}

#: Extension aliases that map onto one of the real targets above.
EXTENSION_ALIASES = {".m4v": "mp4v", ".mp4": "mp4v"}


def get_output_format(key: Optional[str]) -> OutputFormat:
    """Return the :class:`OutputFormat` for *key* (defaults to MP3)."""
    key = (key or "mp3").lower()
    if key in FORMATS_BY_KEY:
        return FORMATS_BY_KEY[key]
    return FORMATS_BY_KEY[EXTENSION_ALIASES.get("." + key, "mp3")]


def format_from_path(path: str) -> Optional[OutputFormat]:
    """Guess the output format from a file name (``None`` when unknown)."""
    extension = os.path.splitext(path)[1].lower()
    if extension in FORMATS_BY_EXTENSION:
        return FORMATS_BY_EXTENSION[extension]
    alias = EXTENSION_ALIASES.get(extension)
    return FORMATS_BY_KEY.get(alias) if alias else None


def formats_for_kind(kind: str) -> Tuple[OutputFormat, ...]:
    """All targets of a family (``"audio"`` or ``"video"``)."""
    return tuple(fmt for fmt in OUTPUT_FORMATS if fmt.kind == kind)


_encoder_cache: dict = {}


def encoder_available(ffmpeg: Optional[str], encoder: str) -> bool:
    """Check (and cache) whether *ffmpeg* ships with *encoder*."""
    if not ffmpeg:
        return False
    if "list" not in _encoder_cache:
        try:
            out = subprocess.run(
                [ffmpeg, "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=30, **_no_window_kwargs(),
            )
            _encoder_cache["list"] = out.stdout or ""
        except Exception:
            _encoder_cache["list"] = ""
    return re.search(rf"\b{re.escape(encoder)}\b", _encoder_cache["list"]) is not None


def format_availability(fmt: OutputFormat, ffmpeg: Optional[str],
                        info: Optional["MediaInfo"], source: str) -> Tuple[bool, str]:
    """Decide whether *fmt* can be used for *info*.

    Returns ``(usable, reason)``; *reason* explains why a target is blocked:

    * the FFmpeg build lacks the required encoder,
    * the target container is the same as the source container,
    * the source has no video stream (so a video target is impossible),
    * the source has no audio stream (so an audio target is impossible).
    """
    if not ffmpeg:
        return False, "FFmpeg was not found."

    if not encoder_available(ffmpeg, fmt.audio_encoder) and not info_has_no_audio_ok(fmt, info):
        return False, f"this FFmpeg build has no '{fmt.audio_encoder}' encoder"
    if fmt.is_video and not encoder_available(ffmpeg, fmt.video_encoder):
        return False, f"this FFmpeg build has no '{fmt.video_encoder}' encoder"

    if info is not None:
        if fmt.is_video and not info.video_codec:
            return False, "the source has no video stream"
        if not fmt.is_video and not info.codec:
            return False, "the source has no audio stream"

    source_extension = os.path.splitext(source)[1].lower() if source else ""
    if source_extension and source_extension == fmt.extension:
        return False, f"the source is already {fmt.extension} - choose another format"
    if source_extension and EXTENSION_ALIASES.get(source_extension) == fmt.key:
        return False, "the source is already in this format - choose another format"

    return True, ""


def info_has_no_audio_ok(fmt: OutputFormat, info: Optional["MediaInfo"]) -> bool:
    """``True`` when the audio encoder is irrelevant (video target, silent file)."""
    return bool(fmt.is_video and info is not None and not info.codec)

# ---------------------------------------------------------------------------
# FFmpeg discovery
# ---------------------------------------------------------------------------

def _candidate_ffmpeg_paths() -> List[str]:
    """Return FFmpeg-related paths that are commonly available on Windows."""
    roots = [
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        os.environ.get("LOCALAPPDATA", ""),
        "/usr/local/bin",
        "/opt/homebrew/bin",
        "/usr/bin",
    ]
    paths: List[str] = []
    for root in roots:
        if not root:
            continue
        paths.append(os.path.join(root, "ffmpeg", "bin", "ffmpeg.exe"))
        paths.append(os.path.join(root, "ffmpeg", "ffmpeg.exe"))
    return paths


def find_ffmpeg() -> Optional[str]:
    """Locate the FFmpeg executable (PATH -> imageio-ffmpeg -> common folders)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe

    # Optional fallback: a self-contained FFmpeg binary shipped with imageio.
    try:  # pragma: no cover - optional dependency
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass

    for candidate in _candidate_ffmpeg_paths():
        if os.path.isfile(candidate):
            return candidate
    return None


def find_ffprobe(ffmpeg_path: Optional[str]) -> Optional[str]:
    """Locate ffprobe next to FFmpeg (or on PATH)."""
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    if ffmpeg_path:
        candidate = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe.exe" if os.name == "nt" else "ffprobe")
        if os.path.isfile(candidate):
            return candidate
        candidate = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe")
        if os.path.isfile(candidate):
            return candidate
    return None


def find_ffplay(ffmpeg_path: Optional[str]) -> Optional[str]:
    """Locate ``ffplay`` (the simple FFmpeg player used for audio previews)."""
    exe = shutil.which("ffplay")
    if exe:
        return exe
    if ffmpeg_path:
        for name in ("ffplay.exe", "ffplay"):
            candidate = os.path.join(os.path.dirname(ffmpeg_path), name)
            if os.path.isfile(candidate):
                return candidate
    return None


def _no_window_kwargs() -> dict:
    """Hide the console window that Windows would otherwise spawn."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


# ---------------------------------------------------------------------------
# Media probing
# ---------------------------------------------------------------------------

@dataclass
class MediaInfo:
    """Description of the first audio stream found inside the source file."""

    path: str = ""
    duration: float = 0.0          # seconds
    codec: str = ""                # e.g. "aac", "mp3" ("" = no audio stream)
    sample_rate: int = 0           # Hz
    channels: int = 0
    channel_layout: str = ""
    bit_rate: int = 0              # bits per second (0 = unknown)
    container: str = ""
    size: int = 0                  # bytes of the source file
    # ---- optional video stream information (empty for audio-only files) ---
    video_codec: str = ""
    video_width: int = 0
    video_height: int = 0
    video_fps: float = 0.0

    # ---- display helpers -------------------------------------------------
    @property
    def duration_text(self) -> str:
        if self.duration <= 0:
            return "unknown"
        minutes, seconds = divmod(self.duration, 60)
        hours, minutes = divmod(int(minutes), 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:04.1f}"
        return f"{minutes:02d}:{seconds:04.1f}"

    @property
    def bit_rate_text(self) -> str:
        return f"{self.bit_rate / 1000:.0f} kbps" if self.bit_rate else "unknown"

    @property
    def sample_rate_text(self) -> str:
        return f"{self.sample_rate} Hz ({self.sample_rate / 1000:.1f} kHz)" if self.sample_rate else "unknown"

    @property
    def video_text(self) -> str:
        if not self.video_codec:
            return "none (audio-only file)"
        size = f" {self.video_width}x{self.video_height}" if self.video_width else ""
        fps = f" @ {self.video_fps:.3f} fps" if self.video_fps else ""
        return f"{self.video_codec.upper()}{size}{fps}"

    @property
    def has_audio(self) -> bool:
        return bool(self.codec)

    @property
    def kind_text(self) -> str:
        if self.video_codec and self.codec:
            return "Video + audio"
        if self.video_codec:
            return "Video only (silent)"
        return "Audio only"

    @property
    def channels_text(self) -> str:
        if not self.channels:
            return "unknown"
        layout = f" - {self.channel_layout}" if self.channel_layout else ""
        return f"{self.channels}{layout}"


def probe_media(path: str, ffprobe: Optional[str], ffmpeg: Optional[str]) -> MediaInfo:
    """Read the audio-stream properties of *path* using ffprobe.

    If ffprobe is unavailable the function falls back to parsing the text
    banner printed by ``ffmpeg -i``.
    """
    info = MediaInfo(path=path)
    try:
        info.size = os.path.getsize(path)
    except OSError:
        info.size = 0

    if ffprobe:
        cmd = [
            ffprobe, "-v", "error", "-hide_banner",
            "-print_format", "json", "-show_format", "-show_streams", path,
        ]
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, **_no_window_kwargs()
            )
            data = json.loads(out.stdout or "{}")
            fmt = data.get("format", {}) or {}
            info.container = _friendly_container(fmt.get("format_name") or "")
            info.duration = _safe_float(fmt.get("duration"))
            if not info.duration:
                info.duration = _safe_float(fmt.get("duration_ts", 0))
            for stream in data.get("streams", []):
                if stream.get("codec_type") != "audio":
                    # Remember the first real video stream (cover art is
                    # reported as a video stream too, so ignore it).
                    if (
                        stream.get("codec_type") == "video"
                        and not info.video_codec
                        and not (stream.get("disposition") or {}).get("attached_pic")
                    ):
                        info.video_codec = stream.get("codec_name", "")
                        info.video_width = int(stream.get("width") or 0)
                        info.video_height = int(stream.get("height") or 0)
                        info.video_fps = _parse_fps(stream.get("avg_frame_rate"))
                    continue
                info.codec = stream.get("codec_name", "unknown")
                info.sample_rate = int(_safe_float(stream.get("sample_rate")))
                info.channels = int(stream.get("channels") or 0)
                info.channel_layout = stream.get("channel_layout", "") or ""
                # Only the *audio* stream bit-rate is relevant here; the
                # container bit-rate would also include the video track.
                info.bit_rate = int(_safe_float(stream.get("bit_rate")))
                if not info.duration:
                    info.duration = _safe_float(stream.get("duration"))
                break
            else:
                # No audio stream: fine as long as there is a video stream
                # (such a file can still be converted to another video format).
                info.codec = ""
                if not info.video_codec:
                    raise RuntimeError(
                        "The file contains neither an audio nor a video stream."
                    )
            return info
        except RuntimeError:
            raise
        except Exception:
            pass  # fall through to the ffmpeg banner parser

    if ffmpeg:
        try:
            proc = subprocess.run(
                [ffmpeg, "-hide_banner", "-i", path],
                capture_output=True, text=True, timeout=60, **_no_window_kwargs(),
            )
            text = (proc.stderr or "") + (proc.stdout or "")
            video_line = next(
                (ln for ln in text.splitlines() if "Video:" in ln), None
            )
            if video_line:
                info.video_codec = _first_match(video_line, r"Video:\s*([A-Za-z0-9_\-]+)", "")
                size_match = re.search(r"(\d{2,5})x(\d{2,5})", video_line)
                if size_match:
                    info.video_width = int(size_match.group(1))
                    info.video_height = int(size_match.group(2))
                fps_match = re.search(r"([\d.]+)\s*fps", video_line)
                if fps_match:
                    info.video_fps = _safe_float(fps_match.group(1))
            audio_line = next(
                (ln for ln in text.splitlines() if "Audio:" in ln), None
            )
            if audio_line is None:
                if not info.video_codec:
                    raise RuntimeError(
                        "The file contains neither an audio nor a video stream."
                    )
                # Video-only source: remember the duration and stop here.
                match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", text)
                if match:
                    info.duration = (int(match.group(1)) * 3600
                                     + int(match.group(2)) * 60 + float(match.group(3)))
                return _fill_container(info)
            info.codec = _first_match(audio_line, r"Audio:\s*([A-Za-z0-9_\-]+)", "unknown").lower()
            match = re.search(r"(\d+)\s*Hz", audio_line)
            info.sample_rate = int(match.group(1)) if match else 0
            match = re.search(r"Hz,\s*([A-Za-z0-9_\(\),\.\s]+?),", audio_line)
            if match:
                info.channel_layout = match.group(1).strip()
            match = re.search(r"(\d+)\s*kb/s", audio_line)
            info.bit_rate = int(match.group(1)) * 1000 if match else 0
            info.channels = _guess_channels(info.channel_layout)
            match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", text)
            if match:
                info.duration = (
                    int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))
                )
            return _fill_container(info)
        except RuntimeError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Could not read the media information: {exc}") from exc

    raise RuntimeError("Neither ffprobe nor ffmpeg could be found.")


def _parse_fps(value) -> float:
    """Convert ffprobe's ``"30000/1001"`` frame-rate notation to a float."""
    if not value:
        return 0.0
    text = str(value)
    if "/" in text:
        numerator, _, denominator = text.partition("/")
        try:
            denominator_value = float(denominator)
            return float(numerator) / denominator_value if denominator_value else 0.0
        except (TypeError, ValueError):
            return 0.0
    return _safe_float(text)


_CONTAINER_BY_EXTENSION = {
    ".mp4": "MP4", ".m4v": "MP4", ".mov": "QuickTime (MOV)", ".mkv": "Matroska (MKV)",
    ".webm": "WebM", ".avi": "AVI", ".ts": "MPEG-TS", ".m2ts": "MPEG-TS",
    ".mts": "MPEG-TS", ".flv": "FLV", ".wmv": "ASF / WMV", ".mpg": "MPEG-PS",
    ".mpeg": "MPEG-PS", ".vob": "MPEG-PS", ".3gp": "3GP", ".ogv": "OGG",
    ".mp3": "MP3", ".m4a": "MP4", ".aac": "AAC (ADTS)", ".wav": "WAV",
    ".flac": "FLAC", ".aiff": "AIFF", ".aif": "AIFF", ".ogg": "OGG",
    ".oga": "OGG", ".opus": "Opus (OGG)", ".wma": "WMA", ".ac3": "AC-3",
    ".ape": "APE", ".mka": "Matroska (MKV)", ".amr": "AMR", ".wv": "WavPack",
}


def _container_from_extension(path: str) -> str:
    """Best-effort container name taken from the file extension."""
    return _CONTAINER_BY_EXTENSION.get(os.path.splitext(path)[1].lower(), "")


def _fill_container(info: "MediaInfo") -> "MediaInfo":
    """Use the extension when the parser could not report a container name."""
    if not info.container or info.container.lower() == "unknown":
        info.container = _container_from_extension(info.path) or "unknown"
    return info


def _friendly_container(format_name: str) -> str:
    """Turn FFmpeg's comma separated container list into one short name."""
    names = [part.strip().lower() for part in format_name.split(",") if part.strip()]
    priority = (
        ("mp4", "MP4"), ("matroska", "Matroska (MKV)"), ("webm", "WebM"),
        ("mov", "QuickTime (MOV)"), ("avi", "AVI"), ("mpegts", "MPEG-TS"),
        ("flv", "FLV"), ("asf", "ASF / WMV"), ("mpeg", "MPEG-PS"),
        ("3gp", "3GP"), ("ogg", "OGG"), ("opus", "Opus (OGG)"),
        ("mp3", "MP3"), ("wav", "WAV"), ("flac", "FLAC"), ("aiff", "AIFF"),
        ("aac", "AAC (ADTS)"), ("wma", "WMA"), ("ape", "APE"),
    )
    for key, label in priority:
        if key in names:
            return label
    return names[0].upper() if names else "unknown"


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _first_match(text: str, pattern: str, default: str = "") -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else default


def _guess_channels(layout: str) -> int:
    layout = (layout or "").lower()
    if not layout:
        return 0
    if layout.startswith("mono"):
        return 1
    if layout.startswith("stereo"):
        return 2
    return layout.count(" ") + 1


# ---------------------------------------------------------------------------
# Wave-form analysis
# ---------------------------------------------------------------------------

def analyse_waveform(path: str, ffmpeg: str,
                     sample_rate: int = PREVIEW_SAMPLE_RATE,
                     columns: int = PREVIEW_COLUMNS
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Decode the audio track and return ``(x, y_min, y_max, duration)``.

    The audio is decoded once, down-mixed to mono and heavily down-sampled
    (``sample_rate`` Hz), which is enough to draw a faithful envelope and keeps
    the analysis fast even for long movies.
    """
    cmd = [
        ffmpeg, "-hide_banner", "-nostdin", "-v", "error",
        "-i", path, "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", str(sample_rate),
        "-f", "s16le", "-acodec", "pcm_s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=600, **_no_window_kwargs())
    if proc.returncode != 0:
        raise RuntimeError(
            (proc.stderr or b"").decode("utf-8", "replace").strip()
            or "FFmpeg could not decode the audio stream."
        )

    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        raise RuntimeError("The decoded audio stream is empty.")

    duration = samples.size / float(sample_rate)
    if samples.size <= columns:
        x = np.arange(samples.size, dtype=np.float32) / sample_rate
        y = samples
        return x, y.copy(), y.copy(), duration

    chunk = samples.size // columns
    trimmed = samples[: chunk * columns].reshape(columns, chunk)
    y_min = trimmed.min(axis=1)
    y_max = trimmed.max(axis=1)
    x = (np.arange(columns, dtype=np.float32) + 0.5) * chunk / float(sample_rate)
    return x, y_min, y_max, duration


# ---------------------------------------------------------------------------
# Conversion engine (shared by the GUI and the command line)
# ---------------------------------------------------------------------------

def build_codec_arguments(preset: str, info: MediaInfo,
                          output_format: Optional[OutputFormat] = None,
                          trimmed: bool = False
                          ) -> Tuple[List[str], List[str]]:
    """Translate a preset key + output format into FFmpeg codec arguments.

    Works for audio-only targets (``-map 0:a:0``) as well as for video targets,
    where the video stream is either copied or re-encoded and the audio track is
    treated exactly like a normal audio conversion.

    Returns ``(arguments, notices)`` where *notices* are human readable remarks
    (stream copies, down-mixes, resampling, ...) for the log.
    """
    fmt = output_format or get_output_format("mp3")
    notices: List[str] = []

    audio_args, audio_notices = _audio_codec_arguments(preset, info, fmt, trimmed)
    notices.extend(audio_notices)

    if not fmt.is_video:
        return audio_args, notices

    # ------------------------------- video part ---------------------------
    video_args, video_notices = _video_codec_arguments(preset, info, fmt, trimmed)
    notices.extend(video_notices)
    return video_args + audio_args, notices


def _audio_codec_arguments(preset: str, info: MediaInfo, fmt: OutputFormat,
                           trimmed: bool) -> Tuple[List[str], List[str]]:
    """Encoder arguments for the audio track of *fmt*."""
    notices: List[str] = []

    # The source may be silent (video-only file) -> no audio track at all.
    if not info.codec:
        return ["-an"], ["The source has no audio stream: the output will be silent."]

    copy_allowed = not (trimmed and not fmt.copy_with_trim)

    # ---- lossless targets ------------------------------------------------
    if fmt.lossless:
        if info.codec in fmt.audio_copyable and copy_allowed:
            notices.append(
                f"The source audio is already {info.codec.upper()}: the stream is "
                f"copied bit-for-bit (no re-encoding, no quality loss)."
            )
            return ["-c:a", "copy"], notices
        if info.codec in fmt.audio_copyable and not copy_allowed:
            notices.append(
                f"A trimmed {fmt.name} stream cannot be copied (the container "
                f"header would keep the duration of the whole file), so the audio is "
                f"re-encoded losslessly."
            )
        else:
            notices.append(
                f"Lossless conversion: the audio is re-encoded without any quality loss "
                f"(bit-rate: {_expected_lossless_rate(info)})."
            )
        return (["-c:a", fmt.audio_codec, *fmt.audio_extra]
                + _rate_channel_arguments(info, fmt, notices)), notices

    # ---- "original quality" ---------------------------------------------
    if preset == "original":
        if info.codec in fmt.audio_copyable and copy_allowed:
            notices.append(
                f"The source audio is already {info.codec.upper()}: the stream is "
                f"copied bit-for-bit into the {fmt.name} container "
                f"(no re-encoding, no quality loss)."
            )
            return ["-c:a", "copy"], notices
        if info.codec in fmt.audio_copyable and not copy_allowed:
            notices.append(
                f"A trimmed {fmt.name} stream cannot be copied, so the audio is "
                f"re-encoded with {fmt.audio_codec} at the source bit-rate."
            )
            if info.bit_rate:
                kbps = max(64, min(fmt.bit_rate_cap, int(round(info.bit_rate / 1000.0))))
                args = ["-c:a", fmt.audio_codec, "-b:a", f"{kbps}k", *fmt.audio_extra]
            else:
                args = ["-c:a", fmt.audio_codec, *fmt.audio_args_for("best"),
                        *fmt.audio_extra]
            return args + _rate_channel_arguments(info, fmt, notices), notices

        if info.bit_rate:
            kbps = max(64, min(fmt.bit_rate_cap, int(round(info.bit_rate / 1000.0))))
            notices.append(
                f"Source audio codec is {info.codec.upper()}; encoded at the source "
                f"bit-rate of {kbps} kbps ({fmt.name} audio is lossy, so the "
                f"result is transparent rather than bit-identical)."
            )
            args = ["-c:a", fmt.audio_codec, "-b:a", f"{kbps}k", *fmt.audio_extra]
        else:
            notices.append(
                "The source bit-rate is not reported by the container; the highest "
                f"quality setting of {fmt.name} is used to stay transparent."
            )
            args = ["-c:a", fmt.audio_codec, *fmt.audio_args_for("best"), *fmt.audio_extra]
        return args + _rate_channel_arguments(info, fmt, notices), notices

    # ---- explicit quality preset ----------------------------------------
    preset_args = list(fmt.audio_args_for(preset))
    notices.append(
        f"Audio encoded with {fmt.audio_codec} ("
        + (" ".join(preset_args) if preset_args else "encoder defaults") + ")."
    )
    args = ["-c:a", fmt.audio_codec, *preset_args, *fmt.audio_extra]
    return args + _rate_channel_arguments(info, fmt, notices), notices


def _video_codec_arguments(preset: str, info: MediaInfo, fmt: OutputFormat,
                           trimmed: bool) -> Tuple[List[str], List[str]]:
    """Encoder arguments for the video stream of a video target."""
    notices: List[str] = []

    if preset == "original" and info.video_codec in fmt.video_copyable:
        notices.append(
            f"The video stream ({info.video_codec.upper()}) is copied without "
            f"re-encoding (remux only)."
        )
        return ["-c:v", "copy"], notices

    if preset == "original":
        # Cannot copy (codec not allowed in this container) -> highest quality.
        quality = fmt.video_args_for("best")
        notices.append(
            f"The video stream cannot be copied into {fmt.name}, so it is "
            f"re-encoded with {fmt.video_codec} at the highest quality setting."
        )
    else:
        quality = fmt.video_args_for(preset)
        notices.append(
            f"Video encoded with {fmt.video_codec} (" + " ".join(quality) + ")."
        )

    if info.video_width and info.video_height:
        # Even dimensions are required by most encoders / players.
        if info.video_width % 2 or info.video_height % 2:
            return (["-c:v", fmt.video_codec, *quality, *fmt.video_extra,
                     "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"], notices)

    return ["-c:v", fmt.video_codec, *quality, *fmt.video_extra], notices


def _expected_lossless_rate(info: MediaInfo) -> str:
    """Rough bit-rate of a lossless conversion (useful for the log)."""
    if not info.sample_rate or not info.channels:
        return "uncompressed"
    bits_per_second = info.sample_rate * info.channels * 16
    return f"~{bits_per_second / 1_000_000:.2f} Mbps"


def _rate_channel_arguments(info: MediaInfo, fmt: OutputFormat,
                            notices: List[str]) -> List[str]:
    """Add ``-ar`` / ``-ac`` when the source exceeds what *fmt* supports."""
    args: List[str] = []

    target_rate = fmt.force_sample_rate
    if not target_rate and fmt.max_sample_rate and info.sample_rate > fmt.max_sample_rate:
        target_rate = fmt.max_sample_rate
    if target_rate and info.sample_rate and info.sample_rate != target_rate:
        args += ["-ar", str(target_rate)]
        notices.append(
            f"Sample-rate {info.sample_rate} Hz is not supported by {fmt.name} audio; "
            f"resampled to {target_rate} Hz."
        )

    if fmt.max_channels and info.channels > fmt.max_channels:
        args += ["-ac", str(fmt.max_channels)]
        notices.append(
            f"{info.channels} channels cannot be stored in {fmt.name} audio; "
            f"down-mixed to {fmt.max_channels} channels."
        )
    return args


def extract_audio(
    source: str,
    destination: str,
    preset: str,
    info: MediaInfo,
    ffmpeg: Optional[str] = None,
    on_progress: Optional[Callable[[float, float, str], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    register_process: Optional[Callable[[Optional[subprocess.Popen]], None]] = None,
    start: float = 0.0,
    end: float = 0.0,
    output_format: Optional[OutputFormat] = None,
) -> Tuple[bool, str]:
    """Convert *source* to an audio or video file at *destination*.

    Parameters
    ----------
    on_progress:
        Called as ``on_progress(percent, processed_seconds, speed_text)``.
        ``percent`` is ``-1`` when the total duration is unknown.
    on_log:
        Called with informational / diagnostic lines.
    register_process:
        Called with the running ``Popen`` object so that the caller can cancel
        the conversion (``None`` is passed once the process is finished).
    start, end:
        Optional trim range in seconds (millisecond precision). ``end`` equal
        to the file duration (or ``0``) means "until the end of the file".
    output_format:
        One of :data:`OUTPUT_FORMATS`; defaults to the format that matches the
        extension of *destination* (MP3 when unknown).
    """
    ffmpeg = ffmpeg or find_ffmpeg()
    if not ffmpeg:
        return False, (
            "FFmpeg was not found. Install it from https://ffmpeg.org/download.html "
            "and make sure the 'ffmpeg' command is available on your PATH."
        )

    log = on_log or (lambda message: None)
    progress = on_progress or (lambda percent, seconds, speed: None)

    fmt = output_format or format_from_path(destination) or get_output_format("mp3")

    # ---- validate the conversion before starting FFmpeg ------------------
    usable, reason = format_availability(fmt, ffmpeg, info, source)
    if not usable:
        return False, (
            f"{fmt.name} output is not possible: {reason}."
        )
    log(f"  . Target  : {fmt.name} ({fmt.kind}) -> {os.path.basename(destination)}")

    # ---- trim range ------------------------------------------------------
    trim_start = max(0.0, float(start or 0.0))
    trim_end = float(end or 0.0)
    if trim_end <= 0:
        trim_end = info.duration
    if trim_end > 0 and trim_end <= trim_start:
        return False, "The end time must be greater than the start time."
    duration = (trim_end - trim_start) if trim_end > 0 else max(0.0, info.duration - trim_start)

    trim_args: List[str] = []
    trimmed = trim_start > 0.0005 or (
        info.duration > 0 and trim_end > 0 and trim_end < info.duration - 0.0005
    )
    if trimmed:
        # Output-side seeking keeps the cut accurate (millisecond precision).
        trim_args += ["-ss", f"{trim_start:.3f}"]
        if duration > 0:
            trim_args += ["-t", f"{duration:.3f}"]
        log(f"  . Trim range: {format_timecode(trim_start)} -> {format_timecode(trim_end)} "
            f"(length {format_timecode(duration)})")

    codec_args, notices = build_codec_arguments(preset, info, fmt, trimmed=trimmed)
    for notice in notices:
        log(f"  . {notice}")


    if fmt.is_video:
        # Keep the first video stream and (when present) the first audio stream.
        stream_args = ["-map", "0:v:0"]
        if info.codec:
            stream_args += ["-map", "0:a:0"]
        stream_args += ["-sn", "-dn"]
    else:
        # Audio only: drop video, subtitles and data streams.
        stream_args = ["-map", "0:a:0", "-vn", "-sn", "-dn"]

    cmd = [
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-i", source,
        *stream_args,
        *trim_args,               # -ss / -t for the selected range
        *codec_args,
        *fmt.extra_muxer_args(),
        "-progress", "pipe:1", "-nostats",
        destination,
    ]
    log("Running: " + " ".join(f'"{c}"' if " " in c else c for c in cmd))

    proc: Optional[subprocess.Popen] = None
    stderr_text = ""
    try:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as err_file:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=err_file,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **_no_window_kwargs(),
            )
            if register_process:
                register_process(proc)

            processed = 0.0
            speed = ""
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                key, _, value = raw_line.strip().partition("=")
                if key == "out_time_us" and value.isdigit():
                    processed = int(value) / 1_000_000.0
                elif key == "out_time_ms" and value.isdigit():
                    # Older FFmpeg releases write microseconds here as well.
                    processed = int(value) / 1_000_000.0
                elif key == "speed":
                    speed = value
                elif key in ("progress", "out_time"):
                    percent = -1.0
                    if duration > 0:
                        percent = max(0.0, min(100.0, processed / duration * 100.0))
                    progress(percent, processed, speed)
            return_code = proc.wait()
            err_file.seek(0)
            stderr_text = err_file.read().strip()
    except FileNotFoundError:
        return False, "FFmpeg executable could not be started."
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"Unexpected error while running FFmpeg: {exc}"
    finally:
        if register_process:
            register_process(None)

    if proc is not None and proc.returncode is not None and proc.returncode < 0:
        _remove_quietly(destination)
        return False, "Conversion cancelled."

    if return_code != 0:
        _remove_quietly(destination)
        message = stderr_text or f"FFmpeg exited with code {return_code}."
        return False, message

    if not os.path.isfile(destination) or os.path.getsize(destination) == 0:
        _remove_quietly(destination)
        return False, "FFmpeg finished but no output file was produced."

    progress(100.0, duration, speed)
    return True, ""


def _remove_quietly(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Background threads
# ---------------------------------------------------------------------------

class AnalyseWorker(QThread):
    """Read media properties and build the wave-form envelope."""

    infoReady = pyqtSignal(object)          # MediaInfo
    waveformReady = pyqtSignal(object)      # (x, y_min, y_max, duration)
    failed = pyqtSignal(str)

    def __init__(self, path: str, ffmpeg: str, ffprobe: Optional[str], parent=None) -> None:
        super().__init__(parent)
        self._path = path
        self._ffmpeg = ffmpeg
        self._ffprobe = ffprobe

    def run(self) -> None:  # noqa: D401 - Qt entry point
        try:
            info = probe_media(self._path, self._ffprobe, self._ffmpeg)
            self.infoReady.emit(info)
            if not info.codec:
                # Silent video: nothing to draw, but the file is still usable
                # as a video source.
                self.waveformReady.emit(None)
                return
            data = analyse_waveform(self._path, self._ffmpeg)
            self.waveformReady.emit(data)
        except Exception as exc:
            self.failed.emit(str(exc))


class ExtractWorker(QThread):
    """Run the conversion in a separate thread so the UI stays responsive."""

    progress = pyqtSignal(float, float, str)   # percent, processed seconds, speed
    log = pyqtSignal(str)
    finishedWithResult = pyqtSignal(bool, str, str)  # ok, output path, message

    def __init__(self, source: str, destination: str, preset: str,
                 info: MediaInfo, ffmpeg: str, parent=None,
                 start: float = 0.0, end: float = 0.0,
                 output_format: Optional[OutputFormat] = None) -> None:
        super().__init__(parent)
        self._source = source
        self._destination = destination
        self._preset = preset
        self._info = info
        self._ffmpeg = ffmpeg
        self._start = start
        self._end = end
        self._output_format = output_format
        self._process: Optional[subprocess.Popen] = None
        self._cancelled = False

    # ---- cancellation ----------------------------------------------------
    def cancel(self) -> None:
        self._cancelled = True
        proc = self._process
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def _register(self, proc: Optional[subprocess.Popen]) -> None:
        self._process = proc

    def run(self) -> None:  # noqa: D401 - Qt entry point
        ok, message = extract_audio(
            source=self._source,
            destination=self._destination,
            preset=self._preset,
            info=self._info,
            ffmpeg=self._ffmpeg,
            on_progress=lambda percent, seconds, speed: self.progress.emit(
                percent, seconds, speed
            ),
            on_log=self.log.emit,
            register_process=self._register,
            start=self._start,
            end=self._end,
            output_format=self._output_format,
        )
        if self._cancelled:
            ok, message = False, "Conversion cancelled."
        self.finishedWithResult.emit(ok, self._destination, message)


class VideoFrameWorker(QThread):
    """Decode a part of a video file and emit its frames one by one.

    FFmpeg is asked for raw RGB24 frames of a small (already scaled) size, so the
    GUI thread only has to wrap the bytes into a ``QImage`` - no Qt multimedia
    backend and no extra codec support is required.
    """

    failed = pyqtSignal(str)
    finishedPlaying = pyqtSignal()

    def __init__(self, ffmpeg: str, source: str, start: float, duration: float,
                 width: int, height: int, parent=None) -> None:
        super().__init__(parent)
        self._ffmpeg = ffmpeg
        self._source = source
        self._start = max(0.0, start)
        self._duration = max(0.05, duration)
        self._width = width
        self._height = height
        self._process: Optional[subprocess.Popen] = None
        self._stop_requested = False
        #: The newest decoded frame - the GUI *pulls* it, so a slow GUI simply
        #: drops frames instead of building up a queue of stale QImages.
        self._latest: Optional[QImage] = None
        self._lock = threading.Lock()

    def take_latest(self) -> Optional[QImage]:
        """Return the newest frame (and forget it). Thread safe."""
        with self._lock:
            image = self._latest
            self._latest = None
        return image

    def stop(self) -> None:
        self._stop_requested = True
        process = self._process
        if process and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass

    def run(self) -> None:  # noqa: D401 - Qt entry point
        width, height = max(2, self._width), max(2, self._height)
        cmd = [
            self._ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-re",                                   # play at native speed
            "-ss", f"{self._start:.3f}",
            "-t", f"{self._duration:.3f}",
            "-i", self._source,
            "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", f"scale={width}:{height}",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
        ]
        try:
            self._process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                **_no_window_kwargs(),
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        frame_bytes = width * height * 3
        stdout = self._process.stdout
        try:
            while not self._stop_requested and stdout is not None:
                data = stdout.read(frame_bytes)
                if not data or len(data) < frame_bytes:
                    break
                image = QImage(data, width, height, width * 3,
                               QImage.Format_RGB888).copy()
                if not image.isNull():
                    with self._lock:
                        self._latest = image
        except Exception as exc:  # pragma: no cover - defensive
            if not self._stop_requested:
                self.failed.emit(str(exc))
        finally:
            process = self._process
            if process and process.poll() is None:
                try:
                    process.terminate()
                except Exception:
                    pass
        if not self._stop_requested:
            self.finishedPlaying.emit()


class VideoCanvas(QWidget):
    """Widget that paints the current video frame, letterboxed.

    The image is drawn in ``paintEvent`` instead of being put into a QLabel:
    a QLabel reports the pixmap size as its size hint, which makes the layout
    (and the surrounding scroll area) grow after every frame - that is the
    "the video window keeps getting bigger" bug. This widget has a *constant*
    size hint, so the layout stays stable.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._image: Optional[QImage] = None
        self._placeholder = "Load a video file to preview it here."
        self.setMinimumSize(420, 236)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

    # ---- geometry ------------------------------------------------------
    def sizeHint(self):  # noqa: N802 - Qt naming
        return QSize(640, 360)

    def minimumSizeHint(self):  # noqa: N802 - Qt naming
        return QSize(420, 236)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt naming
        return False

    # ---- content -------------------------------------------------------
    def setImage(self, image: Optional[QImage]) -> None:  # noqa: N802
        self._image = image
        self.update()

    def setPlaceholder(self, text: str) -> None:  # noqa: N802
        self._placeholder = text
        self._image = None
        self.update()

    def clear(self) -> None:
        self._image = None
        self.update()

    # ---- painting ------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0d0d12"))
        rect = self.rect().adjusted(1, 1, -1, -1)

        if self._image is not None and not self._image.isNull():
            scaled = self._image.size().scaled(rect.size(), Qt.KeepAspectRatio)
            x = rect.x() + (rect.width() - scaled.width()) // 2
            y = rect.y() + (rect.height() - scaled.height()) // 2
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.drawImage(
                QRect(x, y, scaled.width(), scaled.height()), self._image
            )
        else:
            painter.setPen(QColor("#7f7f92"))
            painter.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap, self._placeholder)

        painter.setPen(QColor("#33333f"))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

STYLESHEET = """
QWidget        { background-color: #1b1b22; color: #e6e6ec;
                 font-family: "Segoe UI", "DejaVu Sans", Arial, sans-serif; font-size: 12px; }
QGroupBox      { border: 1px solid #33333f; border-radius: 8px; margin-top: 14px;
                 padding: 10px 10px 8px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #7fc4ff; }
QScrollArea    { border: none; }
QWidget#content { background-color: #1b1b22; }
QLineEdit, QComboBox, QPlainTextEdit {
                 background-color: #23232c; border: 1px solid #3a3a48; border-radius: 6px;
                 padding: 6px 8px; selection-background-color: #2f6fb5; }
QLineEdit:disabled, QComboBox:disabled { color: #8a8a99; }
QComboBox QAbstractItemView { background-color: #23232c; selection-background-color: #2f6fb5; }
QDoubleSpinBox { background-color: #23232c; border: 1px solid #3a3a48; border-radius: 6px;
                 padding: 5px 8px; }
QDoubleSpinBox:disabled { color: #8a8a99; }
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 16px; background-color: #2b2b36;
                 border: none; }
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { background-color: #3a3a48; }
QLabel#infoKey { color: #9a9aac; }
QLabel#infoValue { color: #e6e6ec; }
QLabel#timecode { color: #ffc14d; font-family: "Consolas", "DejaVu Sans Mono", monospace; }
QLabel#selection { color: #9fd0ff; }
QLabel#summary { color: #8fd6a8; background-color: #1f2a24; border: 1px solid #2f4438;
                 border-radius: 6px; padding: 6px 8px; }
QPushButton    { background-color: #2b2b36; border: 1px solid #3d3d4c; border-radius: 6px;
                 padding: 7px 14px; }
QPushButton:hover    { background-color: #353544; }
QPushButton:pressed  { background-color: #20202a; }
QPushButton:disabled { color: #6c6c7a; background-color: #24242c; }
QPushButton#play     { background-color: #1f4d33; border-color: #2f7a4f; font-weight: 600;
                 color: #d9ffe8; }
QPushButton#play:hover      { background-color: #276040; }
QPushButton#play:disabled   { background-color: #24242c; color: #6c6c7a; }
QPushButton#primary  { background-color: #2f6fb5; border-color: #3d82cc; font-weight: 600;
                 color: #ffffff; }
QPushButton#primary:hover   { background-color: #3a7fc9; }
QPushButton#primary:disabled { background-color: #2a3a4a; color: #8fa3b5; }
QProgressBar   { border: 1px solid #3a3a48; border-radius: 6px; height: 18px; color: #e6e6ec;
                 text-align: center; background-color: #23232c; }
QProgressBar::chunk { background-color: #2f6fb5; border-radius: 5px; }
QLabel#hint    { color: #9a9aac; }
QLabel#title   { font-size: 16px; font-weight: 700; color: #ffffff; }
"""


class MainWindow(QMainWindow):
    """The application's single window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1010, 940)
        self.setAcceptDrops(True)

        self._ffmpeg = find_ffmpeg()
        self._ffprobe = find_ffprobe(self._ffmpeg)

        self._source_path: str = ""
        self._info: Optional[MediaInfo] = None
        self._analyse_worker: Optional[AnalyseWorker] = None
        self._extract_worker: Optional[ExtractWorker] = None
        self._plot_items: List[object] = []
        self._last_output: str = ""
        #: Guards the two-way binding between the spin boxes and the plot region.
        self._sync_guard: bool = False
        #: QtMultimedia fallback player (may stay None).
        self._qt_player = None
        self._playback_available = False
        #: Active audio engine: "ffplay", "qt", "winsound", "system" or "".
        self._audio_backend = ""
        self._audio_process = None
        self._audio_start = 0.0
        self._audio_end = 0.0
        #: Timeline origin of the running audio clock.
        self._audio_origin = 0.0
        self._audio_paused = False
        self._audio_paused_at = 0.0
        self._winsound_active = False
        self._qt_play_attempt = False
        self._ffplay_log_path = ""
        self._playback_warned = False
        #: Embedded video preview state.
        self._video_worker = None
        #: Absolute end time of the video stream that is currently decoding.
        self._video_end = 0.0
        #: Unified preview session bookkeeping.
        self._preview_start = 0.0
        self._preview_end = 0.0
        self._preview_paused = False
        self._preview_paused_at = 0.0
        #: True while the system media player is in charge of the audio.
        self._preview_external = False
        #: Transport state machine: idle | playing | paused | external.
        self._transport_state = "idle"
        #: Worker that renders a temporary file for the system player.
        self._preview_worker = None
        #: Temporary files created for previews (removed when the app closes).
        self._preview_files: List[str] = []

        # Clocks / timers used by the two preview engines.
        self._audio_clock = QElapsedTimer()
        self._preview_clock = QElapsedTimer()
        self._preview_ticker = QTimer(self)
        self._preview_ticker.setInterval(100)
        self._preview_ticker.timeout.connect(self._on_preview_tick)
        #: Debounces the "the trim range changed while playing" restarts.
        self._trim_sync_timer = QTimer(self)
        self._trim_sync_timer.setSingleShot(True)
        self._trim_sync_timer.setInterval(320)
        self._trim_sync_timer.timeout.connect(self._apply_trim_to_preview)
        #: Pulls the newest decoded video frame (~30 fps) into the canvas.
        self._frame_ticker = QTimer(self)
        self._frame_ticker.setInterval(33)
        self._frame_ticker.timeout.connect(self._pull_video_frame)

        self._build_ui()
        pg.setConfigOptions(antialias=True)
        self._setup_player()
        play_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        play_shortcut.activated.connect(self._toggle_preview)
        save_log_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_log_shortcut.activated.connect(self._save_log)
        clear_log_shortcut = QShortcut(QKeySequence("Ctrl+L"), self)
        clear_log_shortcut.activated.connect(self._clear_log)
        self._rebuild_format_combo()
        self._on_format_changed()

        if not self._ffmpeg:
            self._append_log(
                "WARNING: FFmpeg was not found on this system. Install it and add it "
                "to your PATH, then restart the application."
            )
            self._extract_button.setEnabled(False)
        else:
            self._append_log(f"FFmpeg  : {self._ffmpeg}")
            audio_ok = [f.name for f in formats_for_kind("audio")
                        if encoder_available(self._ffmpeg, f.audio_encoder)]
            video_ok = [f.name for f in formats_for_kind("video")
                        if encoder_available(self._ffmpeg, f.video_encoder)]
            self._append_log("Audio outputs: " + (", ".join(audio_ok) or "none"))
            self._append_log("Video outputs: " + (", ".join(video_ok) or "none"))
            self._append_log(f"FFprobe : {self._ffprobe or 'not found (fallback parser)'}")
        self._append_log("Select any video or audio file to begin "
                         "(you can also drag & drop it here).")

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        """Create the whole interface inside a vertically scrollable area."""
        # The complete window content is placed inside a QScrollArea, so the
        # application can always be scrolled vertically - even on small screens.
        self._scroll = QScrollArea()
        self._scroll.setObjectName("scrollArea")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setMinimumSize(520, 360)
        self.setCentralWidget(self._scroll)

        content = QWidget()
        content.setObjectName("content")
        self._scroll.setWidget(content)

        root = QVBoxLayout(content)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        title = QLabel(APP_NAME)
        title.setObjectName("title")
        subtitle = QLabel("Convert any video or audio file to any other audio or video format "
                          "(MP4, MKV, MOV, AVI, WebM, TS, MP3, M4A, AAC, WAV, FLAC, OGG, ...), "
                          "optionally trimmed to a time range.")
        subtitle.setObjectName("hint")
        subtitle.setWordWrap(True)
        author = QLabel(f"Version {APP_VERSION}  \u00b7  by {APP_AUTHOR}  \u00b7  {APP_LICENSE} License")
        author.setObjectName("hint")
        root.addWidget(title)
        root.addWidget(author)
        root.addWidget(subtitle)

        # ---- source / destination ---------------------------------------
        files_box = QGroupBox("Files")
        files_box.setMinimumWidth(300)
        files_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        files_layout = QGridLayout(files_box)
        files_layout.setHorizontalSpacing(8)
        files_layout.setVerticalSpacing(8)

        files_layout.addWidget(QLabel("Input file:"), 0, 0)
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("Path to any video or audio file ...")
        self._input_edit.setReadOnly(True)
        files_layout.addWidget(self._input_edit, 0, 1)
        self._browse_input = QPushButton("Browse ...")
        self._browse_input.clicked.connect(self._choose_input)
        files_layout.addWidget(self._browse_input, 0, 2)

        files_layout.addWidget(QLabel("Output file:"), 1, 0)
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Where the converted file will be written ...")
        self._output_edit.textChanged.connect(self._update_extract_state)
        files_layout.addWidget(self._output_edit, 1, 1)
        self._browse_output = QPushButton("Save as ...")
        self._browse_output.clicked.connect(self._choose_output)
        files_layout.addWidget(self._browse_output, 1, 2)

        files_layout.setColumnStretch(1, 1)
        files_layout.setRowStretch(2, 1)      # keep the fields at the top
        drop_hint = QLabel(
            "Tip: you can drag & drop any video or audio file onto this window."
        )
        drop_hint.setObjectName("hint")
        drop_hint.setWordWrap(True)
        files_layout.addWidget(drop_hint, 3, 0, 1, 3)

        # Files and Conversion share one row (groups 1 + 2 side by side).
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(10)
        top_row.addWidget(files_box, 2)

        # ---- quality ------------------------------------------------------
        controls_box = QGroupBox("Conversion")
        controls_box.setMinimumWidth(430)
        controls_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        controls = QGridLayout(controls_box)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(8)

        controls.addWidget(QLabel("Output format:"), 0, 0)
        self._format_combo = QComboBox()
        self._format_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._format_combo.setMinimumContentsLength(16)
        self._format_combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)
        controls.addWidget(self._format_combo, 0, 1, 1, 3)

        controls.addWidget(QLabel("Quality:"), 1, 0)
        self._quality_combo = QComboBox()
        self._quality_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._quality_combo.setMinimumContentsLength(16)
        self._quality_combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._quality_combo.currentIndexChanged.connect(self._update_conversion_summary)
        controls.addWidget(self._quality_combo, 1, 1, 1, 3)

        self._summary_label = QLabel("Load a file to see the conversion plan.")
        self._summary_label.setObjectName("summary")
        self._summary_label.setWordWrap(True)
        self._summary_label.setMinimumWidth(1)
        controls.addWidget(self._summary_label, 2, 0, 1, 4)

        self._extract_button = QPushButton("Extract audio")
        self._extract_button.setObjectName("primary")
        self._extract_button.setEnabled(False)
        self._extract_button.clicked.connect(self._start_extraction)
        controls.addWidget(self._extract_button, 3, 0, 1, 2)

        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._cancel_extraction)
        controls.addWidget(self._cancel_button, 3, 2)

        self._open_folder_button = QPushButton("Open output folder")
        self._open_folder_button.setEnabled(False)
        self._open_folder_button.clicked.connect(self._open_output_folder)
        controls.addWidget(self._open_folder_button, 3, 3)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("%p%")
        controls.addWidget(self._progress, 4, 0, 1, 4)

        self._status_label = QLabel("Idle.")
        self._status_label.setObjectName("hint")
        self._status_label.setWordWrap(True)
        self._status_label.setMinimumWidth(1)
        controls.addWidget(self._status_label, 5, 0, 1, 4)

        controls.setColumnStretch(1, 1)
        controls.setColumnStretch(2, 0)
        top_row.addWidget(controls_box, 3)
        root.addLayout(top_row)

        # ---- trim / time range + media information (groups 3 + 4) --------
        mid_row = QHBoxLayout()
        mid_row.setContentsMargins(0, 0, 0, 0)
        mid_row.setSpacing(10)
        mid_row.addWidget(self._build_trim_box(), 3)

        # ---- media information (two key/value columns) --------------------
        info_box = QGroupBox("Source information")
        info_box.setMinimumWidth(340)
        info_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        info_layout = QGridLayout(info_box)
        info_layout.setHorizontalSpacing(14)
        info_layout.setVerticalSpacing(4)
        info_layout.setColumnMinimumWidth(0, 74)
        info_layout.setColumnMinimumWidth(2, 66)
        info_layout.setColumnMinimumWidth(1, 84)
        info_layout.setColumnMinimumWidth(3, 78)
        info_layout.setColumnStretch(1, 1)
        info_layout.setColumnStretch(3, 1)

        self._info_labels = {}
        rows = (
            ("file", "File"),
            ("kind", "Content"),
            ("container", "Container"),
            ("duration", "Duration"),
            ("video", "Video codec"),
            ("codec", "Audio codec"),
            ("sample_rate", "Sample-rate"),
            ("channels", "Channels"),
            ("bit_rate", "Bit-rate"),
            ("size", "File size"),
        )
        for index, (key, caption) in enumerate(rows):
            row, column = divmod(index, 2)          # two pairs per row
            caption_label = QLabel(f"{caption}:")
            caption_label.setObjectName("infoKey")
            caption_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value = QLabel("-")
            value.setObjectName("infoValue")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value.setWordWrap(True)
            value.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            self._info_labels[key] = value
            info_layout.addWidget(caption_label, row, column * 2)
            info_layout.addWidget(value, row, column * 2 + 1)
        mid_row.addWidget(info_box, 4)
        root.addLayout(mid_row)

        # ---- wave-form ----------------------------------------------------
        wave_box = QGroupBox("Wave-form and trim selection")
        wave_layout = QVBoxLayout(wave_box)
        wave_layout.setContentsMargins(8, 8, 8, 8)
        wave_layout.setSpacing(6)

        self._plot = pg.PlotWidget()
        self._plot.setBackground("#14141b")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setLabel("bottom", "Time", units="s")
        self._plot.setLabel("left", "Amplitude")
        for axis_name in ("left", "bottom"):
            axis = self._plot.getPlotItem().getAxis(axis_name)
            axis.setPen(pg.mkPen("#55556a"))
            axis.setTextPen(pg.mkPen("#c8c8d4"))
        self._plot.setMouseEnabled(x=True, y=False)
        self._plot.setYRange(-1.05, 1.05, padding=0)
        self._plot.setMinimumHeight(112)
        self._plot.setMaximumHeight(190)
        self._plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Draggable selection used to pick the trim range with the mouse.
        self._trim_region = pg.LinearRegionItem(values=(0.0, 1.0), movable=True)
        self._trim_region.setBrush(pg.mkBrush(74, 168, 255, 45))
        self._trim_region.setHoverBrush(pg.mkBrush(74, 168, 255, 75))
        for line in self._trim_region.lines:
            line.setPen(pg.mkPen("#ffc14d", width=2))
        self._trim_region.setZValue(20)
        self._trim_region.sigRegionChanged.connect(self._sync_spins_from_region)
        self._trim_region_added = False

        # Moving play-head of the unified preview player.
        self._playhead = pg.InfiniteLine(pos=0.0, angle=90, movable=False,
                                         pen=pg.mkPen("#ff5252", width=2))
        self._playhead.setZValue(30)
        self._playhead.hide()
        self._playhead_added = False

        wave_layout.addWidget(self._plot)

        wave_hint = QLabel(
            "Drag the two orange handles (or the shaded area) to select the range with the "
            "mouse, or type exact start/end times with millisecond precision below. "
            "Both streams are driven by the same transport and always play only the "
            "selected range; the red line on the wave-form follows the preview."
        )
        wave_hint.setObjectName("hint")
        wave_hint.setWordWrap(True)
        wave_layout.addWidget(wave_hint)
        root.addWidget(wave_box)

        # ---- previews (independent video and audio players) ---------------
        root.addWidget(self._build_preview_box())

        # ---- log ----------------------------------------------------------
        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(8, 8, 8, 8)

        log_buttons = QHBoxLayout()
        log_buttons.setContentsMargins(0, 0, 0, 0)
        log_buttons.setSpacing(8)
        log_buttons.addStretch(1)
        self._save_log_button = QPushButton("Save log ...")
        self._save_log_button.setToolTip("Write the log to a text file (Ctrl+S)")
        self._save_log_button.clicked.connect(self._save_log)
        log_buttons.addWidget(self._save_log_button)
        self._clear_log_button = QPushButton("Clear log")
        self._clear_log_button.setToolTip("Remove all log lines (Ctrl+L)")
        self._clear_log_button.clicked.connect(self._clear_log)
        log_buttons.addWidget(self._clear_log_button)
        log_layout.addLayout(log_buttons)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(2000)
        self._log_view.setFont(QFont("Consolas", 9))
        self._log_view.setMinimumHeight(140)
        log_layout.addWidget(self._log_view)
        root.addWidget(log_box, stretch=1)

    # ------------------------------------------------------- preview widgets
    def _build_preview_box(self) -> QGroupBox:
        """Create the *unified* preview player (video + audio in one section).

        One transport (play / pause / stop / loop / volume / position) drives the
        embedded video canvas and the audio engine together, so both start at the
        same moment and stay in step. Files without a video stream simply show
        the audio player, files without audio show only the video.
        """
        box = QGroupBox("Preview player (selected range)")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ---- embedded video canvas --------------------------------------
        self._video_canvas = VideoCanvas()
        self._video_canvas.setPlaceholder("Load a video file to preview it here.")
        layout.addWidget(self._video_canvas, stretch=1)

        # ---- single transport row ---------------------------------------
        transport = QHBoxLayout()
        transport.setContentsMargins(0, 0, 0, 0)
        transport.setSpacing(8)

        self._play_button = QPushButton("\u25b6  Play")
        self._play_button.setObjectName("play")
        self._play_button.setEnabled(False)
        self._play_button.setToolTip("Play the selected range - shortcut: Space")
        self._play_button.clicked.connect(self._toggle_preview)
        transport.addWidget(self._play_button)

        self._pause_button = QPushButton("\u23f8  Pause")
        self._pause_button.setEnabled(False)
        self._pause_button.clicked.connect(self._pause_preview)
        transport.addWidget(self._pause_button)

        self._stop_button = QPushButton("\u23f9  Stop")
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self._stop_preview)
        transport.addWidget(self._stop_button)

        self._loop_checkbox = QCheckBox("Loop")
        self._loop_checkbox.setEnabled(False)
        transport.addWidget(self._loop_checkbox)

        volume_label = QLabel("Volume:")
        volume_label.setObjectName("hint")
        transport.addWidget(volume_label)
        self._volume_slider = QSlider(Qt.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(80)
        self._volume_slider.setFixedWidth(110)
        self._volume_slider.setToolTip("Playback volume (0 = mute)")
        self._volume_slider.valueChanged.connect(self._on_volume_changed)
        transport.addWidget(self._volume_slider)

        self._playback_label = QLabel("--:--.---")
        self._playback_label.setObjectName("timecode")
        self._playback_label.setMinimumWidth(84)
        transport.addWidget(self._playback_label)

        self._system_preview_button = QPushButton("Open in system player")
        self._system_preview_button.setEnabled(False)
        self._system_preview_button.setToolTip(
            "Render only the selected range to a temporary audio file and open it "
            "with the default media player of your system."
        )
        self._system_preview_button.clicked.connect(self._preview_in_system_player)
        transport.addWidget(self._system_preview_button)
        transport.addStretch(1)
        layout.addLayout(transport)

        # ---- status line -------------------------------------------------
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(10)
        self._engine_label = QLabel("audio engine: -")
        self._engine_label.setObjectName("hint")
        status_row.addWidget(self._engine_label)
        self._stream_label = QLabel("")
        self._stream_label.setObjectName("hint")
        status_row.addWidget(self._stream_label)
        status_row.addStretch(1)
        layout.addLayout(status_row)

        self._transport_hint = QLabel("")
        self._transport_hint.setObjectName("summary")
        layout.addWidget(self._transport_hint)

        hint = QLabel(
            "One transport drives the embedded video and the audio engine together, so "
            "they start and stop at the same moment. Only the trimmed range is played "
            "(Space = play / pause). If an audio engine cannot start, the log reports why "
            "and the next one is tried automatically."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return box

    # ------------------------------------------------------- preview engine
    # A single preview session drives both streams:
    #   * video  -> VideoFrameWorker (ffmpeg -re ... rawvideo -> QImage -> canvas)
    #   * audio  -> one of several engines, chosen automatically:
    #       1. ffplay            (comes with FFmpeg, exact seek)
    #       2. QtMultimedia      (QMediaPlayer)
    #       3. winsound          (Windows, plays a rendered WAV asynchronously)
    #       4. system player     (renders the range and opens the OS player)
    def _preview_dimensions(self) -> Tuple[int, int]:
        """Frame size used for the embedded preview (small and even)."""
        width = self._info.video_width if self._info and self._info.video_width else 640
        height = self._info.video_height if self._info and self._info.video_height else 360
        if height <= 0 or width <= 0:
            return 640, 360
        scale = min(1.0, 640.0 / float(width))
        new_w = int(round(width * scale / 2.0)) * 2
        new_h = int(round(height * scale / 2.0)) * 2
        return max(64, new_w), max(64, new_h)

    # ---- engine initialisation -------------------------------------------
    def _setup_player(self) -> None:
        """Prepare the audio engines (ffplay preferred, Qt as fallback)."""
        self._ffplay = find_ffplay(self._ffmpeg)
        self._audio_process = None

        if self._ffplay:
            self._playback_available = True
            self._append_log(f"Audio engine 1: ffplay ({self._ffplay})")
        else:
            self._append_log("Audio engine 1: ffplay not found (ships with FFmpeg).")

        if MULTIMEDIA_AVAILABLE:
            try:
                self._qt_player = QMediaPlayer(self)
                self._qt_player.positionChanged.connect(self._on_player_position)
                self._qt_player.stateChanged.connect(self._on_player_state)
                self._qt_player.mediaStatusChanged.connect(self._on_media_status)
                try:
                    self._qt_player.error.connect(self._on_player_error)
                except Exception:
                    pass
                self._qt_player.setVolume(self._volume_slider.value())
                self._playback_available = True
                self._append_log("Audio engine 2: QtMultimedia available")
            except Exception as exc:  # pragma: no cover - defensive
                self._qt_player = None
                self._append_log(f"Audio engine 2: QtMultimedia unavailable ({exc})")
        else:
            self._qt_player = None
            self._append_log("Audio engine 2: QtMultimedia module not installed")

        if os.name == "nt":
            self._append_log("Audio engine 3: winsound available")
        self._append_log("Audio engine 4: system media player (always available)")
        if not self._playback_available:
            self._append_log(
                "NOTE: no direct audio engine found - the selection will be rendered "
                "and handed to the system media player."
            )

    def _set_player_source(self, path: str) -> None:
        """Reset every preview engine and point the Qt player at *path*."""
        self._stop_preview(silent=True)
        if self._qt_player is not None:
            try:
                self._qt_player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
            except Exception:
                pass
        self._update_playback_controls()

    # ---- available engines ----------------------------------------------
    def _audio_engine_candidates(self) -> List[str]:
        candidates: List[str] = []
        if self._ffplay:
            candidates.append("ffplay")
        if self._qt_player is not None:
            candidates.append("qt")
        if os.name == "nt":
            candidates.append("winsound")
        candidates.append("system")
        return candidates

    @staticmethod
    def _backend_label(backend: str) -> str:
        return {
            "ffplay": "ffplay",
            "qt": "QtMultimedia",
            "winsound": "Windows audio (winsound)",
            "system": "system media player",
        }.get(backend, backend or "-")

    # ---- state helpers ---------------------------------------------------
    def _has_video(self) -> bool:
        return bool(self._info and self._info.video_codec)

    def _has_audio(self) -> bool:
        return bool(self._info and self._info.has_audio)

    def _video_running(self) -> bool:
        return self._video_worker is not None and self._video_worker.isRunning()

    def _audio_running(self) -> bool:
        if self._audio_backend == "ffplay":
            return self._audio_process is not None and self._audio_process.poll() is None
        if self._audio_backend == "qt":
            return self._is_playing() or self._is_paused()
        if self._audio_backend == "winsound":
            return self._winsound_active
        return False

    def _preview_running(self) -> bool:
        return self._video_running() or self._audio_running() or self._preview_paused

    def _update_playback_controls(self) -> None:
        """Drive the transport buttons from a strict state machine.

        States
        ------
        ``idle``     : nothing playing          -> Play enabled, Pause/Stop off
        ``playing``  : preview running           -> Play off, Pause/Stop enabled
        ``paused``   : preview paused            -> Play ("Resume") + Stop enabled
        ``external`` : system player took over   -> Play off, Pause off, Stop on
        """
        has_source = bool(self._source_path) and self._info is not None
        playable = (
            has_source and bool(self._ffmpeg)
            and (self._has_audio() or self._has_video())
        )
        state = self._transport_state

        self._play_button.setEnabled(playable and state in ("idle", "paused"))
        self._pause_button.setEnabled(playable and state == "playing")
        self._stop_button.setEnabled(playable and state in ("playing", "paused", "external"))
        self._play_button.setText(
            "\u25b6  Resume" if state == "paused" else "\u25b6  Play"
        )
        self._loop_checkbox.setEnabled(playable)
        self._volume_slider.setEnabled(playable and self._has_audio())
        self._system_preview_button.setEnabled(
            has_source and self._has_audio() and bool(self._ffmpeg)
            and (self._preview_worker is None or not self._preview_worker.isRunning())
        )

        self._video_canvas.setVisible(not has_source or self._has_video())
        if has_source:
            if self._has_video():
                self._stream_label.setText(
                    f"streams: video {self._info.video_width}x{self._info.video_height}"
                    + (f" + audio {self._info.codec.upper()}"
                       if self._has_audio() else " (no audio)")
                )
            elif self._has_audio():
                self._stream_label.setText(f"streams: audio only ({self._info.codec.upper()})")
            else:
                self._stream_label.setText("streams: none")
            if not self._has_video():
                self._video_canvas.setPlaceholder(
                    "Audio file - no video stream\n\n"
                    "Use Play to listen to the selected range."
                )
            if state == "idle":
                self._transport_hint.setText(self._idle_hint())
            elif state == "paused":
                self._transport_hint.setText("Paused - press Resume to continue")
            elif state == "external":
                self._transport_hint.setText("Playing in the system media player")
            else:
                self._transport_hint.setText("Playing the selected range")
        else:
            self._video_canvas.setPlaceholder("Load a file to preview it here.")
            self._transport_hint.setText("")

    def _idle_hint(self) -> str:
        start, end = self._trim_range()
        if self._info is None:
            return ""
        return (
            f"Ready - Play will render {format_timecode(start)} -> "
            f"{format_timecode(end)}"
        )

    def _set_transport_state(self, state: str) -> None:
        self._transport_state = state
        self._update_playback_controls()

    # ---- transport -------------------------------------------------------
    def _toggle_preview(self) -> None:
        # Do not hijack the space bar while the user is typing in a field.
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QPlainTextEdit, QDoubleSpinBox)):
            return
        if self._transport_state == "playing":
            self._pause_preview()
        elif self._transport_state == "paused":
            self._resume_preview()
        elif self._transport_state == "idle":
            self._start_preview()
        else:                       # external
            self._stop_preview()

    def _start_preview(self, start_at: Optional[float] = None) -> None:
        """Start video and audio together at the beginning of the selection."""
        if self._info is None or not self._ffmpeg:
            return
        self._stop_preview(silent=True)
        start, end = self._trim_range()
        if start_at is not None:
            start = max(start, min(start_at, end - 0.05))
        self._preview_start = start
        self._preview_end = max(start + 0.05, end)
        self._preview_paused = False
        self._preview_external = False
        self._transport_state = "playing"
        self._preview_clock.restart()
        self._preview_ticker.start()

        # The audio engine is started *first*: it needs a moment to spin up, so
        # this is what makes the sound start together with the first frames.
        if self._has_audio():
            self._start_audio_engine(start, self._preview_end)
        else:
            self._engine_label.setText("audio engine: - (no audio stream)")

        if self._has_video():
            self._start_video_engine(start, self._preview_end - start)
        else:
            self._video_canvas.setPlaceholder(
                "Audio file - no video stream\n\n"
                "Playing the selected range ..."
            )

        self._playhead.setPos(start)
        self._playhead.show()
        self._playback_label.setText(format_timecode(start))
        self._append_log(
            f"Preview {format_timecode(start)} -> {format_timecode(self._preview_end)}"
            + (" (video + audio)" if self._has_video() and self._has_audio() else "")
        )
        self._update_playback_controls()

    def _pause_preview(self) -> None:
        if not self._preview_running():
            return
        self._preview_paused_at = self._current_position()
        self._preview_paused = True
        self._transport_state = "paused"
        self._preview_ticker.stop()
        self._stop_video_engine()
        self._pause_audio_engine()
        self._update_playback_controls()

    def _resume_preview(self) -> None:
        self._preview_paused = False
        self._start_preview(self._preview_paused_at)

    def _stop_preview(self, silent: bool = False) -> None:
        self._stop_video_engine()
        self._stop_audio_engine()
        self._preview_ticker.stop()
        self._preview_paused = False
        self._preview_external = False
        self._transport_state = "idle"
        self._preview_paused_at = self._preview_start
        self._trim_sync_timer.stop()
        self._playhead.hide()
        self._playback_label.setText("--:--.---")
        if self._has_video():
            self._video_canvas.setPlaceholder(
                "Press Play to preview the selected range\n"
                f"({self._info.video_width}x{self._info.video_height})"
            )
        if not silent:
            self._engine_label.setText("audio engine: -")
        self._update_playback_controls()

    def _current_position(self) -> float:
        """Position of the preview, used for the read-out and the play-head."""
        if self._audio_backend in ("ffplay", "winsound") and self._audio_clock.isValid():
            if self._audio_running():
                return self._audio_origin + self._audio_clock.elapsed() / 1000.0
        if self._audio_backend == "qt" and self._audio_running():
            try:
                return self._qt_player.position() / 1000.0
            except Exception:
                pass
        if self._preview_clock.isValid() and (self._video_running() or self._preview_paused):
            return self._preview_start + min(
                self._preview_clock.elapsed() / 1000.0,
                self._preview_end - self._preview_start,
            )
        return self._preview_start

    def _on_preview_tick(self) -> None:
        """Update the read-out, the play-head and the end-of-range handling."""
        position = self._current_position()
        self._playback_label.setText(format_timecode(position))
        self._playhead.setPos(position)
        self._playhead.show()

        if self._preview_external and not self._video_running():
            self._preview_ticker.stop()
            return

        video_done = self._has_video() and not self._video_running() and not self._preview_paused
        audio_done = self._has_audio() and not self._audio_running() and not self._preview_paused
        reached_end = position >= self._preview_end - 0.05
        finished = reached_end or (video_done and (audio_done or not self._has_audio()))

        if not finished:
            return
        if self._loop_checkbox.isChecked():
            self._start_preview()
        else:
            self._stop_preview()

    # ---- audio engines ---------------------------------------------------
    def _start_audio_engine(self, start: float, end: float) -> None:
        self._audio_start, self._audio_end = start, end
        self._audio_origin = start
        self._engine_label.setText("audio engine: starting ...")
        self._try_audio_backends(self._audio_engine_candidates())

    def _try_audio_backends(self, candidates: List[str]) -> None:
        for backend in candidates:
            if self._start_with_backend(backend):
                self._audio_backend = backend
                self._engine_label.setText(f"audio engine: {self._backend_label(backend)}")
                return
        self._engine_label.setText("audio engine: none")
        self._append_log(
            "NOTE: no audio engine could play the selection - use 'Open in system player'."
        )

    def _start_with_backend(self, backend: str) -> bool:
        if backend == "ffplay":
            return self._play_with_ffplay()
        if backend == "qt":
            return self._play_with_qt()
        if backend == "winsound":
            return self._render_selection("wav", self._play_with_winsound)
        if backend == "system":
            return self._render_selection("mp3", self._play_with_system_engine)
        return False

    def _play_with_system_engine(self, path: str) -> None:
        """The OS player takes over -> the internal transport goes idle."""
        self._audio_backend = "system"
        self._preview_external = True
        self._transport_state = "external"
        self._engine_label.setText(f"audio engine: {self._backend_label('system')}")
        self._append_log(
            "The selection was rendered to a temporary file and handed to the system "
            "media player (internal pause/loop are not available for this engine)."
        )
        self._open_with_system_player(path)
        if not self._has_video():
            self._preview_ticker.stop()
            self._playback_label.setText("external")
            self._playhead.hide()
        self._update_playback_controls()

    def _play_with_ffplay(self) -> bool:
        if not self._ffplay:
            return False
        duration = self._audio_end - self._audio_start
        log_file = tempfile.NamedTemporaryFile(prefix="ffplay_", suffix=".log", delete=False)
        log_file.close()
        self._ffplay_log_path = log_file.name
        cmd = [
            self._ffplay, "-nodisp", "-autoexit", "-loglevel", "error",
            "-volume", str(int(self._volume_slider.value())),
            "-ss", f"{self._audio_start:.3f}", "-t", f"{duration:.3f}",
            self._source_path,
        ]
        try:
            with open(self._ffplay_log_path, "wb") as handle:
                self._audio_process = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=handle,
                    stdin=subprocess.DEVNULL, **_no_window_kwargs(),
                )
        except Exception as exc:
            self._append_log(f"NOTE: ffplay could not start ({exc}).")
            self._audio_process = None
            return False
        self._audio_clock.restart()
        QTimer.singleShot(900, self._verify_ffplay_started)
        return True

    def _verify_ffplay_started(self) -> None:
        process = self._audio_process
        if process is None or self._audio_backend != "ffplay":
            return
        if process.poll() is None:
            return
        reason = self._read_ffplay_error()
        self._append_log(
            "NOTE: ffplay could not play the selection"
            + (f" ({reason})" if reason else "")
            + " - trying the next audio engine."
        )
        self._audio_process = None
        remaining = [b for b in self._audio_engine_candidates() if b != "ffplay"]
        self._try_audio_backends(remaining)

    def _read_ffplay_error(self) -> str:
        path = getattr(self, "_ffplay_log_path", "")
        if not path or not os.path.isfile(path):
            return ""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                return handle.read().strip().replace("\n", " ")[-200:]
        except OSError:
            return ""

    def _play_with_qt(self) -> bool:
        if self._qt_player is None:
            return False
        try:
            self._qt_play_attempt = True
            self._qt_player.setVolume(int(self._volume_slider.value()))
            self._qt_player.setPosition(int(round(self._audio_start * 1000.0)))
            self._qt_player.play()
        except Exception as exc:
            self._append_log(f"NOTE: QtMultimedia could not play the file ({exc}).")
            return False
        self._audio_clock.restart()
        QTimer.singleShot(900, self._verify_qt_started)
        return True

    def _verify_qt_started(self) -> None:
        if self._audio_backend != "qt" or not self._qt_play_attempt:
            return
        self._qt_play_attempt = False
        if self._is_playing() or self._is_paused():
            return
        message = ""
        try:
            message = self._qt_player.errorString() or ""
        except Exception:
            pass
        self._append_log(
            "NOTE: QtMultimedia is not usable"
            + (f" ({message})" if message else "")
            + " - trying the next audio engine."
        )
        remaining = [b for b in self._audio_engine_candidates() if b != "qt"]
        self._try_audio_backends(remaining)

    def _play_with_winsound(self, path: str) -> None:
        try:
            import winsound  # type: ignore

            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as exc:
            self._append_log(f"NOTE: winsound playback failed ({exc}) - using the system player.")
            self._try_audio_backends(["system"])
            return
        self._winsound_active = True
        self._audio_backend = "winsound"
        self._audio_clock.restart()
        self._engine_label.setText(f"audio engine: {self._backend_label('winsound')}")
        self._append_log(
            "Audio preview through winsound (volume/pause are not supported by this engine)."
        )
        self._update_playback_controls()

    def _render_selection(self, format_key: str, on_ready) -> bool:
        """Render the selected range to a temporary file, then call *on_ready*."""
        fmt = get_output_format(format_key)
        temp_dir = tempfile.mkdtemp(prefix="media_converter_preview_")
        target = os.path.join(temp_dir, "preview" + fmt.extension)
        self._preview_files.append(target)
        self._append_log(
            f"Rendering the selection to a temporary {fmt.name} file for the "
            f"{self._backend_label('winsound' if format_key == 'wav' else 'system')} engine ..."
        )
        worker = ExtractWorker(
            self._source_path, target, "320", self._info, self._ffmpeg or "", self,
            start=self._audio_start, end=self._audio_end, output_format=fmt,
        )
        self._preview_worker = worker

        def finished(ok: bool, path: str, message: str) -> None:
            self._preview_worker = None
            if not ok:
                self._append_log(f"NOTE: preview rendering failed ({message}).")
                self._update_playback_controls()
                return
            try:
                on_ready(path)
            except Exception as exc:  # pragma: no cover - defensive
                self._append_log(f"NOTE: preview playback failed ({exc}).")

        worker.finishedWithResult.connect(finished)
        worker.start()
        return True

    def _pause_audio_engine(self) -> None:
        if not self._audio_running():
            return
        self._audio_paused_at = self._current_position()
        self._audio_paused = True
        if self._audio_backend == "qt":
            try:
                self._qt_player.pause()
            except Exception:
                pass
        elif self._audio_backend == "ffplay":
            self._terminate_audio_process()
        else:
            self._append_log(
                "NOTE: this audio engine cannot pause - playback stopped."
            )
            self._stop_audio_engine()

    def _resume_audio_engine(self) -> None:
        self._audio_paused = False
        if self._audio_backend == "qt":
            try:
                self._qt_player.play()
                return
            except Exception:
                pass
        self._start_audio_engine(self._audio_paused_at, self._audio_end)

    def _terminate_audio_process(self) -> None:
        process = self._audio_process
        self._audio_process = None
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=2)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass

    def _stop_audio_engine(self) -> None:
        self._preview_external = False
        self._terminate_audio_process()
        self._audio_paused = False
        self._audio_paused_at = self._audio_start
        if self._winsound_active:
            try:
                import winsound  # type: ignore

                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
            self._winsound_active = False
        if self._qt_player is not None:
            try:
                if self._qt_player.state() != QMediaPlayer.StoppedState:
                    self._qt_player.stop()
            except Exception:
                pass
        self._audio_backend = ""

    def _on_volume_changed(self, value: int) -> None:
        if self._qt_player is not None:
            try:
                self._qt_player.setVolume(int(value))
            except Exception:
                pass
        if self._audio_backend == "ffplay" and self._audio_running():
            # ffplay has no runtime volume control -> restart at the same spot.
            resume_at = self._current_position()
            self._terminate_audio_process()
            self._start_audio_engine(resume_at, self._preview_end)

    # ---- Qt fallback player signals --------------------------------------
    def _is_playing(self) -> bool:
        if not self._qt_player or QMediaPlayer is None:
            return False
        try:
            return self._qt_player.state() == QMediaPlayer.PlayingState
        except Exception:
            return False

    def _is_paused(self) -> bool:
        if not self._qt_player or QMediaPlayer is None:
            return False
        try:
            return self._qt_player.state() == QMediaPlayer.PausedState
        except Exception:
            return False

    def _on_player_state(self, *_args) -> None:
        self._update_playback_controls()

    def _on_player_error(self, *_args) -> None:
        message = ""
        try:
            message = self._qt_player.errorString()
        except Exception:
            pass
        if message:
            self._append_log(f"NOTE: Qt audio preview: {message}")

    def _on_media_status(self, status) -> None:
        if not self._qt_player or QMediaPlayer is None:
            return
        if status == QMediaPlayer.EndOfMedia and self._audio_backend == "qt":
            if self._loop_checkbox.isChecked():
                self._start_preview()
            else:
                self._stop_preview()

    def _on_player_position(self, milliseconds: int) -> None:
        """Position updates of the Qt engine (ffplay uses the clock instead)."""
        if self._audio_backend != "qt":
            return
        position = milliseconds / 1000.0
        self._playback_label.setText(format_timecode(position))
        self._playhead.setPos(position)
        self._playhead.show()
        if self._is_playing() and position >= self._audio_end - 0.02:
            if self._loop_checkbox.isChecked():
                self._start_preview()
            else:
                self._stop_preview()

    # ---- video engine ----------------------------------------------------
    def _start_video_engine(self, start: float, duration: float) -> None:
        self._stop_video_engine()
        self._video_end = start + duration
        width, height = self._preview_dimensions()
        self._video_worker = VideoFrameWorker(
            self._ffmpeg, self._source_path, start, duration, width, height, self,
        )
        self._video_worker.failed.connect(self._on_video_failed)
        self._video_worker.finishedPlaying.connect(self._on_video_finished)
        self._video_worker.start()
        self._frame_ticker.start()

    def _pull_video_frame(self) -> None:
        worker = self._video_worker
        if worker is None:
            self._frame_ticker.stop()
            return
        image = worker.take_latest()
        if image is not None:
            self._video_canvas.setImage(image)

    def _stop_video_engine(self) -> None:
        self._frame_ticker.stop()
        worker = self._video_worker
        self._video_worker = None
        if worker is not None:
            worker.stop()
            worker.wait(2000)

    def _on_video_failed(self, message: str) -> None:
        self._append_log(f"NOTE: the embedded video preview failed: {message}")
        self._video_worker = None
        self._video_canvas.setPlaceholder("The embedded video preview failed.\n"
                                          "See the log for details.")

    def _on_video_finished(self) -> None:
        """The decoder reached the end of the selection."""
        if self._preview_paused:
            return
        self._video_worker = None
        self._frame_ticker.stop()
        if self._preview_external:
            # The system player is still responsible for the audio - keep the
            # session visible instead of tearing it down.
            self._playback_label.setText("external")
            self._update_playback_controls()
            return
        if not self._has_audio() or not self._audio_running():
            # Nothing else is playing -> finish the whole preview.
            if self._loop_checkbox.isChecked():
                self._start_preview()
            else:
                self._stop_preview()

    def _sync_preview_with_trim(self) -> None:
        """React *live* to trim changes while the preview is running.

        * the end of the range always follows the handle immediately;
        * if the play-head falls outside the new range the preview restarts at
          the new start (or stops when looping is off and the end moved back);
        * the audio engine is restarted with a short debounce, because engines
          like ffplay take their range as a fixed ``-ss/-t`` argument.
        """
        if self._info is None:
            return
        start, end = self._trim_range()
        self._preview_end = end
        if self._transport_state == "idle":
            self._transport_hint.setText(self._idle_hint())
            return
        if self._transport_state != "playing":
            return

        position = self._current_position()
        if position < start - 0.02:
            # The left handle passed the play-head -> continue from the new start.
            self._trim_sync_timer.stop()
            self._start_preview(start)
            return
        if position > end - 0.02:
            # The right handle passed the play-head -> end the preview (or loop).
            self._trim_sync_timer.stop()
            if self._loop_checkbox.isChecked():
                self._start_preview(start)
            else:
                self._stop_preview()
            return
        # Still inside the range: the video keeps running, but the audio engine
        # must be restarted so that it stops exactly at the new end.
        if self._has_audio() and self._transport_state == "playing":
            self._trim_sync_timer.start()

    def _apply_trim_to_preview(self) -> None:
        """Apply a new trim range to the running preview (debounced).

        When the video is already running it is left untouched (no frame
        hiccup) and only the audio engine is restarted with the new range.
        """
        if self._transport_state != "playing":
            return
        start, end = self._trim_range()
        self._preview_end = end
        position = max(start, min(self._current_position(), max(start, end - 0.05)))
        video_covers_range = self._video_running() and self._video_end >= end - 0.02
        if video_covers_range and self._has_audio():
            # The picture keeps playing; only the sound needs the new range.
            self._stop_audio_engine()
            self._start_audio_engine(position, end)
        else:
            self._start_preview(position)

    # ---- system player preview -------------------------------------------
    def _preview_in_system_player(self) -> None:
        """Render the selected range and open it with the system's media player."""
        if not self._source_path or self._info is None or not self._ffmpeg:
            return
        if not self._info.has_audio:
            QMessageBox.information(self, APP_NAME,
                                    "The file has no audio stream to preview.")
            return
        start, end = self._trim_range()
        temp_dir = tempfile.mkdtemp(prefix="media_converter_preview_")
        target = os.path.join(temp_dir, "preview.mp3")
        self._preview_files.append(target)
        self._append_log(
            f"Preparing a system preview of {format_timecode(start)} -> "
            f"{format_timecode(end)} ..."
        )
        self._system_preview_button.setEnabled(False)
        self._preview_worker = ExtractWorker(
            self._source_path, target, "320", self._info, self._ffmpeg, self,
            start=start, end=end, output_format=get_output_format("mp3"),
        )
        self._preview_worker.finishedWithResult.connect(self._on_preview_ready)
        self._preview_worker.start()

    def _on_preview_ready(self, ok: bool, path: str, message: str) -> None:
        self._preview_worker = None
        if not ok:
            self._append_log(f"Preview failed: {message}")
            QMessageBox.critical(self, APP_NAME, f"Could not create the preview:\n\n{message}")
            self._update_playback_controls()
            return
        self._append_log(f"Preview file ready: {path}")
        self._open_with_system_player(path)
        self._update_playback_controls()

    def _open_with_system_player(self, path: str) -> None:
        """Open *path* with the default application of the operating system."""
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:  # pragma: no cover - depends on the desktop
            self._append_log(f"NOTE: could not start the system player ({exc}).")
            QMessageBox.information(
                self, APP_NAME,
                f"The preview was written to:\n{path}\n\n"
                "It could not be opened automatically - open it manually.",
            )

    # ------------------------------------------------------------- log file
    def _save_log(self) -> None:
        """Write the current log content to a text file chosen by the user."""
        text = self._log_view.toPlainText()
        if not text.strip():
            QMessageBox.information(self, APP_NAME, "The log is empty - nothing to save.")
            return
        suggested = os.path.join(
            os.path.dirname(os.path.abspath(self._source_path)) if self._source_path
            else os.path.expanduser("~"),
            "conversion-log.txt",
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the log as", suggested, "Text file (*.txt);;Log file (*.log);;All files (*)"
        )
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".txt"
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self._log_view.toPlainText() + "\n")
        except OSError as exc:
            QMessageBox.critical(self, APP_NAME, f"Could not write the log file:\n{exc}")
            return
        self._append_log(f"Log saved to: {path}")

    def _clear_log(self) -> None:
        """Empty the log view."""
        self._log_view.clear()

    # --------------------------------------------------------- trim widgets
    def _build_trim_box(self) -> QGroupBox:
        """Create the group box holding the trim (time range) controls."""
        box = QGroupBox("Trim (select a part of the file)")
        box.setMinimumWidth(360)
        box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        layout = QGridLayout(box)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        # Start / end on two compact rows, the selection summary on the right.
        layout.addWidget(QLabel("Start time:"), 0, 0)
        self._start_spin = self._make_time_spin()
        layout.addWidget(self._start_spin, 0, 1)
        self._start_label = QLabel(format_timecode(0.0))
        self._start_label.setObjectName("timecode")
        layout.addWidget(self._start_label, 0, 2)

        layout.addWidget(QLabel("End time:"), 1, 0)
        self._end_spin = self._make_time_spin()
        layout.addWidget(self._end_spin, 1, 1)
        self._end_label = QLabel(format_timecode(0.0))
        self._end_label.setObjectName("timecode")
        layout.addWidget(self._end_label, 1, 2)

        self._selection_label = QLabel("Selection: whole file")
        self._selection_label.setObjectName("selection")
        self._selection_label.setWordWrap(True)
        self._selection_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self._selection_label, 0, 3, 2, 1)

        self._trim_reset_button = QPushButton("Reset")
        self._trim_reset_button.setToolTip("Select the whole file again")
        self._trim_reset_button.setEnabled(False)
        self._trim_reset_button.clicked.connect(self._reset_trim)
        layout.addWidget(self._trim_reset_button, 2, 0)

        self._trim_start_button = QPushButton("Start 0")
        self._trim_start_button.setToolTip("Set the start time to 0")
        self._trim_start_button.setEnabled(False)
        self._trim_start_button.clicked.connect(lambda: self._start_spin.setValue(0.0))
        layout.addWidget(self._trim_start_button, 2, 1)

        self._trim_end_button = QPushButton("File end")
        self._trim_end_button.setToolTip("Set the end time to the end of the file")
        self._trim_end_button.setEnabled(False)
        self._trim_end_button.clicked.connect(self._set_end_to_file_end)
        layout.addWidget(self._trim_end_button, 2, 2)

        hint = QLabel(
            "Millisecond precision (e.g. 12.345); the exact time code is shown next "
            "to each field. Trimming applies to the preview and the export."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint, 3, 0, 1, 4)

        layout.setColumnStretch(3, 1)
        layout.setRowStretch(4, 1)
        return box

    def _make_time_spin(self) -> QDoubleSpinBox:
        """Create a spin box for a time value in seconds (0.001 s steps)."""
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setSingleStep(0.100)
        spin.setSuffix(" s")
        spin.setRange(0.0, 86400.0)
        spin.setKeyboardTracking(False)
        spin.setAccelerated(True)
        spin.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        spin.setMinimumWidth(96)
        spin.setEnabled(False)
        spin.valueChanged.connect(self._sync_region_from_spins)
        return spin

    def _reset_trim(self) -> None:
        """Select the whole file again."""
        duration = self._info.duration if self._info else 0.0
        if duration <= 0:
            return
        self._sync_guard = True
        self._start_spin.setValue(0.0)
        self._end_spin.setValue(duration)
        self._sync_guard = False
        self._trim_region.setRegion((0.0, duration))
        self._update_trim_labels()

    def _set_end_to_file_end(self) -> None:
        duration = self._info.duration if self._info else 0.0
        if duration > self._start_spin.value():
            self._end_spin.setValue(duration)

    def _sync_region_from_spins(self, *_args) -> None:
        """The user typed a value -> move the highlighted region on the plot."""
        if getattr(self, "_sync_guard", False):
            return
        start = self._start_spin.value()
        end = self._end_spin.value()
        if end <= start:
            # Keep the range valid: the second handle follows the first one.
            self._sync_guard = True
            end = min(start + 0.001, self._end_spin.maximum())
            self._end_spin.setValue(end)
            self._sync_guard = False
        self._trim_region.setRegion((start, end))
        self._update_trim_labels()
        self._sync_preview_with_trim()

    def _sync_spins_from_region(self, *_args) -> None:
        """The user dragged the region -> update the numeric fields."""
        if getattr(self, "_sync_guard", False):
            return
        start, end = self._trim_region.getRegion()
        if end <= start:
            end = start + 0.001
        self._sync_guard = True
        self._start_spin.setValue(round(start, 3))
        self._end_spin.setValue(round(end, 3))
        self._sync_guard = False
        self._update_trim_labels()
        self._sync_preview_with_trim()

    def _update_trim_labels(self) -> None:
        """Refresh the time-code labels and the "selection" summary."""
        start = self._start_spin.value()
        end = self._end_spin.value()
        self._start_label.setText(format_timecode(start))
        self._end_label.setText(format_timecode(end))
        length = max(0.0, end - start)
        total = self._info.duration if self._info else 0.0
        milliseconds = int(round(length * 1000))
        if total > 0 and abs(length - total) < 0.0005:
            self._selection_label.setText(f"Selection: whole file ({format_timecode(length)})")
        elif self._info is not None and total > 0:
            self._selection_label.setText(
                f"Selection: {format_timecode(length)} - {length:.3f} s / {milliseconds} ms"
            )
        else:
            self._selection_label.setText("Selection: whole file")
        self._update_extract_state()

    def _set_trim_enabled(self, enabled: bool) -> None:
        """Enable/disable all trim controls at once (used while converting)."""
        for widget in (
            self._start_spin,
            self._end_spin,
            self._trim_reset_button,
            self._trim_start_button,
            self._trim_end_button,
        ):
            widget.setEnabled(enabled)
        self._trim_region.setMovable(enabled)

    def _trim_range(self) -> Tuple[float, float]:
        """Return the currently selected ``(start, end)`` in seconds."""
        start = self._start_spin.value()
        end = self._end_spin.value()
        if end <= start:
            end = start + 0.001
        return start, end

    # ------------------------------------------------------------- helpers
    def _append_log(self, message: str) -> None:
        self._log_view.appendPlainText(message)
        self._log_view.verticalScrollBar().setValue(
            self._log_view.verticalScrollBar().maximum()
        )

    def _set_info_labels(self, info: MediaInfo) -> None:
        values = {
            "file": os.path.basename(info.path),
            "kind": info.kind_text,
            "container": info.container or "unknown",
            "duration": info.duration_text,
            "video": info.video_text,
            "codec": info.codec.upper(),
            "sample_rate": info.sample_rate_text,
            "channels": info.channels_text,
            "bit_rate": info.bit_rate_text,
            "size": _human_size(info.size),
        }
        for key, label in self._info_labels.items():
            label.setText(values.get(key, "-"))

    def _clear_waveform(self) -> None:
        for item in self._plot_items:
            self._plot.removeItem(item)
        self._plot_items.clear()

    def _draw_waveform(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray, float]) -> None:
        x, y_min, y_max, duration = data
        self._clear_waveform()
        pen = pg.mkPen("#4aa8ff", width=1)
        curve_max = pg.PlotDataItem(x, y_max, pen=pen)
        curve_min = pg.PlotDataItem(x, y_min, pen=pen)
        brush = pg.mkBrush(QColor(74, 168, 255, 110))
        fill = pg.FillBetweenItem(curve_max, curve_min, brush=brush)
        self._plot.addItem(fill)
        self._plot.addItem(curve_min)
        self._plot.addItem(curve_max)
        self._plot_items.extend([fill, curve_min, curve_max])
        self._plot.setYRange(-1.05, 1.05, padding=0)
        self._plot.enableAutoRange(axis="x")

        # Attach the draggable trim selection once and keep it in sync.
        # NOTE: the real duration is ``duration`` - the last plotted x value is
        # only the centre of the last bin and would shorten the file slightly.
        total = float(duration)
        self._plot.setXRange(0, max(total if total > 0 else (float(x[-1]) if x.size else 1.0), 0.1),
                             padding=0)
        if total > 0:
            self._trim_region.setBounds((0.0, total))
        if not self._trim_region_added:
            self._plot.addItem(self._trim_region, ignoreBounds=True)
            self._trim_region_added = True
        if not self._playhead_added:
            self._plot.addItem(self._playhead, ignoreBounds=True)
            self._playhead_added = True
        self._playhead.hide()
        if total > 0 and self._info is not None and self._info.duration <= 0:
            # Duration came from the decoded audio, so the trim controls can
            # be enabled even when the container did not report it.
            self._apply_duration_to_trim(total)

    # ----------------------------------------------------------- callbacks
    def _choose_input(self) -> None:
        """Open a file dialog that offers every format FFmpeg can read."""
        start_dir = os.path.dirname(self._source_path) or os.path.expanduser("~")
        video_filter = "Video files (" + " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS) + ")"
        audio_filter = "Audio files (" + " ".join(f"*{ext}" for ext in AUDIO_EXTENSIONS) + ")"
        media_filter = ("All supported media (" + " ".join(f"*{ext}" for ext in MEDIA_EXTENSIONS)
                        + " " + " ".join(f"*{ext.upper()}" for ext in MEDIA_EXTENSIONS) + ")")
        filters = ";;".join((
            f"{media_filter}",
            video_filter,
            audio_filter,
            "All files (*)",
        ))
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a video or audio file", start_dir, filters
        )
        if path:
            self._load_source(path)

    def _choose_output(self) -> None:
        """Ask for the destination path; the chosen extension selects the format."""
        fmt = self._current_format()
        suggested = self._output_edit.text().strip() or self._suggested_output_path()
        if not suggested:
            suggested = os.path.join(os.path.expanduser("~"), "output" + fmt.extension)
        filters = ";;".join(
            [f"{f.key.upper()}  (*{f.extension})" for f in formats_for_kind("audio")]
            + [f"{f.key.upper()} video  (*{f.extension})" for f in formats_for_kind("video")]
            + ["All files (*)"]
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the extracted audio as", suggested, filters
        )
        if not path:
            return

        chosen = format_from_path(path)
        if chosen is None:
            # No (or unknown) extension typed by the user -> append the current one.
            path += fmt.extension
        elif chosen.key != fmt.key:
            # The user picked another extension -> switch the format combo too.
            index = self._format_combo.findData(chosen.key)
            if index >= 0:
                self._format_combo.setCurrentIndex(index)
        self._output_edit.setText(path)
        self._update_extract_state()

    def _suggested_output_path(self) -> str:
        if not self._source_path:
            return ""
        base, _ = os.path.splitext(self._source_path)
        return base + self._current_format().extension

    # -------------------------------------------------------------- formats
    def _rebuild_format_combo(self) -> None:
        """Fill the output-format combo with every target + compatibility state.

        Audio targets come first, then the video targets. Entries that cannot
        be used for the current source file are shown greyed-out with a tooltip
        that explains why (missing encoder, source already in that format,
        audio-only source for a video target, ...).
        """
        previous = self._format_combo.currentData()
        self._format_combo.blockSignals(True)
        self._format_combo.clear()
        model = self._format_combo.model()

        for kind, title in (("audio", "Audio output"), ("video", "Video output")):
            self._format_combo.addItem(f"--- {title} ---", None)
            separator_index = self._format_combo.count() - 1
            item = model.item(separator_index)
            item.setEnabled(False)

            for fmt in formats_for_kind(kind):
                usable, reason = format_availability(
                    fmt, self._ffmpeg, self._info, self._source_path
                )
                label = fmt.label if usable else f"{fmt.label}   [{reason}]"
                self._format_combo.addItem(label, fmt.key)
                index = self._format_combo.count() - 1
                item = model.item(index)
                item.setEnabled(usable)
                if not usable:
                    item.setToolTip(reason)

        # Restore the previous selection when it is still usable, otherwise pick
        # the first usable entry.
        self._format_combo.blockSignals(False)
        target_index = -1
        if previous:
            candidate = self._format_combo.findData(previous)
            if candidate >= 0 and model.item(candidate).isEnabled():
                target_index = candidate
        if target_index < 0:
            for index in range(self._format_combo.count()):
                if self._format_combo.itemData(index) and model.item(index).isEnabled():
                    target_index = index
                    break
        if target_index < 0:
            target_index = next(
                (i for i in range(self._format_combo.count())
                 if self._format_combo.itemData(i)), 0
            )
        self._format_combo.setCurrentIndex(target_index)

    def _rebuild_quality_combo(self) -> None:
        """Show the presets that make sense for the selected target family."""
        fmt = self._current_format()
        presets = VIDEO_PRESETS if fmt.is_video else AUDIO_PRESETS
        previous = self._quality_combo.currentData()
        self._quality_combo.blockSignals(True)
        self._quality_combo.clear()
        for key, label in presets:
            self._quality_combo.addItem(label, key)
        index = max(0, self._quality_combo.findData(previous))
        self._quality_combo.setCurrentIndex(index)
        self._quality_combo.blockSignals(False)

    def _current_format(self) -> OutputFormat:
        """The output format currently selected in the combo box."""
        key = self._format_combo.currentData()
        return get_output_format(key or "mp3")

    def _replace_output_extension(self, path: str, extension: str) -> str:
        base, current = os.path.splitext(path)
        if current.lower() == extension.lower():
            return path
        return base + extension

    def _on_format_changed(self, *_args) -> None:
        """React to a new output format (extension, presets, action text)."""
        if self._format_combo.currentData() is None:
            # The user landed on a separator entry - jump to the next real one.
            for index in range(self._format_combo.currentIndex() + 1,
                               self._format_combo.count()):
                if self._format_combo.itemData(index):
                    self._format_combo.setCurrentIndex(index)
                    return
            return

        fmt = self._current_format()
        self._rebuild_quality_combo()

        lossless = fmt.lossless and not fmt.is_video
        self._quality_combo.setEnabled(not lossless)
        self._quality_combo.setToolTip(
            "This output format is lossless - the quality presets do not apply."
            if lossless else ""
        )
        self._extract_button.setText(
            f"Convert to {fmt.name}" if fmt.is_video else f"Extract {fmt.name}"
        )
        current_output = self._output_edit.text().strip()
        if current_output:
            self._output_edit.setText(
                self._replace_output_extension(current_output, fmt.extension)
            )
        self._update_conversion_summary()
        self._update_extract_state()

    def _update_conversion_summary(self) -> None:
        """Describe what the selected conversion will do, in one line."""
        if self._info is None or not self._source_path:
            self._summary_label.setText("Load a file to see the conversion plan.")
            return

        info = self._info
        fmt = self._current_format()
        preset = self._quality_combo.currentData() or "original"
        preset_label = self._quality_combo.currentText()
        source = f"{os.path.splitext(self._source_path)[1].upper().lstrip('.')} source"
        if info.video_codec:
            source += f" ({info.video_codec.upper()} video + {info.codec.upper() if info.codec else 'no'} audio)"
        else:
            source += f" ({info.codec.upper()} audio)"

        if fmt.is_video:
            if preset == "original" and info.video_codec in fmt.video_copyable:
                video_part = f"{info.video_codec.upper()} copy"
            else:
                video_part = f"{fmt.video_short or fmt.video_codec} re-encoded"
            audio_part = "no audio" if not info.codec else f"{fmt.audio_codec} audio"
            plan = f"{source}  ->  {fmt.name} : {video_part} + {audio_part}  [{preset_label}]"
        else:
            if preset == "original" and info.codec in fmt.audio_copyable:
                audio_part = f"{info.codec.upper()} copy (no re-encoding)"
            elif fmt.lossless:
                audio_part = "lossless re-encode"
            else:
                audio_part = f"{fmt.audio_codec} re-encode"
            plan = f"{source}  ->  {fmt.name} : audio only, {audio_part}  [{preset_label}]"
        self._summary_label.setText(plan)

    def _load_source(self, path: str) -> None:
        if self._extract_worker is not None and self._extract_worker.isRunning():
            QMessageBox.information(self, APP_NAME, "Please wait for the current conversion to finish.")
            return
        if not os.path.isfile(path):
            QMessageBox.warning(self, APP_NAME, f"File not found:\n{path}")
            return

        self._source_path = path
        self._input_edit.setText(path)
        self._output_edit.setText(self._suggested_output_path())
        for label in self._info_labels.values():
            label.setText("-")
        self._info = None
        self._rebuild_format_combo()
        self._update_conversion_summary()
        self._clear_waveform()
        self._set_player_source(path)
        self._update_playback_controls()
        self._open_folder_button.setEnabled(False)
        self._extract_button.setEnabled(False)
        self._progress.setValue(0)
        self._status_label.setText("Analysing the source file ...")
        self._append_log(f"Opening: {path}")

        if not self._ffmpeg:
            self._status_label.setText("FFmpeg is missing - cannot analyse the file.")
            return

        self._analyse_worker = AnalyseWorker(path, self._ffmpeg, self._ffprobe, self)
        self._analyse_worker.infoReady.connect(self._on_info_ready)
        self._analyse_worker.waveformReady.connect(self._on_waveform_ready)
        self._analyse_worker.failed.connect(self._on_analyse_failed)
        self._analyse_worker.finished.connect(self._on_analyse_finished)
        self._analyse_worker.start()

    def _on_info_ready(self, info: MediaInfo) -> None:
        self._info = info
        self._set_info_labels(info)
        self._append_log(
            f"Audio stream: {info.codec.upper()}, {info.sample_rate} Hz, "
            f"{info.channels} channel(s), {info.bit_rate_text}, duration {info.duration_text}"
        )
        self._apply_duration_to_trim(info.duration)
        self._rebuild_format_combo()
        self._on_format_changed()
        self._update_conversion_summary()
        self._update_playback_controls()
        self._update_extract_state()

    def _apply_duration_to_trim(self, duration: float) -> None:
        """Configure the trim controls for the duration of the loaded file."""
        has_duration = duration > 0
        for spin in (self._start_spin, self._end_spin):
            spin.setEnabled(has_duration)
        for button in (self._trim_reset_button, self._trim_start_button, self._trim_end_button):
            button.setEnabled(has_duration)

        if not has_duration:
            self._selection_label.setText("Selection: whole file (duration unknown)")
            self._update_trim_labels()
            return

        self._sync_guard = True
        for spin in (self._start_spin, self._end_spin):
            spin.setRange(0.0, duration)
        self._start_spin.setValue(0.0)
        self._end_spin.setValue(duration)
        self._sync_guard = False

        self._trim_region.setBounds((0.0, duration))
        self._trim_region.setRegion((0.0, duration))
        self._update_trim_labels()

    def _on_waveform_ready(self, data) -> None:
        if data is None:
            self._clear_waveform()
            self._plot.setTitle("No audio stream - nothing to display", color="#9a9aac")
            self._append_log("The file has no audio stream (silent video): "
                             "only video conversions are possible.")
            self._status_label.setText("Ready (no audio).")
            return
        self._plot.setTitle("")
        self._draw_waveform(data)
        self._status_label.setText("Ready.")

    def _on_analyse_failed(self, message: str) -> None:
        self._status_label.setText("Analysis failed.")
        self._append_log(f"ERROR: {message}")
        QMessageBox.critical(self, APP_NAME, f"Could not analyse the file:\n\n{message}")

    def _on_analyse_finished(self) -> None:
        self._analyse_worker = None
        self._update_extract_state()

    def _update_extract_state(self) -> None:
        trim_valid = True
        if self._info is not None:
            start, end = self._trim_range()
            trim_valid = (end - start) >= 0.001
        ready = bool(
            self._source_path
            and self._info is not None
            and self._output_edit.text().strip()
            and self._ffmpeg
            and trim_valid
            and format_availability(
                self._current_format(), self._ffmpeg, self._info, self._source_path
            )[0]
        )
        running = self._extract_worker is not None and self._extract_worker.isRunning()
        self._extract_button.setEnabled(ready and not running)

    # ------------------------------------------------------------ conversion
    def _start_extraction(self) -> None:
        if not self._source_path or self._info is None:
            return
        output = self._output_edit.text().strip()
        if not output:
            self._choose_output()
            output = self._output_edit.text().strip()
            if not output:
                return

        fmt = self._current_format()
        usable, reason = format_availability(fmt, self._ffmpeg, self._info, self._source_path)
        if not usable:
            QMessageBox.critical(
                self, APP_NAME,
                f"{fmt.name} output is not possible:\n\n{reason}.\n\n"
                "Please choose another output format.",
            )
            self._rebuild_format_combo()
            return
        # The extension always follows the selected format.
        fixed_output = self._replace_output_extension(output, fmt.extension)
        if fixed_output != output:
            output = fixed_output
            self._output_edit.setText(output)

        out_dir = os.path.dirname(os.path.abspath(output)) or "."
        if not os.path.isdir(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except OSError as exc:
                QMessageBox.critical(self, APP_NAME, f"Cannot create the output folder:\n{exc}")
                return
        if os.path.exists(output):
            answer = QMessageBox.question(
                self,
                APP_NAME,
                f"The file already exists:\n{output}\n\nOverwrite it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        preset = self._quality_combo.currentData()
        start_time, end_time = self._trim_range()
        total = self._info.duration
        is_trimmed = start_time > 0.0005 or (total > 0 and end_time < total - 0.0005)

        self._append_log("")
        self._append_log(f"--- Converting to: {output}")
        self._append_log(f"    Target: {fmt.name} ({fmt.kind})  "
                         f"[{fmt.video_codec or fmt.audio_codec}]")
        self._append_log(f"    Preset: {self._quality_combo.currentText()}")
        if is_trimmed:
            self._append_log(
                f"    Trim  : {format_timecode(start_time)} -> {format_timecode(end_time)} "
                f"(length {format_timecode(max(0.0, end_time - start_time))})"
            )
        else:
            self._append_log("    Trim  : whole file")
        self._progress.setRange(0, 0 if self._info.duration <= 0 else 100)
        self._progress.setValue(0)
        self._status_label.setText("Converting ...")
        self._extract_button.setEnabled(False)
        self._browse_input.setEnabled(False)
        self._browse_output.setEnabled(False)
        self._quality_combo.setEnabled(False)
        self._format_combo.setEnabled(False)
        self._set_trim_enabled(False)
        self._stop_preview(silent=True)
        self._cancel_button.setEnabled(True)

        self._extract_worker = ExtractWorker(
            self._source_path, output, preset, self._info, self._ffmpeg or "", self,
            start=start_time, end=end_time, output_format=fmt,
        )
        self._extract_worker.progress.connect(self._on_progress)
        self._extract_worker.log.connect(self._append_log)
        self._extract_worker.finishedWithResult.connect(self._on_extraction_finished)
        self._extract_worker.start()

    def _on_progress(self, percent: float, seconds: float, speed: str) -> None:
        if percent < 0:
            self._status_label.setText(
                f"Converting ... {seconds:6.1f} s processed" + (f" (speed {speed})" if speed else "")
            )
            return
        self._progress.setRange(0, 100)
        self._progress.setValue(int(round(percent)))
        self._status_label.setText(
            f"Converting ... {percent:5.1f}%   ({seconds:6.1f} s processed"
            + (f", speed {speed}" if speed else "")
            + ")"
        )

    def _cancel_extraction(self) -> None:
        if self._extract_worker is not None and self._extract_worker.isRunning():
            self._append_log("Cancelling ...")
            self._cancel_button.setEnabled(False)
            self._extract_worker.cancel()

    def _on_extraction_finished(self, ok: bool, output: str, message: str) -> None:
        self._extract_worker = None
        self._browse_input.setEnabled(True)
        self._browse_output.setEnabled(True)
        self._format_combo.setEnabled(True)
        self._quality_combo.setEnabled(not (self._current_format().lossless
                                            and not self._current_format().is_video))
        self._set_trim_enabled(self._info is not None and self._info.duration > 0)
        self._cancel_button.setEnabled(False)
        self._progress.setRange(0, 100)

        if ok:
            self._progress.setValue(100)
            size = os.path.getsize(output) if os.path.isfile(output) else 0
            self._status_label.setText(f"Done - {os.path.basename(output)} ({_human_size(size)})")
            self._append_log(f"SUCCESS: {output} ({_human_size(size)})")
            self._open_folder_button.setEnabled(True)
            self._last_output = output
        else:
            self._progress.setValue(0)
            self._status_label.setText("Conversion failed." if message != "Conversion cancelled."
                                       else "Cancelled.")
            self._append_log(f"FAILED: {message}")
            if message != "Conversion cancelled.":
                QMessageBox.critical(self, APP_NAME, f"Conversion failed:\n\n{message}")
        self._update_extract_state()

    def _open_output_folder(self) -> None:
        path = getattr(self, "_last_output", "") or self._output_edit.text().strip()
        folder = os.path.dirname(os.path.abspath(path)) if path else ""
        if not folder or not os.path.isdir(folder):
            return
        if os.name == "nt":
            os.startfile(folder)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])

    # -------------------------------------------------------- drag and drop
    def dragEnterEvent(self, event) -> None:
        """Accept any local file - FFmpeg decides whether it has audio."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path and os.path.isfile(path):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and os.path.isfile(path):
                self._load_source(path)
                event.acceptProposedAction()
                return
        event.ignore()

    # ------------------------------------------------------------ shutdown
    def _shutdown_previews(self) -> None:
        """Stop every preview engine/thread (safe to call more than once)."""
        try:
            self._stop_preview(silent=True)
        except Exception:
            pass
        try:
            self._stop_video_engine()
        except Exception:
            pass
        worker = self._preview_worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(3000)
        self._preview_worker = None

    def closeEvent(self, event) -> None:
        if self._extract_worker is not None and self._extract_worker.isRunning():
            answer = QMessageBox.question(
                self, APP_NAME,
                "A conversion is still running. Stop it and quit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self._extract_worker.cancel()
            self._extract_worker.wait(5000)
        if self._analyse_worker is not None and self._analyse_worker.isRunning():
            self._analyse_worker.wait(3000)
        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._preview_worker.cancel()
            self._preview_worker.wait(3000)
        self._stop_preview(silent=True)
        for path in self._preview_files:
            try:
                if os.path.isfile(path):
                    os.remove(path)
                folder = os.path.dirname(path)
                if os.path.isdir(folder) and not os.listdir(folder):
                    os.rmdir(folder)
            except OSError:
                pass
        event.accept()


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def format_timecode(seconds: float) -> str:
    """Format *seconds* as ``HH:MM:SS.mmm`` (millisecond precision)."""
    if seconds is None or seconds < 0:
        seconds = 0.0
    milliseconds = int(round(seconds * 1000.0))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _human_size(num_bytes: float) -> str:
    if not num_bytes:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    index = 0
    value = float(num_bytes)
    while value >= 1024 and index < len(units) - 1:
        value /= 1024.0
        index += 1
    return f"{value:.2f} {units[index]}"


# ---------------------------------------------------------------------------
# Command line mode (head-less, same engine as the GUI)
# ---------------------------------------------------------------------------

def run_cli(argv: List[str]) -> int:
    """Head-less conversion mode (same engine as the GUI)."""
    parser = argparse.ArgumentParser(
        prog="Media_Converter.py",
        description="Convert any video/audio file into any other audio or video format, "
                    "optionally trimming the time range.",
        epilog="Audio targets : " + ", ".join(f.key for f in formats_for_kind("audio")) +
               "\nVideo targets : " + ", ".join(f.key for f in formats_for_kind("video")),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version",
        version=f"{APP_NAME} {APP_VERSION} - by {APP_AUTHOR} ({APP_LICENSE} License)",
    )
    parser.add_argument("input", help="source file (any video/audio format FFmpeg can read)")
    parser.add_argument(
        "-o", "--output",
        help="destination audio file; its extension selects the format "
             "(default: same name with .mp3)",
    )
    parser.add_argument(
        "-f", "--format", default=None,
        choices=[fmt.key for fmt in OUTPUT_FORMATS],
        help="output format (default: taken from the -o extension, else mp3)",
    )
    parser.add_argument(
        "-p", "--preset", default="original",
        choices=[key for key, _ in AUDIO_PRESETS],
        help="quality preset: original, best, vbr2, 320, 256, 192, 128 "
             "(video targets also accept best/vbr2/320/128) (default: original)",
    )
    parser.add_argument(
        "--start", type=float, default=0.0,
        help="trim: start time in seconds (millisecond precision, e.g. 12.250)",
    )
    parser.add_argument(
        "--end", type=float, default=0.0,
        help="trim: end time in seconds (0 = end of the file)",
    )
    args = parser.parse_args(argv)

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("ERROR: FFmpeg was not found. Install it and add it to your PATH.", file=sys.stderr)
        return 2
    ffprobe = find_ffprobe(ffmpeg)

    source = os.path.abspath(args.input)
    if not os.path.isfile(source):
        print(f"ERROR: file not found: {source}", file=sys.stderr)
        return 2

    # Output format: --format wins, then the extension of -o, then MP3.
    if args.format:
        fmt = get_output_format(args.format)
    elif args.output:
        fmt = format_from_path(args.output) or get_output_format("mp3")
    else:
        fmt = get_output_format("mp3")

    output = args.output or (os.path.splitext(source)[0] + fmt.extension)
    base, extension = os.path.splitext(output)
    if extension.lower() != fmt.extension:
        output = (base if extension else output) + fmt.extension
    output = os.path.abspath(output)

    try:
        info = probe_media(source, ffprobe, ffmpeg)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    usable, reason = format_availability(fmt, ffmpeg, info, source)
    if not usable:
        print(f"ERROR: {fmt.name} output is not possible: {reason}.", file=sys.stderr)
        return 2

    print(f"Source      : {source}")
    print(f"Content     : {info.kind_text}")
    if info.codec:
        print(f"Audio       : {info.codec.upper()}, {info.sample_rate} Hz, "
              f"{info.channels} ch, {info.bit_rate_text}, {info.duration_text}")
    if info.video_codec:
        print(f"Video       : {info.video_text}")
    print(f"Destination : {output}")
    print(f"Format      : {fmt.name} ({fmt.kind}, "
          f"{fmt.video_codec or fmt.audio_codec})")
    print(f"Preset      : {args.preset}")
    if args.start > 0 or args.end > 0:
        print(f"Trim        : {format_timecode(args.start)} -> "
              f"{format_timecode(args.end or info.duration)}")

    last_line = [-1.0]

    def on_progress(percent: float, seconds: float, speed: str) -> None:
        if percent < 0:
            print(f"\r  {seconds:8.1f} s processed", end="", flush=True)
            return
        step = int(percent // 5)
        if step != last_line[0]:
            last_line[0] = step
            print(f"\r  progress: {percent:5.1f}%  ({seconds:8.1f} s"
                  + (f", speed {speed})" if speed else ")"), end="", flush=True)

    ok, message = extract_audio(
        source=source, destination=output, preset=args.preset, info=info,
        ffmpeg=ffmpeg, on_progress=on_progress, on_log=lambda m: print(m),
        start=args.start, end=args.end, output_format=fmt,
    )
    print()
    if not ok:
        print(f"ERROR: {message}", file=sys.stderr)
        return 1
    print(f"Done: {output} ({_human_size(os.path.getsize(output))})")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] in ("--nogui", "--cli"):
        return run_cli(argv[1:])
    if argv and argv[0] in ("--version", "-V"):
        print(f"{APP_NAME} {APP_VERSION} - by {APP_AUTHOR} ({APP_LICENSE} License)")
        return 0
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(APP_AUTHOR)
    app.setStyleSheet(STYLESHEET)

    window = MainWindow()
    app.aboutToQuit.connect(window._shutdown_previews)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
