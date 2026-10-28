"""Decoding and writing audio through ffmpeg.

Everything is decoded to 44.1 kHz stereo and held as float32 in int16 units,
so lossless samples survive the round trip bit for bit.
"""

import json
import subprocess
from dataclasses import dataclass, field

import numpy as np
from scipy.io import wavfile

RATE = 44100
LOSSLESS_CODECS = {"flac", "alac", "wavpack", "ape", "tta", "mlp", "truehd"}


@dataclass
class Source:
    path: str
    codec: str
    bitrate: int
    samples: np.ndarray = field(repr=False)

    @property
    def lossless(self):
        return self.codec.startswith("pcm_") or self.codec in LOSSLESS_CODECS

    @property
    def name(self):
        return self.path.rsplit("/", 1)[-1]


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name,bit_rate:format=bit_rate",
         "-of", "json", path],
        capture_output=True, check=True, text=True).stdout
    info = json.loads(out)
    stream = info["streams"][0]
    bitrate = stream.get("bit_rate") or info.get("format", {}).get("bit_rate") or 0
    return stream["codec_name"], int(bitrate)


def decode(path):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0",
         "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "2", "-ar", str(RATE), "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).reshape(-1, 2).astype(np.float32)


def load(path):
    codec, bitrate = probe(path)
    return Source(path, codec, bitrate, decode(path))


def write_wav(path, samples):
    pcm = np.clip(np.round(samples), -32768, 32767).astype(np.int16)
    wavfile.write(path, RATE, pcm)
