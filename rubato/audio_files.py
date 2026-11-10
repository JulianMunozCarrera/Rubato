"""Small helpers for reading and writing sound files and running the engine on them."""

import time

import numpy as np
import soundfile as sf

from .config import SAMPLE_RATE, BLOCK


def load(path):
    """Read a sound file as one channel at our sample rate."""
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        # Simple resample. Good enough for testing.
        new_length = int(len(audio) * SAMPLE_RATE / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, new_length), np.arange(len(audio)), audio)
    return audio.astype(np.float32)


def save(path, audio):
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak  # stop it from distorting
    sf.write(path, audio, SAMPLE_RATE)


def run_engine(engine, audio):
    """
    Feed audio through the engine one block at a time, like a live mic would.
    Returns (what Rubato played, how long each block took to process).
    """
    played = np.zeros(len(audio), dtype=np.float32)
    took = []
    for start in range(0, len(audio) - BLOCK + 1, BLOCK):
        began = time.perf_counter()
        played[start:start + BLOCK] = engine.process(audio[start:start + BLOCK])
        took.append(time.perf_counter() - began)
    return played, np.array(took)
