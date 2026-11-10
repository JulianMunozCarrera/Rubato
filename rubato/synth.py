"""
Makes the actual sound for the bass and the drums.

No sample files are needed. Every sound is built from simple math
(sine waves and noise), so the project works anywhere.
"""

import numpy as np

from .config import SAMPLE_RATE, BLOCK


def midi_to_hz(midi):
    """Note number to frequency. Note 69 is the A at 440 Hz."""
    return 440.0 * 2 ** ((midi - 69) / 12)


class BassSynth:
    """
    A simple bass sound. It plays one note at a time.

    We tell it ahead of time "start this note at sample number X".
    That is how notes land exactly on the beat instead of whenever
    the code gets around to it.
    """

    def __init__(self):
        self.events = []      # list of (sample number, frequency or None for stop)
        self.position = 0     # how many samples we have made so far
        self.freq = 0.0
        self.angle = 0.0      # where we are in the wave
        self.volume = 0.0     # current loudness of the note
        self.playing = False

    def note_on(self, at_sample, midi):
        self.events.append((int(at_sample), midi_to_hz(midi)))

    def note_off(self, at_sample):
        self.events.append((int(at_sample), None))

    def render(self):
        """Make the next block of bass sound."""
        out = np.zeros(BLOCK, dtype=np.float32)
        self.events.sort(key=lambda e: e[0])
        start = 0
        end_of_block = self.position + BLOCK

        while start < BLOCK:
            # Play up to the next event, or to the end of the block.
            stop = BLOCK
            if self.events and self.events[0][0] < end_of_block:
                stop = max(start, self.events[0][0] - self.position)
            if stop > start:
                out[start:stop] = self._make(stop - start)
                start = stop
            if self.events and self.events[0][0] <= self.position + start:
                _, freq = self.events.pop(0)
                if freq is None:
                    self.playing = False
                else:
                    self.freq, self.playing, self.volume = freq, True, 1.0

        self.position = end_of_block
        return out

    def _make(self, count):
        if self.volume < 1e-4 and not self.playing:
            return np.zeros(count, dtype=np.float32)

        angles = self.angle + 2 * np.pi * self.freq * np.arange(1, count + 1) / SAMPLE_RATE
        self.angle = float(angles[-1] % (2 * np.pi))

        # A pure sine is hard to hear on small speakers, so we mix in two
        # higher copies of the note to give it some bite.
        wave = np.sin(angles) + 0.4 * np.sin(2 * angles) + 0.15 * np.sin(3 * angles)

        # Held notes fade slowly. Released notes fade fast (about 30 ms)
        # so they do not click when they stop.
        fade_per_sample = 0.99997 if self.playing else 0.9985
        fades = self.volume * fade_per_sample ** np.arange(1, count + 1)
        self.volume = float(fades[-1])
        return (0.3 * wave * fades).astype(np.float32)


def _kick():
    """A kick drum is a low tone that quickly drops in pitch."""
    t = np.arange(int(0.25 * SAMPLE_RATE)) / SAMPLE_RATE
    pitch = 45 + 110 * np.exp(-t * 35)
    return np.sin(2 * np.pi * np.cumsum(pitch) / SAMPLE_RATE) * np.exp(-t * 14)


def _snare():
    """A snare is a burst of noise plus a short tone."""
    t = np.arange(int(0.2 * SAMPLE_RATE)) / SAMPLE_RATE
    noise = np.random.default_rng(1).standard_normal(len(t))
    return 0.6 * noise * np.exp(-t * 22) + 0.5 * np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30)


def _hat():
    """A hi-hat is very short, bright noise."""
    t = np.arange(int(0.05 * SAMPLE_RATE)) / SAMPLE_RATE
    noise = np.random.default_rng(2).standard_normal(len(t))
    bright = np.diff(noise, prepend=0)  # removes the low part of the noise
    return 0.35 * bright * np.exp(-t * 90)


class DrumKit:
    """Plays drum sounds at exact sample positions."""

    def __init__(self):
        self.sounds = {"kick": _kick(), "snare": _snare(), "hat": _hat()}
        self.tape = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)  # 2 seconds of upcoming drum audio
        self.position = 0

    def hit(self, at_sample, name, loudness=1.0):
        sound = self.sounds[name] * loudness
        offset = max(0, int(at_sample) - self.position)
        room = len(self.tape) - offset
        if room > 0:
            self.tape[offset:offset + len(sound)] += sound[:room].astype(np.float32)

    def render(self):
        """Take the next block off the front of the tape."""
        out = self.tape[:BLOCK].copy()
        self.tape[:-BLOCK] = self.tape[BLOCK:]
        self.tape[-BLOCK:] = 0
        self.position += BLOCK
        return 0.5 * out
