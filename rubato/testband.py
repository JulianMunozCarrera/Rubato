"""
A fake band for testing.

We can't bring a real guitarist into an automated test, so this file
makes guitar-like audio from a list of chords and a tempo. Because we
made it, we know exactly where every beat and chord is. That lets us
check if Rubato got them right.
"""

import numpy as np

from .config import SAMPLE_RATE


def pluck(midi, seconds, loudness=1.0):
    """One plucked string: a note with overtones that fades out."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    freq = 440.0 * 2 ** ((midi - 69) / 12)
    sound = np.zeros_like(t)
    for overtone in range(1, 7):
        # Higher overtones are quieter and fade faster, like a real string.
        sound += np.sin(2 * np.pi * freq * overtone * t) / overtone * np.exp(-t * (2.5 + overtone))
    attack = np.minimum(1.0, t / 0.003)  # very short fade in so it does not click
    return loudness * sound * attack


def chord_notes(root, quality):
    """Six notes for a guitar-style chord."""
    third = 3 if quality == "min" else 4
    base = 40 + (root - 4) % 12  # lowest root on or above the low E string
    return [base, base + 7, base + 12, base + 12 + third, base + 19, base + 24]


def make_band(chords_per_bar, bpm_at, seed=0):
    """
    chords_per_bar: list of chords, one per bar, like [(0, "maj"), (7, "maj")]
    bpm_at:         function that gives the tempo at a beat number.
                    Lets us test speeding up and slowing down.

    Returns (audio, beat_times, chord_for_each_beat).
    """
    random = np.random.default_rng(seed)
    total_beats = len(chords_per_bar) * 4

    # Work out when every beat happens. Start 0.5 s in.
    beat_times = [0.5]
    for beat in range(total_beats):
        beat_times.append(beat_times[-1] + 60.0 / bpm_at(beat))

    audio = np.zeros(int((beat_times[-1] + 2.0) * SAMPLE_RATE), dtype=np.float32)
    chord_for_beat = []

    # A common strumming pattern. Each pair is (position in the beat, loudness).
    strums_in_bar = [(0, 1.0), (1, 0.7), (1.5, 0.45), (2, 0.9), (2.5, 0.45), (3, 0.7), (3.5, 0.5)]

    for bar, chord in enumerate(chords_per_bar):
        chord_for_beat += [chord] * 4
        for where, loudness in strums_in_bar:
            beat = bar * 4 + int(where)
            length = beat_times[beat + 1] - beat_times[beat]
            when = beat_times[beat] + (where % 1) * length
            when += random.normal(0, 0.006)  # real players are never perfectly on time
            for string, midi in enumerate(chord_notes(*chord)):
                # The pick hits each string a few ms after the last one.
                start = int((when + string * 0.004) * SAMPLE_RATE)
                note = pluck(midi, 0.8, loudness * random.uniform(0.8, 1.0) * 0.12)
                audio[start:start + len(note)] += note[:len(audio) - start]

    return audio, np.array(beat_times[:-1]), chord_for_beat


def render_midi(midi, seconds):
    """
    Turn a MIDI song (loaded with pretty_midi) into audio, using the same
    plucked sound as the fake band. Only the first 'seconds' are made.
    """
    audio = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
    for instrument in midi.instruments:
        for note in instrument.notes:
            if note.start >= seconds:
                continue
            sound = pluck(note.pitch, min(1.2, note.end - note.start + 0.3), note.velocity / 127 * 0.08)
            start = int(note.start * SAMPLE_RATE)
            audio[start:start + len(sound)] += sound[:len(audio) - start]
    return audio
