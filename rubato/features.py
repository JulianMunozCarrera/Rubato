"""
Turns raw audio into two simple things the rest of the program can use:

1. Onset strength: "did something just get hit or strummed?"
   We use this to find the beat.
2. Chroma: "how loud is each of the 12 notes (C, C#, D ...) right now?"
   We use this to find the chord.
"""

import numpy as np

from .config import SAMPLE_RATE, BLOCK


class OnsetDetector:
    """
    Gives one number per block. A big number means a new sound just started
    (a strum, a drum hit, a piano chord). A small number means nothing new.

    How it works: we look at the frequencies in the sound right now and
    compare them to the block before. If lots of frequencies got louder,
    something new was played.
    """

    WINDOW = 1024

    def __init__(self):
        self.buffer = np.zeros(self.WINDOW, dtype=np.float32)
        self.fade = np.hanning(self.WINDOW).astype(np.float32)
        self.last_spectrum = np.zeros(self.WINDOW // 2 + 1, dtype=np.float32)

    def process(self, block):
        # Slide the newest block into our short memory of recent audio.
        self.buffer[:-BLOCK] = self.buffer[BLOCK:]
        self.buffer[-BLOCK:] = block

        # FFT tells us how much of each frequency is in the sound.
        spectrum = np.abs(np.fft.rfft(self.buffer * self.fade))

        # Square root squashes the values a little. Loud strums still score
        # higher than soft ones, which helps us tell the main beats apart
        # from the lighter strums in between.
        spectrum = np.sqrt(spectrum)

        # Only count frequencies that got LOUDER. Getting quieter is not a new hit.
        change = spectrum - self.last_spectrum
        strength = float(np.sum(change[change > 0]))

        self.last_spectrum = spectrum
        return strength


class ChromaExtractor:
    """
    Measures how strong each of the 12 notes is in the last ~0.19 seconds.

    The result is a list of 12 numbers, one for each note name.
    If the band plays a C chord, the numbers for C, E and G will be the biggest.
    """

    WINDOW = 8192

    def __init__(self):
        self.buffer = np.zeros(self.WINDOW, dtype=np.float32)
        self.fade = np.hanning(self.WINDOW).astype(np.float32)
        self.note_map = self._build_note_map()

        # A second table that only keeps the low notes (below middle C).
        # The lowest note of a chord is usually its root, so the low notes
        # are a strong hint about which chord it is.
        freqs = np.fft.rfftfreq(self.WINDOW, 1.0 / SAMPLE_RATE)
        self.low_note_map = self.note_map * (freqs < 262)

    def _build_note_map(self):
        """
        Build a table that says "this FFT frequency belongs to this note".
        We only use 65 Hz to 2100 Hz, which is where chords live.
        """
        freqs = np.fft.rfftfreq(self.WINDOW, 1.0 / SAMPLE_RATE)
        table = np.zeros((12, len(freqs)), dtype=np.float32)
        for i, freq in enumerate(freqs):
            if freq < 65 or freq > 2100:
                continue
            # Convert the frequency to a piano key number (69 = the A at 440 Hz).
            key = 69 + 12 * np.log2(freq / 440.0)
            nearest = int(round(key))
            # Frequencies that sit right on a note count more than ones in between.
            closeness = np.exp(-0.5 * ((key - nearest) / 0.25) ** 2)
            table[nearest % 12, i] = closeness
        return table

    def push(self, block):
        """Add the newest block of audio."""
        self.buffer[:-BLOCK] = self.buffer[BLOCK:]
        self.buffer[-BLOCK:] = block

    def compute(self):
        """
        Return two lists of 12 note strengths for the audio we have right now:
        one using all notes, and one using only the low notes.
        """
        spectrum = np.abs(np.fft.rfft(self.buffer * self.fade))
        spectrum = np.sqrt(spectrum)  # soften so one loud note does not drown the rest
        return self.note_map @ spectrum, self.low_note_map @ spectrum
