"""
Figures out which chord the band is playing, what key the song is in,
and guesses which chord is coming next.
"""

import numpy as np

# A major chord is the root note, the note 4 steps up, and the note 7 steps up.
# A minor chord is the same but the middle note is 3 steps up.
MAJOR_SHAPE = [0, 4, 7]
MINOR_SHAPE = [0, 3, 7]


def _build_chord_shapes():
    """Make the "ideal" note pattern for all 24 chords (12 major + 12 minor)."""
    shapes, labels = [], []
    for quality, shape in (("maj", MAJOR_SHAPE), ("min", MINOR_SHAPE)):
        for root in range(12):
            pattern = np.zeros(12)
            for step in shape:
                pattern[(root + step) % 12] = 1.0
            pattern[root] = 1.2  # the root note is usually the strongest
            shapes.append(pattern / np.linalg.norm(pattern))
            labels.append((root, quality))
    return np.array(shapes), labels


CHORD_SHAPES, CHORD_LABELS = _build_chord_shapes()


CHORD_ROOTS = np.array([root for root, _ in CHORD_LABELS])


def _same_scale(values):
    """Shift and scale 12 numbers so different loudness levels can be compared."""
    values = values - values.min()
    size = np.linalg.norm(values)
    return values / size if size > 1e-9 else values


def score_chords(chroma, low_chroma=None):
    """
    Compare the 12 note strengths to every chord pattern.
    Returns 24 scores. Higher score = better match.

    low_chroma: the same thing but only for low notes. If a chord's root
    is strong down low, that chord gets a bonus. Without this, C major
    often gets mixed up with E minor, because they share two notes.
    """
    scores = CHORD_SHAPES @ _same_scale(chroma)
    if low_chroma is not None:
        scores = scores + 0.3 * _same_scale(low_chroma)[CHORD_ROOTS]
    return scores


class ChordTracker:
    """
    Collects note strengths while the band plays, then answers
    "what chord was that?" for any stretch of time.
    """

    KEEP = 400  # how many recent readings to remember

    def __init__(self):
        self.times = []
        self.chromas = []
        self.low_chromas = []
        self.loudness = []
        self.last = (None, None)

    def add(self, time, chroma, low_chroma, loudness):
        self.times.append(time)
        self.chromas.append(chroma)
        self.low_chromas.append(low_chroma)
        self.loudness.append(loudness)
        if len(self.times) > self.KEEP:
            del self.times[:100], self.chromas[:100], self.low_chromas[:100], self.loudness[:100]

    def chord_between(self, start, end, stick_to=None):
        """
        Best chord for the audio between two times.

        stick_to: the chord we think is playing. We only switch away from
        it if the new chord is clearly better. This stops the answer from
        flickering between two close chords.
        """
        inside = [i for i, t in enumerate(self.times) if start <= t < end]
        if not inside:
            return None
        if max(self.loudness[i] for i in inside) < 0.003:
            return (None, None)  # silence, so no chord

        scores = score_chords(np.mean([self.chromas[i] for i in inside], axis=0),
                              np.mean([self.low_chromas[i] for i in inside], axis=0))
        best = int(np.argmax(scores))
        if stick_to in CHORD_LABELS:
            if scores[best] - scores[CHORD_LABELS.index(stick_to)] < 0.03:
                return stick_to
        return CHORD_LABELS[best]


class KeyFinder:
    """
    Finds the key of the song (like "G major").

    It keeps a slow running average of which notes get used the most,
    then compares that to the known "fingerprint" of every key.
    The fingerprints come from a classic music psychology study
    (Krumhansl and Kessler, 1982) where people rated how well each
    note fits a key.

    Heads up: every major key has a minor key with the exact same notes
    (C major and A minor, for example). Telling those two apart is hard,
    so sometimes this gives you the other one of the pair.
    """

    MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    def __init__(self):
        self.average = np.zeros(12)

    def add(self, chroma):
        self.average = 0.995 * self.average + 0.005 * chroma

    def key(self):
        if self.average.sum() < 1e-9:
            return (None, None)
        best, best_score = (None, None), -2.0
        for root in range(12):
            for quality, fingerprint in (("maj", self.MAJOR), ("min", self.MINOR)):
                score = np.corrcoef(self.average, np.roll(fingerprint, root))[0, 1]
                if score > best_score:
                    best, best_score = (root, quality), score
        return best


class SongMemory:
    """
    Remembers the chords played so far and guesses the next one.

    Bands repeat themselves. If the song went C, G, Am, F before and we
    just heard C, G, Am again, the next chord is probably F.

    This matters a lot. Without it the bass only finds out about a chord
    change AFTER hearing it, so it would always play the old chord for a
    moment. With it, the bass lands on the new chord at the same time
    as the band.
    """

    def __init__(self):
        self.chords = []  # one chord per beat

    def add(self, chord):
        self.chords.append(chord)

    def guess_next(self, and_then=None):
        """
        Guess the chord for the next beat.

        and_then: pretend this chord was just played too. Handy for asking
        "if this beat is G, what comes after it?"
        """
        chords = self.chords[-512:]
        if and_then is not None:
            chords = chords + [and_then]
        if not chords:
            return (None, None)
        n = len(chords)

        # Take the last few beats and look for the same run earlier in the song.
        # Try a long run first (more sure), then shorter ones.
        for length in (16, 8, 4):
            if n <= length:
                continue
            recent = chords[n - length:]
            for end in range(n - 1, length - 1, -1):  # search backwards, newest first
                if chords[end - length:end] == recent:
                    return chords[end]

        # Never seen this before. Safest guess: the chord stays the same.
        return chords[-1]
