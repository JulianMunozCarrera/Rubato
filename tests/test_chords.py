import numpy as np

from bandmate.config import SAMPLE_RATE, BLOCK
from bandmate.features import ChromaExtractor
from bandmate.chords import score_chords, CHORD_LABELS, SongMemory, KeyFinder
from bandmate.testband import pluck, chord_notes


def hear(chord):
    """Play a chord into the chroma extractor and return what it measured."""
    audio = sum(pluck(m, 0.5, 0.1) for m in chord_notes(*chord)).astype(np.float32)
    chroma = ChromaExtractor()
    for start in range(0, 8192, BLOCK):
        chroma.push(audio[start:start + BLOCK])
    return chroma.compute()


def test_names_every_major_and_minor_chord():
    for chord in CHORD_LABELS:
        notes, low_notes = hear(chord)
        best = CHORD_LABELS[int(np.argmax(score_chords(notes, low_notes)))]
        assert best == chord


def test_memory_guesses_the_next_chord_in_a_loop():
    C, G, Am, F = (0, "maj"), (7, "maj"), (9, "min"), (5, "maj")
    memory = SongMemory()
    for chord in [C, G, Am, F, C, G, Am, F, C, G, Am]:
        for _ in range(4):  # 4 beats per chord
            memory.add(chord)
    assert memory.guess_next() == F


def test_memory_says_same_chord_when_it_has_no_idea():
    memory = SongMemory()
    memory.add((2, "min"))
    assert memory.guess_next() == (2, "min")


def test_finds_the_key():
    finder = KeyFinder()
    for chord in [(0, "maj"), (5, "maj"), (7, "maj"), (9, "min")] * 3:
        notes, _ = hear(chord)
        for _ in range(60):
            finder.add(notes)
    # C major and A minor use the exact same notes, so either answer is fine.
    assert finder.key() in [(0, "maj"), (9, "min")]
