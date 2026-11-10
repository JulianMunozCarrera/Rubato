import numpy as np

from rubato.config import BLOCK_TIME
from rubato.engine import JamEngine
from rubato.testband import make_band
from rubato.audio_files import run_engine

C, G, Am, F = (0, "maj"), (7, "maj"), (9, "min"), (5, "maj")


def play(use_memory=True, tempo=lambda beat: 110):
    audio, beats, chords = make_band([C, G, Am, F] * 6, tempo)
    engine = JamEngine(use_memory=use_memory)
    played, took = run_engine(engine, audio)
    return engine, played, took, beats, chords


def share_on_right_chord(engine, beats, chords):
    when = dict(engine.log["beats"])
    marks = [chord == chords[int(np.argmin(abs(beats - when[n])))]
             for n, chord in engine.log["played_over"] if when[n] > 6]
    return np.mean(marks)


def test_follows_the_fake_band():
    engine, played, _, beats, chords = play()
    ours = [t for _, t in engine.log["beats"] if t > 6]
    on_time = [min(abs(beats - t)) < 0.07 for t in ours]
    assert np.mean(on_time) > 0.9
    assert abs(engine.beat.bpm - 110) < 2
    assert np.all(np.isfinite(played))
    assert len(engine.log["notes"]) > 20


def test_memory_gets_the_bass_to_the_chord_on_time():
    engine, _, _, beats, chords = play(use_memory=True)
    with_memory = share_on_right_chord(engine, beats, chords)
    engine, _, _, beats, chords = play(use_memory=False)
    without = share_on_right_chord(engine, beats, chords)
    assert with_memory > 0.9
    assert with_memory > without + 0.1


def test_follows_a_band_that_speeds_up():
    engine, _, _, beats, _ = play(tempo=lambda beat: 100 + 25 * beat / 96)
    assert abs(engine.beat.bpm - 125) < 4


def test_is_fast_enough_for_live_use():
    _, _, took, _, _ = play()
    # Each block must be done well before the next one arrives.
    assert np.percentile(took, 99) < BLOCK_TIME / 2
