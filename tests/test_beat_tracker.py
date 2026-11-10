import numpy as np

from bandmate.config import SAMPLE_RATE
from bandmate.engine import JamEngine
from bandmate.audio_files import run_engine


def click_track(bpm, seconds):
    """A metronome: one short beep per beat."""
    audio = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
    beats = np.arange(0.5, seconds - 1, 60.0 / bpm)
    t = np.arange(1323) / SAMPLE_RATE
    beep = 0.5 * np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 200)
    for beat in beats:
        start = int(beat * SAMPLE_RATE)
        audio[start:start + len(beep)] += beep
    return audio, beats


def test_finds_the_tempo_of_a_metronome():
    audio, _ = click_track(120, 20)
    engine = JamEngine(bass=False, drums=False)
    run_engine(engine, audio)
    assert abs(engine.beat.bpm - 120) < 1.0


def test_beats_land_within_15_ms_of_the_clicks():
    audio, beats = click_track(96, 25)
    engine = JamEngine(bass=False, drums=False)
    run_engine(engine, audio)
    ours = [t for _, t in engine.log["beats"] if t > 6]
    assert len(ours) > 20
    errors = [min(abs(beats - t)) for t in ours]
    assert np.median(errors) < 0.015


def test_keeps_counting_when_the_band_stops():
    audio, _ = click_track(120, 12)
    silence = np.zeros(SAMPLE_RATE * 4, dtype=np.float32)
    engine = JamEngine(bass=False, drums=False)
    run_engine(engine, np.concatenate([audio, silence]))
    last_beat = engine.log["beats"][-1][1]
    assert last_beat > 15.0  # still announcing beats during the silence
    assert abs(engine.beat.bpm - 120) < 2.0
