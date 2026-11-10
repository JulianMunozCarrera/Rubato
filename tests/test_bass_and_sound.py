import numpy as np

from rubato.config import BLOCK
from rubato.bass_brain import BassBrain, choice_to_midi, NUM_CHOICES, NO_CHANGE, FIRST_NOTE
from rubato.synth import BassSynth, DrumKit
from rubato.drums import drum_hits


def test_model_gives_a_valid_choice_every_step():
    brain = BassBrain()
    for step in range(64):
        choice = brain.next_choice("maj", step % 8, NO_CHANGE, step % 8 == 0)
        assert 0 <= choice < NUM_CHOICES


def test_model_likes_the_root_when_the_chord_changes():
    # Not a rule we wrote. The model learned this from the songs.
    roots = 0
    for seed in range(50):
        brain = BassBrain(seed=seed)
        if brain.next_choice("maj", 0, NO_CHANGE, True) == FIRST_NOTE:
            roots += 1
    assert roots > 35


def test_notes_stay_in_bass_range():
    for choice in range(FIRST_NOTE, NUM_CHOICES):
        for root in range(12):
            midi = choice_to_midi(choice, root, 36)
            assert 28 <= midi <= 48
            assert midi % 12 == (root + choice - FIRST_NOTE) % 12


def test_bass_note_starts_on_the_exact_sample():
    synth = BassSynth()
    synth.note_on(700, 40)
    audio = np.concatenate([synth.render() for _ in range(4)])
    assert np.all(audio[:700] == 0)
    assert np.any(audio[700:720] != 0)


def test_drum_hit_starts_on_the_exact_sample():
    kit = DrumKit()
    kit.hit(1000, "snare")
    audio = np.concatenate([kit.render() for _ in range(4)])
    assert np.all(audio[:1000] == 0)
    assert audio[1000] != 0 or audio[1001] != 0
    assert len(audio) == 4 * BLOCK


def test_drums_back_off_when_the_band_is_quiet():
    quiet = [name for name, _ in drum_hits(0, 0, energy=0.05)]
    loud = [name for name, _ in drum_hits(0, 0, energy=0.9)]
    assert quiet == ["hat"]
    assert "kick" in loud
