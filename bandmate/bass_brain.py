"""
The part that decides which bass note to play next.

This is a small neural network (a GRU) trained on 909 real pop songs.
It learned things like "play the root note when the chord changes" and
"walk up to the next chord just before it arrives".

The network never sees real note names. Everything is counted from the
chord's root note. So "0" means the root, "7" means the fifth, and so on.
That way what it learns in one key works in every key.

Each step, the model picks one of 14 choices:
    0        = silence
    1        = keep holding the last note
    2 to 13  = play a new note, 0 to 11 steps above the chord's root
"""

import os

import numpy as np
import onnxruntime as ort

REST = 0
HOLD = 1
FIRST_NOTE = 2
NUM_CHOICES = 14

# What the model gets told each step (see scripts/train_bass.py):
#   last choice it made, chord type, place in the bar,
#   where the next chord is, and whether the chord just changed.
QUALITY_ID = {None: 0, "maj": 1, "min": 2}
NO_CHANGE = 12

DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "bass.onnx")


class BassBrain:
    def __init__(self, model_path=DEFAULT_MODEL, seed=0, boldness=0.7):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # one thread is plenty for a model this small
        self.session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])

        # The model's short term memory. It gets passed back in every step.
        memory_shape = self.session.get_inputs()[1].shape
        self.memory = np.zeros(memory_shape, dtype=np.float32)

        self.last_choice = REST
        self.random = np.random.default_rng(seed)
        self.boldness = boldness  # low = safe and predictable, high = more adventurous

    def next_choice(self, quality, place_in_bar, next_root_distance, chord_changed):
        """Ask the model what to play on this step. Returns a number 0 to 13."""
        inputs = np.array(
            [[self.last_choice, QUALITY_ID[quality], place_in_bar, next_root_distance, int(chord_changed)]],
            dtype=np.int64,
        )
        scores, self.memory = self.session.run(None, {"inputs": inputs, "memory_in": self.memory})
        scores = scores[0].astype(np.float64)

        # Turn scores into chances, then pick one. We only choose between
        # the 3 best options so it never plays something really wrong.
        chances = np.exp((scores - scores.max()) / self.boldness)
        top = np.argsort(chances)[-3:]
        chances = chances[top] / chances[top].sum()
        choice = int(self.random.choice(top, p=chances))

        self.last_choice = choice
        return choice


def choice_to_midi(choice, chord_root, last_midi):
    """
    Turn the model's choice into a real note number for a bass guitar.
    We pick the octave closest to the last note so the line flows
    and does not jump around.
    """
    note_name = (chord_root + choice - FIRST_NOTE) % 12
    options = [m for m in range(28, 49) if m % 12 == note_name]  # low E up to C
    return min(options, key=lambda m: (abs(m - last_midi), m))
