"""
All the important numbers live here, so there is one place to change them.
"""

# How many audio samples we get per second. 44100 is the normal CD quality rate.
SAMPLE_RATE = 44100

# We handle audio in small chunks called "blocks".
# 512 samples is about 11.6 ms of sound. Small blocks = low delay.
BLOCK = 512

# How long one block lasts, in seconds.
BLOCK_TIME = BLOCK / SAMPLE_RATE

# Slowest and fastest tempo we try to follow (beats per minute).
MIN_BPM = 60
MAX_BPM = 200

# The bass plays 2 notes per beat at most (these are called 8th notes).
STEPS_PER_BEAT = 2

# We assume 4 beats in a bar, like most rock and pop songs.
BEATS_PER_BAR = 4

# Note names, starting from C. Index 0 = C, 1 = C#, and so on.
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def chord_name(root, quality):
    """Turn (0, 'maj') into 'C', and (9, 'min') into 'Am'."""
    if root is None:
        return "none"
    return NOTE_NAMES[root] + ("m" if quality == "min" else "")
