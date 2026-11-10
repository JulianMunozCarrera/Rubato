"""
Makes the demo clip in the demo folder using the fake guitar band.
The band changes key halfway through and the bandmate follows.

    python scripts/make_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bandmate.engine import JamEngine
from bandmate.testband import make_band
from bandmate.audio_files import run_engine, save

C, D, G, A, B, F = 0, 2, 7, 9, 11, 5
first_half = [(C, "maj"), (G, "maj"), (A, "min"), (F, "maj")] * 4
second_half = [(D, "maj"), (A, "maj"), (B, "min"), (G, "maj")] * 4

# Starts at 100 BPM, speeds up to 118 by the end.
band, _, _ = make_band(first_half + second_half, lambda beat: 100 + 18 * beat / 128)
engine = JamEngine()
played, _ = run_engine(engine, band)

os.makedirs("demo", exist_ok=True)
save("demo/band_only.flac", band)
save("demo/band_plus_bandmate.flac", band + played)
print("Saved demo/band_only.flac and demo/band_plus_bandmate.flac")
for when, midi, chord in engine.log["notes"][:12]:
    print(f"  {when:6.2f}s  note {midi}  over {chord}")
