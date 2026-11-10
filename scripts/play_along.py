"""
Let the bandmate play along with a recording instead of a live mic.

    python scripts/play_along.py my_band.wav
    python scripts/play_along.py my_band.wav --out with_bandmate.wav

Handy for trying it without a sound card, and for making demo clips.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bandmate.engine import JamEngine
from bandmate.audio_files import load, save, run_engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording")
    parser.add_argument("--out", default="with_bandmate.wav")
    parser.add_argument("--only-bandmate", action="store_true", help="save the bass and drums without the band")
    args = parser.parse_args()

    band = load(args.recording)
    engine = JamEngine()
    played, _ = run_engine(engine, band)

    save(args.out, played if args.only_bandmate else band + played)
    print(f"Tempo at the end: {engine.beat.bpm:.1f} BPM")
    print(f"Bass notes played: {len(engine.log['notes'])}")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
