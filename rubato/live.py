"""
Run Rubato live with a microphone and speakers.

    python -m rubato.live

Use headphones! If the bass and drums come out of speakers, the mic
hears them too and Rubato starts listening to itself.
"""

import argparse
import time

import numpy as np

from .config import SAMPLE_RATE, BLOCK
from .engine import JamEngine


def main():
    import sounddevice as sd  # imported here so the rest of the project works without a sound card

    parser = argparse.ArgumentParser()
    parser.add_argument("--delay-ms", type=float, default=None,
                        help="time for sound to go out and come back in. Leave empty to use what the sound card reports.")
    parser.add_argument("--no-bass", action="store_true")
    parser.add_argument("--no-drums", action="store_true")
    parser.add_argument("--volume", type=float, default=0.8)
    args = parser.parse_args()

    engine = None

    def on_audio(mic, speakers, frames, when, status):
        # The sound card calls this every 11.6 ms. It has to be quick.
        if engine is None:
            speakers[:] = 0
            return
        speakers[:, 0] = args.volume * engine.process(mic[:, 0])

    with sd.Stream(samplerate=SAMPLE_RATE, blocksize=BLOCK, channels=1, dtype="float32",
                   latency="low", callback=on_audio) as stream:
        if args.delay_ms is None:
            delay = sum(stream.latency)  # mic delay + speaker delay
        else:
            delay = args.delay_ms / 1000
        engine = JamEngine(sound_delay=delay, bass=not args.no_bass, drums=not args.no_drums)

        print(f"Sound delay: {delay * 1000:.0f} ms. Start playing! Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.25)
                print("\r" + engine.status() + "      ", end="", flush=True)
        except KeyboardInterrupt:
            print("\nBye!")


if __name__ == "__main__":
    main()
