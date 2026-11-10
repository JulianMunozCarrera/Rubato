"""
Checks how well the bandmate follows a band, and prints a report.

Part 1 uses our fake guitar band (see bandmate/testband.py).
Part 2 (optional) uses real pop songs from the POP909 test set, turned
into audio with a simple synth. Those songs were never used in training.

    python scripts/evaluate.py
    python scripts/evaluate.py --pop909 data/POP909-Dataset/POP909
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bandmate.config import SAMPLE_RATE, BLOCK_TIME
from bandmate.engine import JamEngine
from bandmate.testband import make_band, pluck
from bandmate.audio_files import run_engine, save

C, D, E, F, G, A, B = 0, 2, 4, 5, 7, 9, 11


def score_beats(our_beats, true_beats, skip_first=6.0, window=0.07):
    """
    Compare our beats to the real ones.
    A beat counts as correct if it is within 70 ms of a real beat.
    That window is the standard one used in beat tracking research.
    We skip the first few seconds because the tracker is still learning the tempo.
    """
    ours = np.array([t for t in our_beats if t >= skip_first])
    true = np.array([t for t in true_beats if t >= skip_first])
    if len(ours) == 0 or len(true) == 0:
        return {"f_score": 0.0, "typical_error_ms": None}
    used, errors = set(), []
    for t in ours:
        nearest = int(np.argmin(np.abs(true - t)))
        if abs(true[nearest] - t) <= window and nearest not in used:
            used.add(nearest)
            errors.append(abs(true[nearest] - t))
    hits = len(errors)
    precision, recall = hits / len(ours), hits / len(true)
    f_score = 2 * precision * recall / (precision + recall) if hits else 0.0
    return {"f_score": round(f_score, 3),
            "typical_error_ms": round(1000 * float(np.median(errors)), 1) if errors else None}


def score_chords(engine, true_beats, chord_at_beat, skip_first=6.0):
    """
    Two chord scores:
      heard:  did we name the chord right after hearing the whole beat?
      on_time: was the bass ALREADY on the right chord when the beat started?
    """
    def right_chord(beat_time):
        nearest = int(np.argmin(np.abs(true_beats - beat_time)))
        return chord_at_beat[nearest]

    when = dict(engine.log["beats"])
    result = {}
    for name, key in (("heard", "heard"), ("on_time", "played_over")):
        marks = [chord == right_chord(when[n]) for n, chord in engine.log[key]
                 if n in when and when[n] >= skip_first]
        result[name] = round(float(np.mean(marks)), 3) if marks else None
    return result


def fake_band_tests():
    loop = [(C, "maj"), (G, "maj"), (A, "min"), (F, "maj")]
    loop_up = [(D, "maj"), (A, "maj"), (B, "min"), (G, "maj")]
    return {
        "steady 110 BPM": (loop * 8, lambda beat: 110),
        "speeding up 95 to 130 BPM": (loop * 10, lambda beat: 95 + 35 * min(1, beat / 140)),
        "slowing down 125 to 90 BPM": (loop * 8, lambda beat: 125 - 35 * min(1, beat / 110)),
        "key change C to D": (loop * 5 + loop_up * 5, lambda beat: 104),
        "sudden jump 100 to 132 BPM": (loop * 10, lambda beat: 100 if beat < 64 else 132),
    }


def render_midi(midi, seconds):
    """Turn a MIDI song into audio using the same plucked sound as the fake band."""
    audio = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
    for instrument in midi.instruments:
        for note in instrument.notes:
            if note.start >= seconds:
                continue
            sound = pluck(note.pitch, min(1.2, note.end - note.start + 0.3), note.velocity / 127 * 0.08)
            start = int(note.start * SAMPLE_RATE)
            audio[start:start + len(sound)] += sound[:len(audio) - start]
    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop909", help="path to the POP909 folder (optional)")
    parser.add_argument("--songs", type=int, default=30)
    parser.add_argument("--demo", help="folder to save a listenable mix of each fake band test")
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args()
    report = {"fake_band": {}, "speed": {}}
    block_times = []

    print("Part 1: fake guitar band")
    for name, (chords, tempo) in fake_band_tests().items():
        audio, beats, chord_at_beat = make_band(chords, tempo)
        row = {}
        for label, use_memory in (("with_memory", True), ("without_memory", False)):
            engine = JamEngine(use_memory=use_memory)
            played, took = run_engine(engine, audio)
            block_times.append(took)
            row[label] = score_chords(engine, beats, chord_at_beat)["on_time"]
            if use_memory:
                row.update(score_beats([t for _, t in engine.log["beats"]], beats))
                row["chord_heard"] = score_chords(engine, beats, chord_at_beat)["heard"]
                if args.demo:
                    os.makedirs(args.demo, exist_ok=True)
                    save(os.path.join(args.demo, name.replace(" ", "_") + ".wav"), audio + played)
        report["fake_band"][name] = row
        print(f"  {name:28s} beat F {row['f_score']:.2f} | off by {row['typical_error_ms']} ms | "
              f"chord heard {row['chord_heard']:.2f} | bass on right chord at the beat: "
              f"{row['with_memory']:.2f} with memory, {row['without_memory']:.2f} without")

    took = np.concatenate(block_times) * 1000
    report["speed"] = {
        "block_ms": round(BLOCK_TIME * 1000, 1),
        "typical_ms": round(float(np.median(took)), 3),
        "slow_case_ms": round(float(np.percentile(took, 99)), 3),
        "worst_ms": round(float(np.max(took)), 3),
    }
    print(f"\nSpeed: each block is {report['speed']['block_ms']} ms of audio. Processing took "
          f"{report['speed']['typical_ms']} ms typically, {report['speed']['slow_case_ms']} ms "
          f"in the slowest 1%, {report['speed']['worst_ms']} ms worst.")

    if args.pop909:
        import pretty_midi
        from train_bass import read_chord
        names = json.load(open("models/bass_training.json"))["test_song_names"][:args.songs]
        print(f"\nPart 2: {len(names)} real pop songs the model never saw (first 60 s each)")
        f_scores, relaxed, errors, heard, right_tempo = [], [], [], [], 0
        for n in names:
            folder = os.path.join(args.pop909, n)
            midi = pretty_midi.PrettyMIDI(os.path.join(folder, n + ".mid"))
            audio = render_midi(midi, 60.0)
            beats = np.loadtxt(os.path.join(folder, "beat_midi.txt"))[:, 0]
            beats = beats[beats < 60.0]  # we only made the first 60 seconds of audio
            labels = [(float(a), float(b), read_chord(c)) for a, b, c in
                      (line.split() for line in open(os.path.join(folder, "chord_midi.txt")))]
            chord_at_beat = [next((c for a, b, c in labels if a <= t + 0.01 < b), (None, None)) for t in beats]

            engine = JamEngine()
            run_engine(engine, audio)
            ours = [t for _, t in engine.log["beats"]]
            s = score_beats(ours, beats)
            f_scores.append(s["f_score"])
            # Also score against the same song counted twice as fast or half
            # as fast. Tapping your foot at double speed is not really wrong.
            between = (beats[:-1] + beats[1:]) / 2
            relaxed.append(max(s["f_score"],
                               score_beats(ours, np.sort(np.r_[beats, between]))["f_score"],
                               score_beats(ours, beats[::2])["f_score"],
                               score_beats(ours, beats[1::2])["f_score"]))
            if s["typical_error_ms"] is not None:
                errors.append(s["typical_error_ms"])
            c = score_chords(engine, beats, np.array(chord_at_beat, dtype=object))["heard"]
            if c is not None:
                heard.append(c)
            true_bpm = 60 / np.median(np.diff(beats))
            if abs(engine.beat.bpm / true_bpm - 1) < 0.04:
                right_tempo += 1
        report["pop909"] = {
            "songs": len(names),
            "mean_beat_f_score": round(float(np.mean(f_scores)), 3),
            "median_beat_f_score": round(float(np.median(f_scores)), 3),
            "mean_beat_f_score_allowing_double_or_half": round(float(np.mean(relaxed)), 3),
            "typical_error_ms": round(float(np.median(errors)), 1),
            "chord_heard": round(float(np.mean(heard)), 3),
            "songs_with_right_tempo": right_tempo,
        }
        print(" ", report["pop909"])

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
