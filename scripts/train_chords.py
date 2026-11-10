"""
Trains the chord model.

The simple way to name a chord is to compare the notes you hear against
a fixed pattern for each chord. That breaks on real songs, because
players rarely hit all the notes of a chord at once.

So this script trains a small network to do it instead:
  1. Take real pop songs (POP909) and turn them into audio.
  2. For every beat, measure how strong each of the 12 notes is.
  3. Teach the network to go from those measurements to the chord label.

Run it like this:
    python scripts/train_chords.py --data data/POP909-Dataset/POP909
"""

import argparse
import json
import os
import sys

import numpy as np
import pretty_midi
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from rubato.config import SAMPLE_RATE, BLOCK
from rubato.features import ChromaExtractor
from rubato.chords import CHORD_LABELS, chord_model_input, score_chords
from rubato.testband import render_midi
from train_bass import read_chord

SECONDS_PER_SONG = 75.0


def song_to_beats(folder):
    """
    Returns one entry per beat: (chord label, notes, low notes, notes of
    the first half of the beat, low notes of the first half).
    We keep the first half too, because on stage the program sometimes
    has to decide before the beat is over.
    """
    number = os.path.basename(folder)
    audio = render_midi(pretty_midi.PrettyMIDI(os.path.join(folder, number + ".mid")), SECONDS_PER_SONG)
    beats = np.loadtxt(os.path.join(folder, "beat_midi.txt"))[:, 0]
    beats = beats[beats < SECONDS_PER_SONG - 1]
    labels = []
    for line in open(os.path.join(folder, "chord_midi.txt")):
        start, end, label = line.split()
        labels.append((float(start), float(end), read_chord(label)))

    # Measure the notes the same way the live engine does.
    chroma = ChromaExtractor()
    times, notes, low_notes = [], [], []
    for i, start in enumerate(range(0, len(audio) - BLOCK + 1, BLOCK)):
        chroma.push(audio[start:start + BLOCK])
        if i % 2 == 1:
            n, low = chroma.compute()
            times.append((start + BLOCK) / SAMPLE_RATE - chroma.WINDOW / SAMPLE_RATE / 2)
            notes.append(n)
            low_notes.append(low)
    times, notes, low_notes = np.array(times), np.array(notes), np.array(low_notes)

    rows = []
    for start, end in zip(beats[:-1], beats[1:]):
        chord = next((c for a, b, c in labels if a <= start + 0.01 < b), (None, None))
        whole = (times >= start + 0.03) & (times < end)
        first_half = (times >= start + 0.03) & (times < (start + end) / 2)
        if not whole.any() or not first_half.any():
            rows.append(None)
            continue
        rows.append((chord, notes[whole].mean(0), low_notes[whole].mean(0),
                     notes[first_half].mean(0), low_notes[first_half].mean(0)))
    return rows


def make_examples(songs, with_half_beats):
    """Turn the per beat rows into (48 numbers, chord number) pairs."""
    inputs, answers = [], []
    for rows in songs:
        before = None
        for row in rows:
            if row is None:
                before = None
                continue
            chord, notes, low, half_notes, half_low = row
            if before is not None and chord[0] is not None:
                versions = [(notes, low)] + ([(half_notes, half_low)] if with_half_beats else [])
                for n, l in versions:
                    inputs.append(chord_model_input(n, l, *before))
                    answers.append(CHORD_LABELS.index(chord))
            before = (notes, low)
    return np.array(inputs, dtype=np.float32), np.array(answers)


def shift_key(inputs, answers, steps):
    """
    Move every example up by some number of notes. A C chord moved up 2
    becomes a D chord. This gives us 12 times more training data for free
    and makes sure the model is equally good in every key.
    """
    moved = np.concatenate([np.roll(inputs[:, i:i + 12], steps, axis=1) for i in (0, 12, 24, 36)], axis=1)
    quality, root = answers // 12, (answers % 12 + steps) % 12
    return moved, quality * 12 + root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to the POP909 folder")
    parser.add_argument("--out", default="models/chords.onnx")
    parser.add_argument("--songs", type=int, default=260, help="how many songs to train on")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    # Use the same test songs as the bass model, so no test song is ever trained on.
    test_names = json.load(open("models/bass_training.json"))["test_song_names"]
    all_names = sorted(f for f in os.listdir(args.data) if f.isdigit())
    train_names = [n for n in all_names if n not in set(test_names)][::3][:args.songs]
    test_names = test_names[:30]

    print(f"Turning {len(train_names) + len(test_names)} songs into audio. This takes a few minutes.")
    train_x, train_y = make_examples([song_to_beats(os.path.join(args.data, n)) for n in train_names], True)
    test_x, test_y = make_examples([song_to_beats(os.path.join(args.data, n)) for n in test_names], False)
    print(f"{len(train_x)} training beats, {len(test_x)} test beats")

    # A plain network: 48 numbers in, two hidden layers, 24 chord scores out.
    net = nn.Sequential(nn.Linear(48, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 24))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

    def accuracy():
        net.eval()
        with torch.no_grad():
            return float((net(torch.tensor(test_x)).argmax(1).numpy() == test_y).mean())

    for epoch in range(args.epochs):
        net.train()
        order = np.random.permutation(len(train_x))
        for i in range(0, len(order), 128):
            batch = order[i:i + 128]
            x, y = shift_key(train_x[batch], train_y[batch], np.random.randint(12))
            x = torch.tensor(x)
            x = x * (1 + 0.1 * torch.randn(x.shape))  # a little noise so it does not memorise
            loss = nn.functional.cross_entropy(net(x), torch.tensor(y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch + 1}: test accuracy {accuracy():.3f}")

    # How does plain pattern matching do on the exact same beats?
    pattern = float(np.mean([int(np.argmax(score_chords(x[:12], x[12:24]))) == y for x, y in zip(test_x, test_y)]))
    results = {"model_accuracy": round(accuracy(), 4), "pattern_matching_accuracy": round(pattern, 4),
               "train_songs": len(train_names), "test_songs": len(test_names)}
    print(json.dumps(results, indent=2))

    net.eval()
    torch.onnx.export(net, (torch.zeros(1, 48),), args.out, input_names=["notes"],
                      output_names=["scores"], dynamo=False)
    with open(os.path.splitext(args.out)[0] + "_training.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
