"""
Trains the bass model.

What it does, step by step:
  1. Reads 909 pop songs (the POP909 dataset). Each song has the notes
     that were played plus labels for the beats and chords.
  2. Pulls out the lowest notes of the piano part. That is the bass line.
  3. Turns each song into a list of small steps (2 per beat).
  4. Trains a small network to guess the next bass step.
  5. Saves the model as an ONNX file so it runs fast without PyTorch.

Run it like this:
    git clone https://github.com/music-x-lab/POP909-Dataset data/POP909-Dataset
    python scripts/train_bass.py --data data/POP909-Dataset/POP909
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
from bandmate.bass_brain import REST, HOLD, FIRST_NOTE, NUM_CHOICES, NO_CHANGE, QUALITY_ID

NOTE_NUMBER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
MEMORY_SIZE = 96
HIGHEST_BASS_NOTE = 54  # anything above this is not a bass note


def read_chord(label):
    """Turn a label like 'F#:min7/5' into (6, 'min'). 'N' means no chord."""
    if label == "N":
        return (None, None)
    name, kind = label.split(":")
    root = NOTE_NUMBER[name[0]]
    root += name.count("#") - name.count("b")
    quality = "min" if kind.startswith(("min", "dim", "hdim")) else "maj"
    return (root % 12, quality)


def song_to_steps(folder):
    """
    Turn one song into a table with one row per step.
    Each row: [choice, chord type, place in bar, distance to next chord, chord changed]
    """
    number = os.path.basename(folder)
    midi = pretty_midi.PrettyMIDI(os.path.join(folder, number + ".mid"))
    piano = [i for i in midi.instruments if i.name.strip().upper() == "PIANO"]
    if not piano:
        return None
    low_notes = sorted((n for n in piano[0].notes if n.pitch <= HIGHEST_BASS_NOTE), key=lambda n: n.start)

    beats = np.loadtxt(os.path.join(folder, "beat_midi.txt"))
    chords = []
    for line in open(os.path.join(folder, "chord_midi.txt")):
        start, end, label = line.split()
        chords.append((float(start), float(end), read_chord(label)))

    def chord_at(time):
        for start, end, chord in chords:
            if start <= time < end:
                return chord
        return (None, None)

    # Build the list of step times: on each beat and halfway to the next one.
    step_times, places = [], []
    beat_in_bar = 0
    for i in range(len(beats) - 1):
        if beats[i, 2] == 1:
            beat_in_bar = 0
        start, end = beats[i, 0], beats[i + 1, 0]
        for half in (0, 1):
            step_times.append(start + half * (end - start) / 2)
            places.append((beat_in_bar % 4) * 2 + half)
        beat_in_bar += 1

    rows = []
    sounding_until = -1.0
    last_chord = (None, None)
    note_index = 0
    for s, time in enumerate(step_times):
        step_length = (step_times[s + 1] - time) if s + 1 < len(step_times) else 0.3
        chord = chord_at(time + 0.01)

        # Find bass notes that start close to this step.
        window_start, window_end = time - step_length / 2, time + step_length / 2
        while note_index < len(low_notes) and low_notes[note_index].start < window_start:
            note_index += 1
        starting = []
        j = note_index
        while j < len(low_notes) and low_notes[j].start < window_end:
            starting.append(low_notes[j])
            j += 1

        if chord[0] is None:
            choice = REST
        elif starting:
            lowest = min(starting, key=lambda n: n.pitch)
            choice = FIRST_NOTE + (lowest.pitch - chord[0]) % 12
            sounding_until = lowest.end
        elif sounding_until > time + 0.02:
            choice = HOLD
        else:
            choice = REST

        # Which chord is playing at the start of the next beat?
        steps_to_next_beat = 2 - (places[s] % 2)
        ahead = s + steps_to_next_beat
        distance = NO_CHANGE
        if ahead < len(step_times) and chord[0] is not None:
            upcoming = chord_at(step_times[ahead] + 0.01)
            if upcoming != chord and upcoming[0] is not None:
                distance = (upcoming[0] - chord[0]) % 12

        rows.append([choice, QUALITY_ID[chord[1]], places[s], distance, int(chord != last_chord)])
        last_chord = chord
    return np.array(rows, dtype=np.int64)


class BassNet(nn.Module):
    """
    The model. Each input number gets turned into a small list of numbers
    (an "embedding"), they all go through a GRU (a layer with memory),
    and out come 14 scores, one for each possible choice.
    """

    def __init__(self):
        super().__init__()
        self.last_choice = nn.Embedding(NUM_CHOICES, 16)
        self.quality = nn.Embedding(3, 4)
        self.place = nn.Embedding(8, 8)
        self.distance = nn.Embedding(13, 8)
        self.changed = nn.Embedding(2, 2)
        self.gru = nn.GRU(16 + 4 + 8 + 8 + 2, MEMORY_SIZE, batch_first=True)
        self.out = nn.Linear(MEMORY_SIZE, NUM_CHOICES)

    def forward(self, inputs, memory=None):
        # inputs has shape (songs, steps, 5)
        x = torch.cat([
            self.last_choice(inputs[..., 0]),
            self.quality(inputs[..., 1]),
            self.place(inputs[..., 2]),
            self.distance(inputs[..., 3]),
            self.changed(inputs[..., 4]),
        ], dim=-1)
        x, memory = self.gru(x, memory)
        return self.out(x), memory


class OneStep(nn.Module):
    """Wraps the model so it handles a single step. This is what we export."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, memory_in):
        scores, memory_out = self.net(inputs.unsqueeze(1), memory_in)
        return scores[:, 0], memory_out


def make_examples(song, length=64):
    """
    Cut a song into pieces of 64 steps.
    Input = what the model knows. Target = the choice it should make.
    """
    targets = song[:, 0]
    inputs = song.copy()
    inputs[1:, 0] = targets[:-1]  # the model sees the PREVIOUS choice, not the answer
    inputs[0, 0] = REST
    pieces = []
    for start in range(0, len(song) - length, length // 2):
        pieces.append((inputs[start:start + length], targets[start:start + length]))
    return pieces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to the POP909 folder")
    parser.add_argument("--out", default="models/bass.onnx")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    folders = sorted(f for f in os.listdir(args.data) if f.isdigit())
    songs = {}
    for f in folders:
        steps = song_to_steps(os.path.join(args.data, f))
        if steps is not None and len(steps) > 100:
            songs[f] = steps
    names = sorted(songs)
    # The last 10% of songs are kept away from training. We test on those.
    split = int(len(names) * 0.9)
    train_songs, test_songs = names[:split], names[split:]
    print(f"{len(train_songs)} songs for training, {len(test_songs)} for testing")

    def to_tensors(song_names):
        pieces = [p for n in song_names for p in make_examples(songs[n])]
        return (torch.tensor(np.stack([p[0] for p in pieces])),
                torch.tensor(np.stack([p[1] for p in pieces])))

    train_x, train_y = to_tensors(train_songs)
    test_x, test_y = to_tensors(test_songs)

    net = BassNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()

    def accuracy(x, y):
        net.eval()
        with torch.no_grad():
            guess = net(x)[0].argmax(-1)
        return float((guess == y).float().mean())

    for epoch in range(args.epochs):
        net.train()
        order = torch.randperm(len(train_x))
        for i in range(0, len(order), 64):
            batch = order[i:i + 64]
            x, y = train_x[batch].clone(), train_y[batch]
            # 30% of the time, hide where the next chord is. On stage we
            # do not always know it, so the model has to cope without it.
            hide = torch.rand(x.shape[:2]) < 0.3
            x[..., 3][hide] = NO_CHANGE
            scores, _ = net(x)
            loss = loss_fn(scores.reshape(-1, NUM_CHOICES), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch + 1}: loss {loss.item():.3f}, test accuracy {accuracy(test_x, test_y):.3f}")

    # Compare against two simple rules, to prove the model learned something.
    flat_x, flat_y = test_x.reshape(-1, 5), test_y.reshape(-1)
    most_common = int(torch.bincount(train_y.reshape(-1)).argmax())
    always_same = float((flat_y == most_common).float().mean())
    rule = torch.where(flat_x[:, 4] == 1, torch.tensor(FIRST_NOTE), torch.tensor(HOLD))
    rule = torch.where(flat_x[:, 1] == 0, torch.tensor(REST), rule)
    simple_rule = float((rule == flat_y).float().mean())
    results = {
        "model_accuracy": round(accuracy(test_x, test_y), 4),
        "always_pick_most_common": round(always_same, 4),
        "root_on_chord_change_else_hold": round(simple_rule, 4),
        "train_songs": len(train_songs),
        "test_songs": len(test_songs),
        "test_song_names": test_songs,
    }
    print(json.dumps({k: v for k, v in results.items() if k != "test_song_names"}, indent=2))

    net.eval()
    torch.onnx.export(
        OneStep(net),
        (torch.zeros(1, 5, dtype=torch.long), torch.zeros(1, 1, MEMORY_SIZE)),
        args.out,
        input_names=["inputs", "memory_in"],
        output_names=["scores", "memory_out"],
        dynamo=False,
    )
    with open(os.path.splitext(args.out)[0] + "_training.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
