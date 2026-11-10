# AI Bandmate

A program that listens to a band and plays bass and drums along with it, live.

You play. It figures out your tempo, your chords and your key while you are playing, then joins in. If you speed up, it speeds up. If you change key, it follows.

I play in a band, so I wanted to build the bandmate who never misses practice.

## Listen

- [`demo/band_only.flac`](demo/band_only.flac): a guitar on its own. It speeds up from 100 to 118 BPM and changes key halfway.
- [`demo/band_plus_bandmate.flac`](demo/band_plus_bandmate.flac): the same guitar with the bandmate playing along.

The guitar in the demo is computer made (see "What is not done yet").

## The hard part

Most music AI makes a song and hands it to you when it is done. This one has to answer every 11.6 milliseconds, and it can never ask the band to wait.

That creates a problem. If you wait to hear a beat before playing on it, you are already late. Same for chords. So the bandmate does what human musicians do. It predicts.

- **Beats:** it learns the tempo, then schedules each note before the beat lands.
- **Chords:** bands repeat themselves. It remembers the chords so far, and when it hears a pattern it knows, it guesses the next chord before you play it.

The chord guessing made the biggest difference. Without it, the bass is on the right chord at the start of the beat 76% of the time. With it, 94 to 97%.

## How it works

```
mic -> find hits -> tempo and beat ----\
    -> find notes -> chord and key -----> plan next step -> bass model -> sound out
                  -> song memory ------/                 -> drum rules
```

| Part | File | What it does |
|---|---|---|
| Ears | `bandmate/features.py` | Turns audio into "something was just hit" and "these notes are playing" |
| Beat | `bandmate/beat_tracker.py` | Finds the tempo and predicts when the next beat lands |
| Chords | `bandmate/chords.py` | Names the chord, finds the key, guesses the next chord |
| Bass | `bandmate/bass_brain.py` | A small neural network picks the next bass note |
| Drums | `bandmate/drums.py` | Simple rules. Plays softer when the band is quiet, adds fills |
| Sound | `bandmate/synth.py` | Makes the bass and drum sounds from scratch |
| Engine | `bandmate/engine.py` | Runs all of the above for every block of audio |

### The bass model

A small GRU network (about 40,000 numbers to learn, 167 KB on disk). I trained it on [POP909](https://github.com/music-x-lab/POP909-Dataset), which is 909 pop songs with the beats and chords labelled.

It never sees real note names. Every note is counted from the root of the chord. So what it learns in C also works in F sharp.

It runs through ONNX Runtime. One step takes about 0.03 ms.

## Results

Everything here comes from `scripts/evaluate.py`. You can run it yourself.

### Bass model (91 songs it never saw in training)

| Method | Picks the same next step as the real song |
|---|---|
| Always pick the most common step | 40.5% |
| Hand written rule (root on chord change, else hold) | 52.7% |
| **The model** | **72.6%** |

### Following a fake guitar band

A beat counts as right if it is within 70 ms of the real one. That is the normal window in beat tracking research. The first 6 seconds are skipped because it is still learning the tempo.

| Test | Beats right (F score) | Typical timing error | Bass on right chord, with memory | Without memory |
|---|---|---|---|---|
| Steady 110 BPM | 0.98 | 21 ms | 96% | 76% |
| Speeding up 95 to 130 BPM | 0.98 | 34 ms | 97% | 76% |
| Slowing down 125 to 90 BPM | 0.99 | 6 ms | 96% | 76% |
| Key change C to D | 0.99 | 21 ms | 94% | 76% |
| Sudden jump 100 to 132 BPM | 0.93 | 20 ms | 94% | 76% |

With a plain metronome click the timing error is under 5 ms.

### Following real songs

30 POP909 songs that were kept out of training, first 60 seconds each. The songs are MIDI files turned into audio with a simple synth.

| Measure | Result |
|---|---|
| Beats right (F score) | 0.69 |
| Beats right if double or half tempo also counts | 0.82 |
| Songs where it found the exact tempo | 15 of 30 |
| Chord named right, per beat | 37% |

This is much harder than the fake band and the numbers show it. See the next section.

### Speed

Each block is 11.6 ms of audio. Processing one block takes 0.12 ms typically and 0.56 ms in the slowest 1% of cases. Measured on a 2 core cloud CPU.

## What is not done yet

I would rather be upfront about this.

- **Not tested with a real band yet.** All numbers come from computer made audio. The live mic mode is written but I have not measured it on stage. That is the next step.
- **Chords on real songs are weak (37%).** It names chords by matching notes against simple patterns. That works on strummed chords and struggles when a piano plays one note at a time. A trained chord model is the obvious fix.
- **Slow songs get counted double.** A 70 BPM ballad is often followed at 140. The beats still line up, there are just twice as many.
- **4 beats per bar only.** No waltzes.
- **Drums are rules.** Only the bass is learned.
- **Use headphones.** If the bandmate comes out of speakers, the mic hears it and it starts following itself.

## Run it

```bash
git clone https://github.com/JulianMunozCarrera/ai-bandmate
cd ai-bandmate
pip install -r requirements.txt
```

Run the tests:

```bash
python -m pytest
```

Make the demo clip:

```bash
python scripts/make_demo.py
```

Play along with any recording:

```bash
python scripts/play_along.py my_band.wav --out with_bandmate.wav
```

Play live (mic in, headphones out):

```bash
pip install sounddevice
python -m bandmate.live
```

Retrain the bass model:

```bash
pip install torch pretty_midi onnx
git clone https://github.com/music-x-lab/POP909-Dataset data/POP909-Dataset
python scripts/train_bass.py --data data/POP909-Dataset/POP909
```

Run the full report:

```bash
python scripts/evaluate.py --pop909 data/POP909-Dataset/POP909
```

## Why this is a bit like a brain

I started in neuroscience before moving to computer science. When people tap along to music, they tap slightly before the beat, not after it. The brain predicts the beat instead of reacting to it. This project ended up working the same way, because reacting is too slow for a computer too.

## Credits

- [POP909 dataset](https://github.com/music-x-lab/POP909-Dataset) by Music X Lab (MIT license)
- Key fingerprints from Krumhansl and Kessler (1982)
