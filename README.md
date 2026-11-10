# Rubato

An AI bandmate. It listens to a band and plays bass and drums along with it, live.

You play. It figures out your tempo, your chords and your key while you are playing, then joins in. If you speed up, it speeds up. If you change key, it follows.

**Why "Rubato"?** It is a music word. It means speeding up and slowing down on purpose to add feeling. Human players do it all the time and a normal backing track can't keep up. This can.

I play in a band, so I wanted to build the bandmate who never misses practice.

## Listen

- [`demo/band_only.flac`](demo/band_only.flac): a guitar on its own. It speeds up from 100 to 118 BPM and changes key halfway.
- [`demo/band_plus_rubato.flac`](demo/band_plus_rubato.flac): the same guitar with Rubato playing along.

Heads up: the guitar in the demo is computer made. Rubato has not been trained or tested on real recordings yet. That is coming soon (see "What is not done yet").

## The hard part

Most music AI makes a song and hands it to you when it is done. This one has to answer every 11.6 milliseconds, and it can never ask the band to wait.

That creates a problem. If you wait to hear a beat before playing on it, you are already late. Same for chords. So Rubato does what human musicians do. It predicts.

- **Beats:** it learns the tempo, then schedules each note before the beat lands.
- **Chords:** bands repeat themselves. It remembers the chords so far, and when it hears a pattern it knows, it guesses the next chord before you play it.

On a looping chord progression, the guessing takes the bass from being on the right chord about 75% of the time to 90 to 96%.

## How it works

```
mic -> find hits  -> tempo and beat ----\
    -> find notes -> chord model, key ---> plan next step -> bass model -> sound out
                  -> song memory -------/                 -> drum rules
```

| Part | File | What it does |
|---|---|---|
| Ears | `rubato/features.py` | Turns audio into "something was just hit" and "these notes are playing" |
| Beat | `rubato/beat_tracker.py` | Finds the tempo and predicts when the next beat lands |
| Chords | `rubato/chords.py` | Names the chord, finds the key, guesses the next chord |
| Bass | `rubato/bass_brain.py` | A small neural network picks the next bass note |
| Drums | `rubato/drums.py` | Simple rules. Plays softer when the band is quiet, adds fills |
| Sound | `rubato/synth.py` | Makes the bass and drum sounds from scratch |
| Engine | `rubato/engine.py` | Runs all of the above for every block of audio |

### The two models

Both are trained on [POP909](https://github.com/music-x-lab/POP909-Dataset), which is 909 pop songs with the beats and chords labelled. Both are tiny on purpose, because they have to run in real time. Both run through ONNX Runtime.

**Bass model.** A GRU network with about 41,000 parameters. It picks the next bass note. It never sees real note names. Every note is counted from the root of the chord, so what it learns in C also works in F sharp. One step takes about 0.03 ms.

**Chord model.** A plain network with about 16,000 parameters. It looks at how strong each of the 12 notes is during a beat and names the chord. I added it because my first version, which matched notes against fixed chord patterns, was only right about half the time on real songs.

## Results

Everything here comes from `scripts/evaluate.py` and the two training scripts. You can run them yourself.

### The models on songs they never saw

| Bass: picks the same next step as the real song | |
|---|---|
| Always pick the most common step | 40.5% |
| Hand written rule (root on chord change, else hold) | 52.7% |
| **Bass model** | **72.6%** |

| Chords: names the right chord for a beat | |
|---|---|
| Pattern matching | 56.1% |
| **Chord model** | **79.9%** |

### Following a fake guitar band

A beat counts as right if it is within 70 ms of the real one. That is the normal window in beat tracking research. The first 6 seconds are skipped because it is still learning the tempo.

| Test | Beats right (F score) | Typical timing error | Bass on right chord, with memory | Without memory |
|---|---|---|---|---|
| Steady 110 BPM | 0.98 | 21 ms | 96% | 76% |
| Speeding up 95 to 130 BPM | 0.98 | 34 ms | 94% | 76% |
| Slowing down 125 to 90 BPM | 0.99 | 6 ms | 96% | 76% |
| Key change C to D | 0.99 | 21 ms | 90% | 73% |
| Sudden jump 100 to 132 BPM | 0.92 | 20 ms | 92% | 75% |

With a plain metronome click the timing error is under 5 ms.

### Following real songs, start to finish

This is the whole system running on 30 POP909 songs that were kept out of training, first 60 seconds each. The songs are MIDI files turned into audio with a simple synth.

| Measure | Result |
|---|---|
| Beats right (F score) | 0.70 |
| Beats right if double or half tempo also counts | 0.83 |
| Songs where it found the exact tempo | 16 of 30 |
| Songs where it found the tempo, or double, or half | 25 of 30 |
| Chord named right, with the chord model | 78% |
| Chord named right, with pattern matching | 52% |
| Bass already on the right chord when the beat starts | 59% |

Real songs are harder than the fake band and the numbers show it. See the next section.

### Speed

Each block is 11.6 ms of audio. Processing one block takes 0.14 ms typically and 0.65 ms in the slowest 1% of cases. Measured on a 2 core cloud CPU.

## What is not done yet

I would rather be upfront about this.

- **Not trained or tested on real sound yet.** Everything so far uses computer made audio. The models learned from songs played by a simple synth, and all the numbers above come from that same kind of audio. The live mic mode is written but I have not measured it with a real band.
- **Real recordings are coming soon.** Next I am recording my own band, testing Rubato on those recordings, and retraining the chord model on real instruments. I will post the new numbers here, good or bad.
- **Slow songs get counted double.** A 70 BPM ballad is often followed at 140. The beats still line up, there are just twice as many. On 5 of 30 songs the tempo was just wrong.
- **Chord guessing needs repetition.** It works great on a 4 chord loop. On real songs, where the chords wander more, the bass is on the right chord at the start of the beat only 59% of the time. It fixes itself half a beat later, once it has heard the chord.
- **4 beats per bar only.** No waltzes.
- **Drums are rules.** Only the bass and the chords are learned.
- **Use headphones.** If Rubato comes out of speakers, the mic hears it and it starts following itself.

## Run it

```bash
git clone https://github.com/JulianMunozCarrera/rubato
cd rubato
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
python scripts/play_along.py my_band.wav --out with_rubato.wav
```

Play live (mic in, headphones out):

```bash
pip install sounddevice
python -m rubato.live
```

Retrain the models:

```bash
pip install torch pretty_midi onnx
git clone https://github.com/music-x-lab/POP909-Dataset data/POP909-Dataset
python scripts/train_bass.py --data data/POP909-Dataset/POP909
python scripts/train_chords.py --data data/POP909-Dataset/POP909
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
