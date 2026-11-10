"""
The engine connects everything.

Every 11.6 ms it gets a small block of audio from the band and has to
hand back a block of bass and drums. In that time it:

  1. Listens      (find hits, find notes, measure loudness)
  2. Understands  (tempo, beat position, chord, key)
  3. Plans ahead  (which chord is coming, what the bass should play)
  4. Plays        (makes the sound)
"""

import numpy as np

from .config import SAMPLE_RATE, BLOCK, BLOCK_TIME, BEATS_PER_BAR, chord_name
from .features import OnsetDetector, ChromaExtractor
from .beat_tracker import BeatTracker
from .chords import ChordTracker, KeyFinder, SongMemory
from .bass_brain import BassBrain, choice_to_midi, REST, HOLD, NO_CHANGE
from .synth import BassSynth, DrumKit
from .drums import drum_hits


class JamEngine:
    def __init__(self, sound_delay=0.0, use_memory=True, seed=0, bass=True, drums=True):
        """
        sound_delay: how long it takes for sound to get from this program
                     to the speakers and back into the mic, in seconds.
                     We play everything this much EARLY so it lands on time.
        use_memory:  turn the chord guessing on or off (off is only for testing).
        """
        self.sound_delay = sound_delay
        self.use_memory = use_memory
        self.play_bass = bass
        self.play_drums = drums

        self.onsets = OnsetDetector()
        self.chroma = ChromaExtractor()
        self.beat = BeatTracker()
        self.chords = ChordTracker()
        self.key = KeyFinder()
        self.memory = SongMemory()
        self.brain = BassBrain(seed=seed)
        self.bass = BassSynth()
        self.drums = DrumKit()

        self.blocks = 0
        self.loud_now = 0.0       # smoothed loudness of the band
        self.loud_max = 1e-6      # loudest the band has been lately

        self.chord = (None, None)        # chord the bass is playing over
        self.heard_chord = (None, None)  # last chord we actually heard
        self.step_chord = (None, None)   # chord used on the previous step
        self.last_midi = 36
        self.beat_started = None         # (beat number, time) of the current beat
        self.offbeat = None              # (time, beat number) of the next "and" between beats
        self.bar_votes = np.zeros(BEATS_PER_BAR)

        # A diary of what happened. Used for testing and for the live display.
        self.log = {"beats": [], "heard": [], "played_over": [], "notes": []}

    @property
    def now(self):
        return self.blocks * BLOCK_TIME

    @property
    def energy(self):
        """0 = band is quiet, 1 = band is as loud as it has been."""
        return min(1.0, self.loud_now / self.loud_max)

    def process(self, block):
        """Take one block of band audio, return one block of bass + drums."""
        block = np.asarray(block, dtype=np.float32)
        self.blocks += 1

        # 1. Listen
        self.beat.update(self.onsets.process(block))
        self.chroma.push(block)
        loudness = float(np.sqrt(np.mean(block ** 2)))
        self.loud_now += 0.02 * (loudness - self.loud_now)
        self.loud_max = max(self.loud_max * 0.9999, self.loud_now)

        if self.blocks % 2 == 0:
            notes, low_notes = self.chroma.compute()
            # The chroma looks at the last 0.19 s, so its "middle" is half of that ago.
            middle = self.now - self.chroma.WINDOW / SAMPLE_RATE / 2
            self.chords.add(middle, notes, low_notes, self.loud_now)
            self.key.add(notes)

        # 2 and 3. Understand and plan.
        # We plan a little into the future so notes are ready before they are needed.
        plan_until = self.now + self.sound_delay + 2 * BLOCK_TIME
        if self.offbeat and self.offbeat[0] <= plan_until:
            self._on_offbeat(*self.offbeat)
        if self.beat.beat_due(plan_until):
            self._on_beat(*self.beat.take_beat())

        # 4. Play
        out = np.zeros(BLOCK, dtype=np.float32)
        bass, drums = self.bass.render(), self.drums.render()
        if self.play_bass:
            out += bass
        if self.play_drums:
            out += drums
        return out

    def _on_beat(self, number, when):
        """Runs once per beat, just before the beat lands."""
        # Now that the last beat is over, work out which chord it was.
        if self.beat_started is not None:
            last_number, last_time = self.beat_started
            heard = self.chords.chord_between(last_time + 0.03, when, stick_to=self.heard_chord)
            if heard is not None:
                if heard != self.heard_chord:
                    # Chords usually change on beat 1. Counting where changes
                    # happen tells us where the bar starts.
                    self.bar_votes[last_number % BEATS_PER_BAR] += 1
                self.heard_chord = heard
            self.memory.add(self.heard_chord)
            self.log["heard"].append((last_number, self.heard_chord))

        # Guess the chord for the beat that is about to start.
        self.chord = self.memory.guess_next() if self.use_memory else self.heard_chord

        self.beat_started = (number, when)
        self.offbeat = (when + self.beat.period / 2, number)
        self.log["beats"].append((number, when))
        self.log["played_over"].append((number, self.chord))
        self._play_step(when, number, half=0)

    def _on_offbeat(self, when, number):
        """Runs halfway between two beats (the "and" when you count 1 and 2 and)."""
        self.offbeat = None
        # By now we have heard a bit of this beat. If our guess was wrong, fix it.
        heard = self.chords.chord_between(self.beat_started[1] + 0.03, self.now, stick_to=self.chord)
        if heard is not None:
            self.chord = heard
        self._play_step(when, number, half=1)

    def _play_step(self, when, number, half):
        bar_start = int(np.argmax(self.bar_votes))
        beat_in_bar = (number - bar_start) % BEATS_PER_BAR
        bar_number = (number - bar_start) // BEATS_PER_BAR
        place = beat_in_bar * 2 + half

        # Which sample should this sound start on? Play early by sound_delay.
        at_sample = int(round((when - self.sound_delay) * SAMPLE_RATE))

        for name, loudness in drum_hits(place, bar_number, self.energy):
            self.drums.hit(at_sample, name, loudness * (0.5 + 0.5 * self.energy))

        root, quality = self.chord
        changed = self.chord != self.step_chord
        self.step_chord = self.chord
        if root is None:
            # No chord, so the bass stays quiet.
            self.bass.note_off(at_sample)
            self.brain.last_choice = REST
            return

        # Tell the model where the next chord is, if we think it will change.
        upcoming = self.memory.guess_next(and_then=self.chord) if self.use_memory else self.chord
        distance = NO_CHANGE
        if upcoming != self.chord and upcoming[0] is not None:
            distance = (upcoming[0] - root) % 12

        choice = self.brain.next_choice(quality, place, distance, changed)
        if choice == REST:
            self.bass.note_off(at_sample)
        elif choice != HOLD:
            self.last_midi = choice_to_midi(choice, root, self.last_midi)
            self.bass.note_on(at_sample, self.last_midi)
            self.log["notes"].append((when, self.last_midi, chord_name(root, quality)))

    def status(self):
        """One line of text describing what the engine thinks is going on."""
        if not self.beat.locked:
            return "listening..."
        return (f"{self.beat.bpm:5.1f} BPM | chord {chord_name(*self.chord):4s} | "
                f"key {chord_name(*self.key.key()):4s} | energy {self.energy:.2f}")
