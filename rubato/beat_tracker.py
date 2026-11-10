"""
Finds the beat of the music while it is being played.

The hard part: we can't wait to hear a beat and then play on it, because by
then it's too late. So we do what a real drummer does. We learn the tempo,
then PREDICT when the next beat will land and get ready before it happens.

Two questions get answered here:
1. How fast is the band playing?  (the tempo)
2. Exactly when do the beats land? (the phase)
"""

import numpy as np

from .config import BLOCK_TIME, MIN_BPM, MAX_BPM

# We notice a hit a tiny bit after it really happened, because we need to
# hear some of it first. This number was measured with a click track test
# (see tests/test_beat_tracker.py) and we subtract it to fix the timing.
ONSET_DELAY = 0.008


class BeatTracker:
    HISTORY_SECONDS = 6.0   # how much recent music we remember
    UPDATE_EVERY = 8        # redo the math every 8 blocks (about 93 ms)
    EXTRA_PROOF = [(2, 0.5), (3, 0.33), (4, 0.25)]  # (how many beats apart, how much it counts)
    USUAL_BEAT = 0.6        # seconds per beat that we lean towards (100 BPM)
    USUAL_SPREAD = 0.6      # how strongly we lean. Smaller = stronger.

    def __init__(self):
        self.size = int(self.HISTORY_SECONDS / BLOCK_TIME)
        self.history = np.zeros(self.size, dtype=np.float32)
        self.blocks_seen = 0

        self.period = 0.5            # seconds per beat (0.5 = 120 BPM)
        self.next_beat_time = None   # when we think the next beat will land
        self.beat_count = 0          # how many beats we have announced so far
        self.locked = False          # True once we trust our tempo

        # Used to make sure a new tempo is real before we jump to it.
        self._candidate = None
        self._candidate_votes = 0
        self._wrong_spot_votes = 0

        # Shortest and longest gap between beats, counted in blocks.
        self.min_lag = int(60.0 / MAX_BPM / BLOCK_TIME)
        self.max_lag = int(60.0 / MIN_BPM / BLOCK_TIME)

    @property
    def bpm(self):
        return 60.0 / self.period

    @property
    def now(self):
        return self.blocks_seen * BLOCK_TIME

    def update(self, onset_strength):
        """Call this once per block with the number from OnsetDetector."""
        self.history[:-1] = self.history[1:]
        self.history[-1] = onset_strength
        self.blocks_seen += 1

        # Wait until we have heard 3 seconds, then update every few blocks.
        if self.blocks_seen * BLOCK_TIME < 3.0:
            return
        if self.blocks_seen % self.UPDATE_EVERY != 0:
            return

        recent = self.history[-min(self.blocks_seen, self.size):]
        recent = recent - recent.mean()

        # If there's basically no sound, keep counting at the old tempo.
        # A good drummer keeps time when the guitarist stops.
        if np.max(np.abs(recent)) < 1e-3:
            return

        self._update_tempo(recent)
        self._update_phase(recent)

    def _update_tempo(self, recent):
        """
        Find the tempo by checking how well the music lines up with a
        delayed copy of itself. If you delay a song by exactly one beat,
        the hits line up again. That delay is the beat length.
        """
        n = len(recent)
        spectrum = np.fft.rfft(recent, 2 * n)
        match = np.fft.irfft(np.abs(spectrum) ** 2)[:n]  # match[d] = how well a delay of d lines up

        lags = np.arange(self.min_lag, self.max_lag + 1)
        score = match[lags].copy()

        # A real beat also lines up at 2, 3 and 4 times the delay, so we
        # add those as extra proof. Each one counts a bit less.
        for times, weight in self.EXTRA_PROOF:
            further = lags * times
            ok = further < n
            score[ok] += weight * match[further[ok]]

        # Prefer tempos near the middle of the range, where most songs live.
        # This helps us not pick half speed or double speed by mistake.
        periods = lags * BLOCK_TIME
        score *= np.exp(-0.5 * (np.log2(periods / self.USUAL_BEAT) / self.USUAL_SPREAD) ** 2)

        best = int(np.argmax(score))
        lag = float(lags[best])

        # Sharpen the answer by checking 4 beats apart. A small error over
        # one beat becomes easy to see over four.
        far = int(round(lag * 4))
        if far + 4 < n:
            around = np.arange(far - 3, far + 4)
            peak = int(around[np.argmax(match[around])])
            lag = _fine_peak(match, peak) / 4.0

        new_period = lag * BLOCK_TIME

        if not self.locked:
            # First time: need to see the same tempo a few times in a row.
            if self._candidate and abs(new_period / self._candidate - 1) < 0.05:
                self._candidate_votes += 1
            else:
                self._candidate, self._candidate_votes = new_period, 1
            if self._candidate_votes >= 5:
                self.period = new_period
                self.locked = True
        elif abs(new_period / self.period - 1) < 0.08:
            # Small change (band speeding up or slowing down): follow smoothly.
            self.period += 0.3 * (new_period - self.period)
            self._candidate_votes = 0
        else:
            # Big change: only jump if we keep seeing the same new tempo
            # for about 1.5 seconds.
            if self._candidate and abs(new_period / self._candidate - 1) < 0.05:
                self._candidate_votes += 1
            else:
                self._candidate, self._candidate_votes = new_period, 1
            if self._candidate_votes >= 16:
                self.period = new_period
                self._candidate_votes = 0

    def _update_phase(self, recent):
        """
        We know how far apart beats are. Now find where exactly they land.

        We slide a row of evenly spaced markers over the recent music and
        look for the position where the markers sit on the loudest hits.
        """
        if not self.locked:
            return

        beat_blocks = self.period / BLOCK_TIME
        loud = np.maximum(recent, 0)
        end = len(loud) - 1

        shifts = np.arange(int(np.ceil(beat_blocks)))
        total = np.zeros(len(shifts))
        for k in range(6):  # look at the last 6 beats
            spots = end - shifts - int(round(k * beat_blocks))
            good = spots >= 0
            # Newer beats count more than older ones.
            total[good] += (0.75 ** k) * loud[spots[good]]

        if self.next_beat_time is None:
            # First time: just take the strongest position.
            shift = _fine_peak(total, int(np.argmax(total)))
            self.next_beat_time = self.now - shift * BLOCK_TIME - ONSET_DELAY + self.period
            while self.next_beat_time < self.now:
                self.next_beat_time += self.period
            return

        # Where do we EXPECT the last beat to be, counted in blocks ago?
        expected = self.next_beat_time - self.period
        expected_shift = ((self.now - ONSET_DELAY - expected) / BLOCK_TIME) % beat_blocks

        # Only look close to where we expect the beat. Music has lots of
        # notes between the beats, and we don't want to hop onto those.
        distance = np.abs(shifts - expected_shift)
        distance = np.minimum(distance, beat_blocks - distance)
        nearby = distance <= 0.2 * beat_blocks
        near_best = int(np.argmax(np.where(nearby, total, -1)))
        any_best = int(np.argmax(total))

        # But if a spot far away keeps being much stronger for about
        # 1.5 seconds, we were probably counting on the wrong notes. Move there.
        if not nearby[any_best] and total[any_best] > 1.4 * total[near_best]:
            self._wrong_spot_votes += 1
        else:
            self._wrong_spot_votes = 0
        moving = self._wrong_spot_votes >= 16
        best = any_best if moving else near_best

        shift = _fine_peak(total, best)
        heard_beat = self.now - shift * BLOCK_TIME - ONSET_DELAY

        # Compare where we heard the beat to where we expected it.
        error = (heard_beat - expected) % self.period
        if error > self.period / 2:
            error -= self.period

        if moving:
            self.next_beat_time += error
            self._wrong_spot_votes = 0
            return

        # Nudge our prediction part of the way. Not all the way, or one
        # sloppy strum would throw us off.
        self.next_beat_time += 0.3 * error
        self.period += 0.05 * error

        # Safety net: never let the tempo drift outside the range we support.
        self.period = float(np.clip(self.period, 60.0 / MAX_BPM, 60.0 / MIN_BPM))

    def beat_due(self, until):
        """True if the next beat lands before the time 'until'."""
        return self.locked and self.next_beat_time is not None and self.next_beat_time <= until

    def take_beat(self):
        """Announce the next beat. Returns (beat number, time it lands)."""
        number, when = self.beat_count, self.next_beat_time
        self.beat_count += 1
        self.next_beat_time += self.period
        return number, when


def _fine_peak(values, i):
    """
    The true peak is often between two points. This fits a small curve
    through the peak and its two neighbours to guess the exact spot.
    """
    if i <= 0 or i >= len(values) - 1:
        return float(i)
    left, mid, right = values[i - 1], values[i], values[i + 1]
    bottom = left - 2 * mid + right
    if bottom == 0:
        return float(i)
    offset = 0.5 * (left - right) / bottom
    return i + float(np.clip(offset, -1, 1))
