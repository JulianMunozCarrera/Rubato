"""
Decides which drums to hit on each step.

The drums follow simple rules instead of a neural network:
  - Kick on beats 1 and 3, snare on beats 2 and 4, hi-hat on every step.
  - If the band plays quietly, the drums back off.
  - Every 4th bar ends with a short fill, like a real drummer would do.
"""


def drum_hits(place_in_bar, bar_number, energy):
    """
    place_in_bar: 0 to 7 (there are 8 steps in a bar)
    bar_number:   which bar we are in
    energy:       0.0 (band is quiet) to 1.0 (band is loud)

    Returns a list like [("kick", 1.0), ("hat", 0.5)].
    """
    hits = []
    on_the_beat = place_in_bar % 2 == 0

    # Hi-hat keeps time. Louder on the beat, softer in between.
    hits.append(("hat", 0.9 if on_the_beat else 0.5))

    # When the band is very quiet, just keep time with the hi-hat.
    if energy < 0.15:
        return hits

    last_bar_of_four = bar_number % 4 == 3
    if last_bar_of_four and place_in_bar >= 6 and energy > 0.4:
        # Fill: two quick snare hits to lead into the next section.
        hits.append(("snare", 0.8 if place_in_bar == 6 else 1.0))
        return hits

    if place_in_bar in (0, 4):
        hits.append(("kick", 1.0))
    if place_in_bar in (2, 6):
        hits.append(("snare", 0.9))
    if place_in_bar == 5 and energy > 0.6:
        hits.append(("kick", 0.7))  # extra kick when the band is going hard
    return hits
