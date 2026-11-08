"""
MoodEngine: slow-moving internal state that decides *how* the slime behaves,
not *what* it does. It never moves anything itself -- gl_main.py reads the
current mood's rate multipliers (profile()) to scale its existing behavior
loops, and asks wants_sleep()/should_wake() to drive the sleep state.

Inputs:
  - time of day   -> `energy` drifts toward a circadian target (low at night)
  - neglect       -> `boredom` climbs the longer nothing interacts with it
  - being poked   -> `annoyance` jumps per poke / rude wake-up, decays slowly

Moods: content, sleepy, bored, jumpy, annoyed.
"""
import time
from datetime import datetime


# (hour, target energy) -- linearly interpolated between points
_CIRCADIAN = [
    (0.0, 0.12), (5.0, 0.12), (8.0, 0.50), (10.0, 0.72),
    (17.0, 0.70), (21.0, 0.50), (23.0, 0.25), (24.0, 0.12),
]


def circadian_energy(hour_float=None):
    if hour_float is None:
        now = datetime.now()
        hour_float = now.hour + now.minute / 60.0
    for (h0, e0), (h1, e1) in zip(_CIRCADIAN, _CIRCADIAN[1:]):
        if h0 <= hour_float <= h1:
            f = (hour_float - h0) / (h1 - h0)
            return e0 + (e1 - e0) * f
    return 0.5


# Multipliers applied to the widget's existing behavior loops.
# wander/look/twitch/stretch/lurk scale how OFTEN things happen (both the
# chance and the delay between checks); hop_h / hop_dur scale the hop itself
# (hop_dur > 1 = slower, lazier hops).
PROFILES = {
    "content": dict(wander=1.0, hop_h=1.0, hop_dur=1.0, look=1.0, twitch=1.0, stretch=1.0, lurk=1.0),
    "sleepy":  dict(wander=0.35, hop_h=0.6, hop_dur=1.3, look=0.5, twitch=0.3, stretch=2.5, lurk=0.3),
    "bored":   dict(wander=0.7, hop_h=0.8, hop_dur=1.15, look=1.8, twitch=1.5, stretch=1.5, lurk=1.6),
    "jumpy":   dict(wander=2.0, hop_h=1.3, hop_dur=0.8, look=1.6, twitch=2.2, stretch=0.7, lurk=1.3),
    "annoyed": dict(wander=0.8, hop_h=0.9, hop_dur=0.75, look=0.7, twitch=1.8, stretch=0.5, lurk=0.2),
}


class MoodEngine:
    MOODS = ("content", "sleepy", "bored", "jumpy", "annoyed")

    SLEEP_IDLE_SECONDS = 90.0   # must be left alone this long before dozing off
    BOREDOM_FULL_SECONDS = 480.0
    POKE_ANNOYANCE = 0.25
    RUDE_WAKE_ANNOYANCE = 0.40
    ANNOYANCE_DECAY = 0.025     # per second
    MIN_DWELL = 15.0            # seconds before a non-annoyed mood may change

    def __init__(self):
        now = time.monotonic()
        self.energy = circadian_energy()
        self.boredom = 0.0
        self.annoyance = 0.0
        self.mood = "content"
        self.last_interaction = now
        self._mood_since = now
        self._forced_until = 0.0
        self._circ_target = self.energy
        self._circ_timer = 0.0

    # ---- events from the widget ----

    def note_interaction(self):
        self.last_interaction = time.monotonic()

    def note_poke(self):
        self.note_interaction()
        self.annoyance = min(1.0, self.annoyance + self.POKE_ANNOYANCE)

    def note_rude_wake(self):
        self.note_interaction()
        self.annoyance = min(1.0, self.annoyance + self.RUDE_WAKE_ANNOYANCE)

    def note_hop(self):
        self.energy = max(0.0, self.energy - 0.004)

    def force(self, mood, seconds=90.0):
        """Pin a mood for a while -- mostly for testing from the tray."""
        if mood in PROFILES:
            self.mood = mood
            self._mood_since = time.monotonic()
            self._forced_until = self._mood_since + seconds

    # ---- queries ----

    def profile(self):
        return PROFILES[self.mood]

    def idle_seconds(self):
        return time.monotonic() - self.last_interaction

    def wants_sleep(self):
        return self.mood == "sleepy" and self.idle_seconds() >= self.SLEEP_IDLE_SECONDS

    def should_wake(self):
        return self.mood != "sleepy"

    # ---- per-frame update ----

    def update(self, dt):
        now = time.monotonic()

        self._circ_timer += dt
        if self._circ_timer >= 30.0:
            self._circ_timer = 0.0
            self._circ_target = circadian_energy()
        self.energy += (self._circ_target - self.energy) * min(1.0, dt * 0.01)

        self.boredom = min(1.0, self.idle_seconds() / self.BOREDOM_FULL_SECONDS)
        self.annoyance = max(0.0, self.annoyance - dt * self.ANNOYANCE_DECAY)

        if now < self._forced_until:
            return

        new = self._derive()
        if new != self.mood:
            # annoyance can interrupt anything immediately; everything else
            # waits out a short dwell so moods don't flicker at thresholds
            if new == "annoyed" or self.mood == "annoyed" or now - self._mood_since >= self.MIN_DWELL:
                self.mood = new
                self._mood_since = now

    def _derive(self):
        if self.mood == "annoyed" and self.annoyance > 0.25:
            return "annoyed"
        if self.annoyance >= 0.55:
            return "annoyed"
        # hysteresis: once sleepy, stay sleepy until energy is clearly back up
        if self.energy < 0.28 or (self.mood == "sleepy" and self.energy < 0.36):
            return "sleepy"
        if self.energy > 0.62 and self.boredom > 0.45:
            return "jumpy"
        if self.boredom > 0.6:
            return "bored"
        return "content"