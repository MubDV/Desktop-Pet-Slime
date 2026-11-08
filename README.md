# Desktop Slime -- Phase 1

Transparent, always-on-top blob that sits on your desktop, hops around on
its own, and reacts to drags/pokes with spring-physics squash and stretch.
No sprites -- the whole shape is 8 boundary points on damped springs,
redrawn every frame, so the motion stays fluid instead of stepping through
fixed poses.

## Run it

```
pip install -r requirements.txt
python main.py
```

There's also a 3D OpenGL version (rounded-cube mesh, shader-based, with the
mood engine wired in):

```
python gl_main.py
```

Quit via the tray icon (bottom-right system tray).

## What's in here

- `slime.py` -- `SlimeBody`: spring-driven rounded-square silhouette
  (Minecraft-slime shape, smoothed) with squash/stretch and per-point
  jiggle.
- `main.py` -- transparent frameless window, 60fps physics/render loop,
  autonomous hop-wander behavior, drag-to-fling, poke-to-jiggle, tray icon.
- `gl_main.py` -- `Slime3DWidget`: the OpenGL version. Same springs from
  `slime.py`, but rendered as a 3D rounded cube with GLSL shaders (incl. a
  twist deformation), plus the tray icon. Reads the mood engine to scale
  its behavior.
- `mood.py` -- `MoodEngine`: slow-moving internal state (time of day,
  neglect, pokes) -> moods (content, sleepy, bored, jumpy, annoyed). It
  doesn't move anything itself; it hands out rate multipliers and
  sleep/wake decisions.
- `mesh_gen.py` -- builds the rounded-box mesh (box + fillet radius, so no
  cusps at low roundness).
- `matutils.py` -- small numpy matrix helpers (translate/scale/rotate) for
  the GL side.

## Known rough edges (expected at this stage)

- Window is pinned to `availableGeometry()` of the primary screen only --
  multi-monitor wandering isn't handled yet.
- No taskbar-edge walking yet -- it just hops along one Y level.
- Color/size are hardcoded in `SlimeWidget.__init__` for now.

## Next phases (not built yet)

1. **Mood engine** -- *(built in `mood.py`, used by `gl_main.py`)*
   time-of-day + idle-duration + poke-frequency driving both physics feel
   (jumpy vs sleepy vs "don't fuck with me"). Still to do: hooking it into
   commentary tone.
2. **Window watcher** -- `pywin32` foreground window title polling.
3. **Commentary** -- Gemini Flash call (Google AI Studio key) behind a
   swappable provider interface, speech-bubble widget, local fallback
   lines.
4. **Split/merge** -- spawn 3-4 reduced-scale `SlimeWidget` instances that
   run their own hop loop for N seconds, then animate back to the origin
   and despawn into the original.
5. **Settings panel** -- tray-launched window for color/size, persisted to
   a small JSON config.

Say the word and we'll build whichever of these next.
