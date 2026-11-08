import ctypes
import math
import random
import sys

import numpy as np
from OpenGL import GL as gl
from PySide6.QtCore import Qt, QTimer, QPointF, QElapsedTimer
from PySide6.QtGui import QSurfaceFormat, QIcon, QPixmap, QPainter, QColor, QAction, QImage, QPen
from PySide6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PySide6.QtOpenGLWidgets import QOpenGLWidget

import matutils as mu
from mesh_gen import build_rounded_cube
from mood import MoodEngine
from slime import Spring

VERT_SRC = """
#version 330 core
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform mat3 normal_mat;
uniform float twist;  // radians of extra rotation per unit of local height

out vec3 v_normal;
out vec3 v_world_pos;

void main() {
    // twist deformation: rotate each vertex around Y by an amount that
    // scales with its own height, so top and bottom spin oppositely and
    // the middle eases between -- a wring, not a rigid spin. Applied in
    // local space before the model matrix.
    float angle = twist * in_pos.y;
    float c = cos(angle);
    float s = sin(angle);
    vec3 twisted_pos = vec3(
        c * in_pos.x - s * in_pos.z,
        in_pos.y,
        s * in_pos.x + c * in_pos.z
    );
    vec3 twisted_normal = vec3(
        c * in_normal.x - s * in_normal.z,
        in_normal.y,
        s * in_normal.x + c * in_normal.z
    );

    vec4 world = model * vec4(twisted_pos, 1.0);
    v_world_pos = world.xyz;
    v_normal = normalize(normal_mat * twisted_normal);
    gl_Position = proj * view * world;
}
"""

FRAG_SRC = """
#version 330 core
in vec3 v_normal;
in vec3 v_world_pos;
out vec4 frag_color;

uniform vec3 base_color;
uniform vec3 light_dir;
uniform vec3 view_pos;
uniform float alpha;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(-light_dir);
    vec3 V = normalize(view_pos - v_world_pos);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 32.0);
    float rim = pow(1.0 - max(dot(N, V), 0.0), 2.0);

    vec3 color = base_color * 0.35            // ambient
                + base_color * diff * 0.7      // diffuse
                + vec3(1.0) * spec * 0.6       // specular highlight
                + base_color * rim * 0.5;      // rim light -- reads as translucent edge

    frag_color = vec4(color, alpha);
}
"""

DECAL_VERT_SRC = """
#version 330 core
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;

uniform mat4 mvp;
out vec2 v_uv;

void main() {
    v_uv = in_uv;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

DECAL_FRAG_SRC = """
#version 330 core
in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D tex;

void main() {
    frag_color = texture(tex, v_uv);
}
"""


class Slime3DWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)

        self.win_size = 230  # more than the object needs unrotated -- see
        # eye distance below, this is padding for rotated/stretched overflow
        self.resize(self.win_size, self.win_size)
        screen = QApplication.primaryScreen().geometry()  # full screen, not availableGeometry() --
        # availableGeometry() excludes the taskbar's reserved space even when
        # it's auto-hidden, which is why it was floating above the real edge
        self.floor_y = screen.bottom() - self.win_size
        self.ENTRY_DROP_HEIGHT = 46  # spawns this far above rest, then drops in
        self.move(screen.center().x(), self.floor_y - self.ENTRY_DROP_HEIGHT)

        self.color = (0.22, 0.72, 0.22)  # outer shell -- saturated, glassy
        self.inner_color = (0.62, 0.88, 0.52)  # inner core -- lighter, more solid
        self.inner_scale = 0.74
        self.roundness = 0.10  # flatter still -- more boxy, less pillowy

        # physics springs -- same scalar Spring class as the 2D version,
        # just driving real 3D scale/rotation now instead of a 2D path
        self.squish_x = Spring(1.0, 220, 16)
        self.squish_y = Spring(1.0, 220, 16)
        self.squish_z = Spring(1.0, 220, 16)
        # collapsed to nearly nothing from the very first frame (target
        # stays 1.0 from construction above) -- growth starts immediately,
        # so there's never a frame where it renders full-size before
        # shrinking, which was the actual source of the flicker
        self.squish_x.value = self.squish_y.value = self.squish_z.value = 0.04
        self.turn_angle = Spring(0.0, 90, 14)  # BODY turn -- only part of a look
        self.face_extra_turn = Spring(0.0, 150, 13)  # HEAD/face leads the rest of it,
        # applied only to the decal so it visibly turns further/faster than the body,
        # the way a person's head turns more than their torso for a quick look
        self.shift_x = Spring(0.0, 150, 14)  # tiny weight-shift, synced to turn_angle
        self.lean = Spring(0.0, 200, 16)  # subtle X-axis tilt, idle "breathing" --
        # also directly driven (bypassing spring easing) for the entry tumble, see below
        self.roll = Spring(0.0, 140, 13)  # Z-axis tilt -- the "leaning head" diamond
        # tilt from the lurk peek, distinct from lean (X) and turn_angle (Y)
        self.twist = Spring(0.0, 55, 6)  # wring deformation

        self.facing = 1
        self.idle_time = 0.0
        self.breathe_scale = 1.0  # slow idle expand/contract, layered on top of squash

        # hop state
        self.hopping = False
        self.hop_t = 0.0
        self.hop_duration = 0.55
        self.hop_start = QPointF()
        self.hop_end = QPointF()
        self.max_hop_height = 60

        # fall state
        self.falling = False
        self.fall_velocity = 0.0
        self.gravity = 2200.0
        # fastest a drop can realistically hit on THIS screen -- y=0 to the
        # floor. Used to scale landing-squash intensity so a short monitor
        # doesn't max out the squish on every drop the way a flat velocity
        # threshold did.
        self.max_fall_velocity = math.sqrt(2 * self.gravity * max(1, self.floor_y))

        # drag state
        self.dragging = False
        self.drag_offset = QPointF()
        self.last_mouse_pos = QPointF()
        self.press_pos = QPointF()
        self.last_drag_velocity = QPointF()

        # lurk state -- peeking diagonally out of a top screen corner
        self.lurking = False
        self.lurk_phase = None  # 'diving' | 'in' | 'hold' | 'out' | 'startled_out'
        self.lurk_t = 0.0
        self.lurk_start_x = 0.0
        self.lurk_start_y = 0.0
        self.lurk_target_x = 0.0
        self.lurk_target_y = 0.0
        self.lurk_offscreen_x = 0.0
        self.lurk_offscreen_y = 0.0
        self.lurk_side = 1
        self.lurk_hold_duration = 1.5
        self.dive_start_y = 0.0
        self.dive_end_y = 0.0

        # spawn state -- hopping up into view from below the screen edge,
        # used whenever it returns from a lurk
        self.spawning = False
        self.spawn_t = 0.0
        self.spawn_start_x = 0.0
        self.spawn_y_start = 0.0
        self.spawn_y_end = 0.0

        # entry state -- first launch only: spawns slightly above rest,
        # tumbles, and drops into place while growing from nothing. All
        # driven by direct value writes handled every tick from frame one
        # (see tick()/handled in the entering branch), never by moving an
        # already-shown window in a jump, which is what caused the flicker.
        self.entering = True
        self.entry_t = 0.0
        self.entry_duration = 0.55

        # mood / sleep -- see mood.py. `sleeping` is the visible behavior
        # state (like hopping/lurking); `mood` supplies rates and triggers;
        # face_kind picks which face texture paintGL draws.
        self.mood = MoodEngine()
        self.sleeping = False
        self.face_kind = "normal"

        self.elapsed = QElapsedTimer()
        self.elapsed.start()
        self.last_ns = self.elapsed.nsecsElapsed()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

        # self-rescheduling cycles with randomized delays instead of fixed
        # QTimer intervals -- a metronomic tick is the fastest way to make
        # something feel robotic, so every behavior loop below picks its
        # own next delay each time it fires
        QTimer.singleShot(random.randint(1200, 3000), self._wander_cycle)
        QTimer.singleShot(random.randint(2000, 4500), self._look_cycle)
        QTimer.singleShot(random.randint(1500, 3200), self._twitch_cycle)
        QTimer.singleShot(random.randint(14000, 30000), self._stretch_cycle)
        # lurking is meant to be rare -- long interval, low odds each check,
        # so it isn't guessable/rhythmic; averages out to roughly once every
        # few minutes rather than a fixed schedule
        QTimer.singleShot(random.randint(25000, 50000), self._lurk_cycle)


        self.squish_x.value = self.squish_y.value = self.squish_z.value = 0.04
        self.squish_x.velocity = self.squish_y.velocity = self.squish_z.velocity = 0.0
        self._squash(1.0, 1.0, 1.0)
        self.twist.kick(random.choice([-1, 1]) * 320)

    # ---------------- GL setup ----------------

    def initializeGL(self):
        # No depth test: this is one small translucent character, not a
        # scene needing real occlusion sorting. Draw order alone (outer,
        # inner, face) gives correct back-to-front layering here.
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        # cull back-facing triangles -- without this, the translucent
        # blending was showing the mesh's own interior back wall bleeding
        # through, which is almost certainly the "slice" artifact. If this
        # makes the slime vanish or look inside-out instead, the winding
        # guess was backwards -- flip GL_BACK to GL_FRONT below and it
        # should resolve immediately.
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)

        self.program = self._compile_program(VERT_SRC, FRAG_SRC)

        verts, normals, indices = build_rounded_cube(resolution=48, roundness=self.roundness)
        self.index_count = int(indices.size)
        interleaved = np.hstack([verts, normals]).astype(np.float32)

        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, gl.GL_STATIC_DRAW)

        ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

        stride = 6 * 4
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(12))
        gl.glEnableVertexAttribArray(1)

        gl.glBindVertexArray(0)

        self._setup_face_decal()

    def _compile_program(self, vsrc, fsrc):
        def compile_shader(src, stype):
            sh = gl.glCreateShader(stype)
            gl.glShaderSource(sh, src)
            gl.glCompileShader(sh)
            if not gl.glGetShaderiv(sh, gl.GL_COMPILE_STATUS):
                raise RuntimeError(gl.glGetShaderInfoLog(sh).decode())
            return sh

        vs = compile_shader(vsrc, gl.GL_VERTEX_SHADER)
        fs = compile_shader(fsrc, gl.GL_FRAGMENT_SHADER)
        prog = gl.glCreateProgram()
        gl.glAttachShader(prog, vs)
        gl.glAttachShader(prog, fs)
        gl.glLinkProgram(prog)
        if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
            raise RuntimeError(gl.glGetProgramInfoLog(prog).decode())
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        return prog

    def _setup_face_decal(self):
        """A small unlit textured quad -- the eyes+mouth -- parented to the
        inner cube's front face. Kept as a flat decal rather than trying to
        UV-map the face onto the curved mesh, which is simpler and matches
        how the reference model actually does it (a flat texture stuck on
        the front face)."""
        self.decal_program = self._compile_program(DECAL_VERT_SRC, DECAL_FRAG_SRC)

        quad = np.array(
            [
                # x,    y,    z,   u,   v
                -0.5, -0.5, 0.0, 0.0, 1.0,
                0.5, -0.5, 0.0, 1.0, 1.0,
                0.5, 0.5, 0.0, 1.0, 0.0,
                -0.5, 0.5, 0.0, 0.0, 0.0,
            ],
            dtype=np.float32,
        )
        quad_idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        self.decal_index_count = int(quad_idx.size)

        self.decal_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.decal_vao)

        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, quad.nbytes, quad, gl.GL_STATIC_DRAW)

        ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, quad_idx.nbytes, quad_idx, gl.GL_STATIC_DRAW)

        stride = 5 * 4
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(12))
        gl.glEnableVertexAttribArray(1)
        gl.glBindVertexArray(0)

        # one face texture per expression -- drawn once with QPainter and
        # uploaded as GL textures; paintGL picks one via self.face_kind
        self.decal_texes = {
            kind: self._make_face_texture(kind)
            for kind in ("normal", "sleepy", "closed", "annoyed")
        }

    def _make_face_texture(self, kind):
        size = 256
        img = QImage(size, size, QImage.Format_RGBA8888)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setPen(Qt.NoPen)
        dark = QColor(25, 60, 20, 255)
        p.setBrush(dark)
        eye = size * 0.14
        lx, rx = int(size * 0.27), int(size * 0.59)

        def eyes(y, h):
            p.drawRect(lx, int(y), int(eye), int(h))
            p.drawRect(rx, int(y), int(eye), int(h))

        if kind == "normal":
            eyes(size * 0.32, eye)
            p.drawRect(int(size * 0.42), int(size * 0.60), int(size * 0.16), int(size * 0.06))
        elif kind == "sleepy":
            # heavy lids: eyes squashed down from the top, small flat mouth
            h = eye * 0.45
            eyes(size * 0.32 + eye - h, h)
            p.drawRect(int(size * 0.44), int(size * 0.61), int(size * 0.12), int(size * 0.04))
        elif kind == "closed":
            eyes(size * 0.40, eye * 0.2)
            p.drawRect(int(size * 0.45), int(size * 0.62), int(size * 0.10), int(size * 0.04))
        elif kind == "annoyed":
            eyes(size * 0.35, eye * 0.7)
            pen = QPen(dark, size * 0.035)
            pen.setCapStyle(Qt.FlatCap)
            p.setPen(pen)
            # brows slanting down toward the middle, plus a flat frown
            p.drawLine(int(size * 0.25), int(size * 0.25), int(size * 0.43), int(size * 0.33))
            p.drawLine(int(size * 0.75), int(size * 0.25), int(size * 0.57), int(size * 0.33))
            p.setPen(Qt.NoPen)
            p.drawRect(int(size * 0.40), int(size * 0.63), int(size * 0.20), int(size * 0.045))
        p.end()

        tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        buf = bytes(img.constBits())
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, size, size, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, buf
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return tex

    def resizeGL(self, w, h):
        gl.glViewport(0, 0, w, h)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        aspect = max(1, self.width()) / max(1, self.height())
        proj = mu.perspective(35, aspect, 0.1, 20.0)
        # camera pulled back from 4.2 -> 5.7: a square turned 45 degrees has
        # a wider footprint than square-on (up to ~1.41x), and drag-stretch
        # squash can add another ~1.6x on top of that -- the old framing
        # only had ~1.32x headroom, so either alone could clip against the
        # window edge. win_size was enlarged to match so the object's
        # actual on-screen size is unchanged, it just has more invisible
        # padding around it now.
        eye = (0, 0.4, 5.7)
        # look straight at the true center -- no tilt. (Previously I aimed
        # the camera above center to push the object down in frame, but
        # that tilts the whole view instead of giving a clean linear shift,
        # which is why it under-delivered and still left a gap. Fixed below
        # by translating the OBJECT down in world space instead, which maps
        # to screen pixels exactly, not approximately.)
        view = mu.look_at(eye, (0, 0, 0), (0, 1, 0))

        body = mu.mul(
            mu.rotate_z(self.roll.value),
            mu.rotate_x(self.lean.value),
            mu.rotate_y(self.turn_angle.value),
            mu.scale(
                self.squish_x.value * self.breathe_scale,
                self.squish_y.value * self.breathe_scale,
                self.squish_z.value * self.breathe_scale,
            ),
        )
        # weight-shift (side-to-side) plus a fixed downward seat offset so
        # the object's resting bottom edge sits flush against the window's
        # bottom edge -- which is itself pinned to the real screen edge
        # (floor_y). Vertical overflow only ever happens upward (hop
        # stretch); landing squash compresses evenly toward the object's
        # own center, so its bottom edge never needed padding below it.
        SEAT_OFFSET = 0.62  # world units -- tuned for win_size=230, eye z=5.7.
        # Was 0.78, which pushed the object's own bottom edge past the
        # window's rendering bounds, clipping it against the window edge
        # itself (not the screen edge). Pulled back to leave a safety
        # margin -- it sits a little higher than flush, trading a small
        # visible gap for guaranteed no-clip.
        model = mu.mul(mu.translate(self.shift_x.value, -SEAT_OFFSET, 0), body)
        # not the exact inverse-transpose -- close enough for our mild
        # squash ranges, and much cheaper than a full normal-matrix inverse
        normal_mat = model[:3, :3]

        gl.glUseProgram(self.program)
        gl.glUniform1f(gl.glGetUniformLocation(self.program, "twist"), math.radians(self.twist.value))
        self._set_mat4("model", model)
        self._set_mat4("view", view)
        self._set_mat4("proj", proj)
        self._set_mat3("normal_mat", normal_mat)
        gl.glUniform3f(gl.glGetUniformLocation(self.program, "base_color"), *self.color)
        gl.glUniform3f(gl.glGetUniformLocation(self.program, "light_dir"), -0.4, -1.0, -0.3)
        gl.glUniform3f(gl.glGetUniformLocation(self.program, "view_pos"), *eye)
        gl.glUniform1f(gl.glGetUniformLocation(self.program, "alpha"), 0.6)  # outer: translucent shell

        gl.glBindVertexArray(self.vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self.index_count, gl.GL_UNSIGNED_INT, None)

        # inner cube: same mesh, scaled down, more opaque/lighter -- the
        # solid "core" the reference images show through the outer shell
        inner_model = mu.mul(model, mu.scale(self.inner_scale, self.inner_scale, self.inner_scale))
        self._set_mat4("model", inner_model)
        self._set_mat3("normal_mat", inner_model[:3, :3])
        gl.glUniform3f(gl.glGetUniformLocation(self.program, "base_color"), *self.inner_color)
        gl.glUniform1f(gl.glGetUniformLocation(self.program, "alpha"), 0.9)
        gl.glDrawElements(gl.GL_TRIANGLES, self.index_count, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        # face decal -- a flat textured quad stuck to the inner cube's front
        # face. It inherits the body's transform (moves/squashes with it)
        # but ALSO gets its own extra Y rotation on top -- the "head turns
        # further than the torso" effect -- applied before the push-to-front
        # translate so it swings/orbits around the front face convincingly
        # rather than just spinning in place
        decal_model = mu.mul(
            model,
            mu.rotate_y(self.face_extra_turn.value),
            mu.translate(0, -0.05 * self.inner_scale, self.inner_scale * 1.01),
            mu.scale(self.inner_scale * 0.92, self.inner_scale * 0.92, 1.0),
        )
        mvp = mu.mul(proj, view, decal_model)

        gl.glUseProgram(self.decal_program)
        loc = gl.glGetUniformLocation(self.decal_program, "mvp")
        gl.glUniformMatrix4fv(loc, 1, gl.GL_TRUE, mvp.astype(np.float32))
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.decal_texes[self.face_kind])
        gl.glUniform1i(gl.glGetUniformLocation(self.decal_program, "tex"), 0)

        # no culling here -- guessed at the quad's winding order and it's
        # not worth debugging blind; drawing both sides means worst case
        # you glimpse a mirrored face when it's turned away, which is a
        # minor visual nitpick, not a "the face is just gone" bug
        gl.glBindVertexArray(self.decal_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self.decal_index_count, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

    def _set_mat4(self, name, mat):
        loc = gl.glGetUniformLocation(self.program, name)
        gl.glUniformMatrix4fv(loc, 1, gl.GL_TRUE, mat.astype(np.float32))

    def _set_mat3(self, name, mat3):
        loc = gl.glGetUniformLocation(self.program, name)
        gl.glUniformMatrix3fv(loc, 1, gl.GL_TRUE, mat3.astype(np.float32))

    # ---------------- physics / animation loop ----------------

    def tick(self):
        now = self.elapsed.nsecsElapsed()
        dt = min((now - self.last_ns) / 1e9, 0.033)
        self.last_ns = now

        self.mood.update(dt)
        self._mood_step()

        # weight-shift tracks wherever the body is currently turned (not the
        # target -- the current eased value), so it lags in and settles out
        # with the same natural timing as the turn itself
        self.shift_x.target = self.turn_angle.value * 0.0016

        for s in (
            self.squish_x, self.squish_y, self.squish_z,
            self.turn_angle, self.face_extra_turn, self.shift_x,
            self.lean, self.roll, self.twist,
        ):
            s.update(dt)

        if self.entering:
            self.entry_t += dt / self.entry_duration
            t = min(self.entry_t, 1.0)

            # drops in from slightly above its resting spot
            drop_eased = 1 - (1 - t) ** 2
            y = (self.floor_y - self.ENTRY_DROP_HEIGHT) + self.ENTRY_DROP_HEIGHT * drop_eased
            self.move(int(self.x()), int(y))

            # cartoon tumble while it drops -- a direct value write, not
            # spring-eased, since a spring can't do a continuous
            # multi-hundred-degree spin. Slight overshoot past 360 so the
            # landing has a little correction wobble once the spring
            # retakes control, same language as every other landing here.
            roll_eased = 1 - (1 - t) ** 3
            self.lean.value = 370 * roll_eased

            if t >= 1.0:
                self.entering = False
                # normalize 370 down to its true equivalent angle (~10)
                # before handing back to the spring -- otherwise the spring
                # doesn't know 370 and 10 look identical, and physically
                # travels the whole 370-unit gap back to 0, which plays out
                # as almost one more full spin instead of a small wobble
                self.lean.value = ((self.lean.value + 180) % 360) - 180
                self.lean.velocity = 0.0
                self.lean.target = 0.0
                self._squash(1.15, 0.8, 1.15)  # landing bounce
                QTimer.singleShot(120, self._relax)

        elif self.spawning:
            self._update_spawn(dt)

        elif self.lurking:
            self._update_lurk(dt)

        elif self.falling:
            self.fall_velocity += self.gravity * dt
            new_y = self.y() + self.fall_velocity * dt
            if new_y >= self.floor_y:
                new_y = self.floor_y
                impact = min(1.0, self.fall_velocity / self.max_fall_velocity)
                self._squash(1.15 + 0.35 * impact, 0.85 - 0.35 * impact, 1.15 + 0.35 * impact)
                QTimer.singleShot(100 + int(80 * impact), self._relax)
                self.falling = False
                self.fall_velocity = 0.0
            self.move(int(self.x()), int(new_y))

        elif self.hopping:
            self.hop_t += dt / self.hop_duration
            landed = self.hop_t >= 1.0
            t = min(self.hop_t, 1.0)

            x = self.hop_start.x() + (self.hop_end.x() - self.hop_start.x()) * t
            base_y = self.hop_start.y() + (self.hop_end.y() - self.hop_start.y()) * t
            height = self.max_hop_height * 4 * t * (1 - t)
            self.move(int(x), int(base_y - height))

            if 0.05 < t < 0.5:
                stretch = 1.0 + 0.3 * (1 - abs(t - 0.25) * 4)
                self.squish_y.target = max(1.0, stretch)
                self.squish_x.target = min(1.0, 2.0 - stretch)
                self.squish_z.target = min(1.0, 2.0 - stretch)

            if landed:
                self.hopping = False
                self._squash(1.25, 0.72, 1.25)
                QTimer.singleShot(90, self._relax)
        elif not self.dragging:
            self.idle_time += dt
            if self.sleeping:
                # slumped, slow deep breathing; squash targets are re-asserted
                # every tick so a stray delayed _relax() can't un-slump it
                self._squash(1.08, 0.88, 1.08)
                self.lean.target = 5
                self.breathe_scale = 1.0 + 0.04 * math.sin(self.idle_time * 0.5)
            else:
                lean_amp, breathe_freq = {
                    "sleepy": (2, 0.6), "jumpy": (4, 1.6), "annoyed": (1, 1.2),
                }.get(self.mood.mood, (3, 0.9))
                self.lean.target = lean_amp * math.sin(self.idle_time * 1.3)
                # slow, subtle "breathing" -- a gentle 2.5% expand/contract,
                # separate from and on top of the interaction-driven squash
                self.breathe_scale = 1.0 + 0.025 * math.sin(self.idle_time * breathe_freq)
        else:
            self.breathe_scale = 1.0

        self.update()

    # ---------------- mood / sleep ----------------

    def _rate(self, key):
        return self.mood.profile()[key]

    def _delay(self, lo, hi, key):
        """Randomized delay (ms), shortened/lengthened by the mood's rate."""
        return int(random.randint(lo, hi) / max(0.2, self._rate(key)))

    def _grounded_idle(self):
        return not (self.hopping or self.dragging or self.falling or self.lurking or self.spawning or self.entering or self.sleeping)

    def _mood_step(self):
        if self.sleeping:
            if self.mood.should_wake():
                self._wake(startled=False)
        elif self.mood.wants_sleep() and self._grounded_idle():
            self._fall_asleep()

        if self.sleeping:
            self.face_kind = "closed"
        else:
            self.face_kind = {"sleepy": "sleepy", "annoyed": "annoyed"}.get(self.mood.mood, "normal")

    def _fall_asleep(self):
        self.sleeping = True
        self._set_look(0.0)
        self.roll.target = 0.0

    def _wake(self, startled):
        self.sleeping = False
        if startled:
            # poked awake: flinch + twist, and it holds a grudge (annoyance)
            self.mood.note_rude_wake()
            self._squash(1.28, 0.66, 1.28)
            self.twist.kick(random.choice([-1, 1]) * random.uniform(300, 480))
            QTimer.singleShot(100, self._relax)
        else:
            # woke up on its own: stretch
            self._squash(0.8, 1.38, 0.8)
            QTimer.singleShot(300, lambda: self._squash(1.12, 0.85, 1.12))
            QTimer.singleShot(480, self._relax)

    def force_sleep(self):
        """Tray debug: doze off right now (if it's free to)."""
        self.mood.force("sleepy", seconds=120.0)
        if not self.sleeping and self._grounded_idle():
            self._fall_asleep()

    def _squash(self, x, y, z):
        self.squish_x.target = x
        self.squish_y.target = y
        self.squish_z.target = z

    def _relax(self):
        self.squish_x.target = 1.0
        self.squish_y.target = 1.0
        self.squish_z.target = 1.0

    def _wander_cycle(self):
        busy = self.hopping or self.dragging or self.falling or self.lurking or self.spawning or self.entering or self.sleeping
        if not busy and random.random() < min(0.95, 0.55 * self._rate("wander")):
            self.start_hop()
        QTimer.singleShot(self._delay(1200, 4200, "wander"), self._wander_cycle)

    def _look_cycle(self):
        self.maybe_look_around()
        QTimer.singleShot(self._delay(1800, 5500, "look"), self._look_cycle)

    def _twitch_cycle(self):
        """Small restless movement well short of a full glance -- a tiny
        twist, sometimes, for nothing in particular. Keeps it from ever
        looking perfectly still between the bigger behaviors."""
        busy = self.hopping or self.dragging or self.falling or self.lurking or self.spawning or self.entering or self.sleeping
        if not busy and random.random() < min(0.9, 0.35 * self._rate("twitch")):
            self.twist.kick(random.uniform(-70, 70))
        QTimer.singleShot(self._delay(1400, 3200, "twitch"), self._twitch_cycle)

    def _stretch_cycle(self):
        """A rare full-body stretch, like it's been sitting too long."""
        busy = self.hopping or self.dragging or self.falling or self.lurking or self.spawning or self.entering or self.sleeping
        if not busy and random.random() < min(0.95, 0.6 * self._rate("stretch")):
            self._squash(0.8, 1.38, 0.8)
            QTimer.singleShot(280, lambda: self._squash(1.12, 0.85, 1.12))
            QTimer.singleShot(460, self._relax)
        QTimer.singleShot(self._delay(16000, 35000, "stretch"), self._stretch_cycle)

    def _lurk_cycle(self):
        """Rare peek-in from off-screen along a random side edge, at a
        random height -- not tied to the floor at all, so it can appear
        anywhere along the edge, not just down where it normally sits."""
        busy = self.hopping or self.dragging or self.falling or self.lurking or self.spawning or self.entering or self.sleeping
        if not busy and random.random() < min(0.5, 0.12 * self._rate("lurk")):
            self._start_lurk()
        QTimer.singleShot(self._delay(25000, 50000, "lurk"), self._lurk_cycle)

    def _start_lurk(self):
        """First phase: a visible dive off the bottom edge, disappearing
        downward -- not an invisible teleport. Once fully off-screen,
        _prepare_lurk_peek repositions it for a diagonal corner peek."""
        self.lurk_side = random.choice([-1, 1])  # -1 = left corner, 1 = right corner
        self.lurking = True
        self.lurk_phase = "diving"
        self.lurk_t = 0.0
        self.dive_start_y = self.y()
        screen = QApplication.primaryScreen().geometry()
        self.dive_end_y = screen.bottom() + self.win_size + 20
        self._squash(1.22, 0.68, 1.22)  # crouch down before rolling off the edge

    def _prepare_lurk_peek(self):
        """Now off-screen below and invisible -- reposition to a TOP corner
        and peek in diagonally (both X and Y move together), not just
        slide in from one side at a random height. Matches an actual
        corner-peek pose: poking out of the corner at an angle."""
        screen = QApplication.primaryScreen().geometry()
        side = self.lurk_side
        w = self.win_size
        peek_fraction = random.uniform(0.55, 0.75)

        # fully off-screen on both axes -- off the chosen side AND above the top
        off_x = screen.right() + 5 if side == 1 else screen.left() - w - 5
        off_y = screen.top() - w - 5

        # target: pokes diagonally out of that exact top corner
        target_x = screen.right() - int(w * peek_fraction) if side == 1 else screen.left() - int(w * (1 - peek_fraction))
        target_y = screen.top() - int(w * (1 - peek_fraction))

        self.move(int(off_x), int(off_y))
        self._relax()

        self.lurk_offscreen_x = off_x
        self.lurk_offscreen_y = off_y
        self.lurk_start_x = off_x
        self.lurk_start_y = off_y
        self.lurk_target_x = target_x
        self.lurk_target_y = target_y
        self.lurk_t = 0.0
        self.lurk_phase = "in"

        self._set_peek_look(side)

    # body only carries this fraction of a normal look; see _set_peek_look
    # for the very different split used while lurking
    BODY_TURN_SHARE = 0.4

    def _set_look(self, angle):
        angle = max(-90.0, min(90.0, angle))
        self.turn_angle.target = angle * self.BODY_TURN_SHARE
        self.face_extra_turn.target = angle * (1.0 - self.BODY_TURN_SHARE)

    def _set_peek_look(self, side):
        """The corner-peek look: face stays close to forward-facing (only a
        modest yaw, nothing like a normal glance's edge-on extremes) while
        a genuine roll (Z-axis tilt) does the actual diagonal "leaning out
        of the corner" work -- matching a real corner peek, where you'd
        still clearly see the face, just tilted."""
        body_angle = side * random.uniform(15, 32)
        self.turn_angle.target = body_angle
        face_total = side * random.uniform(4, 14)  # face stays close to forward
        self.face_extra_turn.target = face_total - body_angle
        self.roll.target = side * random.uniform(25, 40)
        self.twist.kick(math.copysign(random.uniform(120, 250), side))

    def _update_lurk(self, dt):
        if self.lurk_phase == "diving":
            self.lurk_t += dt / 0.35
            t = min(self.lurk_t, 1.0)
            eased = t * t  # accelerating down, like rolling off the edge
            y = self.dive_start_y + (self.dive_end_y - self.dive_start_y) * eased
            self.move(int(self.x()), int(y))
            if t >= 1.0:
                self._prepare_lurk_peek()

        elif self.lurk_phase == "in":
            self.lurk_t += dt / 0.45
            t = min(self.lurk_t, 1.0)
            eased = 1 - (1 - t) ** 3  # ease-out: fast start, gentle settle
            x = self.lurk_start_x + (self.lurk_target_x - self.lurk_start_x) * eased
            y = self.lurk_start_y + (self.lurk_target_y - self.lurk_start_y) * eased
            self.move(int(x), int(y))
            if t >= 1.0:
                self.lurk_phase = "hold"
                self.lurk_t = 0.0
                self.lurk_hold_duration = random.uniform(1.2, 2.4)

        elif self.lurk_phase == "hold":
            self.lurk_t += dt
            if self.lurk_t >= self.lurk_hold_duration:
                self.lurk_phase = "out"
                self.lurk_t = 0.0
                self.lurk_start_x = self.x()
                self.lurk_start_y = self.y()
                self.lurk_target_x = self.lurk_offscreen_x
                self.lurk_target_y = self.lurk_offscreen_y

        elif self.lurk_phase in ("out", "startled_out"):
            duration = 0.22 if self.lurk_phase == "startled_out" else 0.4
            self.lurk_t += dt / duration
            t = min(self.lurk_t, 1.0)
            eased = t * t  # ease-in: accelerating away, like retreating
            x = self.lurk_start_x + (self.lurk_target_x - self.lurk_start_x) * eased
            y = self.lurk_start_y + (self.lurk_target_y - self.lurk_start_y) * eased
            self.move(int(x), int(y))
            if t >= 1.0:
                self.lurking = False
                self.lurk_phase = None
                self._set_look(0.0)
                self.roll.target = 0.0
                screen = QApplication.primaryScreen().geometry()
                new_x = random.randint(screen.left(), screen.right() - self.win_size)
                self._start_spawn(x=new_x)

    def _startle_and_retreat(self):
        """Caught mid-peek: a quick flinch, then bail immediately at a
        faster, more panicked pace than the normal casual retreat."""
        self._squash(1.28, 0.66, 1.28)
        self.twist.kick(math.copysign(random.uniform(400, 600), self.lurk_side))
        QTimer.singleShot(90, self._relax)
        self.lurk_phase = "startled_out"
        self.lurk_t = 0.0
        self.lurk_start_x = self.x()
        self.lurk_start_y = self.y()
        self.lurk_target_x = self.lurk_offscreen_x
        self.lurk_target_y = self.lurk_offscreen_y

    def _start_spawn(self, x=None):
        """Hop up into view from below the screen edge -- used on first
        launch and whenever it returns from a lurk, instead of just
        appearing or teleporting."""
        screen = QApplication.primaryScreen().geometry()
        if x is None:
            x = random.randint(screen.left(), screen.right() - self.win_size)
        self.spawn_start_x = x
        self.spawn_y_start = screen.bottom() + self.win_size + 20
        self.spawn_y_end = self.floor_y
        self.spawn_t = 0.0
        self.spawning = True
        self.move(int(x), int(self.spawn_y_start))
        self._squash(0.72, 1.32, 0.72)  # launch crouch/stretch, like a hop takeoff

    def _update_spawn(self, dt):
        self.spawn_t += dt / 0.5
        t = min(self.spawn_t, 1.0)
        eased = 1 - (1 - t) ** 3
        y = self.spawn_y_start + (self.spawn_y_end - self.spawn_y_start) * eased
        self.move(int(self.spawn_start_x), int(y))

        if 0.1 < t < 0.6:
            stretch = 1.0 + 0.25 * (1 - abs(t - 0.3) * 3)
            self.squish_y.target = max(1.0, stretch)
            self.squish_x.target = min(1.0, 2.0 - stretch)
            self.squish_z.target = min(1.0, 2.0 - stretch)

        if t >= 1.0:
            self.spawning = False
            self._squash(1.22, 0.75, 1.22)  # landing squash
            QTimer.singleShot(90, self._relax)

    def maybe_look_around(self):
        """Idle glance: a quick, varied partial turn -- angle, hold time, and
        a touch of twist all randomized per glance so no two look the same.
        Clamped to +/-90 degrees so it never turns far enough to show its
        back (see _glance)."""
        if self.hopping or self.dragging or self.falling or self.lurking or self.spawning or self.entering or self.sleeping:
            return
        if random.random() < min(0.9, 0.4 * self._rate("look")):
            side = random.choice([-1, 1])
            angle = side * random.uniform(35, 85)
            self._glance(angle, hold_ms=random.randint(450, 950))

    def _glance(self, angle, hold_ms=650):
        """Turn to face `angle` degrees (clamped to +/-90), split between
        body and head, with a little twist along for the ride, then settle
        after hold_ms -- unless something else (a hop, a drag) takes over."""
        self._set_look(angle)
        if abs(angle) > 5:
            self.twist.kick(math.copysign(random.uniform(120, 320), angle))
        QTimer.singleShot(hold_ms, self._settle_look)

    def _settle_look(self):
        """After a glance: usually recenter, but not always -- sometimes it
        stays mildly turned like it's still half-interested in something,
        and occasionally it double-takes to the other side first. A robotic
        always-recenter-to-dead-ahead is exactly what this avoids."""
        if self.hopping or self.dragging or self.falling or self.lurking or self.spawning or self.entering or self.sleeping:
            return
        roll = random.random()
        if roll < 0.22:
            self._set_look(random.uniform(-22, 22))  # doesn't fully recenter
        elif roll < 0.35:
            other_side = -1 if self.turn_angle.target >= 0 else 1
            self._glance(other_side * random.uniform(30, 70), hold_ms=random.randint(350, 600))
        else:
            self._set_look(0.0)

    def start_hop(self):
        screen = QApplication.primaryScreen().geometry()
        self.hop_start = QPointF(self.x(), self.floor_y)
        dx = random.uniform(-180, 180)
        target_x = min(max(self.x() + dx, screen.left()), screen.right() - self.width())
        self.hop_end = QPointF(target_x, self.floor_y)

        # randomized per-hop so no two hops feel the same
        self.hop_duration = random.uniform(0.42, 0.72) * self._rate("hop_dur")
        self.max_hop_height = random.uniform(42, 88) * self._rate("hop_h")
        self.mood.note_hop()

        new_facing = 1 if target_x >= self.x() else -1
        self.facing = new_facing
        # a partial glance toward the hop direction, not a full turn-around --
        # varied each time, always within the +/-90 clamp so it's never
        # showing its back mid-hop
        self._glance(new_facing * random.uniform(40, 85), hold_ms=int(self.hop_duration * 1000) + 200)

        self.hop_t = 0.0
        self.hopping = True
        crouch = random.uniform(0.68, 0.82)
        self._squash(crouch, 2.0 - crouch, crouch)

    # ---------------- mouse interaction ----------------

    def mousePressEvent(self, e):
        if self.lurking:
            # caught mid-peek -- startled flinch, then retreat and reappear
            # via the normal spawn hop-up, rather than being dragged around
            # still half off-screen
            self._startle_and_retreat()
            return
        self.mood.note_interaction()
        if self.sleeping:
            # poked awake -- flinch, don't start a drag
            self._wake(startled=True)
            return
        self.dragging = True
        gp = e.globalPosition()
        self.drag_offset = gp - QPointF(self.x(), self.y())
        self.last_mouse_pos = gp
        self.press_pos = gp
        self._squash(1.15, 0.85, 1.15)

    def mouseMoveEvent(self, e):
        if not self.dragging:
            return
        gp = e.globalPosition()
        new_pos = gp - self.drag_offset
        vel = gp - self.last_mouse_pos
        self.last_mouse_pos = gp
        self.last_drag_velocity = vel

        # clamp to stay fully on-screen -- the mouse itself can go wherever,
        # but letting the WINDOW cross the monitor edge is what was clipping
        # it (nothing outside the physical display gets rendered, window or
        # not). Lower bound is the floor line rather than the screen bottom
        # so it can't be dragged deeper than its resting position.
        screen = QApplication.primaryScreen().geometry()
        clamped_x = min(max(int(new_pos.x()), screen.left()), screen.right() - self.width())
        clamped_y = min(max(int(new_pos.y()), screen.top()), self.floor_y)
        self.move(clamped_x, clamped_y)

        speed = math.hypot(vel.x(), vel.y())
        stretch = min(0.6, speed / 20)
        self.squish_y.target = max(0.6, 1.0 - stretch)
        self.squish_x.target = min(1.6, 1.0 + stretch)
        self.squish_z.target = min(1.6, 1.0 + stretch)

    def mouseReleaseEvent(self, e):
        if not self.dragging:
            return
        self.dragging = False
        self._relax()

        gp = e.globalPosition()
        moved = math.hypot(gp.x() - self.press_pos.x(), gp.y() - self.press_pos.y())
        if moved < 6:
            self.mood.note_poke()
            self._squash(1.2, 0.78, 1.2)
            QTimer.singleShot(120, self._relax)
            return

        if self.y() < self.floor_y:
            self.falling = True
            self.fall_velocity = max(0.0, self.last_drag_velocity.y() * 30)


def make_tray_icon(slime: Slime3DWidget) -> QSystemTrayIcon:
    pix = QPixmap(32, 32)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(90, 200, 90))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(2, 2, 28, 28, 8, 8)
    p.end()

    tray = QSystemTrayIcon(QIcon(pix))
    menu = QMenu()
    # debug: pin a mood from the tray so you don't have to wait for the clock
    mood_menu = menu.addMenu("Mood (debug)")
    for name in MoodEngine.MOODS:
        act = QAction(name.capitalize(), mood_menu)
        act.triggered.connect(lambda _=False, n=name: slime.mood.force(n))
        mood_menu.addAction(act)
    sleep_act = QAction("Sleep now", mood_menu)
    sleep_act.triggered.connect(slime.force_sleep)
    mood_menu.addAction(sleep_act)

    quit_action = QAction("Quit")
    quit_action.triggered.connect(QApplication.instance().quit)
    menu.addAction(quit_action)
    tray.setContextMenu(menu)
    tray.setToolTip("Desktop Slime (3D)")
    tray.show()
    return tray


def main():
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setAlphaBufferSize(8)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    slime = Slime3DWidget()
    slime.show()
    tray = make_tray_icon(slime)  # noqa: F841 -- keep reference alive

    # ---------------- mood test cases ----------------
    # Uncomment ONE at a time. Each pins that mood for 1 hour (instead of the
    # tray's 90s) so it overrides the clock/idle/poke-derived mood while you
    # watch it. The 1500ms delay lets the entry animation finish first.

    # --- CONTENT: baseline, rates all 1.0x, normal face ---
    # QTimer.singleShot(1500, lambda: slime.mood.force("content", seconds=3600))

    # --- SLEEPY: slower/lower hops, rare twitches, half-lidded face. ---
    # (it only actually dozes off after 90s of no clicks -- use SLEEPING below
    # to skip the wait)
    # QTimer.singleShot(1500, lambda: slime.mood.force("sleepy", seconds=3600))

    # --- SLEEPING: sleepy + immediately dozes off (slumped, closed eyes, slow
    # breathing). Click it to test the startled wake-up. ---
    # QTimer.singleShot(1500, slime.force_sleep)

    # --- BORED: more glancing/twitching, sluggish hops, more frequent lurks ---
    # QTimer.singleShot(1500, lambda: slime.mood.force("bored", seconds=3600))

    # --- JUMPY: ~2x wander rate, taller/faster hops, lots of twitching ---
    # QTimer.singleShot(1500, lambda: slime.mood.force("jumpy", seconds=3600))

    # --- ANNOYED: angry-brow face, quick snappy hops, twitchy, hides less ---
    # QTimer.singleShot(1500, lambda: slime.mood.force("annoyed", seconds=3600))

    # --- ANNOYED via real triggers (no forcing): poke it ~3 times quickly,
    # or click it while it's asleep (after SLEEPING above) -- that's the
    # "rude wake-up" path and should flip it to annoyed within a couple clicks.

    sys.exit(app.exec())


if __name__ == "__main__":
    main()