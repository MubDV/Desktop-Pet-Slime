import math
import random
import sys

from PySide6.QtCore import Qt, QTimer, QPointF, QElapsedTimer
from PySide6.QtGui import QPainter, QColor, QLinearGradient, QBrush, QPen, QIcon, QPixmap, QAction
from PySide6.QtWidgets import QApplication, QWidget, QSystemTrayIcon, QMenu

from slime import SlimeBody, Spring


class SlimeWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)

        self.margin = 40  # padding around the blob for shadow/jiggle overflow
        self.body = SlimeBody(size=140, color=(90, 200, 90))
        win_size = self.body.base_size + self.margin * 2
        self.resize(win_size, win_size)

        screen = QApplication.primaryScreen().availableGeometry()
        self.floor_y = screen.bottom() - win_size - 20  # "ground" it always falls back to
        self.move(screen.center().x(), self.floor_y)

        # hop state
        self.hopping = False
        self.hop_t = 0.0
        self.hop_duration = 0.55
        self.hop_start = QPointF()
        self.hop_end = QPointF()
        self.max_hop_height = 60
        self.facing = 1  # +1 = facing/showing right side, -1 = left
        self.visual_facing = 1  # what's currently rendered (lags facing during a turn)

        # fall state -- used after a drag-drop that leaves it above the floor
        self.falling = False
        self.fall_velocity = 0.0
        self.gravity = 2200.0  # px/s^2

        # turn: a pivot around the vertical axis, faked in 2D by shrinking
        # horizontal width toward a thin edge-on sliver then expanding back
        # out facing the new direction -- NOT a tilt/roll, which would look
        # like it's swinging its head sideways instead of turning around.
        self.turning = False
        self.turn_t = 1.0
        self.turn_duration = 0.22

        # idle: a tiny roll (rotation) for a subtle "breathing" wobble at rest only
        self.rotation = Spring(0.0, stiffness=200, damping=16)
        self.idle_time = 0.0

        # drag state
        self.dragging = False
        self.drag_offset = QPointF()
        self.last_mouse_pos = QPointF()
        self.press_pos = QPointF()
        self.last_drag_velocity = QPointF()

        # fixed-ish timestep loop, decoupled from any single animation
        self.elapsed = QElapsedTimer()
        self.elapsed.start()
        self.last_ns = self.elapsed.nsecsElapsed()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)  # ~60fps

        self.wander_timer = QTimer(self)
        self.wander_timer.timeout.connect(self.maybe_start_hop)
        self.wander_timer.start(2500)

    # ---------------- physics / animation loop ----------------

    def tick(self):
        now = self.elapsed.nsecsElapsed()
        dt = min((now - self.last_ns) / 1e9, 0.033)
        self.last_ns = now

        self.body.update(dt)

        if self.falling:
            self.fall_velocity += self.gravity * dt
            new_y = self.y() + self.fall_velocity * dt
            if new_y >= self.floor_y:
                new_y = self.floor_y
                impact = min(1.0, self.fall_velocity / 1400)  # harder fall = harder squash
                self.body.squash(1.15 + 0.35 * impact, 0.85 - 0.35 * impact)
                self.rotation.target = 0.0
                QTimer.singleShot(100 + int(80 * impact), self.body.relax)
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
                self.body.squish_y.target = max(1.0, stretch)
                self.body.squish_x.target = min(1.0, 2.0 - stretch)

            if landed:
                self.hopping = False
                self.body.squash(1.25, 0.72)
                QTimer.singleShot(90, self.body.relax)
        elif not self.dragging:
            # idle: slow gentle sway, not a hard 0 -- feels alive at rest
            self.idle_time += dt
            self.rotation.target = 4 * math.sin(self.idle_time * 1.3)

        if self.turning:
            self.turn_t += dt / self.turn_duration
            if self.turn_t >= 0.5 and self.visual_facing != self.facing:
                # edge-on crossing point -- flip which side is actually drawn
                self.visual_facing = self.facing
            if self.turn_t >= 1.0:
                self.turn_t = 1.0
                self.turning = False

        self.rotation.update(dt)
        self.update()

    def turn_scale(self):
        """Horizontal scale factor for the vertical-axis turn illusion.
        It's a cube, not a flat card -- there's always depth facing you,
        so width bottoms out around half, not near-zero. What sells the
        "it's a solid cube turning" read isn't the width alone, it's
        pairing this with a shading shift (see turn_side_shade)."""
        if not self.turning:
            return 1.0
        t = min(self.turn_t, 1.0)
        return 0.5 + 0.5 * abs(math.cos(math.pi * t))

    def turn_side_shade(self):
        """0..1 -- how much of the darker 'side face' shading to blend in.
        Peaks at the edge-on midpoint of the turn, fades out at either end,
        so the side face reads as coming into view rather than the whole
        blob just dimming uniformly."""
        if not self.turning:
            return 0.0
        t = min(self.turn_t, 1.0)
        return abs(math.sin(math.pi * t))

    def maybe_start_hop(self):
        if self.hopping or self.dragging or self.falling:
            return
        if random.random() < 0.55:
            self.start_hop()

    def start_hop(self):
        screen = QApplication.primaryScreen().availableGeometry()
        self.hop_start = QPointF(self.x(), self.floor_y)
        dx = random.uniform(-180, 180)
        target_x = min(max(self.x() + dx, screen.left()), screen.right() - self.width())
        self.hop_end = QPointF(target_x, self.floor_y)
        self.facing = 1 if target_x >= self.x() else -1
        if self.facing != self.visual_facing and not self.turning:
            self.turning = True
            self.turn_t = 0.0
        self.hop_t = 0.0
        self.hopping = True
        self.body.squash(0.75, 1.3)  # crouch/launch stretch

    # ---------------- painting ----------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        cx = self.width() / 2
        cy = self.height() / 2 + self.margin * 0.3

        # ground shadow -- shrinks while airborne
        shadow_scale = 1.0
        if self.hopping:
            height_frac = 4 * self.hop_t * (1 - self.hop_t)
            shadow_scale = 1.0 - 0.5 * height_frac
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 60))
        sw = self.body.base_size * 0.7 * shadow_scale
        painter.drawEllipse(QPointF(cx, cy + self.body.base_size * 0.42), sw / 2, sw / 5)

        painter.translate(cx, cy)
        painter.save()
        painter.rotate(self.rotation.value)
        painter.scale(self.turn_scale(), 1.0)

        path = self.body.build_path()
        c = self.body.color
        grad = QLinearGradient(0, -self.body.base_size / 2, 0, self.body.base_size / 2)
        grad.setColorAt(0.0, QColor(c.red(), c.green(), c.blue(), 235))
        grad.setColorAt(1.0, QColor(c.red(), c.green(), c.blue(), 170))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(c.red() // 2, c.green() // 2, c.blue() // 2, 200), 2))
        painter.drawPath(path)

        # side-face shading -- darkens as it turns edge-on, so the cube
        # reads as pivoting to show its side rather than just thinning out
        side_shade = self.turn_side_shade()
        if side_shade > 0.01:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, int(110 * side_shade)))
            painter.drawPath(path)

        # inner inset "core" -- the Minecraft-slime tell
        inner_path = self.body.build_path(scale=0.55)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(c.red(), c.green(), c.blue(), 90))
        painter.drawPath(inner_path)

        # specular highlight
        painter.setBrush(QColor(255, 255, 255, 90))
        hs = self.body.base_size * 0.18
        painter.drawEllipse(
            QPointF(-self.body.base_size * 0.18, -self.body.base_size * 0.22), hs, hs * 0.7
        )

        painter.restore()
        painter.end()

    # ---------------- mouse interaction ----------------

    def mousePressEvent(self, e):
        self.dragging = True
        gp = e.globalPosition()
        self.drag_offset = gp - QPointF(self.x(), self.y())
        self.last_mouse_pos = gp
        self.press_pos = gp
        self.body.squash(1.15, 0.85)

    def mouseMoveEvent(self, e):
        if not self.dragging:
            return
        gp = e.globalPosition()
        new_pos = gp - self.drag_offset
        vel = gp - self.last_mouse_pos
        self.last_mouse_pos = gp
        self.last_drag_velocity = vel  # px per tick, used to seed the fall/throw on release
        self.move(int(new_pos.x()), int(new_pos.y()))

        speed = math.hypot(vel.x(), vel.y())
        stretch = min(0.6, speed / 20)
        self.body.squish_y.target = max(0.6, 1.0 - stretch)
        self.body.squish_x.target = min(1.6, 1.0 + stretch)

    def mouseReleaseEvent(self, e):
        if not self.dragging:
            return
        self.dragging = False
        self.body.relax()

        gp = e.globalPosition()
        moved = math.hypot(gp.x() - self.press_pos.x(), gp.y() - self.press_pos.y())
        if moved < 6:
            # barely moved -> treat as a poke, not a drag
            local = e.position() - QPointF(self.width() / 2, self.height() / 2)
            self.body.poke(local.x(), local.y())
            self.body.squash(1.2, 0.78)
            QTimer.singleShot(120, self.body.relax)
            return

        # dropped from a drag -- if it's above the floor, gravity takes over
        # rather than it just staying wherever the mouse let go
        if self.y() < self.floor_y:
            self.falling = True
            # carry some of the fling's downward momentum into the fall,
            # so flinging it down hard lands harder than a gentle drop
            self.fall_velocity = max(0.0, self.last_drag_velocity.y() * 30)


def make_tray_icon(slime: SlimeWidget) -> QSystemTrayIcon:
    pix = QPixmap(32, 32)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(slime.body.color)
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(2, 2, 28, 28, 8, 8)
    painter.end()

    tray = QSystemTrayIcon(QIcon(pix))
    menu = QMenu()
    quit_action = QAction("Quit")
    quit_action.triggered.connect(QApplication.instance().quit)
    menu.addAction(quit_action)
    tray.setContextMenu(menu)
    tray.setToolTip("Desktop Slime")
    tray.show()
    return tray


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # keep running via tray even if window hides

    slime = SlimeWidget()
    slime.show()
    tray = make_tray_icon(slime)  # noqa: F841 -- keep reference alive

    sys.exit(app.exec())


if __name__ == "__main__":
    main()