"""
SlimeBody: a rounded-square blob built from spring-driven boundary points.

Instead of frame-by-frame sprites, the silhouette is defined by 8 control
points arranged in a superellipse (rounded-square) shape. Each point can be
displaced by a damped spring, and the whole shape can be uniformly
squashed/stretched. Because springs are integrated every frame at 60fps
with continuous velocities, the blob never "snaps" between poses -- it
settles, overshoots, and wobbles the way actual gel would.
"""

import math
import random
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPainterPath, QColor


class Spring:
    """A damped spring that eases `value` toward `target`."""

    def __init__(self, value=0.0, stiffness=180.0, damping=14.0):
        self.value = value
        self.velocity = 0.0
        self.target = value
        self.stiffness = stiffness
        self.damping = damping

    def update(self, dt):
        force = (self.target - self.value) * self.stiffness - self.velocity * self.damping
        self.velocity += force * dt
        self.value += self.velocity * dt

    def kick(self, impulse):
        self.velocity += impulse


class SlimeBody:
    NUM_POINTS = 8
    SQUARENESS = 4  # superellipse exponent; higher = boxier, Minecraft-slime-like

    def __init__(self, size=140, color=(90, 200, 90)):
        self.base_size = size
        self.color = QColor(*color)

        # global squash/stretch (landing, drag velocity, launch crouch)
        self.squish_x = Spring(1.0, stiffness=220, damping=16)
        self.squish_y = Spring(1.0, stiffness=220, damping=16)

        # per-point jiggle offsets for the gooey secondary motion
        self.point_offsets = [
            Spring(0.0, stiffness=140, damping=10) for _ in range(self.NUM_POINTS)
        ]

    def squash(self, x_scale, y_scale):
        self.squish_x.target = x_scale
        self.squish_y.target = y_scale

    def relax(self):
        self.squish_x.target = 1.0
        self.squish_y.target = 1.0

    def poke(self, local_x, local_y):
        """Impulse the boundary points nearest the poke location, inward."""
        angle_hit = math.atan2(local_y, local_x)
        for i, sp in enumerate(self.point_offsets):
            point_angle = (i / self.NUM_POINTS) * 2 * math.pi - math.pi
            diff = abs((point_angle - angle_hit + math.pi) % (2 * math.pi) - math.pi)
            weight = max(0.0, 1.0 - diff / (math.pi / 2))
            sp.kick(-260 * weight)

    def jiggle_all(self, strength=180):
        for sp in self.point_offsets:
            sp.kick(random.uniform(-strength, strength))

    def update(self, dt):
        self.squish_x.update(dt)
        self.squish_y.update(dt)
        for sp in self.point_offsets:
            sp.update(dt)

    def _boundary_points(self, scale=1.0):
        half = self.base_size / 2 * scale
        pts = []
        n = self.SQUARENESS
        for i in range(self.NUM_POINTS):
            angle = (i / self.NUM_POINTS) * 2 * math.pi - math.pi
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            r = half / (abs(cos_a) ** n + abs(sin_a) ** n) ** (1 / n)
            r += self.point_offsets[i].value * scale
            px = cos_a * r * self.squish_x.value
            py = sin_a * r * self.squish_y.value
            pts.append(QPointF(px, py))
        return pts

    def build_path(self, scale=1.0):
        """Smooth closed path through the deformed boundary points."""
        pts = self._boundary_points(scale)
        n = len(pts)
        path = QPainterPath()
        start_mid = QPointF((pts[0].x() + pts[-1].x()) / 2, (pts[0].y() + pts[-1].y()) / 2)
        path.moveTo(start_mid)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            mid = QPointF((p0.x() + p1.x()) / 2, (p0.y() + p1.y()) / 2)
            path.quadTo(p0, mid)
        path.closeSubpath()
        return path
