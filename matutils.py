"""Small numpy-based matrix helpers. Matrices are built for column-vector
convention (v' = M @ v) and uploaded to GL with glUniformMatrix4fv(...,
transpose=GL_TRUE) since numpy is row-major -- see gl_slime.py's _set_mat4.
"""
import numpy as np


def identity():
    return np.eye(4, dtype=np.float32)


def translate(x, y, z):
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = [x, y, z]
    return m


def scale(sx, sy, sz):
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def rotate_y(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


def rotate_x(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    m = np.eye(4, dtype=np.float32)
    m[1, 1] = c
    m[1, 2] = -s
    m[2, 1] = s
    m[2, 2] = c
    return m


def rotate_z(deg):
    """Roll -- rotation around the depth axis, the one that was missing.
    This is what actually produces a "tilted head" diamond-lean on screen;
    rotate_x/rotate_y alone can't, since they rotate around axes that lie
    IN the screen plane rather than through it."""
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return m


def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, center, up):
    eye = np.array(eye, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = [-np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye)]
    return m


def mul(*mats):
    result = mats[0]
    for m in mats[1:]:
        result = result @ m
    return result 