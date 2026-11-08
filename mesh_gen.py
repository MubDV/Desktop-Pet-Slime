"""
Generates a rounded-box mesh -- six flat faces with edges/corners filleted
by a fixed radius (the standard "Minkowski sum of a box and a sphere"
construction). Unlike a superellipsoid, this has no singularities at any
radius -- it stays numerically well-behaved even at very small (boxy)
radii, which is what a low-roundness superellipsoid does NOT do (it
develops genuine cusps as the exponent gets small).
"""
import numpy as np

# (normal_axis, sign) for each of the 6 faces
_FACES = [(2, 1), (2, -1), (0, 1), (0, -1), (1, 1), (1, -1)]


def _round_point(p, r):
    inner = np.clip(p, -(1.0 - r), 1.0 - r)
    d = p - inner
    length = np.linalg.norm(d, axis=-1, keepdims=True)
    safe_len = np.where(length < 1e-9, 1.0, length)
    return inner + r * (d / safe_len)


def build_rounded_cube(resolution=240, roundness=0.28):
    """`roundness` here is the fillet radius as a fraction of the half-extent
    (0 = sharp cube, ~1 = sphere) -- same rough meaning as before, just a
    numerically robust construction underneath."""
    r = min(max(roundness, 0.02), 0.95)
    s = np.linspace(-1.0, 1.0, resolution)
    t = np.linspace(-1.0, 1.0, resolution)
    S, T = np.meshgrid(s, t, indexing="ij")
    rows = cols = resolution

    all_pos, all_norm, all_tris = [], [], []
    vert_offset = 0

    for axis, sign in _FACES:
        other = [a for a in range(3) if a != axis]
        p = np.zeros((rows, cols, 3), dtype=np.float64)
        p[..., other[0]] = S
        p[..., other[1]] = T
        p[..., axis] = float(sign)

        pos = _round_point(p, r)

        # normals via finite differences within this face's own grid
        d_s = np.gradient(pos, axis=0)
        d_t = np.gradient(pos, axis=1)
        n = np.cross(d_s, d_t)
        ref = np.zeros(3)
        ref[axis] = sign
        flip = np.sum(n * ref, axis=-1, keepdims=True) < 0
        n = np.where(flip, -n, n)
        n = n / np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9, None)

        flat_pos = pos.reshape(-1, 3)
        flat_norm = n.reshape(-1, 3)
        all_pos.append(flat_pos)
        all_norm.append(flat_norm)

        for i in range(rows - 1):
            for j in range(cols - 1):
                a = i * cols + j
                b = a + 1
                c = a + cols
                d = c + 1
                for tri in ((a, b, c), (b, d, c)):
                    # per-triangle winding correction: compute the actual
                    # geometric normal and flip if it doesn't point outward.
                    # Self-correcting regardless of face orientation, so
                    # there's no manual per-face winding guess to get wrong.
                    p0, p1, p2 = flat_pos[tri[0]], flat_pos[tri[1]], flat_pos[tri[2]]
                    face_n = np.cross(p1 - p0, p2 - p0)
                    if np.dot(face_n, ref) < 0:
                        tri = (tri[0], tri[2], tri[1])
                    all_tris.append((tri[0] + vert_offset, tri[1] + vert_offset, tri[2] + vert_offset))

        vert_offset += rows * cols

    verts = np.concatenate(all_pos, axis=0).astype(np.float32)
    normals = np.concatenate(all_norm, axis=0).astype(np.float32)
    indices = np.array(all_tris, dtype=np.uint32)
    return verts, normals, indices