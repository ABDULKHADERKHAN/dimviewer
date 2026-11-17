"""Inspection measurements built on the recognised features.

- relate():        distance, angle and Y14.5-style relations between two picked features
- symmetry():      how well two features are centred on a datum
- extras():        extra facts about one feature: draft, chamfer, pattern, slot, wall thickness
- patterns():      groups of equal holes or bosses (bolt circle, row, grid)
- slots():         straight slots made of two half-round ends
- thickness_map(): wall thickness at every triangle, for the colour map

No graphics in this file. All values are measured on the model as loaded: they are
nominal geometry, not an inspection of a real part.
"""
# Part of DimViewer. Copyright (c) 2026 Abdul Khader Khan.
# Free for noncommercial use. License and contact details are in README.md.

import math

import numpy as np

from model import DEG, DIA, axis_name, _circle2d

PAR_DEG = 0.5                     # features within this angle count as parallel / square


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _ang(c):
    """Angle in degrees from a cosine, safe against rounding."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(c)))))


def _radius(m, sel):
    """Radius of a picked round feature, or None."""
    if sel[0] == "pt":
        return None
    f = m.sfit(sel[1]) if sel[0] == "surf" else m.efit(sel[1])
    return f.get("r")


def _xyz(U, v):
    """Three offsets as text."""
    return "%s, %s, %s" % (U.one(abs(v[0])), U.one(abs(v[1])), U.one(abs(v[2])))


def _levels(x, tol):
    """Distinct values in x (values closer than tol are averaged into one), ascending."""
    x = np.sort(x)
    cut = np.flatnonzero(np.diff(x) > tol) + 1
    return np.array([g.mean() for g in np.split(x, cut)])


def _is_line(m, sel):
    return sel[0] == "edge" and m.efit(sel[1])["kind"] == "line"


# ----------------------------------------------------------------------------
# Two features: distances, angles, GD&T-style relations
# ----------------------------------------------------------------------------
def relate(m, a, b, U):
    """Text lines for feature b relative to feature a (a is the reference)."""
    ta, pa, da, La = m._geom(a)
    tb, pb, db, Lb = m._geom(b)
    par, near1 = math.sin(math.radians(PAR_DEG)), math.cos(math.radians(PAR_DEG))
    tiny = 1e-6 * m.diag
    w = pb - pa
    out = []
    if ta == "plane" and tb == "plane":
        dot = float(da @ db)
        c = min(1.0, abs(dot))
        Pb = m._verts(b)
        if c > near1:
            d = abs(float(w @ da))
            if d <= tiny:
                out.append("Coplanar faces (offset %s)" % U.L(d))
            else:
                what = "step height" if dot > 0 else ("thickness, material between" if float(w @ da) < 0
                                                      else "gap, open between")
                out.append("Normal distance, plane to plane: %s  (%s)" % (U.L(d), what))
            out.append("Angle between planes: %.2f%s  (parallel)" % (_ang(c), DEG))
            out.append("Parallelism to [1], measured: " + U.L(float(np.ptp(Pb @ da))))
            if d > tiny:
                mid = (pa + pb) / 2.0
                an = axis_name(da)
                out.append("Mid-plane (symmetry plane) at %s"
                           % ("%s = %s" % (an, U.L(mid["XYZ".index(an)])) if an else U.P(mid)))
        else:
            ang = _ang(dot)                              # between the outward normals, 0 to 180
            out.append("Angle between planes: %.2f%s  (included angle of the faces %.2f%s)"
                       % (min(ang, 180.0 - ang), DEG, 180.0 - ang, DEG))
            out.append("Normal distance, centre of [2] to plane [1]: " + U.L(abs(float(w @ da))))
            if c < par:
                n = db - (db @ da) * da
                n /= np.linalg.norm(n)
                out.append("Perpendicularity to [1], measured: " + U.L(float(np.ptp(Pb @ n))))
    elif {ta, tb} == {"plane", "axis"}:
        pn, pp = (da, pa) if ta == "plane" else (db, pb)
        ad, ap, AL = (db, pb, Lb) if ta == "plane" else (da, pa, La)
        s = min(1.0, abs(float(ad @ pn)))
        ang = math.degrees(math.asin(s))
        dist = abs(float((ap - pp) @ pn))
        if s < par:
            out.append("Axis parallel to plane, normal distance: " + U.L(dist))
            out.append("Angle between axis and plane: %.2f%s" % (ang, DEG))
            if AL:
                out.append("Parallelism of axis, measured: " + U.L(AL * s))
        elif s > near1:
            out.append("Axis perpendicular to plane (%.2f%s)" % (ang, DEG))
            out.append("Feature centre to plane, normal distance: " + U.L(dist))
            if AL:
                out.append("Perpendicularity of axis, measured: %s%s"
                           % (DIA, U.L(AL * math.sqrt(max(0.0, 1 - s * s)))))
        else:
            out.append("Angle between axis and plane: %.2f%s  (%.2f%s from square)"
                       % (ang, DEG, 90.0 - ang, DEG))
            out.append("Feature centre to plane, normal distance: " + U.L(dist))
    elif ta == "axis" and tb == "axis":
        c = min(1.0, abs(float(da @ db)))
        word = "edges" if (_is_line(m, a) and _is_line(m, b)) else "axes"
        out.append("Angle between %s: %.2f%s" % (word, _ang(c), DEG))
        if c > near1:
            off = w - (w @ da) * da
            dist = float(np.linalg.norm(off))
            ra, rb = _radius(m, a), _radius(m, b)
            limit = max(tiny * 10, 0.05 * min(ra, rb)) if (ra and rb) else tiny * 10
            if dist <= limit:
                ends = np.array([pb - db * Lb / 2.0, pb + db * Lb / 2.0]) - pa
                rad = np.linalg.norm(ends - np.outer(ends @ da, da), axis=1)
                out.append("Coaxial. Coaxiality of [2] to [1], measured: %s%s"
                           % (DIA, U.L(2.0 * float(rad.max()))))
                out.append("Concentricity, centre offset: " + U.L(dist))
                if ra and rb and abs(ra - rb) > tiny:
                    out.append("Radial distance between the two: " + U.L(abs(ra - rb)))
            else:
                out.append("Parallel %s, centre to centre (normal) distance: %s" % (word, U.L(dist)))
                out.append("Offset X, Y, Z: " + _xyz(U, off))
        else:
            cr = np.cross(da, db)
            gap = abs(float(w @ cr)) / float(np.linalg.norm(cr))
            out.append("Closest distance between the %s: %s%s"
                       % (word, U.L(gap), "  (they cross)" if gap <= tiny * 10 else ""))
    elif "plane" in (ta, tb):
        pn = da if ta == "plane" else db
        out.append("Normal distance, point to plane: " + U.L(abs(float(w @ pn))))
    elif "axis" in (ta, tb):
        ad = da if ta == "axis" else db
        out.append("Normal distance, point to axis: " + U.L(float(np.linalg.norm(w - (w @ ad) * ad))))
    else:
        out.append("Point to point distance: " + U.L(float(np.linalg.norm(w))))
        out.append("Offset X, Y, Z: " + _xyz(U, w))
        return out
    if "plane" not in (ta, tb):                          # a face centre is not a useful reference point
        out.append("Centre to centre, straight line: %s  (X, Y, Z: %s)"
                   % (U.L(float(np.linalg.norm(w))), _xyz(U, w)))
    if m._approx(a) or m._approx(b):                     # no form values for a stand-in direction
        out = [ln for ln in out if "measured" not in ln and "Mid-plane" not in ln]
    return out


def headline(m, a, b, U, shortest):
    """The three values every pair gets, as (label, value) text pairs: angle, shortest
    distance and normal distance. `shortest` is the result of Model.closest(a, b).
    Curved faces and edges use their average direction; those values are marked approx."""
    ta, pa, da, _ = m._geom(a)
    tb, pb, db, _ = m._geom(b)
    near1 = math.cos(math.radians(PAR_DEG))
    w = pb - pa
    ap = ", approx. (curved)" if (m._approx(a) or m._approx(b)) else ""
    if da is None or db is None:
        angle = "not defined (a point or a round body has no direction)"
    elif ta == tb == "plane":
        t = _ang(da @ db)
        angle = "%.2f%s between the faces%s" % (min(t, 180.0 - t), DEG, ap)
        if abs((180.0 - t) - min(t, 180.0 - t)) > 0.01:
            angle += "  (included %.2f%s)" % (180.0 - t, DEG)
    elif ta == tb:
        angle = "%.2f%s between the %s%s" % (_ang(abs(float(da @ db))), DEG,
                                           "edges" if (a[0] == b[0] == "edge") else "axes", ap)
    else:
        pn, ad = (da, db) if ta == "plane" else (db, da)
        angle = "%.2f%s between the axis and the face%s" % (math.degrees(math.asin(min(1.0, abs(float(ad @ pn))))), DEG, ap)

    if ta == tb == "plane":
        d = abs(float(w @ da))
        normal = "%s  %s%s" % (U.L(d), "face to face" if abs(float(da @ db)) > near1
                               else "middle of [2] to face [1]", ap)
    elif "plane" in (ta, tb):
        pn = da if ta == "plane" else db
        normal = "%s  %s to face%s" % (U.L(abs(float(w @ pn))),
                                    "axis centre" if "axis" in (ta, tb) else "point", ap)
    elif ta == tb == "axis":
        cr = np.cross(da, db)
        if abs(float(da @ db)) > near1:
            normal = "%s  axis to axis%s" % (U.L(float(np.linalg.norm(w - (w @ da) * da))), ap)
        else:
            normal = "%s  common perpendicular%s" % (U.L(abs(float(w @ cr)) / float(np.linalg.norm(cr))), ap)
    elif "axis" in (ta, tb):
        ad = da if ta == "axis" else db
        normal = "%s  point to axis%s" % (U.L(float(np.linalg.norm(w - (w @ ad) * ad))), ap)
    else:
        normal = "%s  centre to centre" % U.L(float(np.linalg.norm(w)))
    dist, p, q = shortest
    out = [("Angle", angle), ("Shortest distance", U.L(dist) + ("  (they touch)" if dist <= 1e-9 * m.diag else "")),
           ("Normal distance", normal)]
    if dist > 1e-9 * m.diag:
        out.insert(2, ("  along X, Y, Z", _xyz(U, q - p)))
    return out


def symmetry(m, datum, a, b, U):
    """Symmetry of the pair a, b about a datum plane or axis: where the middle of the pair
    lies relative to the datum. Returns text lines (empty if the three do not line up)."""
    td, pd, dd, _ = m._geom(datum)
    ta, pa, da, _ = m._geom(a)
    tb, pb, db, _ = m._geom(b)
    if ta != tb or ta == "point" or td == "point":
        return []
    near1 = math.cos(math.radians(PAR_DEG))
    if abs(float(da @ db)) < near1:
        return []
    mid = (pa + pb) / 2.0 - pd
    if ta == "plane":                                    # two parallel faces
        along = abs(float(dd @ da))
        if td == "plane" and along > near1:
            off = abs(float(mid @ da))
        elif td == "axis" and along < math.sin(math.radians(PAR_DEG)):
            off = abs(float(mid @ da))
        else:
            return []
    else:                                                # two parallel axes
        if td == "plane" and abs(float(dd @ da)) < math.sin(math.radians(PAR_DEG)):
            off = abs(float(mid @ dd))
        elif td == "axis" and abs(float(dd @ da)) > near1:
            off = float(np.linalg.norm(mid - (mid @ dd) * dd))
        else:
            return []
    return ["Symmetry of [1] and [2] about datum A, measured: %s  (middle is off by %s)"
            % (U.L(2.0 * off), U.L(off))]


# ----------------------------------------------------------------------------
# Patterns of equal round features
# ----------------------------------------------------------------------------
def _rounds(m):
    """Complete round features: cylinder faces that share an axis and radius are joined,
    so a hole stored as two half faces counts once. Returns a list of dicts."""
    cyl = [(i, m.sfit(i)) for i in range(len(m.regs))]
    cyl = [(i, f) for i, f in cyl if f["kind"] == "cyl"]
    if len(cyl) > 4000:
        return []
    tol = 1e-5 * m.diag
    out, used = [], set()
    for k, (i, f) in enumerate(cyl):
        if i in used:
            continue
        ids, span = [i], f["span"]
        for j, g in cyl[k + 1:]:
            if j in used or g["external"] != f["external"] or abs(g["r"] - f["r"]) > max(tol, 1e-3 * f["r"]):
                continue
            if abs(float(g["axis"] @ f["axis"])) < 0.99996:
                continue
            w = g["point"] - f["point"]
            if np.linalg.norm(w - (w @ f["axis"]) * f["axis"]) <= max(tol, 1e-3 * f["r"]):
                ids.append(j)
                span += g["span"]
        if span >= math.radians(350.0):
            used.update(ids)
            out.append(dict(ids=ids, r=f["r"], axis=f["axis"], point=f["point"], external=f["external"]))
    return out


def patterns(m):
    """Groups of equal round features with parallel axes. Each group is a dict:
    ids (surface numbers), n, r, external, kind (pair, row, circle, grid, group), plus sizes."""
    if hasattr(m, "_patterns"):
        return m._patterns
    m._patterns = out = []
    if m.lazy:
        return out
    rounds = _rounds(m)
    tol = 1e-5 * m.diag
    left = list(range(len(rounds)))
    while left:
        f = rounds[left[0]]
        grp = [k for k in left if rounds[k]["external"] == f["external"]
               and abs(rounds[k]["r"] - f["r"]) <= max(tol, 1e-3 * f["r"])
               and abs(float(rounds[k]["axis"] @ f["axis"])) > 0.99996]
        left = [k for k in left if k not in grp]
        if len(grp) < 2:
            continue
        ax = f["axis"]
        b1 = np.cross(ax, [1.0, 0, 0] if abs(ax[0]) < 0.9 else [0, 1.0, 0])
        b1 /= np.linalg.norm(b1)
        b2 = np.cross(ax, b1)
        C = np.array([rounds[k]["point"] for k in grp])
        q = np.c_[C @ b1, C @ b2]
        keep = [0]                                       # features stacked on one axis count once
        for k in range(1, len(q)):
            if np.linalg.norm(q[keep] - q[k], axis=1).min() > 10 * tol:
                keep.append(k)
        if len(keep) < 2:
            continue
        grp, q = [grp[k] for k in keep], q[keep]
        g = dict(ids=[rounds[k]["ids"][0] for k in grp], all_ids=[i for k in grp for i in rounds[k]["ids"]],
                 n=len(grp), r=f["r"], external=f["external"], axis=ax, kind="group")
        out.append(g)
        n = len(q)
        ptol = max(10 * tol, 1e-4 * float(np.ptp(q, axis=0).max()))
        if n == 2:
            g.update(kind="pair", pitch=float(np.linalg.norm(q[1] - q[0])))
            continue
        c = q.mean(0)
        u, s, vt = np.linalg.svd(q - c)
        t = np.sort((q - c) @ vt[0])
        if np.abs((q - c) @ vt[1]).max() <= ptol:        # all on one line
            steps = np.diff(t)
            g.update(kind="row", pitch=float(steps.mean()), even=bool(np.ptp(steps) <= 10 * ptol),
                     span=float(t[-1] - t[0]))
            continue
        cx, cy, rc = _circle2d(q - c)
        if rc is not None and np.abs(np.hypot(q[:, 0] - c[0] - cx, q[:, 1] - c[1] - cy) - rc).max() <= ptol:
            th = np.sort(np.arctan2(q[:, 1] - c[1] - cy, q[:, 0] - c[0] - cx))
            gaps = np.diff(np.append(th, th[0] + 2 * math.pi))
            if np.ptp(gaps) <= 1e-3:
                g.update(kind="circle", bc=2.0 * rc, step=math.degrees(float(gaps.mean())), even=True)
                continue
        e1 = q[np.argsort(np.linalg.norm(q - q[0], axis=1))[1]] - q[0]   # towards the nearest neighbour
        e1 /= np.linalg.norm(e1)
        e2 = np.array([-e1[1], e1[0]])
        xs, ys = _levels((q - q[0]) @ e1, 10 * ptol), _levels((q - q[0]) @ e2, 10 * ptol)
        if len(xs) * len(ys) == n and len(xs) > 1 and len(ys) > 1:
            px, py = np.diff(xs), np.diff(ys)
            if np.ptp(px) <= 20 * ptol and np.ptp(py) <= 20 * ptol:
                g.update(kind="grid", nx=len(xs), ny=len(ys), px=float(px.mean()), py=float(py.mean()))
                continue
        if rc is not None and np.abs(np.hypot(q[:, 0] - c[0] - cx, q[:, 1] - c[1] - cy) - rc).max() <= ptol:
            g.update(kind="circle", bc=2.0 * rc, even=False)
    return out


def pattern_text(g, U):
    """One line describing a pattern."""
    head = "%dX %s%s %s" % (g["n"], DIA, U.one(2 * g["r"]), "bosses" if g["external"] else "holes")
    k = g["kind"]
    if k == "pair":
        return "%s, centre to centre %s" % (head, U.L(g["pitch"]))
    if k == "row":
        if g["even"]:
            return "%s in a row, pitch %s (end to end %s)" % (head, U.L(g["pitch"]), U.L(g["span"]))
        return "%s in a row, uneven pitch (end to end %s)" % (head, U.L(g["span"]))
    if k == "circle":
        if g["even"]:
            return "%s on a %s%s bolt circle, %.2f%s apart" % (head, DIA, U.L(g["bc"]), g["step"], DEG)
        return "%s on a %s%s bolt circle, uneven spacing" % (head, DIA, U.L(g["bc"]))
    if k == "grid":
        return "%s in a %d x %d grid, pitch %s x %s" % (head, g["nx"], g["ny"], U.one(g["px"]), U.L(g["py"]))
    return "%s, no regular pattern found" % head


# ----------------------------------------------------------------------------
# Slots
# ----------------------------------------------------------------------------
def slots(m):
    """Straight slots: two equal half-round ends that face away from each other.
    Each is a dict: ids (the two end faces), width, centres, length, axis, external."""
    if hasattr(m, "_slots"):
        return m._slots
    m._slots = out = []
    if m.lazy:
        return out
    ends = []
    for i in range(len(m.regs)):
        f = m.sfit(i)
        if f["kind"] == "cyl" and not f["full"] and abs(math.degrees(f["span"]) - 180.0) <= 12.0:
            tc = m.V[m.F[m.regs[i]]].mean(1) - f["point"]
            tc -= np.outer(tc @ f["axis"], f["axis"])
            mid = tc.mean(0)
            ends.append((i, f, mid / max(np.linalg.norm(mid), 1e-300)))
    used = set()
    for k, (i, f, mi) in enumerate(ends):
        best = None
        for j, g, mj in ends[k + 1:]:
            if i in used or j in used or g["external"] != f["external"]:
                continue
            if abs(g["r"] - f["r"]) > 1e-3 * f["r"] or abs(float(g["axis"] @ f["axis"])) < 0.99996:
                continue
            w = g["point"] - f["point"]
            w = w - (w @ f["axis"]) * f["axis"]
            d = float(np.linalg.norm(w))
            if d <= 1e-3 * f["r"]:
                continue
            w /= d
            if float(mi @ w) < -0.9 and float(mj @ w) > 0.9 and (best is None or d < best[1]):
                best = (j, d)
        if best:
            used.update((i, best[0]))
            out.append(dict(ids=[i, best[0]], width=2 * f["r"], centres=best[1], length=best[1] + 2 * f["r"],
                            axis=f["axis"], external=f["external"]))
    return out


# ----------------------------------------------------------------------------
# Wall thickness
# ----------------------------------------------------------------------------
class _Grid:
    """Sorts the triangles into a 3D grid of boxes so that a ray only has to be tested
    against the triangles near its path, not against the whole model."""

    def __init__(self, m):
        T = m.V[m.F]
        lo, hi = T.min(1), T.max(1)
        size = np.maximum(m.hi - m.lo, 1e-9 * max(m.diag, 1e-300))
        cell = max((float(np.prod(size)) * 4.0 / len(T)) ** (1.0 / 3.0), float(size.max()) / 96.0)
        self.cell, self.org = cell, m.lo - cell
        self.n = np.floor((m.hi + cell - self.org) / cell).astype(np.int64) + 1
        a = np.floor((lo - 0.5 * cell - self.org) / cell).astype(np.int64)      # half a box of margin, see near()
        b = np.floor((hi + 0.5 * cell - self.org) / cell).astype(np.int64)
        a, b = np.clip(a, 0, self.n - 1), np.clip(b, 0, self.n - 1)
        d = b - a + 1
        cnt = d.prod(1)
        wide = cnt > 512                                 # a few huge triangles: test them for every ray
        self.always = np.flatnonzero(wide)
        tri = np.flatnonzero(~wide)
        a, d, cnt = a[tri], d[tri], cnt[tri]
        o = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        dj, dk = np.repeat(d[:, 1], cnt), np.repeat(d[:, 2], cnt)
        vi = np.repeat(a[:, 0], cnt) + o // (dj * dk)
        vj = np.repeat(a[:, 1], cnt) + (o // dk) % dj
        vk = np.repeat(a[:, 2], cnt) + o % dk
        box = (vi * self.n[1] + vj) * self.n[2] + vk
        order = np.argsort(box, kind="stable")
        self.tri = np.repeat(tri, cnt)[order]
        self.count = np.bincount(box, minlength=int(self.n.prod()))
        self.start = np.cumsum(self.count) - self.count
        self.reach = float(np.linalg.norm(m.hi - m.lo)) + 2 * cell

    def near(self, o, d):
        """Triangles that the ray from o along d could touch. The ray is sampled every half
        box; with the half-box margin used when filing the triangles, none can be missed."""
        t = np.arange(0.0, self.reach, 0.5 * self.cell)
        v = np.floor((o + t[:, None] * d - self.org) / self.cell).astype(np.int64)
        v = v[((v >= 0) & (v < self.n)).all(1)]
        box = np.unique((v[:, 0] * self.n[1] + v[:, 1]) * self.n[2] + v[:, 2])
        cnt = self.count[box]
        idx = np.repeat(self.start[box] - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))
        return np.concatenate([np.unique(self.tri[idx]), self.always])


def _first_hit(m, O, D, enter=False):
    """Distance from each start point O, along direction D, to the first place the ray
    leaves the part (inf if it never does). With enter=True the ray starts outside and
    the distance is to the first place it enters the part: the open gap in front of a face. Surfaces the ray only grazes (under about 12
    degrees) are skipped, otherwise every sharp corner would read as a thin wall."""
    V0 = m.V[m.F[:, 0]]
    E1, E2 = m.V[m.F[:, 1]] - V0, m.V[m.F[:, 2]] - V0
    out = np.full(len(O), np.inf)
    tmin = 1e-6 * m.diag
    grid = None
    if len(m.F) > 2000:
        if not hasattr(m, "_grid"):
            m._grid = _Grid(m)
        grid = m._grid
    for i in range(len(O)):
        o, d = O[i], D[i]
        c = grid.near(o, d) if grid else slice(None)
        e1, e2 = E1[c], E2[c]
        P = np.cross(d, e2)
        det = (P * e1).sum(1)
        ok = np.abs(det) > 1e-300
        inv = 1.0 / np.where(ok, det, 1.0)
        S = o - V0[c]
        u = (S * P).sum(1) * inv
        Q = np.cross(S, e1)
        v = (Q @ d) * inv
        t = (Q * e2).sum(1) * inv
        face = (m.N[c] @ d) * (-1.0 if enter else 1.0)
        hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > tmin) & (face > 0.2)
        if hit.any():
            out[i] = t[hit].min()
    return out


def _nearest(src, pts):
    """For each point in pts, the number of the nearest point in src."""
    try:
        from scipy.spatial import cKDTree
        return cKDTree(src).query(pts)[1]
    except ImportError:
        pass
    if len(src) > 400:                                   # no SciPy: thin the choice so it stays quick
        keep = np.linspace(0, len(src) - 1, 400).astype(np.int64)
        return keep[_nearest(src[keep], pts)]
    out = np.empty(len(pts), np.int64)
    step = max(1, int(4e6 // max(1, len(src))))
    for i in range(0, len(pts), step):
        out[i:i + step] = ((pts[i:i + step, None, :] - src[None]) ** 2).sum(2).argmin(1)
    return out


def _thickness_at(m, tris, corners=False, gap=False):
    """Wall thickness behind each given triangle: straight in from its middle, square to
    the surface, to where the material ends. corners=True also measures near each corner
    (four values per triangle), which matters on faces made of a few large triangles."""
    T = m.V[m.F[tris]]
    cen = T.mean(1)
    if gap:                                              # open space in front of the face
        return _first_hit(m, cen, m.N[tris], enter=True)
    if not corners:
        return _first_hit(m, cen, -m.N[tris])
    P = np.concatenate([cen] + [0.1 * cen + 0.9 * T[:, k] for k in range(3)])
    return _first_hit(m, P, -np.tile(m.N[tris], (4, 1)))


def face_thickness(m, rid):
    """Wall thickness behind surface rid: (thinnest, typical), or None. Sampled at up to 96 places."""
    key = ("thk", rid)
    cache = m.__dict__.setdefault("_thk", {})
    if key not in cache:
        tris = np.asarray(m.regs[rid])
        tris = tris[np.linspace(0, len(tris) - 1, min(24, len(tris))).astype(np.int64)]
        t = _thickness_at(m, tris, corners=True)
        t = t[np.isfinite(t)]
        cache[key] = (float(t.min()), float(np.median(t))) if len(t) else None
    return cache[key]


def thickness_map(m, most=8000, gap=False):
    """Wall thickness at every triangle. Models up to `most` triangles are measured at every
    triangle; on larger ones a spread of triangles is measured and the rest take the value of
    the nearest measured triangle on the same surface.
    Returns (thickness per triangle, number measured).
    gap=True measures the open space in front of each triangle instead (slot and channel
    widths); faces that look out into free space get NaN."""
    name = "_gmap" if gap else "_tmap"
    if hasattr(m, name):
        return getattr(m, name)
    nT = len(m.F)
    n = int(min(nT, most))
    order = np.argsort(m.labels, kind="stable")
    pick = np.unique(np.concatenate([order[np.linspace(0, nT - 1, n).astype(np.int64)],
                                     [int(np.asarray(r)[np.argmax(m.A[r])]) for r in m.regs]]))
    val = _thickness_at(m, pick, gap=gap)
    out = np.full(nT, np.nan)
    out[pick] = val
    if len(pick) < nT:
        cen = m.V[m.F].mean(1)
        lab = m.labels[pick]
        for rid, tris in enumerate(m.regs):
            tris = np.asarray(tris)
            src = pick[lab == rid]
            if len(src) == len(tris):
                continue
            out[tris] = out[src[_nearest(cen[src], cen[tris])]]
    out[~np.isfinite(out)] = np.nan
    setattr(m, name, (out, len(pick)))
    return getattr(m, name)


# ----------------------------------------------------------------------------
# One feature: extra facts
# ----------------------------------------------------------------------------
def chamfer(m, rid):
    """If flat face rid is a chamfer between two flat faces, returns
    (face width, neighbour 1, angle to it, neighbour 2, angle to it, leg 1, leg 2); else None.
    The legs are only given when the two neighbours are square to each other."""
    f = m.sfit(rid)
    if f["kind"] != "plane":
        return None
    cand = []
    for cid in m.rchains[rid]:
        e = m.efit(cid)
        pair = m.chain_pair[cid]
        other = pair[0] if pair[1] == rid else pair[1]
        if e["kind"] == "line" and other >= 0 and m.sfit(other)["kind"] == "plane":
            cand.append((e["length"], other, e))
    cand.sort(key=lambda c: -c[0])
    if len(cand) < 2:
        return None
    (L1, r1, e1), (L2, r2, e2) = cand[0], cand[1]
    if r1 == r2 or abs(float(e1["dir"] @ e2["dir"])) < 0.9999 or L2 < 0.7 * L1:
        return None
    w = e2["point"] - e1["point"]
    width = float(np.linalg.norm(w - (w @ e1["dir"]) * e1["dir"]))
    n, n1, n2 = f["normal"], m.sfit(r1)["normal"], m.sfit(r2)["normal"]
    a1, a2, a12 = _ang(n @ n1), _ang(n @ n2), _ang(n1 @ n2)
    if width > 0.6 * L1 or a12 < 20.0 or a12 > 160.0:
        return None
    if abs(a1 + a2 - a12) > 1.0 or min(a1, a2) < 5.0:    # the face must lie between its neighbours
        return None
    for r, nr in ((r1, n1), (r2, n2)):                   # and be clearly narrower than both of them
        across = np.cross(nr, e1["dir"])
        if width > 0.5 * float(np.ptp(m.V[np.unique(m.F[m.regs[r]])] @ across)):
            return None
    legs = (width * math.cos(math.radians(a1)), width * math.cos(math.radians(a2))) \
        if abs(a12 - 90.0) < PAR_DEG else (None, None)
    return width, r1, a1, r2, a2, legs[0], legs[1]


def extras(m, sel, U, thickness=True):
    """Extra text lines for one picked feature. thickness=False leaves out the wall
    thickness, which is the slow part when every surface of a large model is listed."""
    if sel[0] != "surf" or m.lazy:
        return []
    rid = sel[1]
    f = m.sfit(rid)
    out = []
    if f["kind"] == "plane":
        ch = chamfer(m, rid)
        if not ch and not axis_name(f["normal"]):
            tilt = [(math.degrees(math.asin(min(1.0, abs(float(f["normal"][i]))))), "XYZ"[i]) for i in range(3)]
            tilt = [t for t in tilt if 0.02 < t[0] <= 30.0]
            if tilt:
                d, ax = min(tilt)
                out.append("Draft angle: %.2f%s from the %s axis" % (d, DEG, ax))
        if ch:
            width, r1, a1, r2, a2, l1, l2 = ch
            if l1 is not None:
                out.append("Chamfer: %s x %s  (%.2f%s to S%d)" % (U.one(l1), U.L(l2), a1, DEG, r1))
            else:
                out.append("Chamfer: %.2f%s to S%d, %.2f%s to S%d" % (a1, DEG, r1, a2, DEG, r2))
            out.append("Chamfer face width: " + U.L(width))
    elif f["kind"] == "cone":
        out.append("Draft angle: %.2f%s per side" % (math.degrees(f["half"]), DEG))
    elif f["kind"] == "cyl":
        for g in patterns(m):
            if rid in g["all_ids"]:
                out.append("Pattern: " + pattern_text(g, U))
        for s in slots(m):
            if rid in s["ids"]:
                out.append("%s: width %s, length %s, end centres %s apart"
                           % ("Obround boss" if s["external"] else "Slot",
                              U.one(s["width"]), U.one(s["length"]), U.L(s["centres"])))
    if thickness and f["kind"] != "sphere":
        t = face_thickness(m, rid)
        if t:
            if abs(t[1] - t[0]) <= 1e-4 * m.diag:
                out.append("Wall thickness behind this face: " + U.L(t[0]))
            else:
                out.append("Wall thickness behind this face: thinnest %s, typical %s" % (U.one(t[0]), U.L(t[1])))
    return out
